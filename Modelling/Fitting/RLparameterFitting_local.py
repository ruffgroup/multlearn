from random import *
import os
import glob
import numpy as np
import scipy.stats
import scipy.io
import pandas as pd
import sys
import pathlib
from pathlib import Path
import platform
import ast
from RLparameterPlotting import Plotting
import argparse

sys.path.append(sys.path[0] + "/..")
from TaskDesign import task_Design


class Fitting:
    def __init__(
        self,
        mainTrials,
        additionalTrials,
        gridCount,
        ID,
        plotting=False,
        ww=5,
        method="valid",
        reps=50, 
        extra=False, 
        dyna=False, 
        init=False, 
        asym=False, 
        transfer=False, 
        dyna_init=False,
        pearce=False,
        pearce_init=False
    ):
        """
        extra: False or True for separate alpha for V1
        dyna: False or True for dynamic learning rate model
        pearce: False or True for Pearce Hall implementation
        init: False or True for initial V0 and V1
        asym : False or True for separate alphas based on reward
        transfer: False or True indicating the amount of discount free parameters for pairs that share a stimulus (depends on extra and asym)
        pearce_init: False or True for initial omega
        """
        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.gridCount = gridCount
        self.ID = ID
        self.statLearnPar = 1
        self.plotting = plotting
        self.extra = extra
        self.asym = asym
        self.transfer = transfer
        self.dyna = dyna
        self.gridCount = gridCount
        self.dyna_init = dyna_init
        self.v_init = init
        self.pearce = pearce
        self.pearce_init = pearce_init

        if self.plotting:
            self.Plot = Plotting(
                self.mainTrials,
                self.additionalTrials,
                self.gridCount,
                self.ID,
                ww=ww,
                method=method,
                reps=reps,
            )

        wanted_dir = "/mnt/d/data/sourcedata/behavior/modified_files"

        # Get savedVals file
        self.savedValsFile = glob.glob(
            os.path.abspath(wanted_dir) + "/*{}_savedValues.csv".format(self.ID)
        )[0]
        self.subjectData = pd.read_csv(self.savedValsFile)
        self.subjectData["stimulusPair"] = self.subjectData["stimulusPair"].apply(
            ast.literal_eval
        )

        if not self.dyna and not self.dyna_init and not self.extra and not self.asym and not self.transfer and not self.v_init:
            self.modelFolder = "basic"
            self.params = ["alpha", "beta"]
        elif self.dyna and not self.dyna_init and not self.extra and not self.asym and not self.transfer and not self.v_init:
            self.modelFolder = "dyna"
            self.params = ["alpha", "beta"]
        elif not self.dyna and not self.dyna_init and not self.extra and self.asym and not self.transfer and not self.v_init:
            self.modelFolder = "asym"
            self.params = ["pos alpha", "beta", "neg alpha"]
        elif not self.dyna and not self.dyna_init and not self.extra and not self.asym and self.transfer and not self.v_init:
            self.modelFolder = "transfer"
            self.params = ["alpha", "beta", "K1"]
        elif self.v_init and not self.dyna and not self.dyna_init and not self.extra and not self.asym and not self.transfer:
            self.modelFolder = "v_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init"]
        elif self.dyna and not self.dyna_init and not self.extra and self.asym and not self.transfer and not self.v_init:
            self.modelFolder = "dynaAsym"
            self.params = ["pos alpha", "beta", "neg alpha"]
        elif self.dyna and not self.dyna_init and not self.extra and not self.asym and self.transfer and not self.v_init:
            self.modelFolder = "dynaTransfer"
            self.params = ["alpha", "beta", "K1"]

        elif not self.dyna and not self.dyna_init and not self.extra and not self.asym and self.transfer and self.v_init:
            self.modelFolder = "transferV_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init", "K1"]
        elif self.dyna and not self.dyna_init and not self.extra and self.asym and not self.transfer and self.v_init:
            self.modelFolder = "dynaAsym"
            self.params = ["pos alpha", "beta", "neg alpha", "V0_init", "V1_init"]
        
    ##

    def modelFitting(
        self
    ):
        print("ID: "+str(self.ID))
        bestFitting_dir = "/mnt/d/data/fittedParametersRecoveredModels"
        newPath = os.path.join(
            bestFitting_dir, "sub-{0}".format(self.ID), self.modelFolder
        )
       
        Path(newPath).mkdir(parents=True, exist_ok=True)

        (
            fitted_alphasPos,
            fitted_alphasNeg,
            fitted_alphas2Pos,
            fitted_alphas2Neg,
            fitted_K1,
            fitted_K2,
            fitted_K3,
            fitted_K4,
            fitted_betas,
            fitted_omegas,
            best_LLs,
        ) = (np.empty((max(self.subjectData.runNumber))) for i in range(11))
        (
            fitted_alphasPos[:],
            fitted_alphasNeg[:],
            fitted_alphas2Pos[:],
            fitted_alphas2Neg[:],
            fitted_K1[:],
            fitted_K2[:],
            fitted_K3[:],
            fitted_K4[:],
            fitted_betas[:],
            fitted_omegas[:],
            best_LLs[:],
        ) = (np.nan for i in range(11))

        fitted_V_option0Inits, fitted_V_option1Inits = (
            np.empty((max(self.subjectData.runNumber), 3, 3)) for i in range(2)
        )

        fitted_V_option0Inits[:], fitted_V_option1Inits[:] = (np.nan for i in range(2))

        NLL_array = np.empty((max(self.subjectData.runNumber), self.gridCount, 12))
        NLL_array[:] = np.nan

        (
            trial_alphasPos,
            trial_alphasNeg,
            trial_alphas2Pos,
            trial_alphas2Neg,
            RPE,
        ) = (
            np.empty(
                (
                    max(self.subjectData.runNumber),
                    self.mainTrials + self.additionalTrials,
                )
            )
            for i in range(5)
        )

        V0 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials,
            )
        )
        V1 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials,
            )
        )
        (
            trial_alphasPos[:],
            trial_alphasNeg[:],
            trial_alphas2Pos[:],
            trial_alphas2Neg[:],
            RPE[:],
            V0[:],
            V1[:],
        ) = (np.nan for i in range(7))

        for run in range(0, max(self.subjectData.runNumber)):
            print("run: "+str(run))
            alphaGrid = np.random.rand(self.gridCount)
            if self.extra and self.asym:
                alphaNegGrid, alpha2PosGrid, alpha2NegGrid = (
                    np.random.rand(self.gridCount) for i in range(3)
                )
            elif self.extra and not self.asym:
                alpha2Grid = np.random.rand(self.gridCount)
            elif self.asym and not self.extra:
                alphaNegGrid = np.random.rand(self.gridCount)

            if self.transfer:
                K1Grid = np.random.rand(self.gridCount)
                if self.extra and self.asym:
                    K3Grid, K4Grid, K2Grid = (
                        np.random.rand(self.gridCount) for i in range(3)
                    )
                elif self.extra and not self.asym:
                    K2Grid = np.random.rand(self.gridCount)
                elif self.asym and not self.extra:
                    K3Grid = np.random.rand(self.gridCount)

            if self.dyna_init:
                omegaGrid = np.random.rand(self.gridCount)

            betaGrid = 0 + 14.0 * np.random.rand(self.gridCount)
            LL_array = np.empty((self.gridCount, 1))
            LL_array[:] = np.nan
            if self.v_init:
                V_option0_randGrid = np.random.rand(self.gridCount, 1)
                V_option0Init_Grid = np.repeat(V_option0_randGrid, 9, axis=1).reshape(
                    (self.gridCount, 3, 3)
                )

                V_option1_randGrid = np.random.rand(self.gridCount, 1)
                V_option1Init_Grid = np.repeat(V_option1_randGrid, 9, axis=1).reshape(
                    (self.gridCount, 3, 3)
                )

            runData = self.subjectData[
                self.subjectData.runNumber == run + 1
            ].reset_index()
            (
                run_trialwise_alphasPos,
                run_trialwise_alphasNeg,
                run_trialwise_alphas2Pos,
                run_trialwise_alphas2Neg,
                run_RPEs,
            ) = (
                np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
                for i in range(5)
            )
            run_V0, run_V1 = (
                np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
                for i in range(2)
            )

            (
                run_trialwise_alphasPos[:],
                run_trialwise_alphasNeg[:],
                run_trialwise_alphas2Pos[:],
                run_trialwise_alphas2Neg[:],
                run_RPEs[:],
                run_V0[:],
                run_V1[:],
            ) = (np.nan for i in range(7))

            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridCount):
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((max(runData.trialNumber), 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((max(runData.trialNumber)))
                actionProb[:] = np.nan
                V_option0, V_option1 = (
                    np.empty((max(runData.trialNumber) + 1, 3, 3)) for i in range(2)
                )
                V_option0[:], V_option1[:] = (np.nan for i in range(2))
                if self.v_init:
                    V_option0[0, :] = V_option0Init_Grid[j]
                    V_option1[0, :] = V_option1Init_Grid[j]
                else:
                    V_option0[0, :] = 0.5
                    V_option1[0, :] = 0.5

                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                if self.extra and not self.asym:
                    alphaPosCheck = alphaNegCheck = alphaGrid[j]
                    alpha2PosCheck = alpha2NegCheck = alpha2Grid[j]
                elif self.asym and not self.extra:
                    alphaPosCheck = alpha2PosCheck = alphaGrid[j]
                    alphaNegCheck = alpha2NegCheck = alphaNegGrid[j]
                elif self.extra and self.asym:
                    alphaPosCheck = alphaGrid[j]
                    alpha2PosCheck = alpha2PosGrid[j]
                    alphaNegCheck = alphaNegGrid[j]
                    alpha2NegCheck = alpha2NegGrid[j]
                else:
                    alphaPosCheck = (
                        alpha2PosCheck
                    ) = alphaNegCheck = alpha2NegCheck = alphaGrid[j]

                if self.transfer:
                    if self.extra and not self.asym:
                        K1Check = K3Check = K1Grid[j]
                        K2Check = K4Check = K2Grid[j]
                    elif self.asym and not self.extra:
                        K1Check = K2Check = K1Grid[j]
                        K3Check = K4Check = K3Grid[j]
                    elif self.asym and self.extra:
                        K1Check = K1Grid[j]
                        K2Check = K2Grid[j]
                        K3Check = K3Grid[j]
                        K4Check = K4Grid[j]
                    else:
                        K1Check = K2Check = K3Check = K4Check = K1Grid[j]

                betaCheck = betaGrid[j]
                # run_V0[j, 0] = V_option0[0, 0, 0]
                # run_V1[j, 0] = V_option1[0, 0, 0]

                if self.dyna:
                    omega = omega2 = omega3 = omega4 = 0
                if self.dyna_init:
                    omega = omega2 = omega3 = omega4 = omegaGrid[j]

                for t in range(0, max(runData.trialNumber)):
                    otherPairs = [
                        p
                        for p in list(runData.stimulusPair.unique())
                        if bool(p[0] == runData.stimulusPair[t][0])
                        ^ bool(p[1] == runData.stimulusPair[t][1])
                    ]

                    # Prob of choosing the 0th and 1st option respectively
                    choiceProb[t, 0] = np.exp(
                        betaCheck * V_option0[((t,) + runData.stimulusPair[t])]
                    ) / (
                        (
                            np.exp(
                                betaCheck * V_option0[((t,) + runData.stimulusPair[t])]
                            )
                        )
                        + (
                            np.exp(
                                betaCheck * V_option1[((t,) + runData.stimulusPair[t])]
                            )
                        )
                    )
                    choiceProb[t, 1] = 1 - choiceProb[t, 0]

                    actionProb[t] = (
                        choiceProb[t, int(runData.action[t])]
                        if ~np.isnan(runData.action[t])
                        else np.nan
                    )

                    if runData.action[t] == 0:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t]
                            - V_option0[(t,) + runData.stimulusPair[t]]
                        )
                        V_option0[t + 1, :] = V_option0[t, :]

                        if runData.reward[t] == 1:
                            if self.dyna or self.dyna_init:
                                omega = (
                                    omega
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega
                                    )
                                    * alphaPosCheck
                                )
                                V_option0[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                                run_trialwise_alphasPos[j, t] = omega
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K1Check * omega * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )
                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + alphaPosCheck * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )

                                run_trialwise_alphasPos[j, t] = alphaPosCheck

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K1Check * alphaPosCheck * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            if self.dyna or self.dyna_init:
                                if not self.asym and not self.extra:
                                    omega2 = omega3 = omega4 = omega
                                elif self.asym and not self.extra:
                                    omega2 = omega
                                elif self.extra and not self.asym:
                                    omega3 = omega

                        else:
                            if self.dyna or self.dyna_init:
                                omega3 = (
                                    omega3
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega3
                                    )
                                    * alphaNegCheck
                                )
                                V_option0[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega3 * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                                run_trialwise_alphasNeg[j, t] = omega3
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K3Check * omega3 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + alphaNegCheck * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )

                                run_trialwise_alphasNeg[j, t] = alphaNegCheck

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K3Check * alphaNegCheck * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0

                            if self.dyna or self.dyna_init:
                                if not self.asym and not self.extra:
                                    omega2 = omega = omega4 = omega3
                                elif self.asym and not self.extra:
                                    omega4 = omega3
                                elif self.extra and not self.asym:
                                    omega = omega3

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif runData.action[t] == 1:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t]
                            - V_option1[(t,) + runData.stimulusPair[t]]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]

                        if runData.reward[t] == 1:
                            if self.dyna or self.dyna_init:
                                omega2 = (
                                    omega2
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega2
                                    )
                                    * alpha2PosCheck
                                )
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega2 * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                                run_trialwise_alphas2Pos[j, t] = omega2
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K2Check * omega2 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alpha2PosCheck * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )

                                run_trialwise_alphas2Pos[j, t] = alpha2PosCheck

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K2Check * alpha2PosCheck * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if self.dyna or self.dyna_init:
                                if not self.asym and not self.extra:
                                    omega3 = omega = omega4 = omega2
                                elif self.asym and not self.extra:
                                    omega = omega2
                                elif self.extra and not self.asym:
                                    omega4 = omega2

                        else:
                            if self.dyna or self.dyna_init:
                                omega4 = (
                                    omega4
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega4
                                    )
                                    * alpha2NegCheck
                                )
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega4 * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                                run_trialwise_alphas2Neg[j, t] = omega4
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K4Check * omega4 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alpha2NegCheck * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )

                                run_trialwise_alphas2Neg[j, t] = alpha2NegCheck

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K4Check * alpha2NegCheck * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if self.dyna or self.dyna_init:
                                if not self.asym and not self.extra:
                                    omega2 = omega = omega3 = omega4
                                elif self.asym and not self.extra:
                                    omega3 = omega4
                                elif self.extra and not self.asym:
                                    omega2 = omega4

                        V_option0[t + 1, :] = V_option0[t, :]
                    else:
                        V_option1[t + 1, :] = V_option1[t, :]
                        V_option0[t + 1, :] = V_option0[t, :]

                    run_RPEs[j, t] = rewardPE[(t,) + runData.stimulusPair[t]]
                    run_V0[j, t] = V_option0[(t,) + runData.stimulusPair[t]]
                    run_V1[j, t] = V_option1[(t,) + runData.stimulusPair[t]]
                negativeLogLikelihood = -np.sum(
                    np.log(actionProb[~np.isnan(actionProb)])
                )
                Likelihood = np.prod(actionProb[~np.isnan(actionProb)])
                NLL_array[run, j, 0] = negativeLogLikelihood
                NLL_array[run, j, 1] = alphaPosCheck
                NLL_array[run, j, 2] = betaCheck
                if self.asym and not self.extra:
                    NLL_array[run, j, 3] = alphaNegCheck
                elif self.extra and not self.asym:
                    NLL_array[run, j, 4] = alpha2PosCheck
                elif self.extra and self.asym:
                    NLL_array[run, j, 3] = alphaNegCheck
                    NLL_array[run, j, 4] = alpha2PosCheck
                    NLL_array[run, j, 5] = alpha2NegCheck
                if self.dyna_init:
                    NLL_array[run, j, 6] = omegaGrid[j]
                if self.v_init:
                    NLL_array[run, j, 7] = V_option0Init_Grid[j][0][0]
                    NLL_array[run, j, 8] = V_option1Init_Grid[j][0][0]
                if self.transfer:
                    NLL_array[run, j, 9] = K1Check
                    if self.extra and not self.asym:
                        NLL_array[run, j, 10] = K2Check
                    elif self.asym and not self.extra:
                        NLL_array[run, j, 11] = K3Check
                    elif self.asym and self.extra:
                        NLL_array[run, j, 10] = K2Check
                        NLL_array[run, j, 11] = K3Check
                        NLL_array[run, j, 12] = K4Check

                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 0])
            maxIndex = np.nanargmax(LL_array[:, 0])
            fitted_alphasPos[run] = NLL_array[run, minIndex, 1]
            np.savetxt(
                fname=newPath + "/alphasPos_" + self.modelFolder + ".tsv",
                X=fitted_alphasPos,
                delimiter=",",
            )
            if self.asym and not self.extra:
                fitted_alphasNeg[run] = NLL_array[run, minIndex, 3]
                np.savetxt(
                    fname=newPath + "/alphasNeg_" + self.modelFolder + ".tsv",
                    X=fitted_alphasNeg,
                    delimiter=",",
                )
            elif self.extra and not self.asym:
                fitted_alphas2Pos[run] = NLL_array[run, minIndex, 4]
                np.savetxt(
                    fname=newPath + "/alphas2Pos_" + self.modelFolder + ".tsv",
                    X=fitted_alphas2Pos,
                    delimiter=",",
                )
            elif self.extra and self.asym:
                fitted_alphasNeg[run] = NLL_array[run, minIndex, 3]
                np.savetxt(
                    fname=newPath + "/alphasNeg_" + self.modelFolder + ".tsv",
                    X=fitted_alphasNeg,
                    delimiter=",",
                )
                fitted_alphas2Pos[run] = NLL_array[run, minIndex, 4]
                np.savetxt(
                    fname=newPath + "/alphas2Pos_" + self.modelFolder + ".tsv",
                    X=fitted_alphas2Pos,
                    delimiter=",",
                )
                fitted_alphas2Neg[run] = NLL_array[run, minIndex, 5]
                np.savetxt(
                    fname=newPath + "/alphas2Neg_" + self.modelFolder + ".tsv",
                    X=fitted_alphas2Neg,
                    delimiter=",",
                )
            fitted_betas[run] = NLL_array[run, minIndex, 2]
            np.savetxt(
                fname=newPath + "/betas_" + self.modelFolder + ".tsv",
                X=fitted_betas,
                delimiter=",",
            )
            if self.v_init:
                fitted_V_option0Inits[run] = NLL_array[run, minIndex, 7]
                fitted_V_option1Inits[run] = NLL_array[run, minIndex, 8]

            if self.dyna_init:
                fitted_omegas[run] = NLL_array[run, minIndex, 6]
            if self.transfer:
                fitted_K1[run] = NLL_array[run, minIndex, 9]
                np.savetxt(
                    fname=newPath + "/K1_" + self.modelFolder + ".tsv", X=fitted_K1, delimiter=","
                )
                if self.extra and not self.asym:
                    fitted_K2[run] = NLL_array[run, minIndex, 10]
                    np.savetxt(
                        fname=newPath + "/K2_" + self.modelFolder + ".tsv",
                        X=fitted_K2,
                        delimiter=",",
                    )
                elif self.asym and not self.extra:
                    fitted_K3[run] = NLL_array[run, minIndex, 11]
                    np.savetxt(
                        fname=newPath + "/K3_" + self.modelFolder + ".tsv",
                        X=fitted_K3,
                        delimiter=",",
                    )
                elif self.extra and self.asym:
                    fitted_K2[run] = NLL_array[run, minIndex, 10]
                    np.savetxt(
                        fname=newPath + "/K2_" + self.modelFolder + ".tsv",
                        X=fitted_K2,
                        delimiter=",",
                    )
                    fitted_K3[run] = NLL_array[run, minIndex, 11]
                    np.savetxt(
                        fname=newPath + "/K3_" + self.modelFolder + ".tsv",
                        X=fitted_K3,
                        delimiter=",",
                    )
                    fitted_K4[run] = NLL_array[run, minIndex, 12]
                    np.savetxt(
                        fname=newPath + "/K4_" + self.modelFolder + ".tsv",
                        X=fitted_K4,
                        delimiter=",",
                    )
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]
            trial_alphasPos[run] = run_trialwise_alphasPos[minIndex]
            trial_alphasNeg[run] = run_trialwise_alphasNeg[minIndex]
            trial_alphas2Pos[run] = run_trialwise_alphas2Pos[minIndex]
            trial_alphas2Neg[run] = run_trialwise_alphas2Neg[minIndex]

        if self.plotting:
            self.Plot.plots_modelFitting(
                NLL_array=NLL_array,
                alphasPos=fitted_alphasPos,
                alphas2Pos=fitted_alphas2Pos,
                alphasNeg=fitted_alphasNeg,
                alphas2Neg=fitted_alphas2Neg,
                betas=fitted_betas,
                omega=fitted_omegas,
                V_option0Inits=fitted_V_option0Inits,
                V_option1Inits=fitted_V_option1Inits,
                K1=fitted_K1,
                K2=fitted_K2,
                K3=fitted_K3,
                K4=fitted_K4,
                saveAs=self.modelFolder,
                extra=self.extra,
                asym=self.asym,
                transfer=self.transfer,
                dyna=self.dyna,
                dyna_init=self.dyna_init
            )

        scipy.io.savemat(
            newPath + "/rpe" + self.modelFolder + ".mat".format(self.ID),
            mdict={"rpe": RPE},
        )

        with open(newPath + "/BIC_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, best_LLs)

        with open(newPath + "/V0_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, V0)

        with open(newPath + "/V1_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, V1)

        
        with open(newPath + "/trialwise_alphasPos_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, trial_alphasPos)
        scipy.io.savemat(
            newPath + "/trialwise_alphasPos_" + self.modelFolder + ".mat".format(self.ID),
            mdict={"trialwise_alphasPos": trial_alphasPos},
        )

        with open(newPath + "/trialwise_alphasNeg_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, trial_alphasNeg)
        scipy.io.savemat(
            newPath + "/trialwise_alphasNeg_" + self.modelFolder + ".mat".format(self.ID),
            mdict={"trialwise_alphasNeg": trial_alphasNeg},
        )

        with open(newPath + "/trialwise_alphas2Pos_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, trial_alphas2Pos)
        scipy.io.savemat(
            newPath + "/trialwise_alphas2Pos_" + self.modelFolder + ".mat".format(self.ID),
            mdict={"trialwise_alphas2Pos": trial_alphas2Pos},
        )

        with open(newPath + "/trialwise_alphas2Neg_" + self.modelFolder + ".npy", "wb") as f:
            np.save(f, trial_alphas2Neg)
        scipy.io.savemat(
            newPath + "/trialwise_alphas2Neg_" + self.modelFolder + ".mat".format(self.ID),
            mdict={"trialwise_alphas2Neg": trial_alphas2Neg},
        )


    # Statistical learning
    def statisticalLearning(self, statLearnPar=1):
        self.statLearnPar = statLearnPar

        self.subjectData = pd.read_csv(self.savedValsFile)
        self.subjectData["stimulusPair"] = self.subjectData["stimulusPair"].apply(
            ast.literal_eval
        )
        ID_beliefs = np.empty((6, self.mainTrials + self.additionalTrials + 1, 3, 3))
        ID_surprise = np.empty((6, self.mainTrials + self.additionalTrials))
        for run in range(0, max(self.subjectData.runNumber)):
            rowBeliefs = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
            rowBeliefs[:] = np.nan
            columnBeliefs = np.empty(
                (self.mainTrials + self.additionalTrials + 1, 3, 3)
            )
            columnBeliefs[:] = np.nan
            beliefsStat = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
            beliefsStat[:] = np.nan
            statCount = np.zeros((self.mainTrials + self.additionalTrials + 1, 3, 3))
            statSurprise = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
            statSurprise[:] = np.nan
            statSurpriseRow = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
            statSurpriseRow[:] = np.nan
            statSurpriseColumn = np.empty(
                (self.mainTrials + self.additionalTrials, 3, 3)
            )
            statSurpriseColumn[:] = np.nan

            runData = self.subjectData[
                self.subjectData.runNumber == run + 1
            ].reset_index()
            rowDen0 = 3 * self.statLearnPar
            rowDen1 = 3 * self.statLearnPar
            rowDen2 = 3 * self.statLearnPar
            columnDen0 = 3 * self.statLearnPar
            columnDen1 = 3 * self.statLearnPar
            columnDen2 = 3 * self.statLearnPar

            for i in range(0, self.mainTrials + self.additionalTrials):
                statCount[i + 1, :] = statCount[i, :]
                statCount[
                    i + 1, runData.stimulusPair[i][0], runData.stimulusPair[i][1]
                ] += 1

            rowBeliefs[0, :] = (
                (self.statLearnPar + statCount[0, :]) / 3 * self.statLearnPar
            )
            columnBeliefs[0, :] = (
                (self.statLearnPar + statCount[0, :]) / 3 * self.statLearnPar
            )
            num = self.statLearnPar + statCount[0, :]
            den = 9 * self.statLearnPar
            # Total statistical beliefs irrespective of rows and columns.
            beliefsStat[0, :] = num / den

            for i in range(1, self.mainTrials + self.additionalTrials + 1):
                rowBeliefs[i, :] = rowBeliefs[i - 1, :]
                # Row beliefs
                if runData.stimulusPair[i - 1][0] == 0:
                    rowDen0 = rowDen0 + 1
                    rowBeliefs[i, 0, :] = (
                        self.statLearnPar + statCount[i, 0, :]
                    ) / rowDen0
                elif runData.stimulusPair[i - 1][0] == 1:
                    rowDen1 = rowDen1 + 1
                    rowBeliefs[i, 1, :] = (
                        self.statLearnPar + statCount[i, 1, :]
                    ) / rowDen1
                else:
                    rowDen2 = rowDen2 + 1
                    rowBeliefs[i, 2, :] = (
                        self.statLearnPar + statCount[i, 2, :]
                    ) / rowDen2
                # column beliefs
                columnBeliefs[i, :] = columnBeliefs[i - 1, :]
                if runData.stimulusPair[i - 1][1] == 0:
                    columnDen0 = columnDen0 + 1
                    columnBeliefs[i, 0, :] = (
                        self.statLearnPar + statCount[i, 0, :]
                    ) / columnDen0
                elif runData.stimulusPair[i - 1][1] == 1:
                    columnDen1 = columnDen1 + 1
                    columnBeliefs[i, 1, :] = (
                        self.statLearnPar + statCount[i, 1, :]
                    ) / columnDen1
                else:
                    columnDen2 = columnDen2 + 1
                    columnBeliefs[i, 2, :] = (
                        self.statLearnPar + statCount[i, 2, :]
                    ) / columnDen2
                num = self.statLearnPar + statCount[i, :]
                den += 1
                # Total statistical beliefs irrespective of rows and columns.
                beliefsStat[i, :] = num / den

            ID_beliefs[run, :, :] = beliefsStat
            trial_surprise = np.empty((self.mainTrials + self.additionalTrials))
            # Surprises calculated from beliefs update
            for i in range(0, self.mainTrials + self.additionalTrials):
                statSurpriseRow[i, :] = np.nan
                statSurpriseRow[(i,) + runData.stimulusPair[i]] = -np.log(
                    rowBeliefs[(i,) + runData.stimulusPair[i]]
                )
                statSurpriseColumn[i, :] = np.nan
                statSurpriseColumn[(i,) + runData.stimulusPair[i]] = -np.log(
                    columnBeliefs[(i,) + runData.stimulusPair[i]]
                )
                statSurprise[i, :] = np.nan
                statSurprise[(i,) + runData.stimulusPair[i]] = -np.log(
                    beliefsStat[(i,) + runData.stimulusPair[i]]
                )
                trial_surprise[i] = statSurprise[(i,) + runData.stimulusPair[i]]
            ID_surprise[run] = trial_surprise


        bestFitting_dir = "/mnt/d/data/fittedParametersRecoveredModels/bestFittingVals"
        newPath = os.path.join(
            bestFitting_dir, "sub-{0}".format(self.ID)
        )
       
        Path(newPath).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(
            newPath + "/spe.mat".format(self.ID), mdict={"spe": ID_surprise}
        )
        
        with open(newPath + "/spe.npy", "wb") as f:
            np.save(f, ID_surprise)
        return ID_beliefs, ID_surprise




if __name__ == "__main__":
    mainTrials = 60
    additionalTrials = 0
    gridsize = 5000
    extra=False 
    dyna=False
    init=False
    asym=False
    transfer=False
    dyna_init=False
    IDs = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    for ID in IDs:
        fitting = Fitting(mainTrials, additionalTrials, gridsize, ID, extra=extra, dyna=dyna, init=init, asym=asym, transfer=transfer, dyna_init=dyna_init)
        fitting.statisticalLearning()
    
