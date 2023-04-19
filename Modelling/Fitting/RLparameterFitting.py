from random import *
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.io
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns
import sys
import pathlib
from pathlib import Path
import re
import platform
import ast
import matplotlib.backends.backend_pdf
from mpl_toolkits.mplot3d import Axes3D
from RLparameterPlotting import Plotting

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
    ):
        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.gridCount = gridCount
        self.ID = ID
        self.statLearnPar = 1
        self.plotting = plotting

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

        wanted_dir = "/data/sourcedata/behavior/modified_files"
        # Get savedVals file
        self.savedValsFile = glob.glob(
            os.path.abspath(wanted_dir) + "/*{}_savedValues.csv".format(self.ID)
        )[0]
        self.subjectData = pd.read_csv(self.savedValsFile)
        self.subjectData["stimulusPair"] = self.subjectData["stimulusPair"].apply(
            ast.literal_eval
        )

    ##

    def basicFitting(self, extra=False, pearce=False, init=False):
        """
        extra: False or True for separate alpha for V1
        pearce: False or True for Pearce Hall implementation
        init: False or True for initial V0 and V1
        """
        fitted_alphas = fitted_alphas2 = np.empty((max(self.subjectData.runNumber)))
        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))

        RPE = np.empty(
            (max(self.subjectData.runNumber), self.mainTrials + self.additionalTrials)
        )
        V0 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials + 1,
            )
        )
        V1 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials + 1,
            )
        )

        fitted_V_option0Inits = fitted_V_option1Inits = np.empty((max(self.subjectData.runNumber), 3, 3))

        NLL_array = np.empty((max(self.subjectData.runNumber), self.gridCount, 6))
        NLL_array[:] = np.nan

        for run in range(0, max(self.subjectData.runNumber)):
            alphaGrid = np.random.rand(self.gridCount, 1)
            if extra:
                alpha2Grid = np.random.rand(self.gridCount, 1)
            betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
            LL_array = np.empty((self.gridCount, 1))
            if init:
                V_option0Init_Grid = V_option1Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
            runData = self.subjectData[
                self.subjectData.runNumber == run + 1
            ].reset_index()
            run_RPEs = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials)
            )
            run_V0 = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials + 1)
            )
            run_V1 = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials + 1)
            )
            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridCount):
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((max(runData.trialNumber), 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((max(runData.trialNumber), 1))
                actionProb[:] = np.nan
                V_option0 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option0[:] = np.nan
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                if init:
                    V_option0[0, :] = V_option0Init_Grid[j]
                    V_option1[0, :] = V_option1Init_Grid[j]
                else:
                    V_option0[0, :] = 0.5
                    V_option1[0, :] = 0.5
                
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                if extra:
                    alpha2Check = alpha2Grid[j]
                betaCheck = betaGrid[j]
                run_V0[j, 0] = V_option0[0, 0, 0]
                run_V1[j, 0] = V_option1[0, 0, 0]
                if pearce:
                    omega = 1
                    omega2 = 1
                for t in range(0, max(runData.trialNumber)):
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

                    actionProb[t, :] = (
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
                        if pearce:
                            omega = (
                                omega
                                + (
                                    abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                    - omega
                                )
                                * alphaCheck
                            )
                            V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                (t,) + runData.stimulusPair[t]
                            ] + omega * (rewardPE[(t,) + runData.stimulusPair[t]])
                        else:
                            V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                (t,) + runData.stimulusPair[t]
                            ] + alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif runData.action[t] == 1:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t]
                            - V_option1[(t,) + runData.stimulusPair[t]]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]
                        if extra:
                            if pearce:
                                omega2 = (
                                    omega2
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega2
                                    )
                                    * alpha2Check
                                )
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega2 * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                            else:
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alpha2Check * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                        else:
                            if pearce:
                                omega = (
                                    omega
                                    + (
                                        abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                        - omega
                                    )
                                    * alphaCheck
                                )
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                            else:
                                V_option1[
                                    (t + 1,) + runData.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alphaCheck * (
                                    rewardPE[(t,) + runData.stimulusPair[t]]
                                )
                        V_option0[t + 1, :] = V_option0[t, :]
                    else:
                        V_option1[t + 1, :] = V_option1[t, :]
                        V_option0[t + 1, :] = V_option0[t, :]

                    run_RPEs[j, t] = rewardPE[(t,) + runData.stimulusPair[t]]
                    run_V0[j, t + 1] = V_option0[(t + 1,) + runData.stimulusPair[t]]
                    run_V1[j, t + 1] = V_option1[(t + 1,) + runData.stimulusPair[t]]
                negativeLogLikelihood = -np.sum(
                    np.log(actionProb[~np.isnan(actionProb)])
                )
                Likelihood = np.prod(actionProb[~np.isnan(actionProb)])
                NLL_array[run, j, 0] = alphaCheck
                NLL_array[run, j, 1] = betaCheck
                NLL_array[run, j, 2] = negativeLogLikelihood
                if extra:
                    NLL_array[run, j, 3] = alpha2Check
                if init:
                    NLL_array[run, j, 4] = V_option0Init_Grid[j][0][0]
                    NLL_array[run, j, 5] = V_option1Init_Grid[j][0][0]
                
                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[:, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            if extra:
                fitted_alphas2[run] = NLL_array[run, minIndex, 3]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            if init:
                fitted_V_option0Inits[run] = V_option0Init_Grid[minIndex]
                fitted_V_option1Inits[run] = V_option1Init_Grid[minIndex]
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]

        if self.plotting:
            if extra and init:
                self.Plot.plots_basicFitting(
                    NLL_array=NLL_array,
                    alphas=fitted_alphas,
                    betas=fitted_betas,
                    pearce=pearce,
                    alphas2=fitted_alphas2,
                    V_option0Inits=fitted_V_option0Inits, 
                    V_option1Inits=fitted_V_option1Inits
                )
            elif extra and not init:
                self.Plot.plots_basicFitting(
                    NLL_array=NLL_array,
                    alphas=fitted_alphas,
                    betas=fitted_betas,
                    pearce=pearce,
                    alphas2=fitted_alphas2,
                )
            elif not extra and init:
                self.Plot.plots_basicFitting(
                    NLL_array=NLL_array,
                    alphas=fitted_alphas,
                    betas=fitted_betas,
                    pearce=pearce,
                    V_option0Inits=fitted_V_option0Inits, 
                    V_option1Inits=fitted_V_option1Inits
                )
            else:
                self.Plot.plots_basicFitting(
                    NLL_array=NLL_array,
                    alphas=fitted_alphas,
                    betas=fitted_betas,
                    pearce=pearce,
                )

        newPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters/sub-{}".format(self.ID),
        )
        Path(newPath).mkdir(parents=True, exist_ok=True)

        if extra and not init:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeExtraPearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeExtra.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_alphas2,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
            )
        elif extra and init:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeExtraInitPearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeExtraInit.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_alphas2,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
                fitted_V_option0Inits,
                fitted_V_option1Inits
            )
        elif not extra and init:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeInitPearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeInit.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
                fitted_V_option0Inits,
                fitted_V_option1Inits
            )
        else:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpePearce.mat".format(self.ID), mdict={"rpe": RPE}
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpe.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return fitted_alphas, fitted_betas, best_LLs, RPE, V0, V1, NLL_array


    # Extra update rule multiple versions

    # In update fitting model, when version is None, this means that the pairs corresponding to the same
    # visual or audio/tactile stimulus gets updated as well in the opposite direction with some
    # different learning rate alpha
    # Meanwhile when version is "two", then there is action dependance, ACTION AND REWARD (4)
    def transferFitting(self, version=None, extra=False, pearce=False, init=False):
        """
        version: None, "two" or "four" for one, two or three alphas for other pairs
        Extra: False or True for separate alpha for V1
        pearce: True or False
        init: True or False
        """
        fitted_alphas = np.empty((max(self.subjectData.runNumber)))
        fitted_otherAlphas = np.empty((max(self.subjectData.runNumber)))
        if version == "two":
            fitted_otherAlphas2 = np.empty((max(self.subjectData.runNumber)))
        elif version == "four":
            fitted_otherAlphas2 = np.empty((max(self.subjectData.runNumber)))
            fitted_otherAlphas3 = np.empty((max(self.subjectData.runNumber)))
            fitted_otherAlphas4 = np.empty((max(self.subjectData.runNumber)))

        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))

        if init:
            fitted_V_option0Inits = np.empty((max(self.subjectData.runNumber), 3, 3))
            fitted_V_option1Inits = np.empty((max(self.subjectData.runNumber), 3, 3))
        NLL_array = np.empty((max(self.subjectData.runNumber), self.gridCount, 5))
        NLL_array[:] = np.nan

        RPE = np.empty(
            (max(self.subjectData.runNumber), self.mainTrials + self.additionalTrials)
        )
        V0 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials + 1,
            )
        )
        V1 = np.empty(
            (
                max(self.subjectData.runNumber),
                self.mainTrials + self.additionalTrials + 1,
            )
        )

        for run in range(0, max(self.subjectData.runNumber)):
            alphaGrid = np.random.rand(self.gridCount, 1)
            alphaOtherGrid = np.random.rand(self.gridCount, 1)
            if version == "two":
                alphaOther2Grid = np.random.rand(self.gridCount, 1)
            elif version == "four":
                alphaOther2Grid = np.random.rand(self.gridCount, 1)
                alphaOther3Grid = np.random.rand(self.gridCount, 1)
                alphaOther4Grid = np.random.rand(self.gridCount, 1)
            betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
            LL_array = np.empty((self.gridCount, 1))
            runData = self.subjectData[
                self.subjectData.runNumber == run + 1
            ].reset_index()
            run_RPEs = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials)
            )
            run_V0 = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials + 1)
            )
            run_V1 = np.empty(
                (self.gridCount, self.mainTrials + self.additionalTrials + 1)
            )
            if init:
                V_option0Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
                V_option1Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridCount):
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((max(runData.trialNumber), 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((max(runData.trialNumber), 1))
                actionProb[:] = np.nan
                V_option0 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option0[:] = np.nan
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                alphaOtherCheck = alphaOtherGrid[j]
                if version == "two":
                    alphaOther2Check = alphaOther2Grid[j]
                elif version == "four":
                    alphaOther2Check = alphaOther2Grid[j]
                    alphaOther3Check = alphaOther3Grid[j]
                    alphaOther4Check = alphaOther4Grid[j]
                betaCheck = betaGrid[j]
                if init:
                    V_option0[0, :] = V_option0Init_Grid[j]
                    V_option1[0, :] = V_option1Init_Grid[j]
                else:
                    V_option0[0, :] = 0.5
                    V_option1[0, :] = 0.5

                run_V0[j, 0] = V_option0[0, 0, 0]
                run_V1[j, 0] = V_option1[0, 0, 0]
                if pearce:
                    omega = 1
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

                    actionProb[t, :] = (
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

                        if pearce:
                            omega = (
                                omega
                                + (
                                    abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                    - omega
                                )
                                * alphaCheck
                            )
                            V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                (t,) + runData.stimulusPair[t]
                            ] + omega * (rewardPE[(t,) + runData.stimulusPair[t]])
                        else:
                            V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                (t,) + runData.stimulusPair[t]
                            ] + alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                        if (
                            version is None
                            or version == "two"
                            or (version == "four" and runData.reward[t] == 1)
                        ):
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alphaOtherCheck
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alphaOtherCheck * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                        else:
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alphaOther2Check
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alphaOther2Check * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                        V_option1[t + 1, :] = V_option1[t, :]

                    elif runData.action[t] == 1:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t]
                            - V_option1[(t,) + runData.stimulusPair[t]]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]
                        if pearce:
                            omega = (
                                omega
                                + (
                                    abs(rewardPE[(t,) + runData.stimulusPair[t]])
                                    - omega
                                )
                                * alphaCheck
                            )
                            V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                (t,) + runData.stimulusPair[t]
                            ] + omega * (rewardPE[(t,) + runData.stimulusPair[t]])
                        else:
                            V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                (t,) + runData.stimulusPair[t]
                            ] + alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                        if version is None:
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alphaOtherCheck
                                    )
                                    V_option1[(t + 1,) + pair] = V_option1[
                                        (t,) + pair
                                    ] + (
                                        omega
                                        * (
                                            1
                                            - runData.reward[t]
                                            - V_option1[(t,) + pair]
                                        )
                                    )
                                else:
                                    V_option1[(t + 1,) + pair] = V_option1[
                                        (t,) + pair
                                    ] + (
                                        alphaOtherCheck
                                        * (
                                            1
                                            - runData.reward[t]
                                            - V_option1[(t,) + pair]
                                        )
                                    )
                        elif version == "two":
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alphaOther2Check
                                    )
                                    V_option1[(t + 1,) + pair] = V_option1[
                                        (t,) + pair
                                    ] + (
                                        omega
                                        * (
                                            1
                                            - runData.reward[t]
                                            - V_option1[(t,) + pair]
                                        )
                                    )
                                else:
                                    V_option1[(t + 1,) + pair] = V_option1[
                                        (t,) + pair
                                    ] + (
                                        alphaOther2Check
                                        * (
                                            1
                                            - runData.reward[t]
                                            - V_option1[(t,) + pair]
                                        )
                                    )
                        else:
                            if runData.reward[t] == 1:
                                for pair in otherPairs:
                                    if pearce:
                                        omega = (
                                            omega
                                            + (abs(rewardPE[(t,) + pair]) - omega)
                                            * alphaOther3Check
                                        )
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] + (
                                            omega
                                            * (
                                                1
                                                - runData.reward[t]
                                                - V_option1[(t,) + pair]
                                            )
                                        )
                                    else:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] + (
                                            alphaOther3Check
                                            * (
                                                1
                                                - runData.reward[t]
                                                - V_option1[(t,) + pair]
                                            )
                                        )
                            else:
                                for pair in otherPairs:
                                    if pearce:
                                        omega = (
                                            omega
                                            + (abs(rewardPE[(t,) + pair]) - omega)
                                            * alphaOther4Check
                                        )
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] + (
                                            omega
                                            * (
                                                1
                                                - runData.reward[t]
                                                - V_option1[(t,) + pair]
                                            )
                                        )
                                    else:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] + (
                                            alphaOther4Check
                                            * (
                                                1
                                                - runData.reward[t]
                                                - V_option1[(t,) + pair]
                                            )
                                        )

                        V_option0[t + 1, :] = V_option0[t, :]

                    else:
                        V_option0[t + 1, :] = V_option0[t, :]
                        V_option1[t + 1, :] = V_option1[t, :]

                    run_RPEs[j, t] = rewardPE[(t,) + runData.stimulusPair[t]]
                    run_V0[j, t + 1] = V_option0[(t + 1,) + runData.stimulusPair[t]]
                    run_V1[j, t + 1] = V_option1[(t + 1,) + runData.stimulusPair[t]]
                negativeLogLikelihood = -np.sum(
                    np.log(actionProb[~np.isnan(actionProb)])
                )
                Likelihood = np.prod(actionProb[~np.isnan(actionProb)])

                NLL_array[run, j, 0] = alphaCheck
                NLL_array[run, j, 1] = betaCheck
                NLL_array[run, j, 2] = negativeLogLikelihood
                NLL_array[run, j, 3] = alphaOtherCheck
                if version == "two":
                    NLL_array[run, j, 4] = alphaOther2Check
                elif version == "four":
                    NLL_array[run, j, 4] = alphaOther2Check
                    NLL_array[run, j, 5] = alphaOther3Check
                    NLL_array[run, j, 6] = alphaOther4Check
                # NLL_array[run, j, ?] = alpha2Check
                #NLL_array[run, j, ?] = V_option0Init_Grid[j][0][0]
                #NLL_array[run, j, ?] = V_option1Init_Grid[j][0][0]
                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[run, :, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            fitted_otherAlphas[run] = NLL_array[run, minIndex, 2]

            if version == "two":
                fitted_otherAlphas2[run] = NLL_array[minIndex, 4]
            elif version == "four":
                fitted_otherAlphas2[run] = NLL_array[minIndex, 4]
                fitted_otherAlphas3[run] = NLL_array[minIndex, 5]
                fitted_otherAlphas4[run] = NLL_array[minIndex, 6]
            if init:
                fitted_V_option0Inits[run] = V_option0Init_Grid[minIndex]
                fitted_V_option1Inits[run] = V_option1Init_Grid[minIndex]
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]

        newPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters/sub-{}".format(self.ID),
        )
        Path(newPath).mkdir(parents=True, exist_ok=True)

        if version == "two":
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeInit2Pearce.mat".format(self.ID), mdict={"rpe": RPE}
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeInit2.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_otherAlphas,
                fitted_otherAlphas2,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
            )
        elif version == "four":
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeInit4Pearce.mat".format(self.ID), mdict={"rpe": RPE}
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeInit4.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_otherAlphas,
                fitted_otherAlphas2,
                fitted_otherAlphas3,
                fitted_otherAlphas4,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
            )
        else:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeInitPearce.mat".format(self.ID), mdict={"rpe": RPE}
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeInit.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_otherAlphas,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
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

        newPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters/sub-{}".format(self.ID),
        )
        Path(newPath).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(
            newPath + "/spe.mat".format(self.ID), mdict={"spe": ID_surprise}
        )
        return ID_beliefs, ID_surprise


def ma(interval, window_size=10, method="same"):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, method)


# Calls for different fitting and plotting functions
# You can only run ONE model fitting at a time

#

IDs = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "09",
    "10",
    "11",
    "12",
    "14",
    "15",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "63",
    "64",
]

for IDnr in IDs:
    fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=True)

    # # Run stat learning
    # beliefs, surprise = fitted.statisticalLearning(statLearnPar=1)

    # # Run the simplest RL model
    (
        fitted_alphas,
        fitted_alphas2,
        fitted_betas,
        best_LLs,
        RPE,
        V0,
        V1,
        NLL_array,
    ) = fitted.basicFitting(pearce=True, extra=True, init=False)
    # fitted.plots_simplestFitting(ww=10, NLL_array=NLL_array, alphas=fitted_alphas, betas=fitted_betas, method ='valid', reps=50)

# # Run the RL model including initial V0 and V1 as free parameters
# fitted_alphas, fitted_betas, best_LLs, fitted_V_option0Inits, fitted_V_option1Inits, RPE, V0, V1, NLL_array = fitted.valFitting()
# fitted.plots_valueFitting(ww=10, method='same', reps=50)

# # Run the RL model with extra learning rates for the other pairs
# # (if you see pair image0 and audio0, also update other image0 and audio0 pairs)
# # Options: 1) Do not add anything to the call 2) Add version="two" 3) Add version="four"
# fitted.updateFitting()
# fitted.plots_updateFitting(ww=10, method='same', reps=50)
# fitted.updateFitting(version="two")
# fitted.plots_updateFitting(ww=10, method='same', reps=50, version="two")
# fitted.updateFitting(version="four")
# fitted.plots_updateFitting(ww=10, method='same', reps=50, version="four")

# Run the RL model with extra learning rates for sthe other pairs AND initial V0 and V1 as free parameters
# Same options as above; 1) Do not add anything to the call 2) Add version="two" 3) Add version="four"
# fitted.updateInitFitting()
# fitted.plots_updateInitFitting(ww=10, method="same", reps=50)
# fitted.updateInitFitting(version="two")
# fitted.plots_updateInitFitting(ww=10, method="same", reps=50, version="two")
# fitted.updateInitFitting(version="four")
# fitted.plots_updateInitFitting(ww=10, method="same", reps=50, version="four")


# You can always include statistical learning; necessary to get surprise values
# beliefs, surprise = fitted.statisticalLearning(statLearnPar=1)
# fitted.plots_stats(beliefs, surprise)

# Save max log likelihood values if you want to do model comparison later; rename np array yourself (e.g. based on model)
# with open('BIC_new_upInit4.npy', 'wb') as f:
#     np.save(f, fitted.all_LLs)
