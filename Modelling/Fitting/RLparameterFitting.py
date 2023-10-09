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
from multiprocessing import Pool
import signal
from functools import wraps

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

        if platform.system() == "Windows":
            wanted_dir = "/data/sourcedata/behavior/modified_files"
        else:
            wanted_dir = "/Volumes/SDrive/data/sourcedata/behavior/modified_files"

        # Get savedVals file
        self.savedValsFile = glob.glob(os.path.abspath(wanted_dir) + "/*{}_savedValues.csv".format(self.ID))[0]
        self.subjectData = pd.read_csv(self.savedValsFile)
        self.subjectData["stimulusPair"] = self.subjectData["stimulusPair"].apply(ast.literal_eval)

    ##

    def modelFitting(self, saveAs, extra=False, pearce=False, init=False, asym=False, transfer=False):
        """
        extra: False or True for separate alpha for V1
        pearce: False or True for Pearce Hall implementation
        init: False or True for initial V0 and V1
        asym : False or True for separate alphas based on reward
        transfer: False or True indicating the amount of discount free parameters for pairs that share a stimulus (depends on extra and asym)
        """
        if platform.system() == "Windows":
            newPath = os.path.join(
                pathlib.Path(__file__).resolve().parents[3],
                "/data/fittedParameters/sub-{0}/{1}".format(self.ID, saveAs),
            )

        else:
            newPath = os.path.join(
                str(pathlib.Path(__file__).resolve().parents[3])
                + "/data/fittedParameters/sub-{0}/{1}".format(self.ID, saveAs),
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
            best_LLs,
        ) = (np.empty((max(self.subjectData.runNumber))) for i in range(10))
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
            best_LLs[:],
        ) = (np.nan for i in range(10))

        fitted_V_option0Inits, fitted_V_option1Inits = (
            np.empty((max(self.subjectData.runNumber), 3, 3)) for i in range(2)
        )

        fitted_V_option0Inits[:], fitted_V_option1Inits[:] = (np.nan for i in range(2))

        NLL_array = np.empty((max(self.subjectData.runNumber), self.gridCount, 12))
        NLL_array[:] = np.nan

        (
            trial_alphas, 
            RPE,
            ) = (np.empty((max(self.subjectData.runNumber), self.mainTrials + self.additionalTrials)) for i in range(2))

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
        trial_alphas[:], RPE[:], V0[:], V1[:] = (np.nan for i in range(4))

        for run in range(0, max(self.subjectData.runNumber)):
            alphaGrid = np.random.rand(self.gridCount, 1)
            if extra and asym:
                alphaNegGrid, alpha2PosGrid, alpha2NegGrid = (np.random.rand(self.gridCount, 1) for i in range(3))
            elif extra and not asym:
                alpha2Grid = np.random.rand(self.gridCount, 1)
            elif asym and not extra:
                alphaNegGrid = np.random.rand(self.gridCount, 1)

            if transfer:
                K1Grid = np.random.rand(self.gridCount, 1)
                if extra and asym:
                    K3Grid, K4Grid, K2Grid = (np.random.rand(self.gridCount, 1) for i in range(3))
                elif extra and not asym:
                    K2Grid = np.random.rand(self.gridCount, 1)
                elif asym and not extra:
                    K3Grid = np.random.rand(self.gridCount, 1)

            betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
            LL_array = np.empty((self.gridCount, 1))
            LL_array[:] = np.nan
            if init:
                V_option0_rand = np.random.rand(self.gridCount, 1)
                V_option0Init_Grid = np.repeat(V_option0_rand, 9, axis=1).reshape((self.gridCount, 3, 3))

                V_option1_rand = np.random.rand(self.gridCount, 1)
                V_option1Init_Grid = np.repeat(V_option1_rand, 9, axis=1).reshape((self.gridCount, 3, 3))

            runData = self.subjectData[self.subjectData.runNumber == run + 1].reset_index()
            run_trialwise_alphas, run_RPEs = (np.empty((self.gridCount, self.mainTrials + self.additionalTrials)) for i in range(2))
            run_V0, run_V1 = (np.empty((self.gridCount, self.mainTrials + self.additionalTrials)) for i in range(2))

            run_trialwise_alphas[:], run_RPEs[:], run_V0[:], run_V1[:] = (np.nan for i in range(4))

            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridCount):
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((max(runData.trialNumber), 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((max(runData.trialNumber), 1))
                actionProb[:] = np.nan
                V_option0, V_option1 = (np.empty((max(runData.trialNumber) + 1, 3, 3)) for i in range(2))
                V_option0[:], V_option1[:] = (np.nan for i in range(2))
                if init:
                    V_option0[0, :] = V_option0Init_Grid[j]
                    V_option1[0, :] = V_option1Init_Grid[j]
                else:
                    V_option0[0, :] = 0.5
                    V_option1[0, :] = 0.5

                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                if extra and not asym:
                    alphaPosCheck = alphaNegCheck = alphaGrid[j]
                    alpha2PosCheck = alpha2NegCheck = alpha2Grid[j]
                elif asym and not extra:
                    alphaPosCheck = alpha2PosCheck = alphaGrid[j]
                    alphaNegCheck = alpha2NegCheck = alphaNegGrid[j]
                elif extra and asym:
                    alphaPosCheck = alphaGrid[j]
                    alpha2PosCheck = alpha2PosGrid[j]
                    alphaNegCheck = alphaNegGrid[j]
                    alpha2NegCheck = alpha2NegGrid[j]
                else:
                    alphaPosCheck = alpha2PosCheck = alphaNegCheck = alpha2NegCheck = alphaGrid[j]

                if transfer:
                    if extra and not asym:
                        K1Check = K3Check = K1Grid[j]
                        K2Check = K4Check = K2Grid[j]
                    elif asym and not extra:
                        K1Check = K2Check = K1Grid[j]
                        K3Check = K4Check = K3Grid[j]
                    elif asym and extra:
                        K1Check = K1Grid[j]
                        K2Check = K2Grid[j]
                        K3Check = K3Grid[j]
                        K4Check = K4Grid[j]
                    else:
                        K1Check = K2Check = K3Check = K4Check = K1Grid[j]

                betaCheck = betaGrid[j]
                # run_V0[j, 0] = V_option0[0, 0, 0]
                # run_V1[j, 0] = V_option1[0, 0, 0]

                if pearce:
                    omega = omega2 = omega3 = omega4 = 1

                for t in range(0, max(runData.trialNumber)):
                    otherPairs = [
                        p
                        for p in list(runData.stimulusPair.unique())
                        if bool(p[0] == runData.stimulusPair[t][0]) ^ bool(p[1] == runData.stimulusPair[t][1])
                    ]

                    # Prob of choosing the 0th and 1st option respectively
                    choiceProb[t, 0] = np.exp(betaCheck * V_option0[((t,) + runData.stimulusPair[t])]) / (
                        (np.exp(betaCheck * V_option0[((t,) + runData.stimulusPair[t])]))
                        + (np.exp(betaCheck * V_option1[((t,) + runData.stimulusPair[t])]))
                    )
                    choiceProb[t, 1] = 1 - choiceProb[t, 0]

                    actionProb[t, :] = choiceProb[t, int(runData.action[t])] if ~np.isnan(runData.action[t]) else np.nan

                    if runData.action[t] == 0:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t] - V_option0[(t,) + runData.stimulusPair[t]]
                        )
                        V_option0[t + 1, :] = V_option0[t, :]

                        if runData.reward[t] == 1:
                            if pearce:
                                omega = omega + (abs(rewardPE[(t,) + runData.stimulusPair[t]]) - omega) * alphaPosCheck
                                V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega * (rewardPE[(t,) + runData.stimulusPair[t]])
                                run_trialwise_alphas[j,t] = omega
                                if transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[(t,) + pair] - K1Check * omega * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )
                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + alphaPosCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                if transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K1Check * alphaPosCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            if pearce:
                                if not asym and not extra:
                                    omega2 = omega3 = omega4 = omega
                                elif asym and not extra:
                                    omega2 = omega
                                elif extra and not asym:
                                    omega3 = omega

                        else:
                            if pearce:
                                omega3 = (
                                    omega3 + (abs(rewardPE[(t,) + runData.stimulusPair[t]]) - omega3) * alphaNegCheck
                                )
                                V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega3 * (rewardPE[(t,) + runData.stimulusPair[t]])
                                run_trialwise_alphas[j,t] = omega3
                                if transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[(t,) + pair] - K3Check * omega3 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[(t + 1,) + runData.stimulusPair[t]] = V_option0[
                                    (t,) + runData.stimulusPair[t]
                                ] + alphaNegCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                if transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K3Check * alphaNegCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0

                            if pearce:
                                if not asym and not extra:
                                    omega2 = omega = omega4 = omega3
                                elif asym and not extra:
                                    omega4 = omega3
                                elif extra and not asym:
                                    omega = omega3

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif runData.action[t] == 1:
                        rewardPE[(t,) + runData.stimulusPair[t]] = (
                            runData.reward[t] - V_option1[(t,) + runData.stimulusPair[t]]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]

                        if runData.reward[t] == 1:
                            if pearce:
                                omega2 = (
                                    omega2 + (abs(rewardPE[(t,) + runData.stimulusPair[t]]) - omega2) * alpha2PosCheck
                                )
                                V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega2 * (rewardPE[(t,) + runData.stimulusPair[t]])
                                run_trialwise_alphas[j,t] = omega2
                                if transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[(t,) + pair] - K2Check * omega2 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alpha2PosCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                if transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K2Check * alpha2PosCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if pearce:
                                if not asym and not extra:
                                    omega3 = omega = omega4 = omega2
                                elif asym and not extra:
                                    omega = omega2
                                elif extra and not asym:
                                    omega4 = omega2

                        else:
                            if pearce:
                                omega4 = (
                                    omega4 + (abs(rewardPE[(t,) + runData.stimulusPair[t]]) - omega4) * alpha2NegCheck
                                )
                                V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + omega4 * (rewardPE[(t,) + runData.stimulusPair[t]])
                                run_trialwise_alphas[j,t] = omega4
                                if transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[(t,) + pair] - K4Check * omega4 * (
                                            rewardPE[(t,) + runData.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[(t + 1,) + runData.stimulusPair[t]] = V_option1[
                                    (t,) + runData.stimulusPair[t]
                                ] + alpha2NegCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                if transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K4Check * alpha2NegCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if pearce:
                                if not asym and not extra:
                                    omega2 = omega = omega3 = omega4
                                elif asym and not extra:
                                    omega3 = omega4
                                elif extra and not asym:
                                    omega2 = omega4

                        V_option0[t + 1, :] = V_option0[t, :]
                    else:
                        V_option1[t + 1, :] = V_option1[t, :]
                        V_option0[t + 1, :] = V_option0[t, :]

                    run_RPEs[j, t] = rewardPE[(t,) + runData.stimulusPair[t]]
                    run_V0[j, t] = V_option0[(t,) + runData.stimulusPair[t]]
                    run_V1[j, t] = V_option1[(t,) + runData.stimulusPair[t]]
                negativeLogLikelihood = -np.sum(np.log(actionProb[~np.isnan(actionProb)]))
                Likelihood = np.prod(actionProb[~np.isnan(actionProb)])
                NLL_array[run, j, 0] = negativeLogLikelihood
                NLL_array[run, j, 1] = alphaPosCheck
                NLL_array[run, j, 2] = betaCheck
                if asym and not extra:
                    NLL_array[run, j, 3] = alphaNegCheck
                elif extra and not asym:
                    NLL_array[run, j, 4] = alpha2PosCheck
                elif extra and asym:
                    NLL_array[run, j, 3] = alphaNegCheck
                    NLL_array[run, j, 4] = alpha2PosCheck
                    NLL_array[run, j, 5] = alpha2NegCheck
                if init:
                    NLL_array[run, j, 6] = V_option0Init_Grid[j][0][0]
                    NLL_array[run, j, 7] = V_option1Init_Grid[j][0][0]
                if transfer:
                    NLL_array[run, j, 8] = K1Check
                    if extra and not asym:
                        NLL_array[run, j, 9] = K2Check
                    elif asym and not extra:
                        NLL_array[run, j, 10] = K3Check
                    elif asym and extra:
                        NLL_array[run, j, 9] = K2Check
                        NLL_array[run, j, 10] = K3Check
                        NLL_array[run, j, 11] = K4Check

                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 0])
            maxIndex = np.nanargmax(LL_array[:, 0])
            newPath = "bestFittingVals/sub-{0}".format(self.ID)
            Path(newPath).mkdir(parents=True, exist_ok=True)
            fitted_alphasPos[run] = NLL_array[run, minIndex, 1]
            np.savetxt(fname=newPath + "/alphasPos_" + saveAs + ".tsv", X=fitted_alphasPos, delimiter=",")
            if asym and not extra:
                fitted_alphasNeg[run] = NLL_array[run, minIndex, 3]
                np.savetxt(fname=newPath + "/alphasNeg_" + saveAs + ".tsv", X=fitted_alphasNeg, delimiter=",")
            elif extra and not asym:
                fitted_alphas2Pos[run] = NLL_array[run, minIndex, 4]
                np.savetxt(fname=newPath + "/alphas2Pos_" + saveAs + ".tsv", X=fitted_alphas2Pos, delimiter=",")
            elif extra and asym:
                fitted_alphasNeg[run] = NLL_array[run, minIndex, 3]
                np.savetxt(fname=newPath + "/alphasNeg_" + saveAs + ".tsv", X=fitted_alphasNeg, delimiter=",")
                fitted_alphas2Pos[run] = NLL_array[run, minIndex, 4]
                np.savetxt(fname=newPath + "/alphas2Pos_" + saveAs + ".tsv", X=fitted_alphas2Pos, delimiter=",")
                fitted_alphas2Neg[run] = NLL_array[run, minIndex, 5]
                np.savetxt(fname=newPath + "/alphas2Neg_" + saveAs + ".tsv", X=fitted_alphas2Neg, delimiter=",")
            fitted_betas[run] = NLL_array[run, minIndex, 2]
            np.savetxt(fname=newPath + "/betas_" + saveAs + ".tsv", X=fitted_betas, delimiter=",")
            if init:
                fitted_V_option0Inits[run] = V_option0Init_Grid[minIndex]
                fitted_V_option1Inits[run] = V_option1Init_Grid[minIndex]
            if transfer:
                fitted_K1[run] = NLL_array[run, minIndex, 8]
                np.savetxt(fname=newPath + "/K1_" + saveAs + ".tsv", X=fitted_K1, delimiter=",")
                if extra and not asym:
                    fitted_K2[run] = NLL_array[run, minIndex, 9]
                    np.savetxt(fname=newPath + "/K2_" + saveAs + ".tsv", X=fitted_K2, delimiter=",")
                elif asym and not extra:
                    fitted_K3[run] = NLL_array[run, minIndex, 10]
                    np.savetxt(fname=newPath + "/K3_" + saveAs + ".tsv", X=fitted_K3, delimiter=",")
                elif extra and asym:
                    fitted_K2[run] = NLL_array[run, minIndex, 9]
                    np.savetxt(fname=newPath + "/K2_" + saveAs + ".tsv", X=fitted_K2, delimiter=",")
                    fitted_K3[run] = NLL_array[run, minIndex, 10]
                    np.savetxt(fname=newPath + "/K3_" + saveAs + ".tsv", X=fitted_K3, delimiter=",")
                    fitted_K4[run] = NLL_array[run, minIndex, 11]
                    np.savetxt(fname=newPath + "/K4_" + saveAs + ".tsv", X=fitted_K4, delimiter=",")
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]
            trial_alphas[run] = run_trialwise_alphas[minIndex]

        if self.plotting:
            self.Plot.plots_modelFitting(
                NLL_array=NLL_array,
                alphasPos=fitted_alphasPos,
                alphas2Pos=fitted_alphas2Pos,
                alphasNeg=fitted_alphasNeg,
                alphas2Neg=fitted_alphas2Neg,
                betas=fitted_betas,
                pearce=pearce,
                V_option0Inits=fitted_V_option0Inits,
                V_option1Inits=fitted_V_option1Inits,
                K1=fitted_K1,
                K2=fitted_K2,
                K3=fitted_K3,
                K4=fitted_K4,
                saveAs=saveAs,
                extra=extra,
                asym=asym,
                transfer=transfer,
            )

        scipy.io.savemat(
            newPath + "/rpe" + saveAs + ".mat".format(self.ID),
            mdict={"rpe": RPE},
        )

        with open(newPath + "/BIC_" + saveAs + ".npy", "wb") as f:
            np.save(f, best_LLs)

        with open(newPath + "/V0_" + saveAs + ".npy", "wb") as f:
            np.save(f, V0)

        with open(newPath + "/V1_" + saveAs + ".npy", "wb") as f:
            np.save(f, V1)

        if pearce:
            with open(newPath + "/trialwise_alphas_" + saveAs + ".npy", "wb") as f:
                np.save(f, trial_alphas)
            scipy.io.savemat(
            newPath + "/trialwise_alphas_" + saveAs + ".mat".format(self.ID),
            mdict={"trialwise_alphas": trial_alphas},
        )

    # Statistical learning
    def statisticalLearning(self, statLearnPar=1):
        self.statLearnPar = statLearnPar

        self.subjectData = pd.read_csv(self.savedValsFile)
        self.subjectData["stimulusPair"] = self.subjectData["stimulusPair"].apply(ast.literal_eval)
        ID_beliefs = np.empty((6, self.mainTrials + self.additionalTrials + 1, 3, 3))
        ID_surprise = np.empty((6, self.mainTrials + self.additionalTrials))
        for run in range(0, max(self.subjectData.runNumber)):
            rowBeliefs = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
            rowBeliefs[:] = np.nan
            columnBeliefs = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
            columnBeliefs[:] = np.nan
            beliefsStat = np.empty((self.mainTrials + self.additionalTrials + 1, 3, 3))
            beliefsStat[:] = np.nan
            statCount = np.zeros((self.mainTrials + self.additionalTrials + 1, 3, 3))
            statSurprise = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
            statSurprise[:] = np.nan
            statSurpriseRow = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
            statSurpriseRow[:] = np.nan
            statSurpriseColumn = np.empty((self.mainTrials + self.additionalTrials, 3, 3))
            statSurpriseColumn[:] = np.nan

            runData = self.subjectData[self.subjectData.runNumber == run + 1].reset_index()
            rowDen0 = 3 * self.statLearnPar
            rowDen1 = 3 * self.statLearnPar
            rowDen2 = 3 * self.statLearnPar
            columnDen0 = 3 * self.statLearnPar
            columnDen1 = 3 * self.statLearnPar
            columnDen2 = 3 * self.statLearnPar

            for i in range(0, self.mainTrials + self.additionalTrials):
                statCount[i + 1, :] = statCount[i, :]
                statCount[i + 1, runData.stimulusPair[i][0], runData.stimulusPair[i][1]] += 1

            rowBeliefs[0, :] = (self.statLearnPar + statCount[0, :]) / 3 * self.statLearnPar
            columnBeliefs[0, :] = (self.statLearnPar + statCount[0, :]) / 3 * self.statLearnPar
            num = self.statLearnPar + statCount[0, :]
            den = 9 * self.statLearnPar
            # Total statistical beliefs irrespective of rows and columns.
            beliefsStat[0, :] = num / den

            for i in range(1, self.mainTrials + self.additionalTrials + 1):
                rowBeliefs[i, :] = rowBeliefs[i - 1, :]
                # Row beliefs
                if runData.stimulusPair[i - 1][0] == 0:
                    rowDen0 = rowDen0 + 1
                    rowBeliefs[i, 0, :] = (self.statLearnPar + statCount[i, 0, :]) / rowDen0
                elif runData.stimulusPair[i - 1][0] == 1:
                    rowDen1 = rowDen1 + 1
                    rowBeliefs[i, 1, :] = (self.statLearnPar + statCount[i, 1, :]) / rowDen1
                else:
                    rowDen2 = rowDen2 + 1
                    rowBeliefs[i, 2, :] = (self.statLearnPar + statCount[i, 2, :]) / rowDen2
                # column beliefs
                columnBeliefs[i, :] = columnBeliefs[i - 1, :]
                if runData.stimulusPair[i - 1][1] == 0:
                    columnDen0 = columnDen0 + 1
                    columnBeliefs[i, 0, :] = (self.statLearnPar + statCount[i, 0, :]) / columnDen0
                elif runData.stimulusPair[i - 1][1] == 1:
                    columnDen1 = columnDen1 + 1
                    columnBeliefs[i, 1, :] = (self.statLearnPar + statCount[i, 1, :]) / columnDen1
                else:
                    columnDen2 = columnDen2 + 1
                    columnBeliefs[i, 2, :] = (self.statLearnPar + statCount[i, 2, :]) / columnDen2
                num = self.statLearnPar + statCount[i, :]
                den += 1
                # Total statistical beliefs irrespective of rows and columns.
                beliefsStat[i, :] = num / den

            ID_beliefs[run, :, :] = beliefsStat
            trial_surprise = np.empty((self.mainTrials + self.additionalTrials))
            # Surprises calculated from beliefs update
            for i in range(0, self.mainTrials + self.additionalTrials):
                statSurpriseRow[i, :] = np.nan
                statSurpriseRow[(i,) + runData.stimulusPair[i]] = -np.log(rowBeliefs[(i,) + runData.stimulusPair[i]])
                statSurpriseColumn[i, :] = np.nan
                statSurpriseColumn[(i,) + runData.stimulusPair[i]] = -np.log(
                    columnBeliefs[(i,) + runData.stimulusPair[i]]
                )
                statSurprise[i, :] = np.nan
                statSurprise[(i,) + runData.stimulusPair[i]] = -np.log(beliefsStat[(i,) + runData.stimulusPair[i]])
                trial_surprise[i] = statSurprise[(i,) + runData.stimulusPair[i]]
            ID_surprise[run] = trial_surprise

        newPath = os.path.join(
            str(pathlib.Path(__file__).resolve().parents[3]) + "/data/fittedParameters/sub-{}".format(self.ID),
        )
        newPath2 = "bestFittingVals/sub-{0}".format(self.ID)
        Path(newPath).mkdir(parents=True, exist_ok=True)
        Path(newPath2).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(newPath + "/spe.mat".format(self.ID), mdict={"spe": ID_surprise})
        scipy.io.savemat(newPath2 + "/spe.mat".format(self.ID), mdict={"spe": ID_surprise})
        with open(newPath2 + "/spe.npy", "wb") as f:
            np.save(f, ID_surprise)
        return ID_beliefs, ID_surprise


# Calls for different fitting and plotting functions
# You can only run ONE model fitting at a time

configurations = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
#     "07",
    "09",
    "10",
#     "11",
    "12",
    "14",
#     "15",
    "17",
    "18",
#    "19",
#     "20",
#     "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
#     "28",
#    "29",
#   "30",
   "33",
#    "34",
#    "35",
#    "36",
#     "37",
    "38",
    "39",
    "40",
#     "41",
#     "42",
#     "43",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
#     "53",
   "54",
#    "55",
   "56",
   "57",
#    "58",
#     "59",
   "60",
   "61",
#     "62",
#    "63",
#    "64",
]


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler)  # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt()
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt()

    return wrapper


@handle_ctrl_c
def use_fitting(IDnr):
    fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=False)
    fitted.modelFitting(saveAs="PearceExtra", pearce=True, extra=True)


@handle_ctrl_c
def use_fitting2(IDnr):
    fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=False)
    fitted.modelFitting(saveAs="Pearce", pearce=True)


# @handle_ctrl_c
# def use_fitting3(IDnr):
#     fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=True)
#     fitted.modelFitting(
#         saveAs="PearceInitExtraAsymTransfer", pearce=True, init=True, extra=True, asym=True, transfer=True
#     )


# @handle_ctrl_c
# def use_fitting4(IDnr):
#     fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=True)
#     fitted.modelFitting(saveAs="PearceInitAsymTransfer", init=True, asym=True, pearce=True, transfer=True)

# @handle_ctrl_c
# def use_surprise(IDnr):
#     fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=False)
#     fitted.statisticalLearning()

def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True


def init_pool():
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(8, initializer=init_pool)
    results = pool.map(use_fitting, configurations)
    if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
        print("Ctrl-C was entered.")
    pool.close()
    pool.join()

    pool2 = Pool(8, initializer=init_pool)
    results2 = pool2.map(use_fitting2, ['34', '35', '63'])
    if any(map(lambda x: isinstance(x, KeyboardInterrupt), results2)):
        print("Ctrl-C was entered.")
    pool2.close()
    pool2.join()

    # pool3 = Pool(8, initializer=init_pool)
    # results3 = pool3.map(use_fitting3, configurations)
    # if any(map(lambda x: isinstance(x, KeyboardInterrupt), results3)):
    #     print("Ctrl-C was entered.")
    # pool3.close()
    # pool3.join()

    # pool4 = Pool(8, initializer=init_pool)
    # results4 = pool4.map(use_fitting4, configurations)
    # if any(map(lambda x: isinstance(x, KeyboardInterrupt), results4)):
    #     print("Ctrl-C was entered.")
    # pool4.close()
    # pool4.join()

    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    # pool = Pool(8, initializer=init_pool)
    # results = pool.map(use_surprise, configurations)
    # if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
    #     print("Ctrl-C was entered.")
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()
