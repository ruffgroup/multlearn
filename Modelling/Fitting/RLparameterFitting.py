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

    ## Simple fitting

    def simplestFitting(self, pearce=False):
        fitted_alphas = np.empty((max(self.subjectData.runNumber)))
        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))
        NLL_array = np.empty((max(self.subjectData.runNumber), self.gridCount, 3))
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
                V_option0[0, :] = 0.5
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                V_option1[0, :] = 0.5
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                betaCheck = betaGrid[j]
                run_V0[j, 0] = V_option0[0, 0, 0]
                run_V1[j, 0] = V_option1[0, 0, 0]
                if pearce:
                    omega = 1
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
                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[:, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]

        newPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters/sub-{}".format(self.ID),
        )
        Path(newPath).mkdir(parents=True, exist_ok=True)
        if pearce:
            scipy.io.savemat(
                newPath + "/rpeSimplePearce.mat".format(self.ID), mdict={"rpe": RPE}
            )
        else:
            scipy.io.savemat(
                newPath + "/rpeSimple.mat".format(self.ID), mdict={"rpe": RPE}
            )

        if self.plotting:
            self.Plot.plots_simplestFitting(
                NLL_array=NLL_array,
                alphas=fitted_alphas,
                betas=fitted_betas,
                pearce=pearce,
            )

        return fitted_alphas, fitted_betas, best_LLs, RPE, V0, V1, NLL_array

    ## Value fitting

    def valFitting(self, pearce=False):
        fitted_alphas = np.empty((max(self.subjectData.runNumber)))
        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))
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

        fitted_V_option0Inits = np.empty((max(self.subjectData.runNumber), 3, 3))
        fitted_V_option1Inits = np.empty((max(self.subjectData.runNumber), 3, 3))

        for run in range(0, max(self.subjectData.runNumber)):
            runData = self.subjectData[
                self.subjectData.runNumber == run + 1
            ].reset_index()
            alphaGrid = np.random.rand(self.gridCount, 1)
            betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
            LL_array = np.empty((self.gridCount, 1))
            # V_option0Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
            V_option0_rand = np.random.rand(self.gridCount, 1)
            V_option0Init_Grid = np.repeat(V_option0_rand, 9, axis=1).reshape(
                (self.gridCount, 3, 3)
            )
            # V_option1Init_Grid = np.random.uniform(0, 1, (self.gridCount, 3, 3))
            V_option1_rand = np.random.rand(self.gridCount, 1)
            V_option1Init_Grid = np.repeat(V_option1_rand, 9, axis=1).reshape(
                (self.gridCount, 3, 3)
            )

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
                V_option0[0, :] = V_option0Init_Grid[j]
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                V_option1[0, :] = V_option1Init_Grid[j]
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                betaCheck = betaGrid[j]
                run_V0[j, 0] = V_option0[0, 0, 0]
                run_V1[j, 0] = V_option1[0, 0, 0]
                if pearce:
                    omega = 1
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
                NLL_array[run, j, 3] = V_option0Init_Grid[j][0][0]
                NLL_array[run, j, 4] = V_option1Init_Grid[j][0][0]
                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[:, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            best_LLs[run] = LL_array[maxIndex]
            fitted_V_option0Inits[run] = V_option0Init_Grid[minIndex]
            fitted_V_option1Inits[run] = V_option1Init_Grid[minIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]

        newPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters/sub-{}".format(self.ID),
        )
        Path(newPath).mkdir(parents=True, exist_ok=True)
        if pearce:
            scipy.io.savemat(
                newPath + "/rpeValPearce.mat".format(self.ID), mdict={"rpe": RPE}
            )
        else:
            scipy.io.savemat(
                newPath + "/rpeVal.mat".format(self.ID), mdict={"rpe": RPE}
            )

        return (
            fitted_alphas,
            fitted_betas,
            best_LLs,
            fitted_V_option0Inits,
            fitted_V_option1Inits,
            RPE,
            V0,
            V1,
            NLL_array,
        )

    # Extra update rule multiple versions

    # In update fitting model, when version is None, this means that the pairs corresponding to the same
    # visual or audio/tactile stimulus gets updated as well in the opposite direction with some
    # different learning rate alpha2
    # Meanwhile when version is "two", then there is action dependance, ACTION AND REWARD (4)
    def updateFitting(self, version=None, pearce=False):
        fitted_alphas = np.empty((max(self.subjectData.runNumber)))
        fitted_alphas2 = np.empty((max(self.subjectData.runNumber)))
        if version == "two":
            fitted_alphas3 = np.empty((max(self.subjectData.runNumber)))
        elif version == "four":
            fitted_alphas3 = np.empty((max(self.subjectData.runNumber)))
            fitted_alphas4 = np.empty((max(self.subjectData.runNumber)))
            fitted_alphas5 = np.empty((max(self.subjectData.runNumber)))

        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))
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
            alpha2Grid = np.random.rand(self.gridCount, 1)
            if version == "two":
                alpha3Grid = np.random.rand(self.gridCount, 1)
            elif version == "four":
                alpha3Grid = np.random.rand(self.gridCount, 1)
                alpha4Grid = np.random.rand(self.gridCount, 1)
                alpha5Grid = np.random.rand(self.gridCount, 1)
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
                V_option0[0, :] = 0.5
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                V_option1[0, :] = 0.5
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                alpha2Check = alpha2Grid[j]
                if version == "two":
                    alpha3Check = alpha3Grid[j]
                elif version == "four":
                    alpha3Check = alpha3Grid[j]
                    alpha4Check = alpha4Grid[j]
                    alpha5Check = alpha5Grid[j]
                betaCheck = betaGrid[j]

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
                                        * alpha2Check
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alpha2Check * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                        else:
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alpha3Check
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alpha3Check * (
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
                                        * alpha2Check
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
                                        alpha2Check
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
                                        * alpha3Check
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
                                        alpha3Check
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
                                            * alpha4Check
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
                                            alpha4Check
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
                                            * alpha5Check
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
                                            alpha5Check
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
                NLL_array[run, j, 3] = alpha2Check
                if version == "two":
                    NLL_array[run, j, 4] = alpha3Check
                elif version == "four":
                    NLL_array[run, j, 4] = alpha3Check
                    NLL_array[run, j, 5] = alpha4Check
                    NLL_array[run, j, 6] = alpha5Check

                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[run, :, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            fitted_alphas2[run] = NLL_array[run, minIndex, 2]

            if version == "two":
                fitted_alphas3[run] = NLL_array[minIndex, 4]
            elif version == "four":
                fitted_alphas3[run] = NLL_array[minIndex, 4]
                fitted_alphas4[run] = NLL_array[minIndex, 5]
                fitted_alphas5[run] = NLL_array[minIndex, 6]

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
                fitted_alphas2,
                fitted_alphas3,
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
                fitted_alphas2,
                fitted_alphas3,
                fitted_alphas4,
                fitted_alphas5,
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
                fitted_alphas2,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                NLL_array,
            )

    # Update and init V0 and V1

    def updateInitFitting(self, version=None, pearce=False):
        fitted_alphas = np.empty((max(self.subjectData.runNumber)))
        fitted_alphas2 = np.empty((max(self.subjectData.runNumber)))
        if version == "two":
            fitted_alphas3 = np.empty((max(self.subjectData.runNumber)))
        elif version == "four":
            fitted_alphas3 = np.empty((max(self.subjectData.runNumber)))
            fitted_alphas4 = np.empty((max(self.subjectData.runNumber)))
            fitted_alphas5 = np.empty((max(self.subjectData.runNumber)))

        fitted_betas = np.empty((max(self.subjectData.runNumber)))
        best_LLs = np.empty((max(self.subjectData.runNumber)))
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

        fitted_V_option0Inits = np.empty((max(self.subjectData.runNumber), 3, 3))
        fitted_V_option1Inits = np.empty((max(self.subjectData.runNumber), 3, 3))

        for run in range(0, max(self.subjectData.runNumber)):
            alphaGrid = np.random.rand(self.gridCount, 1)
            alpha2Grid = np.random.rand(self.gridCount, 1)
            if version == "two":
                alpha3Grid = np.random.rand(self.gridCount, 1)
            elif version == "four":
                alpha3Grid = np.random.rand(self.gridCount, 1)
                alpha4Grid = np.random.rand(self.gridCount, 1)
                alpha5Grid = np.random.rand(self.gridCount, 1)
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
            # V_option0Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
            V_option0_rand = np.random.rand(self.gridCount, 1)
            V_option0Init_Grid = np.repeat(V_option0_rand, 9, axis=1).reshape(
                (self.gridCount, 3, 3)
            )
            # V_option1Init_Grid = np.random.uniform(0, 1, (self.gridCount, 3, 3))
            V_option1_rand = np.random.rand(self.gridCount, 1)
            V_option1Init_Grid = np.repeat(V_option1_rand, 9, axis=1).reshape(
                (self.gridCount, 3, 3)
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
                V_option0[0, :] = V_option0Init_Grid[j]
                V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                V_option1[:] = np.nan
                V_option1[0, :] = V_option1Init_Grid[j]
                rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                rewardPE[:] = np.nan

                # Checking parameters from the grid
                alphaCheck = alphaGrid[j]
                alpha2Check = alpha2Grid[j]
                if version == "two":
                    alpha3Check = alpha3Grid[j]
                elif version == "four":
                    alpha3Check = alpha3Grid[j]
                    alpha4Check = alpha4Grid[j]
                    alpha5Check = alpha5Grid[j]
                betaCheck = betaGrid[j]

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
                                        * alpha2Check
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alpha2Check * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                        else:
                            for pair in otherPairs:
                                if pearce:
                                    omega = (
                                        omega
                                        + (abs(rewardPE[(t,) + pair]) - omega)
                                        * alpha3Check
                                    )
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + omega * (
                                        1 - runData.reward[t] - V_option0[(t,) + pair]
                                    )
                                else:
                                    V_option0[(t + 1,) + pair] = V_option0[
                                        (t,) + pair
                                    ] + alpha3Check * (
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
                                        * alpha2Check
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
                                        alpha2Check
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
                                        * alpha3Check
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
                                        alpha3Check
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
                                            * alpha4Check
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
                                            alpha4Check
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
                                            * alpha5Check
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
                                            alpha5Check
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
                NLL_array[run, j, 3] = V_option0Init_Grid[j][0][0]
                NLL_array[run, j, 4] = V_option1Init_Grid[j][0][0]
                NLL_array[run, j, 5] = alpha2Check
                if version == "two":
                    NLL_array[run, j, 6] = alpha3Check
                elif version == "four":
                    NLL_array[run, j, 6] = alpha3Check
                    NLL_array[run, j, 7] = alpha4Check
                    NLL_array[run, j, 8] = alpha5Check

                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[run, :, 0])

            fitted_alphas[run] = NLL_array[run, minIndex, 0]
            fitted_betas[run] = NLL_array[run, minIndex, 1]
            fitted_alphas2[run] = NLL_array[run, minIndex, 2]
            fitted_alphas2[run] = NLL_array[run, minIndex, 5]
            if version == "two":
                fitted_alphas3[run] = NLL_array[run, minIndex, 6]
            elif version == "four":
                fitted_alphas3[run] = NLL_array[run, minIndex, 6]
                fitted_alphas4[run] = NLL_array[run, minIndex, 7]
                fitted_alphas5[run] = NLL_array[run, minIndex, 8]
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
                    newPath + "/rpeValInit2Pearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeValInit2.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_alphas2,
                fitted_alphas3,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                fitted_V_option0Inits,
                fitted_V_option1Inits,
                NLL_array,
            )
        elif version == "four":
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeValInit4Pearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeValInit4.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_alphas2,
                fitted_alphas3,
                fitted_alphas4,
                fitted_alphas5,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                fitted_V_option0Inits,
                fitted_V_option1Inits,
                NLL_array,
            )
        else:
            if pearce:
                scipy.io.savemat(
                    newPath + "/rpeValInitPearce.mat".format(self.ID),
                    mdict={"rpe": RPE},
                )
            else:
                scipy.io.savemat(
                    newPath + "/rpeValInit.mat".format(self.ID), mdict={"rpe": RPE}
                )
            return (
                fitted_alphas,
                fitted_alphas2,
                fitted_betas,
                best_LLs,
                RPE,
                V0,
                V1,
                fitted_V_option0Inits,
                fitted_V_option1Inits,
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
    fitted = Fitting(60, 0, 5000, ID=IDnr, plotting=False)

    # # Run stat learning
    # beliefs, surprise = fitted.statisticalLearning(statLearnPar=1)

    # # Run the simplest RL model
    (
        fitted_alphas,
        fitted_betas,
        best_LLs,
        RPE,
        V0,
        V1,
        NLL_array,
    ) = fitted.simplestFitting(pearce=True)

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
