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

sys.path.append(sys.path[0] + '/..')
from TaskDesign import task_Design


class Fitting:

    def __init__(self, mainTrials, additionalTrials, gridCount, ID):
        
        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.gridCount = gridCount
        self.ID = ID
        self.statLearnPar = 1

        if platform.system() == 'Windows':
            wanted_dir = 'T:/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri/data/sourcedata/behavior/modified_files'
        else:
            wanted_dir = '/Volumes/g_econ_department$/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri/data/sourcedata/behavior/modified_files'
        # Get savedVals file
        self.savedValsFile = glob.glob(os.path.abspath(wanted_dir)+"/*{}_savedValues.csv".format(self.ID))[0]
    
    ## Simple fitting

    def plots_simplestFitting(self, ww, NLL_array, alphas, betas, method, reps=50):
        
        print("simple")
        saving_folder = "simple"
        subjectData = pd.read_csv(self.savedValsFile)

        for run in range(0, max(subjectData.runNumber)):
            NLL_run = NLL_array[run]
            alpha = alphas[run]
            beta = betas[run]

            runData = subjectData[subjectData.runNumber == run + 1].reset_index()
            green = runData[runData.combinationConditionalProbability == 0.5].stimulusPair.unique()
            green = [ast.literal_eval(green[0]), ast.literal_eval(green[1]), ast.literal_eval(green[2])]

            attracts = runData.accurate[runData.correctResponse == 0]
            notAttracts = runData.accurate[runData.correctResponse == 1]

            
            A_line = pd.DataFrame(ma(attracts, ww, method)).astype(float).interpolate(option='spline', order=1)
            NA_line = pd.DataFrame(ma(notAttracts, ww, method)).astype(float).interpolate(option='spline', order=1)
            Acc_line = pd.DataFrame(ma(runData.accurate, ww, method)).astype(float).interpolate(option='spline', order=1)


            simA_lines = np.empty((reps, int(self.mainTrials / 2)-ww+1))
            simNA_lines = np.empty((reps, int(self.mainTrials / 2)-ww+1))
            simAcc_lines = np.empty((reps, int(self.mainTrials)-ww+1))

            taskStruct = np.array([list(tuple(ast.literal_eval(x))) for x in runData.stimulusPair])

            feedbackAcc = runData.feedbackAccuracy.astype(int)

            for i in range(reps):
                simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta)
                simulation.taskStructure(taskStruct, green, feedbackAcc)
                simulation.RLloops()

                simAttracts = simulation.accurate[simulation.correctResponse == 0]
                simNotAttracts = simulation.accurate[simulation.correctResponse == 1]
                simC = simulation.accurate
                if simAttracts.shape[0] == 30 & simNotAttracts.shape[0] == 30:
                    simA_lines[i, :] = ma(simAttracts, ww, method)
                    simNA_lines[i, :] = ma(simNotAttracts, ww, method)
                    simAcc_lines[i, :] = ma(simC.flatten(), ww, method)

            if simA_lines.size:
                simA_line = pd.DataFrame(np.mean(simA_lines, axis=0))
                simNA_line = pd.DataFrame(np.mean(simNA_lines, axis=0))
                simAcc_line = pd.DataFrame(np.mean(simAcc_lines, axis=0))

                fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, beta {3}"
                                .format(self.ID, run + 1, np.round(alpha, 2), np.round(beta, 2)))

                ax[0].plot(A_line, label='Real data')
                ax[0].plot(simA_line, label='Simulated data')
                ax[0].set_title("Accurate for 'attracts'")
                ax[0].set_ylim(0, 1.1)
                ax[0].set_ylabel("Accurate")
                ax[0].legend()

                ax[1].plot(NA_line, label='Real data')
                ax[1].plot(simNA_line, label='Simulated data')
                ax[1].set_title("Accurate for 'does not attract'")
                ax[1].set_ylim(0, 1.1)
                ax[1].set_ylabel("Accurate")
                ax[1].legend()

                ax[2].plot(Acc_line, label='Real data')
                ax[2].plot(simAcc_line, label='Simulated data')
                ax[2].set_title("Accurate overall")
                ax[2].set_ylim(0, 1.1)
                ax[2].set_ylabel("Accurate")
                ax[2].legend()

                # Creating figure
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                NLL_run[NLL_run[:, -1] > min(NLL_run[:, -1]) + 10] = np.nan
                ax.scatter(NLL_run[:, 0], NLL_run[:, 1], NLL_run[:, -1], color="green")
                ax.set_xlabel('alpha', fontweight='bold')
                ax.set_ylabel('beta', fontweight='bold')
                ax.set_zlabel('NLL', fontweight='bold')
                ax.set_title("Participant {0}, run {1}: NLL, alpha and beta 3D scatter plot".format(self.ID, run + 1))

                plt.figure(3)
                plt.scatter(NLL_run[:, 0], NLL_run[:, -1])
                plt.xlabel('alpha', fontweight='bold')
                plt.ylabel('NLL', fontweight='bold')
                plt.title("Participant {0}, run {1}: NLL and alpha scatter plot".format(self.ID, run + 1))

                plt.figure(4)
                plt.scatter(NLL_run[:, 1], NLL_run[:, -1])
                plt.xlabel('beta', fontweight='bold')
                plt.ylabel('NLL', fontweight='bold')
                plt.title("Participant {0}, run {1}: NLL and beta scatter plot".format(self.ID, run + 1))


                os.makedirs(saving_folder, exist_ok=True)
                save_name = "{0}_{1}_simplePlots.pdf".format(self.ID, run)
                file_path = os.path.join(saving_folder, save_name)

                pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                for fig in range(1, plt.gcf().number + 1):
                    pdf.savefig(fig)
                pdf.close()

                plt.close('all')

    def simplestFitting(self):
              
        subjectData = pd.read_csv(self.savedValsFile)
        subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
        fitted_alphas = np.empty((max(subjectData.runNumber)))
        fitted_betas = np.empty((max(subjectData.runNumber)))
        best_LLs = np.empty((max(subjectData.runNumber)))
        NLL_array = np.empty((max(subjectData.runNumber), self.gridCount, 3))
        NLL_array[:] = np.nan

        RPE = np.empty((max(subjectData.runNumber), self.mainTrials + self.additionalTrials))
        V0 = np.empty((max(subjectData.runNumber), self.mainTrials + self.additionalTrials + 1))
        V1 = np.empty((max(subjectData.runNumber), self.mainTrials + self.additionalTrials + 1))

        for run in range(0, max(subjectData.runNumber)):

            alphaGrid = np.random.rand(self.gridCount, 1)
            betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
            LL_array = np.empty((self.gridCount, 1))
            runData = subjectData[subjectData.runNumber == run + 1].reset_index()
            run_RPEs = np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
            run_V0 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
            run_V1 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
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
                for t in range(0, max(runData.trialNumber)):
                    # Prob of choosing the 0th and 1st option respectively
                    choiceProb[t, 0] = np.exp(betaCheck * V_option0[
                        ((t,) + runData.stimulusPair[t])]) / ((np.exp(
                        betaCheck * V_option0[
                            ((t,) + runData.stimulusPair[t])])) + (np.exp(
                        betaCheck * V_option1[
                            ((t,) + runData.stimulusPair[t])])))
                    choiceProb[t, 1] = 1 - choiceProb[t, 0]

                    actionProb[t, :] = choiceProb[t, int(runData.action[t])] if ~np.isnan(
                        runData.action[t]) else np.nan

                    if runData.action[t] == 0:
                        rewardPE[(t,) + runData.stimulusPair[t]] = \
                            runData.reward[t] - V_option0[(t,) + runData.stimulusPair[t]]

                        V_option0[t + 1, :] = V_option0[t, :]
                        V_option0[(t + 1,) + runData.stimulusPair[t]] = \
                            V_option0[(t,) + runData.stimulusPair[t]] + \
                            alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif runData.action[t] == 1:
                        rewardPE[(t,) + runData.stimulusPair[t]] = \
                            runData.reward[t] - V_option1[(t,) + runData.stimulusPair[t]]

                        V_option1[t + 1, :] = V_option1[t, :]
                        V_option1[(t + 1,) + runData.stimulusPair[t]] = \
                            V_option1[(t,) + runData.stimulusPair[t]] + \
                            alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])
                        V_option0[t + 1, :] = V_option0[t, :]
                    else:
                        V_option1[t + 1, :] = V_option1[t, :]
                        V_option0[t + 1, :] = V_option0[t, :]

                    run_RPEs[j,t] = rewardPE[(t,) + runData.stimulusPair[t]]
                    run_V0[j, t + 1] = V_option0[(t + 1,) + runData.stimulusPair[t]]
                    run_V1[j, t + 1] = V_option1[(t + 1,) + runData.stimulusPair[t]]
                negativeLogLikelihood = -np.sum(np.log(actionProb[~np.isnan(actionProb)]))
                Likelihood = np.prod(actionProb[~np.isnan(actionProb)])
                NLL_array[run, j, 0] = alphaCheck
                NLL_array[run, j, 1] = betaCheck
                NLL_array[run, j, 2] = negativeLogLikelihood
                LL_array[j, 0] = Likelihood

            minIndex = np.argmin(NLL_array[run, :, 2])
            maxIndex = np.nanargmax(LL_array[:, 0])
            fittedAlpha = NLL_array[run, minIndex, 0]
            fittedBeta = NLL_array[run, minIndex, 1]

            fitted_alphas[run] = fittedAlpha
            fitted_betas[run] = fittedBeta
            best_LLs[run] = LL_array[maxIndex]
            RPE[run] = run_RPEs[minIndex]
            V0[run] = run_V0[minIndex]
            V1[run] = run_V1[minIndex]

        return fitted_alphas, fitted_betas, best_LLs, RPE, V0, V1, NLL_array


    ## Value fitting

    def plots_valueFitting(self, ww, method, reps=50):

        count = 0

        for ID in self.IDs:
            print("now")
            saving_folder = "initVal"
            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))

            for run in range(0, max(subjectData.runNumber)):
                NLL_array = self.NLL_arrays[count, run, :, :]
                alpha = self.all_alphas[count, run]
                beta = self.all_betas[count, run]
                V_option0 = self.all_V_option0Inits[count, run]
                V_option1 = self.all_V_option1Inits[count, run]
                runData = subjectData[subjectData.runNumber == run + 1].reset_index()

                attracts = runData.accurate[runData.correctResponse == 0]
                notAttracts = runData.accurate[runData.correctResponse == 1]

                MostAcc = runData.accurate[runData.combinationConditionalProbability == 0.5]
                MiddleAcc = runData.accurate[runData.combinationConditionalProbability == 0.35]
                LeastAcc = runData.accurate[runData.combinationConditionalProbability == 0.15]

                oppositeReward = np.where(runData.reward != runData.correctResponse)[0]
                oppositeRewardNext = oppositeReward + 1
                oppositeRewardNext = oppositeRewardNext[oppositeRewardNext < 60]
                accOppReward = runData.accurate.loc[oppositeRewardNext]
                oppositeRewardPairs = runData.stimulusPair[oppositeReward]
                nextOppPairAcc = list()
                for pair, idx in zip(oppositeRewardPairs, oppositeReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextOppPairAcc.append(temp.accurate[0])

                avgAccOppReward = np.nanmean(accOppReward)
                avgNextOppPairAcc = np.nanmean(nextOppPairAcc)

                correctReward = np.where(runData.reward == runData.correctResponse)[0]
                correctRewardNext = correctReward + 1
                correctRewardNext = correctRewardNext[correctRewardNext < 60]
                accCorrReward = runData.accurate.loc[correctRewardNext]
                correctRewardPairs = runData.stimulusPair[correctReward]
                nextCorrPairAcc = list()
                for pair, idx in zip(correctRewardPairs, correctReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextCorrPairAcc.append(temp.accurate[0])

                avgAccCorrReward = np.nanmean(accCorrReward)
                avgNextCorrPairAcc = np.nanmean(nextCorrPairAcc)

                A_line = pd.DataFrame(ma(attracts, ww, method)).astype(float).interpolate(option='spline', order=1)
                NA_line = pd.DataFrame(ma(notAttracts, ww, method)).astype(float).interpolate(option='spline', order=1)
                Acc_line = pd.DataFrame(ma(runData.accurate, ww, method)).astype(float).interpolate(option='spline', order=1)

                MoA_Line = pd.DataFrame(MostAcc)
                MiA_Line = pd.DataFrame(MiddleAcc)
                LA_line = pd.DataFrame(LeastAcc)

                simA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simNA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simAcc_lines = np.empty((reps, int(self.mainTrials)))

                taskStruct = np.array([list(tuple(ast.literal_eval(x))) for x in runData.stimulusPair])
                green = runData[runData.combinationConditionalProbability == 0.5].stimulusPair.unique()
                green = [ast.literal_eval(green[0]), ast.literal_eval(green[1]), ast.literal_eval(green[2])]

                if 'feedbackAccuracy' in runData.columns:
                    feedbackAcc = runData.feedbackAccuracy.astype(int)
                else:
                    feedbackAcc = np.array(runData.accurate == runData.reward).astype(int)

                    if len(np.where(feedbackAcc[0:10] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[0:10] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[0:10]))[0][:toFlip]
                        feedbackAcc[idx] = 1
                    if len(np.where(feedbackAcc[10:20] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[10:20] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[10:20]))[0][:toFlip]
                        feedbackAcc[idx + 10] = 1
                    if len(np.where(feedbackAcc[20:30] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[20:30] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[20:30]))[0][:toFlip]
                        feedbackAcc[idx + 20] = 1
                    if len(np.where(feedbackAcc[30:40] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[30:40] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[30:40]))[0][:toFlip]
                        feedbackAcc[idx + 30] = 1
                    if len(np.where(feedbackAcc[40:50] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[40:50] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[40:50]))[0][:toFlip]
                        feedbackAcc[idx + 40] = 1
                    if len(np.where(feedbackAcc[50:60] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[50:60] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[50:60]))[0][:toFlip]
                        feedbackAcc[idx + 50] = 1

                for i in range(reps):
                    simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                             V_option0Init=V_option0, V_option1Init=V_option1)
                    simulation.taskStructure(taskStruct, green, feedbackAcc)
                    # simulation.taskStructure()
                    simulation.RLloops()
                    simulation.statisticalLearning()
                    simAttracts = simulation.accurate[simulation.correctResponse == 0]
                    simNotAttracts = simulation.accurate[simulation.correctResponse == 1]
                    simC = simulation.accurate
                    if simAttracts.shape[0] == 30 & simNotAttracts.shape[0] == 30:
                        simA_lines[i, :] = ma(simAttracts, ww, method)
                        simNA_lines[i, :] = ma(simNotAttracts, ww, method)
                        simAcc_lines[i, :] = ma(simC.flatten(), ww, method)

                if simA_lines.size:
                    simA_line = pd.DataFrame(np.mean(simA_lines, axis=0))
                    simNA_line = pd.DataFrame(np.mean(simNA_lines, axis=0))
                    simAcc_line = pd.DataFrame(np.mean(simAcc_lines, axis=0))

                    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                    fig.suptitle("Participant {0}, run {1}: alpha {2}, beta {3}, V_option0 {4} and V_option1 {5}"
                                 .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                         np.round(V_option0[0, 0], 2),
                                         np.round(V_option1[0, 0], 2)))

                    ax[0].plot(A_line, label='Real data')
                    ax[0].plot(simA_line, label='Simulated data')
                    ax[0].set_title("Accurate for 'attracts'")
                    ax[0].set_ylim(0, 1.1)
                    ax[0].set_ylabel("Accurate")
                    ax[0].legend()

                    ax[1].plot(NA_line, label='Real data')
                    ax[1].plot(simNA_line, label='Simulated data')
                    ax[1].set_title("Accurate for 'does not attract'")
                    ax[1].set_ylim(0, 1.1)
                    ax[1].set_ylabel("Accurate")
                    ax[1].legend()

                    ax[2].plot(Acc_line, label='Real data')
                    ax[2].plot(simAcc_line, label='Simulated data')
                    ax[2].set_title("Accurate overall")
                    ax[2].set_ylim(0, 1.1)
                    ax[2].set_ylabel("Accurate")
                    ax[2].legend()

                    fig2, ax2 = plt.subplots(3, 1, figsize=(12, 12))
                    bars = [np.nanmean(LA_line), np.nanmean(MoA_Line), np.nanmean(MiA_Line)]
                    x = ['0.15', '0.35', '0.5']
                    ax2[0].bar(x, bars)
                    ax2[0].set_title("Accuracy per Conditional Probability")
                    ax2[0].set_ylim(0, 1.1)
                    ax2[0].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[1].bar(x, [avgAccOppReward, avgAccCorrReward])
                    ax2[1].set_title("Accuracy on trial after wrong or true reward")
                    ax2[1].set_ylim(0, 1.1)
                    ax2[1].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[2].bar(x, [avgNextOppPairAcc, avgNextCorrPairAcc])
                    ax2[2].set_title("Accuracy on next occurrence same pair after wrong or true reward")
                    ax2[2].set_ylim(0, 1.1)
                    ax2[2].set_ylabel("Average Accurate")

                    # Creating figure
                    fig = plt.figure(3)
                    ax = fig.add_subplot(111, projection='3d')
                    NLL_array[NLL_array[:, 2] > min(NLL_array[:, 2]) + 20] = np.nan
                    ax.scatter(NLL_array[:, 0], NLL_array[:, 1], NLL_array[:, 2], color="green")
                    ax.set_xlabel('alpha', fontweight='bold')
                    ax.set_ylabel('beta', fontweight='bold')
                    ax.set_zlabel('NLL', fontweight='bold')
                    ax.set_title("Participant {0}, run {1}: NLL, alpha and beta 3D scatter plot".format(ID, run + 1))

                    plt.figure(4)
                    plt.scatter(NLL_array[:, 0], NLL_array[:, 2])
                    plt.xlabel('alpha', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and alpha scatter plot".format(ID, run + 1))

                    plt.figure(5)
                    plt.scatter(NLL_array[:, 1], NLL_array[:, 2])
                    plt.xlabel('beta', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and beta scatter plot".format(ID, run + 1))

                    plt.figure(6)
                    plt.scatter(NLL_array[:,3], NLL_array[:, 2])
                    plt.xlabel('Initial V0', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and Init V0 scatter plot".format(ID, run + 1))

                    plt.figure(7)
                    plt.scatter(NLL_array[:,4], NLL_array[:, 2])
                    plt.xlabel('Initial V1', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and Init V1 scatter plot".format(ID, run + 1))

                    os.makedirs(saving_folder, exist_ok=True)
                    save_name = "{0}_{1}_initValPlots.pdf".format(ID, run)
                    file_path = os.path.join(saving_folder, save_name)

                    pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    for fig in range(1, plt.gcf().number + 1):
                        pdf.savefig(fig)
                    pdf.close()

                    plt.close('all')

            count += 1

        # Average over all participants


    def valFitting(self):

        self.all_alphas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_betas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_LLs = np.empty((len(np.unique(self.IDs)), 6))
        self.all_V_option0Inits = np.empty((len(np.unique(self.IDs)), 6, 3, 3))
        self.all_V_option1Inits = np.empty((len(np.unique(self.IDs)), 6, 3, 3))
        self.NLL_arrays = np.empty((len(np.unique(self.IDs)), 6, self.gridCount, 5))
        self.all_RPEs = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials))
        self.all_V0 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))
        self.all_V1 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))

        count = 0
        for ID in np.unique(self.IDs):

            print(ID)
            print("val")
            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))
            subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
            # subjectInfo = pd.read_csv(str([file[1] for file in self.expInfoFiles if file[0] == ID][0]))
            fitted_alphas = np.empty((1, 6))
            fitted_betas = np.empty((1, 6))
            best_LLs = np.empty((1, 6))
            fitted_V_option0Inits = np.empty((1, 6, 3, 3))
            fitted_V_option1Inits = np.empty((1, 6, 3, 3))

            ID_RPE = np.empty((1, 6, self.mainTrials + self.additionalTrials))
            ID_V0 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))
            ID_V1 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))

            for run in range(0, max(subjectData.runNumber)):

                runData = subjectData[subjectData.runNumber == run + 1].reset_index()
                alphaGrid = np.random.rand(self.gridCount, 1)
                betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
                # V_option0Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
                V_option0_rand = np.random.rand(self.gridCount, 1)
                V_option0Init_Grid = np.repeat(V_option0_rand, 9, axis=1).reshape((self.gridCount, 3, 3))
                # V_option1Init_Grid = np.random.uniform(0, 1, (self.gridCount, 3, 3))
                V_option1_rand = np.random.rand(self.gridCount, 1)
                V_option1Init_Grid = np.repeat(V_option1_rand, 9, axis=1).reshape((self.gridCount, 3, 3))
                NLL_array = np.empty((self.gridCount, 5))
                NLL_array[:] = np.nan
                LL_array = np.empty((self.gridCount, 1))

                run_RPEs = np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
                run_V0 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
                run_V1 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))

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
                    trials_RPE = np.empty((max(runData.trialNumber)))
                    trials_V0 = np.empty((max(runData.trialNumber) + 1))
                    trials_V1 = np.empty((max(runData.trialNumber) + 1))
                    trials_V0[0] = V_option0[0, 0, 0]
                    trials_V1[0] = V_option1[0, 0, 0]

                    for t in range(0, max(runData.trialNumber)):
                        # Prob of choosing the 0th and 1st option respectively
                        choiceProb[t, 0] = np.exp(betaCheck * V_option0[
                            ((t,) + runData.stimulusPair[t])]) / ((np.exp(
                            betaCheck * V_option0[
                                ((t,) + runData.stimulusPair[t])])) + (np.exp(
                            betaCheck * V_option1[
                                ((t,) + runData.stimulusPair[t])])))
                        choiceProb[t, 1] = 1 - choiceProb[t, 0]

                        actionProb[t, :] = choiceProb[t, int(runData.action[t])] if ~np.isnan(
                            runData.action[t]) else np.nan

                        if runData.action[t] == 0:
                            rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - V_option0[(t,) + runData.stimulusPair[t]]

                            V_option0[t + 1, :] = V_option0[t, :]
                            V_option0[(t + 1,) + runData.stimulusPair[t]] = \
                                V_option0[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])

                            V_option1[t + 1, :] = V_option1[t, :]

                        elif runData.action[t] == 1:
                            rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - V_option1[(t,) + runData.stimulusPair[t]]

                            V_option1[t + 1, :] = V_option1[t, :]
                            V_option1[(t + 1,) + runData.stimulusPair[t]] = \
                                V_option1[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (rewardPE[(t,) + runData.stimulusPair[t]])
                            V_option0[t + 1, :] = V_option0[t, :]
                        else:
                            V_option1[t + 1, :] = V_option1[t, :]
                            V_option0[t + 1, :] = V_option0[t, :]

                        trials_RPE[t] = rewardPE[(t,) + runData.stimulusPair[t]]
                        trials_V0[t + 1] = V_option0[(t + 1,) + runData.stimulusPair[t]]
                        trials_V1[t + 1] = V_option1[(t + 1,) + runData.stimulusPair[t]]

                    negativeLogLikelihood = -np.sum(np.log(actionProb[~np.isnan(actionProb)]))
                    Likelihood = np.prod(actionProb[~np.isnan(actionProb)])
                    NLL_array[j, 0] = alphaCheck
                    NLL_array[j, 1] = betaCheck
                    NLL_array[j, 2] = negativeLogLikelihood
                    NLL_array[j, 3] = V_option0Init_Grid[j][0][0]
                    NLL_array[j, 4] = V_option1Init_Grid[j][0][0]
                    LL_array[j, 0] = Likelihood

                    run_RPEs[j] = trials_RPE
                    run_V0[j] = trials_V0
                    run_V1[j] = trials_V1

                minIndex = np.argmin(NLL_array[:, 2])
                maxIndex = np.nanargmax(LL_array[:, 0])
                fittedAlpha = NLL_array[minIndex, 0]
                fittedBeta = NLL_array[minIndex, 1]

                self.NLL_arrays[count, run, :, :] = NLL_array
                fitted_alphas[0, run] = fittedAlpha
                fitted_betas[0, run] = fittedBeta
                best_LLs[0, run] = LL_array[maxIndex]
                fitted_V_option0Inits[0, run] = V_option0Init_Grid[minIndex]
                fitted_V_option1Inits[0, run] = V_option1Init_Grid[minIndex]
                ID_RPE[0, run] = run_RPEs[minIndex]
                ID_V0[0, run] = run_V0[minIndex]
                ID_V1[0, run] = run_V1[minIndex]

            self.all_alphas[count, :] = fitted_alphas
            self.all_betas[count, :] = fitted_betas
            self.all_LLs[count, :] = best_LLs
            self.all_V_option0Inits[count, :] = fitted_V_option0Inits
            self.all_V_option1Inits[count, :] = fitted_V_option1Inits
            self.all_RPEs[count] = ID_RPE
            self.all_V0[count] = ID_V0
            self.all_V1[count] = ID_V1

            count += 1

    # Extra update rule multiple versions

    def plots_updateFitting(self, ww, method, reps=50, version=None):

        count = 0
        for ID in self.IDs:
            print(ID)
            print("update")
            if version == None:
                saving_folder = "updateSimple"
            elif version == "two":
                print("action")
                saving_folder = "updateAction"
            elif version == "four":
                print("actionReward")
                saving_folder = "updateActionReward"

            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))

            for run in range(0, max(subjectData.runNumber)):
                NLL_array = self.NLL_arrays[count, run, :, :]
                alpha = self.all_alphas[count, run]
                beta = self.all_betas[count, run]
                alpha2 = self.all_alphas2[count, run]
                if version == "two":
                    alpha3 = self.all_alphas3[count, run]
                elif version == "four":
                    alpha3 = self.all_alphas3[count, run]
                    alpha4 = self.all_alphas4[count, run]
                    alpha5 = self.all_alphas5[count, run]

                runData = subjectData[subjectData.runNumber == run + 1].reset_index()
                green = runData[runData.combinationConditionalProbability == 0.5].stimulusPair.unique()
                green = [ast.literal_eval(green[0]), ast.literal_eval(green[1]), ast.literal_eval(green[2])]

                attracts = runData.accurate[runData.correctResponse == 0]
                notAttracts = runData.accurate[runData.correctResponse == 1]

                MostAcc = runData.accurate[runData.combinationConditionalProbability == 0.5]
                MiddleAcc = runData.accurate[runData.combinationConditionalProbability == 0.35]
                LeastAcc = runData.accurate[runData.combinationConditionalProbability == 0.15]

                oppositeReward = np.where(runData.reward != runData.correctResponse)[0]
                oppositeRewardNext = oppositeReward + 1
                oppositeRewardNext = oppositeRewardNext[oppositeRewardNext < 60]
                accOppReward = runData.accurate.loc[oppositeRewardNext]
                oppositeRewardPairs = runData.stimulusPair[oppositeReward]
                nextOppPairAcc = list()
                for pair, idx in zip(oppositeRewardPairs, oppositeReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextOppPairAcc.append(temp.accurate[0])

                avgAccOppReward = np.nanmean(accOppReward)
                avgNextOppPairAcc = np.nanmean(nextOppPairAcc)

                correctReward = np.where(runData.reward == runData.correctResponse)[0]
                correctRewardNext = correctReward + 1
                correctRewardNext = correctRewardNext[correctRewardNext < 60]
                accCorrReward = runData.accurate.loc[correctRewardNext]
                correctRewardPairs = runData.stimulusPair[correctReward]
                nextCorrPairAcc = list()
                for pair, idx in zip(correctRewardPairs, correctReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextCorrPairAcc.append(temp.accurate[0])

                avgAccCorrReward = np.nanmean(accCorrReward)
                avgNextCorrPairAcc = np.nanmean(nextCorrPairAcc)

                A_line = pd.DataFrame(ma(attracts, ww, method)).fillna(method='ffill')
                NA_line = pd.DataFrame(ma(notAttracts, ww, method)).fillna(method='ffill')
                Acc_line = pd.DataFrame(ma(runData.accurate, ww, method)).fillna(method='ffill')

                MoA_Line = pd.DataFrame(MostAcc)
                MiA_Line = pd.DataFrame(MiddleAcc)
                LA_line = pd.DataFrame(LeastAcc)

                simA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simNA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simAcc_lines = np.empty((reps, int(self.mainTrials)))

                taskStruct = np.array([list(tuple(ast.literal_eval(x))) for x in runData.stimulusPair])

                if 'feedbackAccuracy' in runData.columns:
                    feedbackAcc = runData.feedbackAccuracy.astype(int)
                else:
                    feedbackAcc = np.array(runData.accurate == runData.reward).astype(int)

                    if len(np.where(feedbackAcc[0:10] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[0:10] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[0:10]))[0][:toFlip]
                        feedbackAcc[idx] = 1
                    if len(np.where(feedbackAcc[10:20] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[10:20] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[10:20]))[0][:toFlip]
                        feedbackAcc[idx + 10] = 1
                    if len(np.where(feedbackAcc[20:30] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[20:30] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[20:30]))[0][:toFlip]
                        feedbackAcc[idx + 20] = 1
                    if len(np.where(feedbackAcc[30:40] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[30:40] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[30:40]))[0][:toFlip]
                        feedbackAcc[idx + 30] = 1
                    if len(np.where(feedbackAcc[40:50] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[40:50] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[40:50]))[0][:toFlip]
                        feedbackAcc[idx + 40] = 1
                    if len(np.where(feedbackAcc[50:60] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[50:60] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[50:60]))[0][:toFlip]
                        feedbackAcc[idx + 50] = 1

                for i in range(reps):
                    if version is None:
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2)
                    elif version == "two":
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2, alpha3=alpha3)
                    else:
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, alpha5=alpha5)
                    simulation.taskStructure(taskStruct, green, feedbackAcc)
                    # simulation.taskStructure()
                    simulation.RLloops()
                    simAttracts = simulation.accurate[simulation.correctResponse == 0]
                    simNotAttracts = simulation.accurate[simulation.correctResponse == 1]
                    simC = simulation.accurate
                    if simAttracts.shape[0] == 30 & simNotAttracts.shape[0] == 30:
                        simA_lines[i, :] = ma(simAttracts, ww, method)
                        simNA_lines[i, :] = ma(simNotAttracts, ww, method)
                        simAcc_lines[i, :] = ma(simC.flatten(), ww, method)

                if simA_lines.size:
                    simA_line = pd.DataFrame(np.mean(simA_lines, axis=0))
                    simNA_line = pd.DataFrame(np.mean(simNA_lines, axis=0))
                    simAcc_line = pd.DataFrame(np.mean(simAcc_lines, axis=0))

                    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                    if version is None:
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2)))
                    elif version == "two":
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}, alpha3 {5}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2), np.round(alpha3, 2)))
                    else:
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}, alpha3 {5}, alpha4 {6}, alpha5 {7}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2), np.round(alpha3, 2),
                                             np.round(alpha4, 2), np.round(alpha5, 2)))

                    ax[0].plot(A_line, label='Real data')
                    ax[0].plot(simA_line, label='Simulated data')
                    ax[0].set_title("Accurate for 'attracts'")
                    ax[0].set_ylim(0, 1.1)
                    ax[0].set_ylabel("Accurate")
                    ax[0].legend()

                    ax[1].plot(NA_line, label='Real data')
                    ax[1].plot(simNA_line, label='Simulated data')
                    ax[1].set_title("Accurate for 'does not attract'")
                    ax[1].set_ylim(0, 1.1)
                    ax[1].set_ylabel("Accurate")
                    ax[1].legend()

                    ax[2].plot(Acc_line, label='Real data')
                    ax[2].plot(simAcc_line, label='Simulated data')
                    ax[2].set_title("Accurate overall")
                    ax[2].set_ylim(0, 1.1)
                    ax[2].set_ylabel("Accurate")
                    ax[2].legend()

                    fig2, ax2 = plt.subplots(3, 1, figsize=(12, 12))
                    bars = [np.nanmean(LA_line), np.nanmean(MoA_Line), np.nanmean(MiA_Line)]
                    x = ['0.15', '0.35', '0.5']
                    ax2[0].bar(x, bars)
                    ax2[0].set_title("Accuracy per Conditional Probability")
                    ax2[0].set_ylim(0, 1.1)
                    ax2[0].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[1].bar(x, [avgAccOppReward, avgAccCorrReward])
                    ax2[1].set_title("Accuracy on trial after wrong or true reward")
                    ax2[1].set_ylim(0, 1.1)
                    ax2[1].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[2].bar(x, [avgNextOppPairAcc, avgNextCorrPairAcc])
                    ax2[2].set_title("Accuracy on next occurrence same pair after wrong or true reward")
                    ax2[2].set_ylim(0, 1.1)
                    ax2[2].set_ylabel("Average Accurate")

                    # Creating figure
                    fig = plt.figure(3)
                    ax = fig.add_subplot(111, projection='3d')
                    NLL_array[NLL_array[:, 2] > min(NLL_array[:, 2]) + 20] = np.nan
                    ax.scatter(NLL_array[:, 0], NLL_array[:, 1], NLL_array[:, 2], color="green")
                    ax.set_xlabel('alpha', fontweight='bold')
                    ax.set_ylabel('beta', fontweight='bold')
                    ax.set_zlabel('NLL', fontweight='bold')
                    ax.set_title("Participant {0}, run {1}: NLL, alpha and beta 3D scatter plot".format(ID, run + 1))

                    plt.figure(4)
                    plt.scatter(NLL_array[:, 0], NLL_array[:, 2])
                    plt.xlabel('alpha', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and alpha scatter plot".format(ID, run + 1))

                    plt.figure(5)
                    plt.scatter(NLL_array[:, 1], NLL_array[:, 2])
                    plt.xlabel('beta', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and beta scatter plot".format(ID, run + 1))

                    plt.figure(6)
                    plt.scatter(NLL_array[:, 3], NLL_array[:, 2])
                    plt.xlabel('alpha2', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and alpha2 scatter plot".format(ID, run + 1))

                    if version == "two":
                        plt.figure(7)
                        plt.scatter(NLL_array[:, 4], NLL_array[:, 2])
                        plt.xlabel('alpha3', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha3 scatter plot".format(ID, run + 1))

                    elif version == "four":
                        plt.figure(7)
                        plt.scatter(NLL_array[:, 4], NLL_array[:, 2])
                        plt.xlabel('alpha3', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha3 scatter plot".format(ID, run + 1))

                        plt.figure(8)
                        plt.scatter(NLL_array[:, 5], NLL_array[:, 2])
                        plt.xlabel('alpha4', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha4 scatter plot".format(ID, run + 1))

                        plt.figure(9)
                        plt.scatter(NLL_array[:, 6], NLL_array[:, 2])
                        plt.xlabel('alpha5', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha5 scatter plot".format(ID, run + 1))


                    os.makedirs(saving_folder, exist_ok=True)

                    if version is None:
                        save_name = "{0}_{1}_simpleUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    elif version == "two":
                        save_name = "{0}_{1}_actionUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    else:
                        save_name = "{0}_{1}_actionRewardUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    
                    for fig in range(1, plt.gcf().number + 1):
                        pdf.savefig(fig)
                    pdf.close()

                    plt.close('all')


            count += 1

# In update fitting model, when version is None, this means that the pairs corresponding to the same
# visual or audio/tactile stimulus gets updated as well in the opposite direction with some 
# different learning rate alpha2
# Meanwhile when version is "two", then there is action dependance, ACTION AND REWARD (4)
    def updateFitting(self, version=None):

        self.all_alphas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_betas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_alphas2 = np.empty((len(np.unique(self.IDs)), 6))
        if version == "two":
            self.all_alphas3 = np.empty((len(np.unique(self.IDs)), 6))
        elif version == "four":
            self.all_alphas3 = np.empty((len(np.unique(self.IDs)), 6))
            self.all_alphas4 = np.empty((len(np.unique(self.IDs)), 6))
            self.all_alphas5 = np.empty((len(np.unique(self.IDs)), 6))
        self.all_LLs = np.empty((len(np.unique(self.IDs)), 6))
        self.NLL_arrays = np.empty((len(np.unique(self.IDs)), 6, self.gridCount, 7))
        self.all_RPEs = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials))
        self.all_V0 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))
        self.all_V1 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))

        count = 0
        for ID in np.unique(self.IDs):
            print(ID)
            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))
            subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
            # subjectInfo = pd.read_csv(str([file[1] for file in self.expInfoFiles if file[0] == ID][0]))
            fitted_alphas = np.empty((1, 6))
            fitted_betas = np.empty((1, 6))
            fitted_alphas2 = np.empty((1, 6))
            if version == "two":
                fitted_alphas3 = np.empty((1, 6))
            elif version == "four":
                fitted_alphas3 = np.empty((1, 6))
                fitted_alphas4 = np.empty((1, 6))
                fitted_alphas5 = np.empty((1, 6))
            best_LLs = np.empty((1, 6))
            ID_RPE = np.empty((1, 6, self.mainTrials + self.additionalTrials))
            ID_V0 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))
            ID_V1 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))

            for run in range(0, max(subjectData.runNumber)):

                alphaGrid = np.random.rand(self.gridCount, 1)
                alpha2Grid = np.random.rand(self.gridCount, 1)
                if version == "two":
                    alpha3Grid = np.random.rand(self.gridCount, 1)
                elif version == "four":
                    alpha3Grid = np.random.rand(self.gridCount, 1)
                    alpha4Grid = np.random.rand(self.gridCount, 1)
                    alpha5Grid = np.random.rand(self.gridCount, 1)
                betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
                NLL_array = np.empty((self.gridCount, 7))
                NLL_array[:] = np.nan
                LL_array = np.empty((self.gridCount, 1))
                runData = subjectData[subjectData.runNumber == run + 1].reset_index()
                run_RPEs = np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
                run_V0 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
                run_V1 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
                # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
                for j in range(0, self.gridCount):
                    # For each point on the grid we instantiate the arrays for the time steps-
                    """Instantiating for the fitting"""

                    self.choiceProb = np.empty((max(runData.trialNumber), 2))
                    self.choiceProb[:] = np.nan
                    self.actionProb = np.empty((max(runData.trialNumber), 1))
                    self.actionProb[:] = np.nan
                    self.V_option0 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                    self.V_option0[:] = np.nan
                    self.V_option0[0, :] = 0.5
                    self.V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                    self.V_option1[:] = np.nan
                    self.V_option1[0, :] = 0.5
                    self.rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                    self.rewardPE[:] = np.nan

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
                    trials_RPE = np.empty((max(runData.trialNumber)))
                    trials_V0 = np.empty((max(runData.trialNumber) + 1))
                    trials_V1 = np.empty((max(runData.trialNumber) + 1))
                    trials_V0[0] = self.V_option0[0, 0, 0]
                    trials_V1[0] = self.V_option1[0, 0, 0]
                    for t in range(0, max(runData.trialNumber)):
                        otherPairs = [p for p in list(runData.stimulusPair.unique())
                                      if bool(p[0] == runData.stimulusPair[t][0]) ^
                                      bool(p[1] == runData.stimulusPair[t][1])]

                        # Prob of choosing the 0th and 1st option respectively
                        self.choiceProb[t, 0] = np.exp(betaCheck * self.V_option0[
                            ((t,) + runData.stimulusPair[t])]) / ((np.exp(
                            betaCheck * self.V_option0[
                                ((t,) + runData.stimulusPair[t])])) + (np.exp(
                            betaCheck * self.V_option1[
                                ((t,) + runData.stimulusPair[t])])))
                        self.choiceProb[t, 1] = 1 - self.choiceProb[t, 0]

                        self.actionProb[t, :] = self.choiceProb[t, int(runData.action[t])] if ~np.isnan(
                            runData.action[t]) else np.nan

                        if runData.action[t] == 0:
                            self.rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - self.V_option0[(t,) + runData.stimulusPair[t]]

                            self.V_option0[t + 1, :] = self.V_option0[t, :]
                            self.V_option0[(t + 1,) + runData.stimulusPair[t]] = \
                                self.V_option0[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (self.rewardPE[(t,) + runData.stimulusPair[t]])

                            if version is None or version == "two" or (version == "four" and runData.reward[t] == 1):
                                for pair in otherPairs:
                                    self.V_option0[(t + 1,) + pair] = \
                                        self.V_option0[(t,) + pair] + \
                                        alpha2Check * (1 - runData.reward[t] - self.V_option0[(t,) + pair])
                            else:
                                for pair in otherPairs:
                                    self.V_option0[(t + 1,) + pair] = \
                                            self.V_option0[(t,) + pair] + \
                                            alpha3Check * (1 - runData.reward[t] - self.V_option0[(t,) + pair])
                            self.V_option1[t + 1, :] = self.V_option1[t, :]

                        elif runData.action[t] == 1:
                            self.rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - self.V_option1[(t,) + runData.stimulusPair[t]]

                            self.V_option1[t + 1, :] = self.V_option1[t, :]
                            self.V_option1[(t + 1,) + runData.stimulusPair[t]] = \
                                self.V_option1[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (self.rewardPE[(t,) + runData.stimulusPair[t]])

                            if version is None:
                                for pair in otherPairs:
                                    self.V_option1[(t + 1,) + pair] = \
                                        self.V_option1[(t,) + pair] + \
                                        (alpha2Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                            elif version == "two":
                                for pair in otherPairs:
                                    self.V_option1[(t + 1,) + pair] = \
                                        self.V_option1[(t,) + pair] + \
                                        (alpha3Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                            else:
                                if runData.reward[t] == 1:
                                    for pair in otherPairs:
                                        self.V_option1[(t + 1,) + pair] = \
                                            self.V_option1[(t,) + pair] + \
                                            (alpha4Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                                else:
                                    for pair in otherPairs:
                                        self.V_option1[(t + 1,) + pair] = \
                                            self.V_option1[(t,) + pair] + \
                                            (alpha5Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))

                            self.V_option0[t + 1, :] = self.V_option0[t, :]

                        else:
                            self.V_option0[t + 1, :] = self.V_option0[t, :]
                            self.V_option1[t + 1, :] = self.V_option1[t, :]

                        trials_RPE[t] = self.rewardPE[(t,) + runData.stimulusPair[t]]
                        trials_V0[t + 1] = self.V_option0[(t + 1,) + runData.stimulusPair[t]]
                        trials_V1[t + 1] = self.V_option1[(t + 1,) + runData.stimulusPair[t]]
                    negativeLogLikelihood = -np.sum(np.log(self.actionProb[~np.isnan(self.actionProb)]))
                    Likelihood = np.prod(self.actionProb[~np.isnan(self.actionProb)])
                    NLL_array[j, 0] = alphaCheck
                    NLL_array[j, 1] = betaCheck
                    NLL_array[j, 2] = negativeLogLikelihood
                    NLL_array[j, 3] = alpha2Check
                    if version == "two":
                        NLL_array[j, 4] = alpha3Check
                    elif version == "four":
                        NLL_array[j, 4] = alpha3Check
                        NLL_array[j, 5] = alpha4Check
                        NLL_array[j, 6] = alpha5Check
                    LL_array[j] = Likelihood
                    run_RPEs[j, :] = trials_RPE
                    run_V0[j, :] = trials_V0
                    run_V1[j, :] = trials_V1

                minIndex = np.argmin(NLL_array[:, 2])
                maxIndex = np.nanargmax(LL_array[:, 0])

                self.NLL_arrays[count, run, :, :] = NLL_array
                fitted_alphas[0, run] = NLL_array[minIndex, 0]
                fitted_betas[0, run] = NLL_array[minIndex, 1]
                fitted_alphas2[0, run] = NLL_array[minIndex, 3]
                if version == "two":
                    fitted_alphas3[0, run] = NLL_array[minIndex, 4]
                elif version == "four":
                    fitted_alphas3[0, run] = NLL_array[minIndex, 4]
                    fitted_alphas4[0, run] = NLL_array[minIndex, 5]
                    fitted_alphas5[0, run] = NLL_array[minIndex, 6]
                best_LLs[0, run] = LL_array[maxIndex]
                ID_RPE[0, run] = run_RPEs[minIndex, :]
                ID_V0[0, run] = run_V0[minIndex, :]
                ID_V1[0, run] = run_V1[minIndex, :]

            self.all_alphas[count, :] = fitted_alphas
            self.all_betas[count, :] = fitted_betas
            self.all_alphas2[count, :] = fitted_alphas2
            if version == "two":
                self.all_alphas3[count, :] = fitted_alphas3
            elif version == "four":
                self.all_alphas3[count, :] = fitted_alphas3
                self.all_alphas4[count, :] = fitted_alphas4
                self.all_alphas5[count, :] = fitted_alphas5
            self.all_RPEs[count] = ID_RPE
            self.all_V0[count] = ID_V0
            self.all_V1[count] = ID_V1
            self.all_LLs[count, :] = best_LLs

            count += 1

# Update and Init V0 and V1

    def plots_updateInitFitting(self, ww, method, reps=50, version=None):

        count = 0
        for ID in self.IDs:
            print(ID)
            print("initUpdate")
            if version == None:
                saving_folder = "initValUpdateSimple"
            elif version == "two":
                saving_folder = "initValUpdateAction"
            elif version == "four":
                saving_folder = "initValUpdateActionReward"

            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))

            for run in range(0, max(subjectData.runNumber)):
                NLL_array = self.NLL_arrays[count, run, :, :]
                alpha = self.all_alphas[count, run]
                beta = self.all_betas[count, run]
                alpha2 = self.all_alphas2[count, run]
                if version == "two":
                    alpha3 = self.all_alphas3[count, run]
                elif version == "four":
                    alpha3 = self.all_alphas3[count, run]
                    alpha4 = self.all_alphas4[count, run]
                    alpha5 = self.all_alphas5[count, run]
                V_option0 = self.all_V_option0Inits[count, run]
                V_option1 = self.all_V_option1Inits[count, run]

                runData = subjectData[subjectData.runNumber == run + 1].reset_index()
                green = runData[runData.combinationConditionalProbability == 0.5].stimulusPair.unique()
                green = [ast.literal_eval(green[0]), ast.literal_eval(green[1]), ast.literal_eval(green[2])]

                attracts = runData.accurate[runData.correctResponse == 0]
                notAttracts = runData.accurate[runData.correctResponse == 1]

                MostAcc = runData.accurate[runData.combinationConditionalProbability == 0.5]
                MiddleAcc = runData.accurate[runData.combinationConditionalProbability == 0.35]
                LeastAcc = runData.accurate[runData.combinationConditionalProbability == 0.15]

                oppositeReward = np.where(runData.reward != runData.correctResponse)[0]
                oppositeRewardNext = oppositeReward + 1
                oppositeRewardNext = oppositeRewardNext[oppositeRewardNext < 60]
                accOppReward = runData.accurate.loc[oppositeRewardNext]
                oppositeRewardPairs = runData.stimulusPair[oppositeReward]
                nextOppPairAcc = list()
                for pair, idx in zip(oppositeRewardPairs, oppositeReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextOppPairAcc.append(temp.accurate[0])

                avgAccOppReward = np.nanmean(accOppReward)
                avgNextOppPairAcc = np.nanmean(nextOppPairAcc)

                correctReward = np.where(runData.reward == runData.correctResponse)[0]
                correctRewardNext = correctReward + 1
                correctRewardNext = correctRewardNext[correctRewardNext < 60]
                accCorrReward = runData.accurate.loc[correctRewardNext]
                correctRewardPairs = runData.stimulusPair[correctReward]
                nextCorrPairAcc = list()
                for pair, idx in zip(correctRewardPairs, correctReward):
                    subset = runData.loc[idx + 1:, ['stimulusPair', 'accurate']]
                    temp = subset[subset.stimulusPair == pair].reset_index()
                    if not temp.empty:
                        nextCorrPairAcc.append(temp.accurate[0])

                avgAccCorrReward = np.nanmean(accCorrReward)
                avgNextCorrPairAcc = np.nanmean(nextCorrPairAcc)

                A_line = pd.DataFrame(ma(attracts, ww, method)).fillna(method='ffill')
                NA_line = pd.DataFrame(ma(notAttracts, ww, method)).fillna(method='ffill')
                Acc_line = pd.DataFrame(ma(runData.accurate, ww, method)).fillna(method='ffill')

                MoA_Line = pd.DataFrame(MostAcc)
                MiA_Line = pd.DataFrame(MiddleAcc)
                LA_line = pd.DataFrame(LeastAcc)

                simA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simNA_lines = np.empty((reps, int(self.mainTrials / 2)))
                simAcc_lines = np.empty((reps, int(self.mainTrials)))

                taskStruct = np.array([list(tuple(ast.literal_eval(x))) for x in runData.stimulusPair])

                if 'feedbackAccuracy' in runData.columns:
                    feedbackAcc = runData.feedbackAccuracy.astype(int)
                else:
                    feedbackAcc = np.array(runData.accurate == runData.reward).astype(int)

                    if len(np.where(feedbackAcc[0:10] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[0:10] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[0:10]))[0][:toFlip]
                        feedbackAcc[idx] = 1
                    if len(np.where(feedbackAcc[10:20] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[10:20] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[10:20]))[0][:toFlip]
                        feedbackAcc[idx + 10] = 1
                    if len(np.where(feedbackAcc[20:30] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[20:30] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[20:30]))[0][:toFlip]
                        feedbackAcc[idx + 20] = 1
                    if len(np.where(feedbackAcc[30:40] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[30:40] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[30:40]))[0][:toFlip]
                        feedbackAcc[idx + 30] = 1
                    if len(np.where(feedbackAcc[40:50] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[40:50] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[40:50]))[0][:toFlip]
                        feedbackAcc[idx + 40] = 1
                    if len(np.where(feedbackAcc[50:60] == 0)[0]) > 2:
                        toFlip = len(np.where(feedbackAcc[50:60] == 0)[0])
                        idx = np.where(np.isnan(runData.accurate[50:60]))[0][:toFlip]
                        feedbackAcc[idx + 50] = 1

                for i in range(reps):
                    if version is None:
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2, V_option0Init=V_option0, V_option1Init=V_option1)
                    elif version == "two":
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2, alpha3=alpha3, V_option0Init=V_option0, V_option1Init=V_option1)
                    else:
                        simulation = task_Design(self.mainTrials, self.additionalTrials, alpha=alpha, beta=beta,
                                                 alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, alpha5=alpha5,
                                                 V_option0Init=V_option0, V_option1Init=V_option1)
                    simulation.taskStructure(taskStruct, green, feedbackAcc)
                    # simulation.taskStructure()
                    simulation.RLloops()
                    simAttracts = simulation.accurate[simulation.correctResponse == 0]
                    simNotAttracts = simulation.accurate[simulation.correctResponse == 1]
                    simC = simulation.accurate
                    if simAttracts.shape[0] == 30 & simNotAttracts.shape[0] == 30:
                        simA_lines[i, :] = ma(simAttracts, ww, method)
                        simNA_lines[i, :] = ma(simNotAttracts, ww, method)
                        simAcc_lines[i, :] = ma(simC.flatten(), ww, method)

                if simA_lines.size:
                    simA_line = pd.DataFrame(np.mean(simA_lines, axis=0))
                    simNA_line = pd.DataFrame(np.mean(simNA_lines, axis=0))
                    simAcc_line = pd.DataFrame(np.mean(simAcc_lines, axis=0))

                    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                    if version is None:
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}, V0Init {5}, V1Init {6}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2), np.round(V_option0[0][0], 2),
                                             np.round(V_option1[0][0], 2)))
                    elif version == "two":
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}, alpha3 {5}, V0Init {6}, V1Init {7}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2), np.round(alpha3, 2),
                                             np.round(V_option0[0][0], 2), np.round(V_option1[0][0], 2)))
                    else:
                        fig.suptitle("MA of binary accuracy for participant {0}, run {1}: alpha {2}, "
                                     "beta {3}, alpha2 {4}, alpha3 {5}, alpha4 {6}, alpha5 {7},"
                                     "V0Init {8}, V1Init {9}"
                                     .format(ID, run + 1, np.round(alpha, 2), np.round(beta, 2),
                                             np.round(alpha2, 2), np.round(alpha3, 2),
                                             np.round(alpha4, 2), np.round(alpha5, 2),
                                             np.round(V_option0[0][0], 2), np.round(V_option1[0][0], 2)))

                    ax[0].plot(A_line, label='Real data')
                    ax[0].plot(simA_line, label='Simulated data')
                    ax[0].set_title("Accurate for 'attracts'")
                    ax[0].set_ylim(0, 1.1)
                    ax[0].set_ylabel("Accurate")
                    ax[0].legend()

                    ax[1].plot(NA_line, label='Real data')
                    ax[1].plot(simNA_line, label='Simulated data')
                    ax[1].set_title("Accurate for 'does not attract'")
                    ax[1].set_ylim(0, 1.1)
                    ax[1].set_ylabel("Accurate")
                    ax[1].legend()

                    ax[2].plot(Acc_line, label='Real data')
                    ax[2].plot(simAcc_line, label='Simulated data')
                    ax[2].set_title("Accurate overall")
                    ax[2].set_ylim(0, 1.1)
                    ax[2].set_ylabel("Accurate")
                    ax[2].legend()

                    fig2, ax2 = plt.subplots(3, 1, figsize=(12, 12))
                    bars = [np.nanmean(LA_line), np.nanmean(MoA_Line), np.nanmean(MiA_Line)]
                    x = ['0.15', '0.35', '0.5']
                    ax2[0].bar(x, bars)
                    ax2[0].set_title("Accuracy per Conditional Probability")
                    ax2[0].set_ylim(0, 1.1)
                    ax2[0].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[1].bar(x, [avgAccOppReward, avgAccCorrReward])
                    ax2[1].set_title("Accuracy on trial after wrong or true reward")
                    ax2[1].set_ylim(0, 1.1)
                    ax2[1].set_ylabel("Average Accurate")

                    x = ['Wrong reward', 'True reward']
                    ax2[2].bar(x, [avgNextOppPairAcc, avgNextCorrPairAcc])
                    ax2[2].set_title("Accuracy on next occurrence same pair after wrong or true reward")
                    ax2[2].set_ylim(0, 1.1)
                    ax2[2].set_ylabel("Average Accurate")

                    # Creating figure
                    fig = plt.figure(3)
                    ax = fig.add_subplot(111, projection='3d')
                    NLL_array[NLL_array[:, 2] > min(NLL_array[:, 2]) + 20] = np.nan
                    ax.scatter(NLL_array[:, 0], NLL_array[:, 1], NLL_array[:, 2], color="green")
                    ax.set_xlabel('alpha', fontweight='bold')
                    ax.set_ylabel('beta', fontweight='bold')
                    ax.set_zlabel('NLL', fontweight='bold')
                    ax.set_title("Participant {0}, run {1}: NLL, alpha and beta 3D scatter plot".format(ID, run + 1))

                    plt.figure(4)
                    plt.scatter(NLL_array[:, 0], NLL_array[:, 2])
                    plt.xlabel('alpha', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and alpha scatter plot".format(ID, run + 1))

                    plt.figure(5)
                    plt.scatter(NLL_array[:, 1], NLL_array[:, 2])
                    plt.xlabel('beta', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and beta scatter plot".format(ID, run + 1))

                    plt.figure(6)
                    plt.scatter(NLL_array[:, 5], NLL_array[:, 2])
                    plt.xlabel('alpha2', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and alpha2 scatter plot".format(ID, run + 1))

                    plt.figure(7)
                    plt.scatter(NLL_array[:, 3], NLL_array[:, 2])
                    plt.xlabel('Initial V0', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and Init V0 scatter plot".format(ID, run + 1))

                    plt.figure(8)
                    plt.scatter(NLL_array[:, 4], NLL_array[:, 2])
                    plt.xlabel('Initial V1', fontweight='bold')
                    plt.ylabel('NLL', fontweight='bold')
                    plt.title("Participant {0}, run {1}: NLL and Init V1 scatter plot".format(ID, run + 1))

                    if version == "two":
                        plt.figure(9)
                        plt.scatter(NLL_array[:, 6], NLL_array[:, 2])
                        plt.xlabel('alpha3', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha3 scatter plot".format(ID, run + 1))

                    elif version == "four":
                        plt.figure(9)
                        plt.scatter(NLL_array[:, 6], NLL_array[:, 2])
                        plt.xlabel('alpha3', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha3 scatter plot".format(ID, run + 1))

                        plt.figure(10)
                        plt.scatter(NLL_array[:, 7], NLL_array[:, 2])
                        plt.xlabel('alpha4', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha4 scatter plot".format(ID, run + 1))

                        plt.figure(11)
                        plt.scatter(NLL_array[:, 8], NLL_array[:, 2])
                        plt.xlabel('alpha5', fontweight='bold')
                        plt.ylabel('NLL', fontweight='bold')
                        plt.title("Participant {0}, run {1}: NLL and alpha5 scatter plot".format(ID, run + 1))


                    os.makedirs(saving_folder, exist_ok=True)

                    if version is None:
                        save_name = "{0}_{1}_initValSimpleUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    elif version == "two":
                        save_name = "{0}_{1}_initValActionUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    else:
                        save_name = "{0}_{1}_initValActionRewardUpdatePlots.pdf".format(ID, run)
                        file_path = os.path.join(saving_folder, save_name)
                        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                    
                    for fig in range(1, plt.gcf().number + 1):
                        pdf.savefig(fig)
                    pdf.close()

                    plt.close('all')

            count += 1

    def updateInitFitting(self, version=None):

        self.all_alphas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_betas = np.empty((len(np.unique(self.IDs)), 6))
        self.all_V_option0Inits = np.empty((len(np.unique(self.IDs)), 6, 3, 3))
        self.all_V_option1Inits = np.empty((len(np.unique(self.IDs)), 6, 3, 3))
        self.all_alphas2 = np.empty((len(np.unique(self.IDs)), 6))
        if version == "two":
            self.all_alphas3 = np.empty((len(np.unique(self.IDs)), 6))
        elif version == "four":
            self.all_alphas3 = np.empty((len(np.unique(self.IDs)), 6))
            self.all_alphas6 = np.empty((len(np.unique(self.IDs)), 6))
            self.all_alphas5 = np.empty((len(np.unique(self.IDs)), 6))
        self.all_LLs = np.empty((len(np.unique(self.IDs)), 6))
        self.NLL_arrays = np.empty((len(np.unique(self.IDs)), 6, self.gridCount, 9))
        self.all_RPEs = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials))
        self.all_V0 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))
        self.all_V1 = np.empty((len(np.unique(self.IDs)), 6, self.mainTrials + self.additionalTrials + 1))

        count = 0
        for ID in np.unique(self.IDs):

            subjectData = pd.read_csv(str([file[1] for file in self.savedValsFiles if file[0] == ID][0]))
            subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
            # subjectInfo = pd.read_csv(str([file[1] for file in self.expInfoFiles if file[0] == ID][0]))
            fitted_alphas = np.empty((1, 6))
            fitted_betas = np.empty((1, 6))
            fitted_alphas2 = np.empty((1, 6))
            if version == "two":
                fitted_alphas3 = np.empty((1, 6))
            elif version == "four":
                fitted_alphas3 = np.empty((1, 6))
                fitted_alphas4 = np.empty((1, 6))
                fitted_alphas5 = np.empty((1, 6))
            fitted_V_option0Inits = np.empty((1, 6, 3, 3))
            fitted_V_option1Inits = np.empty((1, 6, 3, 3))
            best_LLs = np.empty((1, 6))
            ID_RPE = np.empty((1, 6, self.mainTrials + self.additionalTrials))
            ID_V0 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))
            ID_V1 = np.empty((1, 6, self.mainTrials + self.additionalTrials + 1))

            for run in range(0, max(subjectData.runNumber)):

                alphaGrid = np.random.rand(self.gridCount, 1)
                alpha2Grid = np.random.rand(self.gridCount, 1)
                if version == "two":
                    alpha3Grid = np.random.rand(self.gridCount, 1)
                elif version == "four":
                    alpha3Grid = np.random.rand(self.gridCount, 1)
                    alpha4Grid = np.random.rand(self.gridCount, 1)
                    alpha5Grid = np.random.rand(self.gridCount, 1)
                betaGrid = 0 + 15 * np.random.rand(self.gridCount, 1)
                # V_option0Init_Grid = np.random.uniform(0,1,(self.gridCount, 3, 3))
                V_option0_rand = np.random.rand(self.gridCount, 1)
                V_option0Init_Grid = np.repeat(V_option0_rand, 9, axis=1).reshape((self.gridCount, 3, 3))
                # V_option1Init_Grid = np.random.uniform(0, 1, (self.gridCount, 3, 3))
                V_option1_rand = np.random.rand(self.gridCount, 1)
                V_option1Init_Grid = np.repeat(V_option1_rand, 9, axis=1).reshape((self.gridCount, 3, 3))
                NLL_array = np.empty((self.gridCount, 9))
                NLL_array[:] = np.nan
                LL_array = np.empty((self.gridCount, 1))
                runData = subjectData[subjectData.runNumber == run + 1].reset_index()
                run_RPEs = np.empty((self.gridCount, self.mainTrials + self.additionalTrials))
                run_V0 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
                run_V1 = np.empty((self.gridCount, self.mainTrials + self.additionalTrials + 1))
                # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
                for j in range(0, self.gridCount):
                    # For each point on the grid we instantiate the arrays for the time steps-
                    """Instantiating for the fitting"""

                    self.choiceProb = np.empty((max(runData.trialNumber), 2))
                    self.choiceProb[:] = np.nan
                    self.actionProb = np.empty((max(runData.trialNumber), 1))
                    self.actionProb[:] = np.nan
                    self.V_option0 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                    self.V_option0[:] = np.nan
                    self.V_option0[0, :] = V_option0Init_Grid[j]
                    self.V_option1 = np.empty((max(runData.trialNumber) + 1, 3, 3))
                    self.V_option1[:] = np.nan
                    self.V_option1[0, :] = V_option1Init_Grid[j]
                    self.rewardPE = np.empty((max(runData.trialNumber), 3, 3))
                    self.rewardPE[:] = np.nan

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
                    trials_RPE = np.empty((max(runData.trialNumber)))
                    trials_V0 = np.empty((max(runData.trialNumber) + 1))
                    trials_V1 = np.empty((max(runData.trialNumber) + 1))
                    trials_V0[0] = self.V_option0[0, 0, 0]
                    trials_V1[0] = self.V_option1[0, 0, 0]
                    for t in range(0, max(runData.trialNumber)):
                        otherPairs = [p for p in list(runData.stimulusPair.unique())
                                      if bool(p[0] == runData.stimulusPair[t][0]) ^
                                      bool(p[1] == runData.stimulusPair[t][1])]

                        # Prob of choosing the 0th and 1st option respectively
                        self.choiceProb[t, 0] = np.exp(betaCheck * self.V_option0[
                            ((t,) + runData.stimulusPair[t])]) / ((np.exp(
                            betaCheck * self.V_option0[
                                ((t,) + runData.stimulusPair[t])])) + (np.exp(
                            betaCheck * self.V_option1[
                                ((t,) + runData.stimulusPair[t])])))
                        self.choiceProb[t, 1] = 1 - self.choiceProb[t, 0]

                        self.actionProb[t, :] = self.choiceProb[t, int(runData.action[t])] if ~np.isnan(
                            runData.action[t]) else np.nan

                        if runData.action[t] == 0:
                            self.rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - self.V_option0[(t,) + runData.stimulusPair[t]]

                            self.V_option0[t + 1, :] = self.V_option0[t, :]
                            self.V_option0[(t + 1,) + runData.stimulusPair[t]] = \
                                self.V_option0[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (self.rewardPE[(t,) + runData.stimulusPair[t]])

                            if version is None or version == "two" or (version == "four" and runData.reward[t] == 1):
                                for pair in otherPairs:
                                    self.V_option0[(t + 1,) + pair] = \
                                        self.V_option0[(t,) + pair] + \
                                        alpha2Check * (1 - runData.reward[t] - self.V_option0[(t,) + pair])
                            else:
                                for pair in otherPairs:
                                    self.V_option0[(t + 1,) + pair] = \
                                            self.V_option0[(t,) + pair] + \
                                            alpha3Check * (1 - runData.reward[t] - self.V_option0[(t,) + pair])
                            self.V_option1[t + 1, :] = self.V_option1[t, :]


                        elif runData.action[t] == 1:
                            self.rewardPE[(t,) + runData.stimulusPair[t]] = \
                                runData.reward[t] - self.V_option1[(t,) + runData.stimulusPair[t]]

                            self.V_option1[t + 1, :] = self.V_option1[t, :]
                            self.V_option1[(t + 1,) + runData.stimulusPair[t]] = \
                                self.V_option1[(t,) + runData.stimulusPair[t]] + \
                                alphaCheck * (self.rewardPE[(t,) + runData.stimulusPair[t]])

                            if version is None:
                                for pair in otherPairs:
                                    self.V_option1[(t + 1,) + pair] = \
                                        self.V_option1[(t,) + pair] + \
                                        (alpha2Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                            elif version == "two":
                                for pair in otherPairs:
                                    self.V_option1[(t + 1,) + pair] = \
                                        self.V_option1[(t,) + pair] + \
                                        (alpha3Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                            else:
                                if runData.reward[t] == 1:
                                    for pair in otherPairs:
                                        self.V_option1[(t + 1,) + pair] = \
                                            self.V_option1[(t,) + pair] + \
                                            (alpha4Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))
                                else:
                                    for pair in otherPairs:
                                        self.V_option1[(t + 1,) + pair] = \
                                            self.V_option1[(t,) + pair] + \
                                            (alpha5Check * (1 - runData.reward[t] - self.V_option1[(t,) + pair]))

                            self.V_option0[t + 1, :] = self.V_option0[t, :]

                        else:
                            self.V_option0[t + 1, :] = self.V_option0[t, :]
                            self.V_option1[t + 1, :] = self.V_option1[t, :]

                        trials_RPE[t] = self.rewardPE[(t,) + runData.stimulusPair[t]]
                        trials_V0[t + 1] = self.V_option0[(t + 1,) + runData.stimulusPair[t]]
                        trials_V1[t + 1] = self.V_option1[(t + 1,) + runData.stimulusPair[t]]
                    negativeLogLikelihood = -np.sum(np.log(self.actionProb[~np.isnan(self.actionProb)]))
                    Likelihood = np.prod(self.actionProb[~np.isnan(self.actionProb)])
                    NLL_array[j, 0] = alphaCheck
                    NLL_array[j, 1] = betaCheck
                    NLL_array[j, 2] = negativeLogLikelihood
                    NLL_array[j, 3] = V_option0Init_Grid[j][0][0]
                    NLL_array[j, 4] = V_option1Init_Grid[j][0][0]
                    NLL_array[j, 5] = alpha2Check
                    if version == "two":
                        NLL_array[j, 6] = alpha3Check
                    elif version == "four":
                        NLL_array[j, 6] = alpha3Check
                        NLL_array[j, 7] = alpha4Check
                        NLL_array[j, 8] = alpha5Check
                    LL_array[j] = Likelihood
                    run_RPEs[j, :] = trials_RPE
                    run_V0[j, :] = trials_V0
                    run_V1[j, :] = trials_V1

                minIndex = np.argmin(NLL_array[:, 2])
                maxIndex = np.nanargmax(LL_array[:, 0])

                self.NLL_arrays[count, run, :, :] = NLL_array
                fitted_alphas[0, run] = NLL_array[minIndex, 0]
                fitted_betas[0, run] = NLL_array[minIndex, 1]
                fitted_alphas2[0, run] = NLL_array[minIndex, 5]
                if version == "two":
                    fitted_alphas3[0, run] = NLL_array[minIndex, 6]
                elif version == "four":
                    fitted_alphas3[0, run] = NLL_array[minIndex, 6]
                    fitted_alphas4[0, run] = NLL_array[minIndex, 7]
                    fitted_alphas5[0, run] = NLL_array[minIndex, 8]
                fitted_V_option0Inits[0, run] = V_option0Init_Grid[minIndex]
                fitted_V_option1Inits[0, run] = V_option1Init_Grid[minIndex]
                best_LLs[0, run] = LL_array[maxIndex]
                ID_RPE[0, run] = run_RPEs[minIndex, :]
                ID_V0[0, run] = run_V0[minIndex, :]
                ID_V1[0, run] = run_V1[minIndex, :]

            self.all_alphas[count, :] = fitted_alphas
            self.all_betas[count, :] = fitted_betas
            self.all_alphas2[count, :] = fitted_alphas2
            if version == "two":
                self.all_alphas3[count, :] = fitted_alphas3
            elif version == "four":
                self.all_alphas3[count, :] = fitted_alphas3
                self.all_alphas4[count, :] = fitted_alphas4
                self.all_alphas5[count, :] = fitted_alphas5
            self.all_RPEs[count] = ID_RPE
            self.all_V0[count] = ID_V0
            self.all_V1[count] = ID_V1
            self.all_V_option0Inits[count, :] = fitted_V_option0Inits
            self.all_V_option1Inits[count, :] = fitted_V_option1Inits
            self.all_LLs[count, :] = best_LLs
            count += 1

# Statistical learning
    def statisticalLearning(self, statLearnPar=1):

        self.statLearnPar = statLearnPar
        

        subjectData = pd.read_csv(self.savedValsFile)
        subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
        ID_beliefs = np.empty((6, self.mainTrials + self.additionalTrials + 1, 3, 3))
        ID_surprise = np.empty((6, self.mainTrials + self.additionalTrials))
        for run in range(0, max(subjectData.runNumber)):

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

            runData = subjectData[subjectData.runNumber == run + 1].reset_index()
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
                    rowBeliefs[i, 0, :] = (
                                                    self.statLearnPar + statCount[i, 0, :]) / rowDen0
                elif runData.stimulusPair[i - 1][0] == 1:
                    rowDen1 = rowDen1 + 1
                    rowBeliefs[i, 1, :] = (
                                                    self.statLearnPar + statCount[i, 1, :]) / rowDen1
                else:
                    rowDen2 = rowDen2 + 1
                    rowBeliefs[i, 2, :] = (
                                                    self.statLearnPar + statCount[i, 2, :]) / rowDen2
                # column beliefs
                columnBeliefs[i, :] = columnBeliefs[i - 1, :]
                if runData.stimulusPair[i - 1][1] == 0:
                    columnDen0 = columnDen0 + 1
                    columnBeliefs[i, 0, :] = (
                                                        self.statLearnPar + statCount[i, 0, :]) / columnDen0
                elif runData.stimulusPair[i - 1][1] == 1:
                    columnDen1 = columnDen1 + 1
                    columnBeliefs[i, 1, :] = (
                                                        self.statLearnPar + statCount[i, 1, :]) / columnDen1
                else:
                    columnDen2 = columnDen2 + 1
                    columnBeliefs[i, 2, :] = (
                                                        self.statLearnPar + statCount[i, 2, :]) / columnDen2
                num = self.statLearnPar + statCount[i, :]
                den += 1
                # Total statistical beliefs irrespective of rows and columns.
                beliefsStat[i, :] = num / den

            ID_beliefs[run, :, :] = beliefsStat
            trial_surprise = np.empty((self.mainTrials + self.additionalTrials))
            # Surprises calculated from beliefs update
            for i in range(0, self.mainTrials + self.additionalTrials):
                statSurpriseRow[i, :] = np.nan
                statSurpriseRow[(i,) + runData.stimulusPair[i]] = - \
                    np.log(rowBeliefs[(i,) + runData.stimulusPair[i]])
                statSurpriseColumn[i, :] = np.nan
                statSurpriseColumn[(i,) + runData.stimulusPair[i]] = - \
                    np.log(columnBeliefs[(i,) + runData.stimulusPair[i]])
                statSurprise[i, :] = np.nan
                statSurprise[(i,) + runData.stimulusPair[i]] = - \
                    np.log(beliefsStat[(i,) + runData.stimulusPair[i]])
                trial_surprise[i] = statSurprise[(i,) + runData.stimulusPair[i]]
            ID_surprise[run] = trial_surprise

        newPath = os.path.join(pathlib.Path(__file__).parent.resolve(), "fittedParameters/sub-{}".format(self.ID))
        Path(newPath).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(newPath+'/spe.mat'.format(self.ID), mdict={'spe': ID_surprise})
        return ID_beliefs, ID_surprise
        

    def plots_stats(self, beliefs, surprise):

        subjectData = pd.read_csv(self.savedValsFile)
        subjectData['stimulusPair'] = subjectData['stimulusPair'].apply(ast.literal_eval)
        # subjectInfo = pd.read_csv(str([file[1] for file in self.expInfoFiles if file[0] == ID][0]))

        for run in range(0, max(subjectData.runNumber)):
            beliefsStat = beliefs[run]
            statSurprise = surprise[run]

            plt.figure(1)
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 0, 0], label="1A")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 0, 1], label="1B")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 0, 2], label="1C")
            plt.title("Learning of statistical structure (beliefs) by Bayesian observer")

            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 1, 0], label="2A")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 1, 1], label="2B")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 1, 2], label="2C")

            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 2, 0], label="3A")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 2, 1], label="3B")
            plt.plot(range(0, self.mainTrials + 1),
                        beliefsStat[:, 2, 2], label="3C")

            plt.xlabel("trials")
            plt.ylabel("Beliefs of probabilities co-occurence")
            plt.legend(bbox_to_anchor=(0.99, 0.65))
            plt.axhline(y=0.15 * 0.33, color='r', linestyle='-')
            plt.axhline(y=0.35 * 0.33, color='g', linestyle='-')
            plt.axhline(y=0.5 * 0.33, color='b', linestyle='-')

            plt.figure(2)
            plt.plot(statSurprise[~np.isnan(statSurprise)])
            plt.xlabel("trials")
            plt.ylabel("Total surprise")
            # plt.legend(bbox_to_anchor=(0.98, 0.9))
            plt.title("Total statistical surprise signal")

            plt.show()


def ma(interval, window_size = 10, method = 'same'):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, method)

# Calls for different fitting and plotting functions
# You can only run ONE model fitting at a time

fitted = Fitting(60, 0, 5000, ID="58")

# # Run the simplest RL model
#fitted_alphas, fitted_betas, best_LLs, RPE, V0, V1, NLL_array = fitted.simplestFitting()

#fitted.plots_simplestFitting(ww=10, NLL_array=NLL_array, alphas=fitted_alphas, betas=fitted_betas, method ='valid', reps=50)

# # # Run the RL model including initial V0 and V1 as free parameters
#fitted.valFitting()
#fitted.plots_valueFitting(ww=10, method='same', reps=50)

# Run the RL model with extra learning rates for the other pairs
# (if you see pair image0 and audio0, also update other image0 and audio0 pairs)
# Options: 1) Do not add anything to the call 2) Add version="two" 3) Add version="four"
#fitted.updateFitting()
#fitted.plots_updateFitting(ww=10, method='same', reps=50)
#fitted.updateFitting(version="two")
#fitted.plots_updateFitting(ww=10, method='same', reps=50, version="two")
#fitted.updateFitting(version="four")
#fitted.plots_updateFitting(ww=10, method='same', reps=50, version="four")

# Run the RL model with extra learning rates for sthe other pairs AND initial V0 and V1 as free parameters
# Same options as above; 1) Do not add anything to the call 2) Add version="two" 3) Add version="four"
#fitted.updateInitFitting()
#fitted.plots_updateInitFitting(ww=10, method="same", reps=50)
#fitted.updateInitFitting(version="two")
#fitted.plots_updateInitFitting(ww=10, method="same", reps=50, version="two")
#fitted.updateInitFitting(version="four")
#fitted.plots_updateInitFitting(ww=10, method="same", reps=50, version="four")


# You can always include statistical learning; necessary to get surprise values
beliefs, surprise = fitted.statisticalLearning(statLearnPar=1)
#fitted.plots_stats(beliefs, surprise)

# Run to save RPE and surprise values; rename .mat files yourself (e.g. based on model)
#RPE_arr = fitted.all_RPEs
#scipy.io.savemat('rpe_upInit4.mat', mdict={'RPE_arr': RPE_arr})
#SPE_arr = fitted.all_surprise
#scipy.io.savemat('spe_upInit4.mat', mdict={'SPE_arr': SPE_arr})

#Run to save V0 and V1 values; rename .mat files yourself (e.g. based on model)
# V0_arr = fitted.all_V0
# scipy.io.savemat('v0.mat', mdict={'V0_arr': V0_arr})
# V1_arr = fitted.all_V1
# scipy.io.savemat('v1.mat', mdict={'V1_arr': V1_arr})

# Save max log likelihood values if you want to do model comparison later; rename np array yourself (e.g. based on model)
# with open('BIC_new_upInit4.npy', 'wb') as f:
#     np.save(f, fitted.all_LLs)
