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


class Plotting:

    def __init__(self, mainTrials, additionalTrials, gridCount, ID, ww, method):
        
        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.gridCount = gridCount
        self.ID = ID
        self.statLearnPar = 1
        self.ww = ww
        self.method = method
        if platform.system() == 'Windows':
            wanted_dir = '/data/sourcedata/behavior/modified_files'
        else:
            wanted_dir = '/data/sourcedata/behavior/modified_files'
        # Get savedVals file
        self.savedValsFile = glob.glob(os.path.abspath(wanted_dir)+"/*{}_savedValues.csv".format(self.ID))[0]


    # Simple plots

    def plots_simplestFitting(self, ww, NLL_array, alphas, betas, reps=50):
        
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

    # Value plots

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

    # Update plots

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

    # Update and Init V0 and V1 plots

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

    # Stat learning plots

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

