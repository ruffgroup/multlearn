from random import *
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns
sns.set_context('talk')
import sys

#scriptDirectory = os.path.abspath(os.path.dirname(sys.argv[0]))
#sys.path.insert(1, scriptDirectory[0:-9])


# Adding path of module task_Design
sys.path.append(sys.path[0] + '/..')
print(sys.path)

# Importing class task_Design
from TaskDesign import task_Design

taskSimulationList = []
NLL_array_list = []


# Here we can change the trial Numbers and how many parameters we recover. should be a multiple of 60.
mainTrials = 60
additionalTrials = 0
checkingCount = 50

simulatedRLParams = np.empty((checkingCount, 2))
simulatedRLParams[:] = np.nan
recoveredRLParams = np.empty((checkingCount, 2))
recoveredRLParams[:] = np.nan

def recoveringParameters(mainTrials, additionalTrials, checkingCount):
    # For storing what was simulated and recovered

    # Number of times we simulate and make the object taskSimulation by using the task_Design class
    for i in range(0, checkingCount):

        print("checking", i)

        # Simulating an object from the task design using the next 4 lines of code
        taskSimulation = task_Design(mainTrials,  additionalTrials)
        taskSimulation.taskStructure()
        taskSimulation.RLloops()
        taskSimulation.statisticalLearning()

        # Appending the simulated object to a list
        taskSimulationList.append(taskSimulation)

        # Setting the size of the random grid that we check
        gridCount = 5000
        alphaGrid = np.random.uniform(0, 1, (gridCount, 1))
        betaGrid = 0 + 20.0 * np.random.rand(gridCount, 1)
        NLL_array = np.empty((gridCount, 3))
        NLL_array[:] = np.nan

        # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
        for j in range(0, gridCount):
            # For each point on the grid we instantiate the arrays for the time steps-
            """Instantiating for the recovery"""

            choiceProb = np.empty((taskSimulation.mainTrials, 2))
            choiceProb[:] = np.nan
            actionProb = np.empty((taskSimulation.mainTrials, 1))
            actionProb[:] = np.nan
            V_option0 = np.empty((taskSimulation.mainTrials+1, 3, 3))
            V_option0[:] = np.nan
            V_option0[0, :] = 0.5
            V_option1 = np.empty((taskSimulation.mainTrials+1, 3, 3))
            V_option1[:] = np.nan
            V_option1[0, :] = 0.5

            # Checking parameters from the grid
            alphaCheck = alphaGrid[j]
            betaCheck = betaGrid[j]
            for t in range(0, taskSimulation.mainTrials):
                # Prob of chocing the 0th and 1st option respectively
                choiceProb[t, 0] = np.exp(betaCheck*V_option0[((t,)+tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :]))])/((np.exp(
                    betaCheck*V_option0[((t,)+tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :]))]))+(np.exp(betaCheck*V_option1[((t,)+tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :]))])))
                choiceProb[t, 1] = 1 - choiceProb[t, 0]

                actionProb[t, :] = choiceProb[t, int(taskSimulation.action[t])]

                if taskSimulation.action[t] == 0:
                    V_option0[t+1, :] = V_option0[t, :]
                    V_option0[(t+1,) + tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])] = V_option0[(t,) +
                                                                                                                             tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])] + alphaCheck * (taskSimulation.rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])])
                    V_option1[t+1, :] = taskSimulation.V_option1[t, :]
                else:
                    V_option1[t+1, :] = V_option1[t, :]
                    V_option1[(t+1,) + tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])] = V_option1[(t,) +
                                                                                                                             tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])] + alphaCheck * (taskSimulation.rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t+taskSimulation.additionalTrials, :])])
                    V_option0[t+1, :] = V_option0[t, :]

            negativeLogLikelihood = -np.sum(np.log(actionProb))
            NLL_array[j, 0] = alphaCheck
            NLL_array[j, 1] = betaCheck
            NLL_array[j, 2] = negativeLogLikelihood



        minIndex = np.argmin(NLL_array[:, 2])
        recoveredAlpha = alphaGrid[minIndex]
        recoveredBeta = betaGrid[minIndex]

        # plt.figure(1)
        # plt.scatter(NLL_array[:, 0], NLL_array[:, -1])
        # plt.ylim(np.nanmin(NLL_array[:, -1]) - 1, np.nanmin(NLL_array[:, -1]) + 2)
        # plt.title("Run {0}: Simulated alpha {1}, recovered alpha {2}".format(i, np.round(taskSimulation.alpha,2), np.round(recoveredAlpha[0],2)))
        # plt.xlabel('alpha', fontweight='bold')
        # plt.ylabel('NLL', fontweight='bold')
        #
        # plt.figure(2)
        # plt.scatter(NLL_array[:, 1], NLL_array[:, -1])
        # plt.ylim(np.nanmin(NLL_array[:, -1]) - 1, np.nanmin(NLL_array[:, -1]) + 2)
        # plt.title("Run {0}: Simulated beta {1}, recovered beta {2}".format(i, np.round(taskSimulation.beta,2), np.round(recoveredBeta[0],2)))
        # plt.xlabel('beta', fontweight='bold')
        # plt.ylabel('NLL', fontweight='bold')
        # plt.show()

        simulatedRLParams[i, 0] = taskSimulation.alpha
        simulatedRLParams[i, 1] = taskSimulation.beta
        recoveredRLParams[i, 0] = recoveredAlpha
        recoveredRLParams[i, 1] = recoveredBeta
        NLL_array_list.append(NLL_array)

    return simulatedRLParams, recoveredRLParams, NLL_array_list


recoveringParameters(mainTrials, additionalTrials, checkingCount)


alphaRecovery = {
    'simulated alpha': simulatedRLParams[:, 0], 'recovered alpha': recoveredRLParams[:, 0]}

alphaRecovery = pd.DataFrame(alphaRecovery)

corrAlpha = scipy.stats.pearsonr(alphaRecovery['simulated alpha'], alphaRecovery['recovered alpha'])

betaRecovery = {'simulated beta': simulatedRLParams[:,
                                                    1], 'recovered beta': recoveredRLParams[:, 1]}

betaRecovery = pd.DataFrame(betaRecovery)

corrBeta = scipy.stats.pearsonr(betaRecovery['simulated beta'], betaRecovery['recovered beta'])

pearsonCorr = np.empty((checkingCount, 2))
pearsonCorr[:] = np.nan

for i in range(0, checkingCount):
    pearsonCorr[i] = scipy.stats.pearsonr(taskSimulationList[i].statSurprise[additionalTrials:, :, :][~np.isnan(
        taskSimulationList[i].statSurprise[additionalTrials:, :, :])], taskSimulationList[i].rewardPE[:, :, :][~np.isnan(taskSimulationList[i].rewardPE[:, :, :])])


g = sns.lmplot(x="simulated alpha", y="recovered alpha", data=alphaRecovery)
g.fig.subplots_adjust(top=.95)
g.ax.set_title("correlation {:.2f}, p-value{:.2f}".format(corrAlpha[0], corrAlpha[1]))
plt.show()

g = sns.lmplot(x="simulated beta", y="recovered beta", data=betaRecovery)
g.fig.subplots_adjust(top=.95)
g.ax.set_title("correlation {:.2f}, p-value{:.2f}".format(corrBeta[0], corrBeta[1]))
plt.show()


plt.hist(pearsonCorr[:, 0])
plt.title("Histogram of correlation coefficients between surprise and RPE\n 50 subjects, {} + {} trials".format(mainTrials, additionalTrials))
plt.xlabel("pearson correlation coefficients")
plt.ylabel("Frequency")
plt.vlines(0, 0.0, 20.0, "black")
plt.vlines(np.mean(pearsonCorr[:, 0]), 0.0, 20.0, "red")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Histogram.png', dpi=300, bbox_inches='tight')


# del mainTrials
# del additionalTrials
#del checkingCount
