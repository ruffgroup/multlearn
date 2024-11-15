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
import seaborn as sns
import argparse

sys.path.append(sys.path[0] + "/..")
from TaskDesign import task_Design

class Recovery:
    def __init__(self, mainTrials, simulations, gridsize, extra, asym, transfer, pearce):
            
        self.mainTrials = mainTrials
        self.simulations = simulations
        self.extra = extra
        self.asym = asym
        self.transfer = transfer
        self.pearce = pearce
        self.gridsize = gridsize


    def recovery(self):
        
        taskSimulationList = []
        NLL_array_list = []
        simulatedRLParams_list = []
        recoveredRLParams_list = []

        alphaSimArr = np.random.rand(self.simulations, 1)
        if self.extra and self.asym:
            alphaNegSimArr, alpha2PosSimArr, alpha2NegSimArr = (
                np.random.rand(self.simulations, 1) for i in range(3)
            )
        elif self.extra and not self.asym:
            alpha2SimArr = np.random.rand(self.simulations, 1)
        elif self.asym and not self.extra:
            alphaNegSimArr = np.random.rand(self.simulations, 1)

        if self.transfer:
            K1SimArr = np.random.rand(self.simulations, 1)
            if self.extra and self.asym:
                K3SimArr, K4SimArr, K2SimArr = (
                    np.random.rand(self.simulations, 1) for i in range(3)
                )
            elif self.extra and not self.asym:
                K2SimArr = np.random.rand(self.simulations, 1)
            elif self.asym and not self.extra:
                K3SimArr = np.random.rand(self.simulations, 1)

        betaSimArr = 0 + 15 * np.random.rand(self.simulations, 1)

        for i in range(0, self.simulations):
        # Checking parameters from the grid
            alphaPosSim = alpha2PosSim = alphaNegSim = alpha2NegSim = K1Sim = K2Sim = K3Sim = K4Sim = betaSim = V_option0Init = V_option1Init = None
            if self.extra and not self.asym:
                alphaPosSim = alphaNegSim = alphaSimArr[i]
                alpha2PosSim = alpha2NegSim = alpha2SimArr[i]
            elif self.asym and not self.extra:
                alphaPosSim = alpha2PosSim = alphaSimArr[i]
                alphaNegSim = alpha2NegSim = alphaNegSimArr[i]
            elif self.extra and self.asym:
                alphaPosSim = alphaSimArr[i]
                alpha2PosSim = alpha2PosSimArr[i]
                alphaNegSim = alphaNegSimArr[i]
                alpha2NegSim = alpha2NegSimArr[i]
            else:
                alphaPosSim = (
                    alpha2PosSim
                ) = alphaNegSim = alpha2NegSim = alphaSimArr[i]

            if self.transfer:
                if self.extra and not self.asym:
                    K1Sim = K3Sim = K1SimArr[i]
                    K2Sim = K4Sim = K2SimArr[i]
                elif self.asym and not self.extra:
                    K1Sim = K2Sim = K1SimArr[i]
                    K3Sim = K4Sim = K3SimArr[i]
                elif self.asym and self.extra:
                    K1Sim = K1SimArr[i]
                    K2Sim = K2SimArr[i]
                    K3Sim = K3SimArr[i]
                    K4Sim = K4SimArr[i]
                else:
                    K1Sim = K2Sim = K3Sim = K4Sim = K1SimArr[i]

            betaSim = betaSimArr[i]

            taskSimulation = task_Design(self.mainTrials, 0, asym=self.asym, extra=self.extra, pearce=self.pearce, transfer=self.transfer, alphaPos=alphaPosSim, alphaNeg=alphaNegSim, alpha2Pos=alpha2PosSim, alpha2Neg=alpha2NegSim, K1=K1Sim, K2=K2Sim, K3=K3Sim, K4=K4Sim, beta=betaSim, V_option0Init=V_option0Init, V_option1Init=V_option1Init)
            taskSimulation.taskStructure()
            taskSimulation.RLloops()
            taskSimulation.statisticalLearning()

            taskSimulationList.append(taskSimulation)


            alphaGrid = np.random.rand(self.gridsize, 1)
            if self.extra and self.asym:
                alphaNegGrid, alpha2PosGrid, alpha2NegGrid = (
                    np.random.rand(self.gridsize, 1) for i in range(3)
                )
            elif self.extra and not self.asym:
                alpha2Grid = np.random.rand(self.gridsize, 1)
            elif self.asym and not self.extra:
                alphaNegGrid = np.random.rand(self.gridsize, 1)

            if self.transfer:
                K1Grid = np.random.rand(self.gridsize, 1)
                if self.extra and self.asym:
                    K3Grid, K4Grid, K2Grid = (
                        np.random.rand(self.gridsize, 1) for i in range(3)
                    )
                elif self.extra and not self.asym:
                    K2Grid = np.random.rand(self.gridsize, 1)
                elif self.asym and not self.extra:
                    K3Grid = np.random.rand(self.gridsize, 1)

            betaGrid = 0 + 15 * np.random.rand(self.gridsize, 1)
            NLL_array = np.empty((self.gridsize, 12))
            NLL_array[:] = np.nan

            simulatedRLParams = np.empty((self.gridsize, 11))
            simulatedRLParams[:] = np.nan
            recoveredRLParams = np.empty((self.gridsize, 11))
            recoveredRLParams[:] = np.nan

            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridsize):
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((taskSimulation.mainTrials, 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((taskSimulation.mainTrials, 1))
                actionProb[:] = np.nan
                V_option0, V_option1 = (
                    np.empty((taskSimulation.mainTrials + 1, 3, 3)) for i in range(2)
                )
                V_option0[:], V_option1[:] = (np.nan for i in range(2))

                V_option0[0, :] = 0.5
                V_option1[0, :] = 0.5

                rewardPE = np.empty((taskSimulation.mainTrials, 3, 3))
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

                if self.pearce:
                    omega = omega2 = omega3 = omega4 = 1

                for t in range(0, taskSimulation.mainTrials):
                    otherPairs = [
                        p
                        for p in list(taskSimulation.stimulusPair.unique())
                        if bool(p[0] == taskSimulation.stimulusPair[t][0])
                        ^ bool(p[1] == taskSimulation.stimulusPair[t][1])
                    ]

                    # Prob of choosing the 0th and 1st option respectively
                    choiceProb[t, 0] = np.exp(
                        betaCheck * V_option0[((t,) + taskSimulation.stimulusPair[t])]
                    ) / (
                        (
                            np.exp(
                                betaCheck * V_option0[((t,) + taskSimulation.stimulusPair[t])]
                            )
                        )
                        + (
                            np.exp(
                                betaCheck * V_option1[((t,) + taskSimulation.stimulusPair[t])]
                            )
                        )
                    )
                    choiceProb[t, 1] = 1 - choiceProb[t, 0]

                    actionProb[t, :] = (
                        choiceProb[t, int(taskSimulation.action[t])]
                        if ~np.isnan(taskSimulation.action[t])
                        else np.nan
                    )

                    if taskSimulation.action[t] == 0:
                        rewardPE[(t,) + taskSimulation.stimulusPair[t]] = (
                            taskSimulation.reward[t]
                            - V_option0[(t,) + taskSimulation.stimulusPair[t]]
                        )
                        V_option0[t + 1, :] = V_option0[t, :]

                        if taskSimulation.reward[t] == 1:
                            if self.pearce:
                                omega = (
                                    omega
                                    + (
                                        abs(rewardPE[(t,) + taskSimulation.stimulusPair[t]])
                                        - omega
                                    )
                                    * alphaPosCheck
                                )
                                V_option0[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + omega * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K1Check * omega * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )
                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + alphaPosCheck * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )

                                

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K1Check * alphaPosCheck * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            if self.pearce:
                                if not self.asym and not self.extra:
                                    omega2 = omega3 = omega4 = omega
                                elif self.asym and not self.extra:
                                    omega2 = omega
                                elif self.extra and not self.asym:
                                    omega3 = omega

                        else:
                            if self.pearce:
                                omega3 = (
                                    omega3
                                    + (
                                        abs(rewardPE[(t,) + taskSimulation.stimulusPair[t]])
                                        - omega3
                                    )
                                    * alphaNegCheck
                                )
                                V_option0[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + omega3 * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K3Check * omega3 * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0
                            else:
                                V_option0[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option0[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + alphaNegCheck * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )


                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + pair] = V_option0[
                                            (t,) + pair
                                        ] - K3Check * alphaNegCheck * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option0[(t + 1,) + pair] > 1:
                                            V_option0[(t + 1,) + pair] = 1
                                        elif V_option0[(t + 1,) + pair] < 0:
                                            V_option0[(t + 1,) + pair] = 0

                            if self.pearce:
                                if not self.asym and not self.extra:
                                    omega2 = omega = omega4 = omega3
                                elif self.asym and not self.extra:
                                    omega4 = omega3
                                elif self.extra and not self.asym:
                                    omega = omega3

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif taskSimulation.action[t] == 1:
                        rewardPE[(t,) + taskSimulation.stimulusPair[t]] = (
                            taskSimulation.reward[t]
                            - V_option1[(t,) + taskSimulation.stimulusPair[t]]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]

                        if taskSimulation.reward[t] == 1:
                            if self.pearce:
                                omega2 = (
                                    omega2
                                    + (
                                        abs(rewardPE[(t,) + taskSimulation.stimulusPair[t]])
                                        - omega2
                                    )
                                    * alpha2PosCheck
                                )
                                V_option1[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + omega2 * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K2Check * omega2 * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + alpha2PosCheck * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )


                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K2Check * alpha2PosCheck * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if self.pearce:
                                if not self.asym and not self.extra:
                                    omega3 = omega = omega4 = omega2
                                elif self.asym and not self.extra:
                                    omega = omega2
                                elif self.extra and not self.asym:
                                    omega4 = omega2

                        else:
                            if self.pearce:
                                omega4 = (
                                    omega4
                                    + (
                                        abs(rewardPE[(t,) + taskSimulation.stimulusPair[t]])
                                        - omega4
                                    )
                                    * alpha2NegCheck
                                )
                                V_option1[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + omega4 * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )
                           
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K4Check * omega4 * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            else:
                                V_option1[
                                    (t + 1,) + taskSimulation.stimulusPair[t]
                                ] = V_option1[
                                    (t,) + taskSimulation.stimulusPair[t]
                                ] + alpha2NegCheck * (
                                    rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                )

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + pair] = V_option1[
                                            (t,) + pair
                                        ] - K4Check * alpha2NegCheck * (
                                            rewardPE[(t,) + taskSimulation.stimulusPair[t]]
                                        )

                                        if V_option1[(t + 1,) + pair] > 1:
                                            V_option1[(t + 1,) + pair] = 1
                                        elif V_option1[(t + 1,) + pair] < 0:
                                            V_option1[(t + 1,) + pair] = 0
                            if self.pearce:
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


                negativeLogLikelihood = -np.sum(np.log(actionProb))
                NLL_array[j, 0] = negativeLogLikelihood
                NLL_array[j, 1] = alphaPosCheck
                NLL_array[j, 2] = betaCheck
                if self.asym and not self.extra:
                    NLL_array[j, 3] = alphaNegCheck
                elif self.extra and not self.asym:
                    NLL_array[j, 4] = alpha2PosCheck
                elif self.extra and self.asym:
                    NLL_array[j, 3] = alphaNegCheck
                    NLL_array[j, 4] = alpha2PosCheck
                    NLL_array[j, 5] = alpha2NegCheck
                if self.transfer:
                    NLL_array[j, 6] = K1Check
                    if self.extra and not self.asym:
                        NLL_array[j, 7] = K2Check
                    elif self.asym and not self.extra:
                        NLL_array[j, 8] = K3Check
                    elif self.asym and self.extra:
                        NLL_array[j, 7] = K2Check
                        NLL_array[j, 8] = K3Check
                        NLL_array[j, 9] = K4Check

            minIndex = np.argmin(NLL_array[:, 0])
            recoveredAlphaPos = alphaGrid[minIndex]
            recoveredBeta = betaGrid[minIndex]
            if self.asym and not self.extra:
                recoveredAlphaNeg = alphaNegGrid[minIndex]
            elif self.extra and not self.asym:
                recoveredAlpha2Pos = alpha2PosGrid[minIndex]
            elif self.extra and self.asym:
                recoveredAlphaNeg = alphaNegGrid[minIndex]
                recoveredAlpha2Pos = alpha2PosGrid[minIndex]
                recoveredAlpha2Neg = alpha2NegGrid[minIndex]
            if self.transfer:
                recoveredK1 = K1Grid[minIndex]
                if self.extra and not self.asym:
                    recoveredK2 = K2Grid[minIndex]
                elif self.asym and not self.extra:
                    recoveredK3 = K3Grid[minIndex]
                elif self.asym and self.extra:
                    recoveredK2 = K2Grid[minIndex]
                    recoveredK3 = K3Grid[minIndex]
                    recoveredK4 = K4Grid[minIndex]

            simulatedRLParams[i, 0] = alphaPosSim
            recoveredRLParams[i, 0] = recoveredAlphaPos
            simulatedRLParams[i, 1] = betaSim
            recoveredRLParams[i, 1] = recoveredBeta
            if self.asym and not self.extra:
                simulatedRLParams[i, 2] = alphaNegSim
                recoveredRLParams[i, 2] = recoveredAlphaNeg
            elif self.extra and not self.asym:
                simulatedRLParams[i, 2] = alpha2PosSim
                recoveredRLParams[i, 2] = recoveredAlpha2Pos
            elif self.extra and self.asym:
                simulatedRLParams[i, 2] = alphaNegSim
                simulatedRLParams[i, 3] = alpha2PosSim
                simulatedRLParams[i, 4] = alpha2NegSim
                recoveredRLParams[i, 2] = recoveredAlphaNeg
                recoveredRLParams[i, 3] = recoveredAlpha2Pos
                recoveredRLParams[i, 4] = recoveredAlpha2Neg

            if self.transfer:
                simulatedRLParams[i, 5] = K1Sim
                recoveredRLParams[i, 5] = recoveredK1
                if self.extra and not self.asym:
                    simulatedRLParams[i, 6] = K2Sim
                    recoveredRLParams[i, 6] = recoveredK2
                elif self.asym and not self.extra:
                    simulatedRLParams[i, 6] = K3Sim
                    recoveredRLParams[i, 6] = recoveredK3
                elif self.asym and self.extra:
                    simulatedRLParams[i, 6] = K2Sim
                    simulatedRLParams[i, 7] = K3Sim
                    simulatedRLParams[i, 8] = K4Sim
                    recoveredRLParams[i, 6] = recoveredK2
                    recoveredRLParams[i, 7] = recoveredK3
                    recoveredRLParams[i, 8] = recoveredK4
            NLL_array_list.append(NLL_array)
            simulatedRLParams_list.append(simulatedRLParams)
            recoveredRLParams_list.append(recoveredRLParams)
            

        return simulatedRLParams_list, recoveredRLParams_list, NLL_array_list
    

    def recoveryPlot(self, simulatedRLParams, recoveredRLParams, NLL_array_list):

        alphaRecovery = {
    'simulated alpha': simulatedRLParams[:, 0], 'recovered alpha': recoveredRLParams[:, 0]}

        alphaRecovery = pd.DataFrame(alphaRecovery)

        g = sns.lmplot(x="simulated alpha", y="recovered alpha", data=alphaRecovery)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovery of RL parameters")
    parser.add_argument("--mainTrials", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--gridsize", type=int, default=100)
    parser.add_argument("--extra", type=bool, default=False)
    parser.add_argument("--asym", type=bool, default=False)
    parser.add_argument("--transfer", type=bool, default=False)
    parser.add_argument("--pearce", type=bool, default=False)
    args = parser.parse_args()
    Recovery(args.mainTrials, args.simulations, args.gridsize, args.extra, args.asym, args.transfer, args.pearce).recovery()
        