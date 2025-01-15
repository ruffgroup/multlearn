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
#from RLparameterPlotting import Plotting
from multiprocessing import Pool
import signal
from functools import wraps
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

sys.path.append(sys.path[0] + "/..")
from TaskDesign import task_Design

class Recovery:
    def __init__(self, mainTrials, simulations, gridsize, extra, asym, transfer, pearce, dyna, v_init):
            
        self.mainTrials = mainTrials
        self.simulations = simulations
        self.extra = extra
        self.asym = asym
        self.transfer = transfer
        self.pearce = pearce
        self.gridsize = gridsize
        self.dyna = dyna
        self.v_init = v_init

        if not self.dyna and not self.extra and not self.asym and not self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "basic"
            self.params = ["alpha", "beta"]
        elif self.dyna and not self.extra and not self.asym and not self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "dyna"
            self.params = ["alpha", "beta", "omega"]
        elif not self.dyna and self.extra and not self.asym and not self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "extra"
            self.params = ["alpha attract", "beta", "alpha not attract"]
        elif not self.dyna and not self.extra and self.asym and not self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "asym"
            self.params = ["pos alpha", "beta", "neg alpha"]
        elif not self.dyna and not self.extra and not self.asym and self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "transfer"
            self.params = ["alpha", "beta", "K1"]
        elif self.v_init and not self.dyna and not self.extra and not self.asym and not self.transfer and not self.pearce:
            self.modelFolder = "v_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init"]
        elif not self.dyna and not self.extra and not self.asym and not self.transfer and not self.v_init and self.pearce:
            self.modelFolder = "pearce"
            self.params = ["alpha", "beta", "omega"]
        elif not self.dyna and not self.extra and self.asym and not self.transfer and not self.v_init and self.pearce:
            self.modelFolder = "pearceAsym"
            self.params = ["pos alpha", "beta", "neg alpha"]
        elif not self.dyna and not self.extra and self.asym and self.transfer and not self.v_init and self.pearce:
            self.modelFolder = "pearceAsymTransfer"
            self.params = ["pos alpha", "beta", "neg alpha", "K1", "K2"]
        elif not self.dyna and not self.extra and not self.asym and self.transfer and not self.v_init and self.pearce:
            self.modelFolder = "pearceTransfer"
            self.params = ["alpha", "beta", "K1"]
        elif not self.dyna and not self.extra and self.asym and self.transfer and not self.v_init and not self.pearce:
            self.modelFolder = "asymTransfer"
            self.params = ["pos alpha", "beta", "neg alpha", "K1", "K2"]


        elif not self.dyna and not self.extra and self.asym and not self.transfer and self.v_init and not self.pearce:
            self.modelFolder = "asymV_init"
            self.params = ["pos alpha", "beta", "neg alpha", "V0_init", "V1_init"]
        elif not self.dyna and not self.extra and not self.asym and self.transfer and self.v_init and not self.pearce:
            self.modelFolder = "transferV_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init", "K1"]
        
        elif not self.dyna and not self.extra and not self.asym and not self.transfer and self.v_init and self.pearce:
            self.modelFolder = "pearceV_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init"]
        elif not self.dyna and not self.extra and self.asym and not self.transfer and self.v_init and self.pearce:
            self.modelFolder = "pearceAsymV_init"
            self.params = ["pos alpha", "beta", "neg alpha", "V0_init", "V1_init"]
        elif not self.dyna and not self.extra and self.asym and self.transfer and self.v_init and self.pearce:
            self.modelFolder = "pearceAsymTransferV_init"
            self.params = ["pos alpha", "beta", "neg alpha", "V0_init", "V1_init", "K1", "K2"]
        elif not self.dyna and not self.extra and not self.asym and self.transfer and self.v_init and self.pearce:
            self.modelFolder = "pearceTransferV_init"
            self.params = ["alpha", "beta", "V0_init", "V1_init", "K1"]
        elif not self.dyna and not self.extra and self.asym and self.transfer and self.v_init and not self.pearce:
            self.modelFolder = "asymTransferV_init"
            self.params = ["pos alpha", "beta", "neg alpha", "V0_init", "V1_init", "K1", "K2"]


    def recovery(self):
        
        taskSimulationList = []
        NLL_array_list = []

        if self.pearce:
            alphaSimArr = -20 + 40 * np.random.rand(self.simulations)
            if self.extra and self.asym:
                alphaNegSimArr, alpha2PosSimArr, alpha2NegSimArr = (
                    (-20 + 40 * np.random.rand(self.simulations)) for i in range(3)
                )
            elif self.extra and not self.asym:
                alpha2SimArr = -20 + 40 * np.random.rand(self.simulations)
            elif self.asym and not self.extra:
                alphaNegSimArr = -20 + 40 * np.random.rand(self.simulations)
        else:
            alphaSimArr = np.random.rand(self.simulations)
            if self.extra and self.asym:
                alphaNegSimArr, alpha2PosSimArr, alpha2NegSimArr = (
                    np.random.rand(self.simulations) for i in range(3)
                )
            elif self.extra and not self.asym:
                alpha2SimArr = np.random.rand(self.simulations)
            elif self.asym and not self.extra:
                alphaNegSimArr = np.random.rand(self.simulations)

        if self.transfer:
            K1SimArr = np.random.rand(self.simulations)
            if self.extra and self.asym:
                K3SimArr, K4SimArr, K2SimArr = (
                    np.random.rand(self.simulations) for i in range(3)
                )
            elif self.extra and not self.asym:
                K2SimArr = np.random.rand(self.simulations)
            elif self.asym and not self.extra:
                K3SimArr = np.random.rand(self.simulations)

        if self.dyna:
            omegaArr = np.random.rand(self.simulations)
        elif self.pearce:
            omegaArr = -10 + 20 * np.random.rand(self.simulations)

        if self.v_init:
            V_option0_rand = np.random.rand(self.simulations, 1)
            V_option0Init_Arr = np.repeat(V_option0_rand, 9, axis=1).reshape(
                (self.simulations, 3, 3)
            )

            V_option1_rand = np.random.rand(self.simulations, 1)
            V_option1Init_Arr = np.repeat(V_option1_rand, 9, axis=1).reshape(
                (self.simulations, 3, 3)
            )
        

        betaSimArr = 0 + 14.0 * np.random.rand(self.simulations)

        simulatedRLParams = np.empty((self.simulations, 11))
        simulatedRLParams[:] = np.nan
        recoveredRLParams = np.empty((self.simulations, 11))
        recoveredRLParams[:] = np.nan

        for i in range(0, self.simulations):
            print("Started simulation ", i)
        # Checking parameters from the grid
            alphaPosSim = alpha2PosSim = alphaNegSim = alpha2NegSim = K1Sim = K2Sim = K3Sim = K4Sim = betaSim = V_option0Init = V_option1Init = omegaSim = None
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
                alphaPosSim = alpha2PosSim = alphaNegSim = alpha2NegSim = alphaSimArr[i]

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

            if self.dyna or self.pearce:
                omegaSim = omegaArr[i]

            if self.v_init:
                V_option0Init = V_option0Init_Arr[i]
                V_option1Init = V_option1Init_Arr[i]

            taskSimulation = task_Design(self.mainTrials, 0, asym=self.asym, extra=self.extra, dyna=self.dyna, transfer=self.transfer, alphaPos=alphaPosSim, alphaNeg=alphaNegSim, alpha2Pos=alpha2PosSim, alpha2Neg=alpha2NegSim, K1=K1Sim, K2=K2Sim, K3=K3Sim, K4=K4Sim, beta=betaSim, V_option0Init=V_option0Init, V_option1Init=V_option1Init, dyna=self.dyna, omegaInit=omegaSim, pearce=self.pearce)
            taskSimulation.taskStructure()
            taskSimulation.RLloops()
            taskSimulation.statisticalLearning()

            taskSimulationList.append(taskSimulation)

            if self.pearce:
                alphaGrid = -20 + 40 * np.random.rand(self.gridsize)
                if self.extra and self.asym:
                    alphaNegGrid, alpha2PosGrid, alpha2NegGrid = (
                        (-20 + 40 * np.random.rand(self.gridsize)) for i in range(3)
                    )
                elif self.extra and not self.asym:
                    alpha2Grid = -20 + 40 * np.random.rand(self.gridsize)
                elif self.asym and not self.extra:
                    alphaNegGrid = -20 + 40 * np.random.rand(self.gridsize)
            else:
                alphaGrid = np.random.rand(self.gridsize)
                if self.extra and self.asym:
                    alphaNegGrid, alpha2PosGrid, alpha2NegGrid = (
                        np.random.rand(self.gridsize) for i in range(3)
                    )
                elif self.extra and not self.asym:
                    alpha2Grid = np.random.rand(self.gridsize)
                elif self.asym and not self.extra:
                    alphaNegGrid = np.random.rand(self.gridsize)

            if self.transfer:
                K1Grid = np.random.rand(self.gridsize)
                if self.extra and self.asym:
                    K3Grid, K4Grid, K2Grid = (
                        np.random.rand(self.gridsize) for i in range(3)
                    )
                elif self.extra and not self.asym:
                    K2Grid = np.random.rand(self.gridsize)
                elif self.asym and not self.extra:
                    K3Grid = np.random.rand(self.gridsize)

            if self.dyna:
                omegaGrid = np.random.rand(self.gridsize)
            elif self.pearce:
                omegaGrid = -10 + 20 * np.random.rand(self.gridsize)

            if self.v_init:
                V_option0_randGrid = np.random.rand(self.gridsize, 1)
                V_option0Init_Grid = np.repeat(V_option0_randGrid, 9, axis=1).reshape(
                    (self.gridsize, 3, 3)
                )

                V_option1_randGrid = np.random.rand(self.gridsize, 1)
                V_option1Init_Grid = np.repeat(V_option1_randGrid, 9, axis=1).reshape(
                    (self.gridsize, 3, 3)
                )
            betaGrid = 0 + 14.0 * np.random.rand(self.gridsize)
            NLL_array = np.empty((self.gridsize, 12))
            NLL_array[:] = np.nan

            # Simulating from the grid to recover the sum of negative log likelihood of actions from parameters corresponding to each grid value
            for j in range(0, self.gridsize):
                print("Step ", j)
                # For each point on the grid we instantiate the arrays for the time steps-
                """Instantiating for the fitting"""

                choiceProb = np.empty((taskSimulation.mainTrials, 2))
                choiceProb[:] = np.nan
                actionProb = np.empty((taskSimulation.mainTrials))
                actionProb[:] = np.nan
                V_option0, V_option1 = (
                    np.empty((taskSimulation.mainTrials + 1, 3, 3)) for i in range(2)
                )
                V_option0[:], V_option1[:] = (np.nan for i in range(2))

                rewardPE = np.empty((taskSimulation.mainTrials, 3, 3))
                rewardPE[:] = np.nan

                alphaPosCheck = alpha2PosCheck = alphaNegCheck = alpha2NegCheck = K1Check = K2Check = K3Check = K4Check = betaCheck = None
                if self.v_init:
                    V_option0[0, :] = V_option0Init_Grid[j]
                    V_option1[0, :] = V_option1Init_Grid[j]
                else:
                    V_option0[0, :] = 0.5
                    V_option1[0, :] = 0.5
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
                    omega = omega2 = omega3 = omega4 = omegaGrid[j]
                if self.pearce:
                    eta = omegaGrid[j]


                for t in range(0, taskSimulation.mainTrials):

                    otherPairs = [
                        p
                        for p in list(np.unique(taskSimulation.taskStruct, axis=0))
                        if bool(p[0] == taskSimulation.stimulusPair[t, 0]) ^ bool(p[1] == taskSimulation.stimulusPair[t, 1])
                    ]
                    
                    # Prob of choosing the 0th and 1st option respectively
                    choiceProb[t, 0] = np.exp(
                        betaCheck * V_option0[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                    ) / (
                        (
                            np.exp(
                                betaCheck * V_option0[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                            )
                        )
                        + (
                            np.exp(
                                betaCheck * V_option1[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                            )
                        )
                    )
                    choiceProb[t, 1] = 1 - choiceProb[t, 0]

                    actionProb[t] = (
                        choiceProb[t, int(taskSimulation.action[t])]
                        if ~np.isnan(choiceProb[t, int(taskSimulation.action[t])])
                        else np.nan
                    )
                    
                    if taskSimulation.action[t] == 0:
                        rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])] = (
                            taskSimulation.reward[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                            - V_option0[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                        )
                        V_option0[t + 1, :] = V_option0[t, :]

                        if taskSimulation.reward[(t,) + tuple(taskSimulation.stimulusPair[t,:])] == 1:
                            if self.dyna:
                                omega = (
                                    omega
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                        - omega
                                    )
                                    * alphaPosCheck
                                )
                                if omega > 1:
                                    omega = 1
                                elif omega < 0:
                                    omega = 0
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K1Check * omega * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )
                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0
                            elif self.pearce:
                                omega = (
                                    eta
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                    )
                                    * alphaPosCheck
                                )
                                omega = 1/(1 + np.exp(-omega))
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K1Check * omega * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )
                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0
                            
                            else:
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + alphaPosCheck * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )

                                

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K1Check * alphaPosCheck * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0
                            if self.dyna:
                                if not self.asym and not self.extra:
                                    omega2 = omega3 = omega4 = omega
                                elif self.asym and not self.extra:
                                    omega2 = omega
                                elif self.extra and not self.asym:
                                    omega3 = omega

                        else:
                            if self.pearce:
                                omega3 = (
                                    eta
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                    )
                                    * alphaNegCheck
                                )
                                omega3 = 1/(1 + np.exp(-omega3))
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega3 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K3Check * omega3 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0
                            elif self.dyna:
                                omega3 = (
                                    omega3
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                        - omega3
                                    )
                                    * alphaNegCheck
                                )
                                if omega3 > 1:
                                    omega3 = 1
                                elif omega3 < 0:
                                    omega3 = 0
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega3 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K3Check * omega3 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0
                            
                            
                            else:
                                V_option0[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option0[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + alphaNegCheck * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )


                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option0[(t + 1,) + tuple(pair)] = V_option0[
                                            (t,) + tuple(pair)
                                        ] - K3Check * alphaNegCheck * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option0[(t + 1,) + tuple(pair)] > 1:
                                            V_option0[(t + 1,) + tuple(pair)] = 1
                                        elif V_option0[(t + 1,) + tuple(pair)] < 0:
                                            V_option0[(t + 1,) + tuple(pair)] = 0

                            if self.dyna:
                                if not self.asym and not self.extra:
                                    omega2 = omega = omega4 = omega3
                                elif self.asym and not self.extra:
                                    omega4 = omega3
                                elif self.extra and not self.asym:
                                    omega = omega3

                        V_option1[t + 1, :] = V_option1[t, :]

                    elif taskSimulation.action[t] == 1:
                        rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])] = (
                            taskSimulation.reward[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                            - V_option1[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                        )

                        V_option1[t + 1, :] = V_option1[t, :]

                        if taskSimulation.reward[(t,) + tuple(taskSimulation.stimulusPair[t,:])] == 1:
                            if self.dyna:
                                omega2 = (
                                    omega2
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                        - omega2
                                    )
                                    * alpha2PosCheck
                                )
                                if omega2 > 1:
                                    omega2 = 1
                                elif omega2 < 0:
                                    omega2 = 0
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega2 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K2Check * omega2 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            elif self.pearce:
                                omega2 = (
                                    eta
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                    )
                                    * alpha2PosCheck
                                )
                                omega2 = 1/(1 + np.exp(-omega2))
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega2 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K2Check * omega2 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            
                            else:
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + alpha2PosCheck * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )

                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K2Check * alpha2PosCheck * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            if self.dyna:
                                if not self.asym and not self.extra:
                                    omega3 = omega = omega4 = omega2
                                elif self.asym and not self.extra:
                                    omega = omega2
                                elif self.extra and not self.asym:
                                    omega4 = omega2

                        else:
                            if self.dyna:
                                omega4 = (
                                    omega4
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                        - omega4
                                    )
                                    * alpha2NegCheck
                                )
                                if omega4 > 1:
                                    omega4 = 1
                                elif omega4 < 0:
                                    omega4 = 0
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega4 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                           
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K4Check * omega4 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            elif self.pearce:
                                omega4 = (
                                    eta
                                    + (
                                        abs(rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])])
                                    )
                                    * alpha2NegCheck
                                )
                                omega4 = 1/(1 + np.exp(-omega4))
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + omega4 * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                           
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K4Check * omega4 * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            
                            
                            else:
                                V_option1[
                                    (t + 1,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] = V_option1[
                                    (t,) + tuple(taskSimulation.stimulusPair[t,:])
                                ] + alpha2NegCheck * (
                                    rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                )
                                
                                if self.transfer:
                                    for pair in otherPairs:
                                        V_option1[(t + 1,) + tuple(pair)] = V_option1[
                                            (t,) + tuple(pair)
                                        ] - K4Check * alpha2NegCheck * (
                                            rewardPE[(t,) + tuple(taskSimulation.stimulusPair[t,:])]
                                        )

                                        if V_option1[(t + 1,) + tuple(pair)] > 1:
                                            V_option1[(t + 1,) + tuple(pair)] = 1
                                        elif V_option1[(t + 1,) + tuple(pair)] < 0:
                                            V_option1[(t + 1,) + tuple(pair)] = 0
                            if self.dyna:
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
                if self.dyna or self.pearce:
                    NLL_array[j, 6] = omegaGrid[j]
                if self.v_init:
                    NLL_array[j, 7] = V_option0_randGrid[j, 0]
                    NLL_array[j, 8] = V_option1_randGrid[j, 0]
                if self.transfer:
                    NLL_array[j, 9] = K1Check
                    if self.extra and not self.asym:
                        NLL_array[j, 10] = K2Check
                    elif self.asym and not self.extra:
                        NLL_array[j, 11] = K3Check
                    elif self.asym and self.extra:
                        NLL_array[j, 10] = K2Check
                        NLL_array[j, 11] = K3Check
                        NLL_array[j, 12] = K4Check

            minIndex = np.argmin(NLL_array[:, 0])
            recoveredAlphaPos = alphaGrid[minIndex]
            recoveredBeta = betaGrid[minIndex]
            if self.asym and not self.extra:
                recoveredAlphaNeg = alphaNegGrid[minIndex]
            elif self.extra and not self.asym:
                recoveredAlpha2Pos = alpha2Grid[minIndex]
            elif self.extra and self.asym:
                recoveredAlphaNeg = alphaNegGrid[minIndex]
                recoveredAlpha2Pos = alpha2PosGrid[minIndex]
                recoveredAlpha2Neg = alpha2NegGrid[minIndex]
            if self.dyna or self.pearce:
                recoveredOmega = omegaGrid[minIndex]
            if self.v_init:
                recovered_V0 = V_option0_randGrid[minIndex, 0]
                recovered_V1 = V_option1_randGrid[minIndex, 0]
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
                simulatedRLParams[i, 3] = alpha2PosSim
                recoveredRLParams[i, 3] = recoveredAlpha2Pos
            elif self.extra and self.asym:
                simulatedRLParams[i, 2] = alphaNegSim
                simulatedRLParams[i, 3] = alpha2PosSim
                simulatedRLParams[i, 4] = alpha2NegSim
                recoveredRLParams[i, 2] = recoveredAlphaNeg
                recoveredRLParams[i, 3] = recoveredAlpha2Pos
                recoveredRLParams[i, 4] = recoveredAlpha2Neg
            
            if self.dyna or self.pearce:
                simulatedRLParams[i, 5] = omegaSim
                recoveredRLParams[i, 5] = recoveredOmega
            
            if self.v_init:
                simulatedRLParams[i, 6] = V_option0Init[0,0]
                simulatedRLParams[i, 7] = V_option1Init[0,0]
                recoveredRLParams[i, 6] = recovered_V0
                recoveredRLParams[i, 7] = recovered_V1

            if self.transfer:
                simulatedRLParams[i, 8] = K1Sim
                recoveredRLParams[i, 8] = recoveredK1
                if self.extra and not self.asym:
                    simulatedRLParams[i, 9] = K2Sim
                    recoveredRLParams[i, 9] = recoveredK2
                elif self.asym and not self.extra:
                    simulatedRLParams[i, 9] = K3Sim
                    recoveredRLParams[i, 9] = recoveredK3
                elif self.asym and self.extra:
                    simulatedRLParams[i, 9] = K2Sim
                    simulatedRLParams[i, 10] = K3Sim
                    simulatedRLParams[i, 11] = K4Sim
                    recoveredRLParams[i, 9] = recoveredK2
                    recoveredRLParams[i, 10] = recoveredK3
                    recoveredRLParams[i, 11] = recoveredK4
            NLL_array_list.append(NLL_array)
        

        destination = "git/multlearn-sns/Modelling/Recovery/"+self.modelFolder
        if not os.path.exists(destination):
            os.makedirs(destination)

        with open(os.path.join(destination,"simulatedParams.npy"), "wb") as f:
            np.save(f, simulatedRLParams)
        
        with open(os.path.join(destination,"recoveredParams.npy"), "wb") as f:
            np.save(f, recoveredRLParams)

        with open(os.path.join(destination,"NLL_array.npy"), "wb") as f:
            np.save(f, NLL_array_list)
 
        return simulatedRLParams, recoveredRLParams, NLL_array_list
    

    def recoveryPlot(self, simulatedRLParams, recoveredRLParams):

        simulatedRLParams = simulatedRLParams[:, ~np.isnan(simulatedRLParams).any(axis=0)]
        recoveredRLParams = recoveredRLParams[:, ~np.isnan(recoveredRLParams).any(axis=0)]

        for idx, param in enumerate(self.params):
            
            paramRecovery = {'simulated '+param: simulatedRLParams[:,idx], 'recovered '+param: recoveredRLParams[:,idx]}
            paramRecovery = pd.DataFrame(paramRecovery)
            corrParam = scipy.stats.pearsonr(paramRecovery['simulated '+param], paramRecovery['recovered '+param])
        
            g = sns.lmplot(x="simulated "+param, y="recovered "+param, data=paramRecovery)
            # Access the figure
            fig = g.figure
            # Add a title to the Figure
            fig.suptitle("correlation {:.2f}, p-value {:.2f}".format(corrParam[0], corrParam[1]), fontsize=12)
            destination = "git/multlearn-sns/Modelling/Recovery/"+self.modelFolder
            if not os.path.exists(destination):
                os.makedirs(destination)
            plt.savefig(os.path.join(destination,param.replace(" ", "_")+"Recovery.png"))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovery of RL parameters")
    parser.add_argument("--mainTrials", type=int, default=60)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--gridsize", type=int, default=100)
    parser.add_argument("--extra", action='store_true')
    parser.add_argument("--asym", action='store_true')
    parser.add_argument("--transfer", action='store_true')
    parser.add_argument("--pearce", action='store_true')
    parser.add_argument("--dyna", action='store_true')
    parser.add_argument("--v_init", action='store_true')
    args = parser.parse_args()
    recovery = Recovery(args.mainTrials, args.simulations, args.gridsize, args.extra, args.asym, args.transfer, args.pearce, args.dyna, args.v_init)
    simParams, recParams, NLL = recovery.recovery()
    recovery.recoveryPlot(simParams, recParams)

        