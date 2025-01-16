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

sys.path.append(sys.path[0] + "/..")
from TaskDesign import task_Design


class Plotting:
    def __init__(self, mainTrials, additionalTrials, gridCount, ID, ww, method, reps):
        self.mainTrials = mainTrials
        self.additionalTrials = additionalTrials
        self.gridCount = gridCount
        self.ID = ID
        self.statLearnPar = 1
        self.ww = ww
        self.method = method
        self.reps = reps

        wanted_dir = "/mnt/d/data/sourcedata/behavior/modified_files"

        # Get savedVals file
        self.savedValsFile = glob.glob(
            os.path.abspath(wanted_dir) + "/*{}_savedValues.csv".format(self.ID)
        )[0]

    def plots_modelFitting(
        self,
        NLL_array,
        alphasPos,
        alphas2Pos,
        alphasNeg,
        alphas2Neg,
        betas,
        omega,
        V_option0Inits,
        V_option1Inits,
        K1,
        K2,
        K3,
        K4,
        saveAs,
        extra,
        asym,
        transfer,
        pearce,
        dyna,
    ):
        saving_folder = saveAs
        subjectData = pd.read_csv(self.savedValsFile)

        for run in range(0, max(subjectData.runNumber)):
            NLL_run = NLL_array[run]
            alphaPos = alphasPos[run]
            beta = betas[run]
            varList = ["beta: {}".format(np.round(beta, 2))]
            if extra and not asym:
                alphaNeg = alphaPos
                alpha2Neg = alpha2Pos = alphas2Pos[run]
                varList.append(", V0 alpha: {}".format(np.round(alphaPos, 2)))
                varList.append(", V1 alpha: {}".format(np.round(alpha2Pos, 2)))
            elif asym and not extra:
                alpha2Pos = alphaPos
                alpha2Neg = alphaNeg = alphasNeg[run]
                varList.append(", Pos reward alpha: {}".format(np.round(alphaPos, 2)))
                varList.append(", Neg reward alpha: {}".format(np.round(alphaNeg, 2)))
            elif asym and extra:
                alpha2Pos = alphas2Pos[run]
                alphaNeg = alphasNeg[run]
                alpha2Neg = alphas2Neg[run]
                varList.append(", V0 Pos reward alpha: {}".format(np.round(alphaPos, 2)))
                varList.append(", V0 Neg reward alpha: {}".format(np.round(alphaNeg, 2)))
                varList.append(", V1 Pos reward alpha: {}".format(np.round(alpha2Pos, 2)))
                varList.append(", V1 Neg reward alpha: {}".format(np.round(alpha2Neg, 2)))
            else:
                alpha2Pos = alphaNeg = alpha2Neg = alphaPos
                varList.append(", alpha: {}".format(np.round(alphaPos, 2))) 
            if not np.isnan(V_option0Inits).all():
                V_option0 = V_option0Inits[run]
                V_option1 = V_option1Inits[run]
                varList.append("\nV0 init: {0}, V1 init {1}".format(np.round(V_option0[0][0], 2), np.round(V_option1[0][0], 2)))
            else:
                V_option0 = V_option1 = None
            if pearce or dyna:
                omegaRun = omega[run]
                varList.append(", omega: {}".format(np.round(omegaRun,2)))
            else:
                omegaRun = None

            if transfer:
                K1run = K1[run]
                if extra and not asym:
                    K3run = K1run
                    K4run = K2run = K2[run]
                    varList.append(", V0 Kappa: {}".format(np.round(K1run, 2)))
                    varList.append(", V1 Kappa: {}".format(np.round(K2run, 2)))
                elif asym and not extra:
                    K2run = K1run
                    K3run = K4run = K3[run]
                    varList.append(", Pos Kappa: {}".format(np.round(K1run, 2)))
                    varList.append(", Neg Kappa: {}".format(np.round(K3run, 2)))
                elif asym and extra:
                    K2run = K2[run]
                    K3run = K3[run]
                    K4run = K4[run]
                    varList.append(", V0 Pos Kappa: {}".format(np.round(K1run, 2)))
                    varList.append(", V1 Pos Kappa: {}".format(np.round(K2run, 2)))
                    varList.append(", V0 Neg Kappa: {}".format(np.round(K3run, 2)))
                    varList.append(", V1 Neg Kappa: {}".format(np.round(K4run, 2)))
                else:
                    K2run = K3run = K4run = K1run
                    varList.append(", Kappa: {}".format(np.round(K1run, 2)))
            else:
                K1run = K2run = K3run = K4run = None
            

            runData = subjectData[subjectData.runNumber == run + 1].reset_index()
            green = runData[runData.combinationConditionalProbability == 0.5].stimulusPair.unique()
            green = [ast.literal_eval(green[0]), ast.literal_eval(green[1]), ast.literal_eval(green[2])]

            attracts = runData.accurate[runData.correctResponse == 0]
            notAttracts = runData.accurate[runData.correctResponse == 1]

            A_line = (
                pd.DataFrame(ma(attracts, self.ww, self.method)).astype(float).interpolate(option="spline", order=1)
            )
            NA_line = (
                pd.DataFrame(ma(notAttracts, self.ww, self.method)).astype(float).interpolate(option="spline", order=1)
            )
            Acc_line = (
                pd.DataFrame(ma(runData.accurate, self.ww, self.method))
                .astype(float)
                .interpolate(option="spline", order=1)
            )

            simA_lines = np.empty((self.reps, int(self.mainTrials / 2) - self.ww + 1))
            simNA_lines = np.empty((self.reps, int(self.mainTrials / 2) - self.ww + 1))
            simAcc_lines = np.empty((self.reps, int(self.mainTrials) - self.ww + 1))

            taskStruct = np.array([list(tuple(ast.literal_eval(x))) for x in runData.stimulusPair])

            feedbackAcc = runData.feedbackAccuracy.astype(int)

            for i in range(self.reps):
                simulation = task_Design(
                    self.mainTrials,
                    self.additionalTrials,
                    alphaPos=alphaPos,
                    alphaNeg=alphaNeg,
                    alpha2Pos=alpha2Pos,
                    alpha2Neg = alpha2Neg,
                    beta=beta,
                    K1=K1run,
                    K2=K2run,
                    K3=K3run,
                    K4=K4run,
                    omegaInit=omegaRun,
                    V_option0Init=V_option0,
                    V_option1Init=V_option1,
                    dyna=dyna,
                    extra=extra,
                    asym=asym,
                    transfer=transfer,
                    pearce=pearce,
                )
                simulation.taskStructure(taskStruct, green, feedbackAcc)
                simulation.RLloops()

                simAttracts = simulation.accurate[simulation.correctResponse == 0]
                simNotAttracts = simulation.accurate[simulation.correctResponse == 1]
                simC = simulation.accurate
                if simAttracts.shape[0] == 30 & simNotAttracts.shape[0] == 30:
                    simA_lines[i, :] = ma(simAttracts, self.ww, self.method)
                    simNA_lines[i, :] = ma(simNotAttracts, self.ww, self.method)
                    simAcc_lines[i, :] = ma(simC.flatten(), self.ww, self.method)

            if simA_lines.size:
                simA_line = pd.DataFrame(np.mean(simA_lines, axis=0))
                simNA_line = pd.DataFrame(np.mean(simNA_lines, axis=0))
                simAcc_line = pd.DataFrame(np.mean(simAcc_lines, axis=0))

                fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                titleStr = " ".join(varList)
                fig.suptitle(
                    "MA of binary accuracy for participant {0}, run {1}:" .format(
                        self.ID,
                        run + 1
                    ) + " " + titleStr
                )

                ax[0].plot(A_line, label="Real data")
                ax[0].plot(simA_line, label="Simulated data")
                ax[0].set_title("Accurate for 'attracts'")
                ax[0].set_ylim(0, 1.1)
                ax[0].set_ylabel("Accurate")
                ax[0].legend()

                ax[1].plot(NA_line, label="Real data")
                ax[1].plot(simNA_line, label="Simulated data")
                ax[1].set_title("Accurate for 'does not attract'")
                ax[1].set_ylim(0, 1.1)
                ax[1].set_ylabel("Accurate")
                ax[1].legend()

                ax[2].plot(Acc_line, label="Real data")
                ax[2].plot(simAcc_line, label="Simulated data")
                ax[2].set_title("Accurate overall")
                ax[2].set_ylim(0, 1.1)
                ax[2].set_ylabel("Accurate")
                ax[2].legend()

                plt.tight_layout()

                # Creating figure
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection="3d")
                NLL_run[NLL_run[:, 0] > min(NLL_run[:, 0]) + 8] = np.nan
                ax.scatter(NLL_run[:, 1], NLL_run[:, 2], NLL_run[:, 0], color="green")
                ax.set_xlabel("alpha", fontweight="bold")
                ax.set_ylabel("beta", fontweight="bold")
                ax.set_zlabel("NLL", fontweight="bold")
                ax.set_title("Participant {0}, run {1}: NLL, alpha and beta 3D scatter plot".format(self.ID, run + 1))

                plt.tight_layout()

                count = 3
                for var in [
                    (1, alphaPos, "V0 Pos alpha"),
                    (2, beta, "beta"),
                    (3, alphaNeg, "V0 Neg alpha"),
                    (4, alpha2Pos, "V1 Pos alpha"),
                    (5, alpha2Neg, "V1 Neg alpha"),
                    (6, V_option0, "V0"),
                    (7, V_option1, "V1"),
                    (8, K1run, "V0 Pos Kappa"),
                    (9, K2run, "V1 Pos Kappa"),
                    (10, K3run, "V0 Neg Kappa"),
                    (11, K4run, "V1 Neg Kappa"),
                    (12, omegaRun, "omega")
                ]:
                    if var[1] is not None:
                        plt.figure(count)
                        plt.scatter(NLL_run[:, var[0]], NLL_run[:, 0])
                        plt.xlabel("{}".format(var[2]), fontweight="bold")
                        plt.ylabel("NLL", fontweight="bold")
                        plt.title(
                            "Participant {0}, run {1}: NLL and {2} scatter plot".format(self.ID, run + 1, var[2])
                        )
                        plt.tight_layout()
                        count += 1

                os.makedirs(saving_folder, exist_ok=True)
 
                save_name = "{0}_{1}_{2}_Plots.pdf".format(self.ID, run, saveAs)

                file_path = os.path.join(saving_folder, save_name)

                pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
                for fig in range(1, plt.gcf().number + 1):
                    pdf.savefig(fig)
                pdf.close()

                plt.close("all")

    # Stat learning plots

    def plots_stats(self, beliefs, surprise):
        subjectData = pd.read_csv(self.savedValsFile)
        subjectData["stimulusPair"] = subjectData["stimulusPair"].apply(ast.literal_eval)
        # subjectInfo = pd.read_csv(str([file[1] for file in self.expInfoFiles if file[0] == ID][0]))

        for run in range(0, max(subjectData.runNumber)):
            beliefsStat = beliefs[run]
            statSurprise = surprise[run]

            plt.figure(1)
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 0, 0], label="1A")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 0, 1], label="1B")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 0, 2], label="1C")
            plt.title("Learning of statistical structure (beliefs) by Bayesian observer")

            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 1, 0], label="2A")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 1, 1], label="2B")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 1, 2], label="2C")

            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 2, 0], label="3A")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 2, 1], label="3B")
            plt.plot(range(0, self.mainTrials + 1), beliefsStat[:, 2, 2], label="3C")

            plt.xlabel("trials")
            plt.ylabel("Beliefs of probabilities co-occurence")
            plt.legend(bbox_to_anchor=(0.99, 0.65))
            plt.axhline(y=0.15 * 0.33, color="r", linestyle="-")
            plt.axhline(y=0.35 * 0.33, color="g", linestyle="-")
            plt.axhline(y=0.5 * 0.33, color="b", linestyle="-")

            plt.figure(2)
            plt.plot(statSurprise[~np.isnan(statSurprise)])
            plt.xlabel("trials")
            plt.ylabel("Total surprise")
            # plt.legend(bbox_to_anchor=(0.98, 0.9))
            plt.title("Total statistical surprise signal")

            plt.show()


def ma(interval, window_size=10, method="same"):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, method)
