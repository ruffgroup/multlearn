import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import csv
import collections
import shutil

dataPath = "/mnt/d/data/fittedParametersRecoveredModels"
IDs = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "9",
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

models = [
    "basic",
    "v_init",
    "asym",
    "transfer",
    "pearce",
    "pearceAsym",
    "pearceTransfer",
    "transferV_init"
]

listBestFits = list()

BIC_basic = np.zeros(58)
BIC_v_init = np.zeros(58)
BIC_asym = np.zeros(58)
BIC_transfer = np.zeros(58)
BIC_pearce = np.zeros(58)
BIC_pearceAsym = np.zeros(58)
BIC_pearceTransfer = np.zeros(58)
BIC_transferV_init = np.zeros(58)

winning_model = {key: [0, 0] for key in models}

for idx, IDnr in enumerate(IDs):
    newPath = "/mnt/d/data/fittedParametersRecoveredModels/sub-{0}".format(IDnr)
    LL_basic = np.load(newPath + "/basic/BIC_basic.npy")
    LL_v_init = np.load(newPath + "/v_init/BIC_v_init.npy")
    LL_asym = np.load(newPath + "/asym/BIC_asym.npy")
    LL_transfer = np.load(newPath + "/transfer/BIC_transfer.npy")
    LL_transferV_init = np.load(newPath + "/transferV_init/BIC_transferV_init.npy")
    
    LL_pearce = np.load(newPath + "/pearce/BIC_pearce.npy")
    LL_pearceAsym = np.load(newPath + "/pearceAsym/BIC_pearceAsym.npy")
    LL_pearceTransfer = np.load(newPath + "/pearceTransfer/BIC_pearceTransfer.npy")

    BIC_basic[idx] = np.mean([2 * np.log(60) - 2 * np.log(LL_basic[run]) for run in range(6)])
    BIC_v_init[idx] = np.mean([4 * np.log(60) - 2 * np.log(LL_v_init[run]) for run in range(6)])
    BIC_asym[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_asym[run]) for run in range(6)])
    BIC_transfer[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_transfer[run]) for run in range(6)])
    BIC_transferV_init[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_transferV_init[run]) for run in range(6)])
    
    BIC_pearce[idx] = np.mean([2 * np.log(60) - 2 * np.log(LL_pearce[run]) for run in range(6)])
    BIC_pearceAsym[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_pearceAsym[run]) for run in range(6)])
    BIC_pearceTransfer[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_pearceTransfer[run]) for run in range(6)])
    
    best_fitting_arr = [
            BIC_basic[idx],
            BIC_v_init[idx],
            BIC_asym[idx],
            BIC_transfer[idx],
            BIC_transferV_init[idx],
            BIC_pearce[idx],
            BIC_pearceAsym[idx],
            BIC_pearceTransfer[idx],
        ]
    
    best_fitting = np.argmin(best_fitting_arr)
    best_fitting_val = best_fitting_arr[best_fitting]

    #for i in range(5):
    #    shutil.copy(models[best_fitting] + "/" + IDnr + "_" + str(i) + "_" + models[best_fitting] + "_Plots.pdf",
    #                "bestFittingVals/sub-" + IDnr)

    rpeBest = scipy.io.loadmat(
        newPath + "/" + models[best_fitting] + "/rpe" + models[best_fitting] + ".mat"
    )

    scipy.io.savemat(
            newPath + "/rpeBest.mat",
            mdict={"rpe": rpeBest},
        )

    winning_model[models[best_fitting]][0] += 1
    for i in range(len(best_fitting_arr)):
        winning_model[models[i]][1] += best_fitting_arr[i]
    #winning_model[models[best_fitting]][1] += best_fitting_val
    listBestFits.append([IDnr, models[best_fitting], best_fitting_val])


with open("BestFitting.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(listBestFits)

final = dict(sorted(winning_model.items(), key=lambda x: -x[1][0]))

for idx, IDnr in enumerate(IDs):
    newPath = "/mnt/d/data/fittedParametersRecoveredModels/sub-{0}".format(IDnr)
    rpeBestOverall = scipy.io.loadmat(
        newPath + "/" + str(list(final.keys())[0]) + "/rpe" + str(list(final.keys())[0]) + ".mat"
    )

    scipy.io.savemat(
            newPath + "/rpeBestOverall.mat",
            mdict={"rpe": rpeBestOverall},
        )

with open("BestFittingOverall.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(final.items())

print(final)
