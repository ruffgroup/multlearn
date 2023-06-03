import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import csv
import collections

dataPath = os.path.join(pathlib.Path(__file__).resolve().parents[3], "/data/fittedParameters")
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

models = [
    "Basic",
    "Init",
    "Extra",
    "Asym",
    "Transfer",
    "InitExtra",
    "InitAsym",
    "InitTransfer",
    "ExtraAsym",
    "ExtraTransfer",
    "AsymTransfer",
    "InitExtraAsym",
    "InitExtraTransfer",
    "InitAsymTransfer",
    "ExtraAsymTransfer",
    "InitAsymExtraTransfer",
    "Pearce",
    "PearceInit",
    "PearceExtra",
    "PearceAsym",
    "PearceTransfer",
    "PearceInitExtra",
    "PearceInitAsym",
    "PearceInitTransfer",
    "PearceAsymExtra",
    "PearceExtraTransfer",
    "PearceAsymTransfer",
    "PearceInitExtraAsym",
    "PearceInitExtraTransfer",
    "PearceInitAsymTransfer",
    "PearceExtraAsymTransfer",
    "PearceInitExtraAsymTransfer"
]

listBestFits = list()

BIC_Basic = np.zeros(58)
BIC_Init = np.zeros(58)
BIC_Extra = np.zeros(58)
BIC_Asym = np.zeros(58)
BIC_Transfer = np.zeros(58)
BIC_InitExtra = np.zeros(58)
BIC_InitAsym = np.zeros(58)
BIC_InitTransfer = np.zeros(58)
BIC_ExtraAsym = np.zeros(58)
BIC_ExtraTransfer = np.zeros(58)
BIC_AsymTransfer = np.zeros(58)
BIC_InitExtraAsym = np.zeros(58)
BIC_InitExtraTransfer = np.zeros(58)
BIC_InitAsymTransfer = np.zeros(58)
BIC_ExtraAsymTransfer = np.zeros(58)
BIC_InitAsymExtraTransfer = np.zeros(58)

BIC_Pearce = np.zeros(58)
BIC_PearceInit = np.zeros(58)
BIC_PearceExtra = np.zeros(58)
BIC_PearceAsym = np.zeros(58)
BIC_PearceTransfer = np.zeros(58)
BIC_PearceInitExtra = np.zeros(58)
BIC_PearceInitAsym = np.zeros(58)
BIC_PearceInitTransfer = np.zeros(58)
BIC_PearceAsymExtra = np.zeros(58)
BIC_PearceExtraTransfer = np.zeros(58)
BIC_PearceAsymTransfer = np.zeros(58)
BIC_PearceInitExtraAsym = np.zeros(58)
BIC_PearceInitExtraTransfer = np.zeros(58)
BIC_PearceInitAsymTransfer = np.zeros(58)
BIC_PearceExtraAsymTransfer = np.zeros(58)
BIC_PearceInitExtraAsymTransfer = np.zeros(58)

winning_model = {key: [0, 0] for key in models}

for idx, IDnr in enumerate(IDs):
    newPath = os.path.join(
        pathlib.Path(__file__).resolve().parents[3],
        "/data/fittedParameters/sub-{0}".format(IDnr),
    )
    LL_Basic = np.load(newPath + "/Basic/BIC_Basic.npy")
    LL_Init = np.load(newPath + "/Init/BIC_Init.npy")
    LL_Extra = np.load(newPath + "/Extra/BIC_Extra.npy")
    LL_Asym = np.load(newPath + "/Asym/BIC_Asym.npy")
    LL_Transfer = np.load(newPath + "/Transfer/BIC_Transfer.npy")
    LL_InitExtra = np.load(newPath + "/InitExtra/BIC_InitExtra.npy")
    LL_InitAsym = np.load(newPath + "/InitAsym/BIC_InitAsym.npy")
    LL_InitTransfer = np.load(newPath + "/InitTransfer/BIC_InitTransfer.npy")
    LL_ExtraAsym = np.load(newPath + "/ExtraAsym/BIC_ExtraAsym.npy")
    LL_ExtraTransfer = np.load(newPath + "/ExtraTransfer/BIC_ExtraTransfer.npy")
    LL_AsymTransfer = np.load(newPath + "/AsymTransfer/BIC_AsymTransfer.npy")
    LL_InitExtraAsym = np.load(newPath + "/InitExtraAsym/BIC_InitExtraAsym.npy")
    LL_InitExtraTransfer = np.load(newPath + "/InitExtraTransfer/BIC_InitExtraTransfer.npy")
    LL_InitAsymTransfer = np.load(newPath + "/InitAsymTransfer/BIC_InitAsymTransfer.npy")
    LL_ExtraAsymTransfer = np.load(newPath + "/ExtraAsymTransfer/BIC_ExtraAsymTransfer.npy")
    LL_InitAsymExtraTransfer = np.load(newPath + "/InitAsymExtraTransfer/BIC_InitAsymExtraTransfer.npy")

    LL_Pearce = np.load(newPath + "/Pearce/BIC_Pearce.npy")
    LL_PearceInit = np.load(newPath + "/PearceInit/BIC_PearceInit.npy")
    LL_PearceExtra = np.load(newPath + "/PearceExtra/BIC_PearceExtra.npy")
    LL_PearceAsym = np.load(newPath + "/PearceAsym/BIC_PearceAsym.npy")
    LL_PearceTransfer = np.load(newPath + "/PearceTransfer/BIC_PearceTransfer.npy")
    LL_PearceInitExtra = np.load(newPath + "/PearceInitExtra/BIC_PearceInitExtra.npy")
    LL_PearceInitAsym = np.load(newPath + "/PearceInitAsym/BIC_PearceInitAsym.npy")
    LL_PearceInitTransfer = np.load(newPath + "/PearceInitTransfer/BIC_PearceInitTransfer.npy")
    LL_PearceAsymExtra = np.load(newPath + "/PearceAsymExtra/BIC_PearceAsymExtra.npy")
    LL_PearceExtraTransfer = np.load(newPath + "/PearceExtraTransfer/BIC_PearceExtraTransfer.npy")
    LL_PearceAsymTransfer = np.load(newPath + "/PearceAsymTransfer/BIC_PearceAsymTransfer.npy")
    LL_PearceInitExtraAsym = np.load(newPath + "/PearceInitExtraAsym/BIC_PearceInitExtraAsym.npy")
    LL_PearceInitExtraTransfer = np.load(newPath + "/PearceInitExtraTransfer/BIC_PearceInitExtraTransfer.npy")
    LL_PearceInitAsymTransfer = np.load(newPath + "/PearceInitAsymTransfer/BIC_PearceInitAsymTransfer.npy")
    LL_PearceExtraAsymTransfer = np.load(newPath + "/PearceExtraAsymTransfer/BIC_PearceExtraAsymTransfer.npy")
    LL_PearceInitExtraAsymTransfer = np.load(newPath + "/PearceInitExtraAsymTransfer/BIC_PearceInitExtraAsymTransfer.npy")
    BIC_Basic[idx] = np.mean([2 * np.log(60) - 2 * np.log(LL_Basic[run]) for run in range(6)])
    BIC_Init[idx] = np.mean([4 * np.log(60) - 2 * np.log(LL_Init[run]) for run in range(6)])
    BIC_Extra[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_Extra[run]) for run in range(6)])
    BIC_Asym[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_Asym[run]) for run in range(6)])
    BIC_Transfer[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_Transfer[run]) for run in range(6)])
    BIC_InitExtra[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitExtra[run]) for run in range(6)])
    BIC_InitAsym[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitAsym[run]) for run in range(6)])
    BIC_InitTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitTransfer[run]) for run in range(6)])
    BIC_ExtraAsym[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_ExtraAsym[run]) for run in range(6)])
    BIC_ExtraTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_ExtraTransfer[run]) for run in range(6)])
    BIC_AsymTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_AsymTransfer[run]) for run in range(6)])
    BIC_InitExtraAsym[idx] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitExtraAsym[run]) for run in range(6)])
    BIC_InitExtraTransfer[idx] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitExtraTransfer[run]) for run in range(6)])
    BIC_InitAsymTransfer[idx] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitAsymTransfer[run]) for run in range(6)])
    BIC_ExtraAsymTransfer[idx] = np.mean([9 * np.log(60) - 2 * np.log(LL_ExtraAsymTransfer[run]) for run in range(6)])
    BIC_InitAsymExtraTransfer[idx] = np.mean([11 * np.log(60) - 2 * np.log(LL_InitAsymExtraTransfer[run]) for run in range(6)])

    BIC_Pearce[idx] = np.mean([2 * np.log(60) - 2 * np.log(LL_Pearce[run]) for run in range(6)])
    BIC_PearceInit[idx] = np.mean([4 * np.log(60) - 2 * np.log(LL_PearceInit[run]) for run in range(6)])
    BIC_PearceExtra[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceExtra[run]) for run in range(6)])
    BIC_PearceAsym[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceAsym[run]) for run in range(6)])
    BIC_PearceTransfer[idx] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceTransfer[run]) for run in range(6)])
    BIC_PearceInitExtra[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitExtra[run]) for run in range(6)])
    BIC_PearceInitAsym[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitAsym[run]) for run in range(6)])
    BIC_PearceInitTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitTransfer[run]) for run in range(6)])
    BIC_PearceAsymExtra[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceAsymExtra[run]) for run in range(6)])
    BIC_PearceExtraTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceExtraTransfer[run]) for run in range(6)])
    BIC_PearceAsymTransfer[idx] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceAsymTransfer[run]) for run in range(6)])
    BIC_PearceInitExtraAsym[idx] = np.mean([7 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsym[run]) for run in range(6)])
    BIC_PearceInitExtraTransfer[idx] = np.mean([
       7 * np.log(60) - 2 * np.log(LL_PearceInitExtraTransfer[run]) for run in range(6)
    ])
    BIC_PearceInitAsymTransfer[idx] = np.mean([
       7 * np.log(60) - 2 * np.log(LL_PearceInitAsymTransfer[run]) for run in range(6)
    ])
    BIC_PearceExtraAsymTransfer[idx] = np.mean([
       9 * np.log(60) - 2 * np.log(LL_PearceExtraAsymTransfer[run]) for run in range(6)
    ])
    BIC_PearceInitExtraAsymTransfer[idx] = np.mean([
       11 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsymTransfer[run]) for run in range(6)
    ])

    best_fitting_arr = [
            BIC_Basic[idx],
            BIC_Init[idx],
            BIC_Extra[idx],
            BIC_Asym[idx],
            BIC_Transfer[idx],
            BIC_InitExtra[idx],
            BIC_InitAsym[idx],
            BIC_InitTransfer[idx],
            BIC_ExtraAsym[idx],
            BIC_ExtraTransfer[idx],
            BIC_AsymTransfer[idx],
            BIC_InitExtraAsym[idx],
            BIC_InitExtraTransfer[idx],
            BIC_InitAsymTransfer[idx],
            BIC_ExtraAsymTransfer[idx],
            BIC_InitAsymExtraTransfer[idx],
            BIC_Pearce[idx],
            BIC_PearceInit[idx],
            BIC_PearceExtra[idx],
            BIC_PearceAsym[idx],
            BIC_PearceTransfer[idx],
            BIC_PearceInitExtra[idx],
            BIC_PearceInitAsym[idx],
            BIC_PearceInitTransfer[idx],
            BIC_PearceAsymExtra[idx],
            BIC_PearceExtraTransfer[idx],
            BIC_PearceAsymTransfer[idx],
            BIC_PearceInitExtraAsym[idx],
            BIC_PearceInitExtraTransfer[idx],
            BIC_PearceInitAsymTransfer[idx],
            BIC_PearceExtraAsymTransfer[idx],
            BIC_PearceInitExtraAsymTransfer[idx],
        ]
    
    best_fitting = np.argmin(best_fitting_arr)
    best_fitting_val = best_fitting_arr[best_fitting]


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
    newPath = os.path.join(
        pathlib.Path(__file__).resolve().parents[3],
        "/data/fittedParameters/sub-{0}".format(IDnr),
    )
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
