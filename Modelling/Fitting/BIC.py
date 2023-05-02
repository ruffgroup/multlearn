import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import csv


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
    "InitExtraAsymTransfer",
    "Pearce",
    "PearceInit",
    "PearceExtra",
    "PearceAsym",
    "PearceTransfer",
    "PearceInitExtra",
    "PearceInitAsym",
    "PearceInitTransfer",
    "PearceExtraAsym",
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
BIC_InitExtraAsymTransfer = np.zeros(58)

BIC_Pearce = np.zeros(58)
BIC_PearceInit = np.zeros(58)
BIC_PearceExtra = np.zeros(58)
BIC_PearceAsym = np.zeros(58)
BIC_PearceTransfer = np.zeros(58)
BIC_PearceInitExtra = np.zeros(58)
BIC_PearceInitAsym = np.zeros(58)
BIC_PearceInitTransfer = np.zeros(58)
BIC_PearceExtraAsym = np.zeros(58)
BIC_PearceExtraTransfer = np.zeros(58)
BIC_PearceAsymTransfer = np.zeros(58)
BIC_PearceInitExtraAsym = np.zeros(58)
BIC_PearceInitExtraTransfer = np.zeros(58)
BIC_PearceInitAsymTransfer = np.zeros(58)
BIC_PearceExtraAsymTransfer = np.zeros(58)
BIC_PearceInitExtraAsymTransfer = np.zeros(58)

winning_model = {key: 0 for key in models}

for IDnr in IDs:
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
    LL_InitExtraAsymTransfer = np.load(newPath + "/InitExtraAsymTransfer/BIC_InitExtraAsymTransfer.npy")

    LL_Pearce = np.load(newPath + "/Pearce/BIC_Pearce.npy")
    LL_PearceInit = np.load(newPath + "/PearceInit/BIC_PearceInit.npy")
    LL_PearceExtra = np.load(newPath + "/PearceExtra/BIC_PearceExtra.npy")
    LL_PearceAsym = np.load(newPath + "/PearceAsym/BIC_PearceAsym.npy")
    LL_PearceTransfer = np.load(newPath + "/PearceTransfer/BIC_PearceTransfer.npy")
    LL_PearceInitExtra = np.load(newPath + "/PearceInitExtra/BIC_PearceInitExtra.npy")
    LL_PearceInitAsym = np.load(newPath + "/PearceInitAsym/BIC_PearceInitAsym.npy")
    LL_PearceInitTransfer = np.load(newPath + "/PearceInitTransfer/BIC_PearceInitTransfer.npy")
    LL_PearceExtraAsym = np.load(newPath + "/PearceExtraAsym/BIC_PearceExtraAsym.npy")
    LL_PearceExtraTransfer = np.load(newPath + "/PearceExtraTransfer/BIC_PearceExtraTransfer.npy")
    LL_PearceAsymTransfer = np.load(newPath + "/PearceAsymTransfer/BIC_PearceAsymTransfer.npy")
    LL_PearceInitExtraAsym = np.load(newPath + "/PearceInitExtraAsym/BIC_PearceInitExtraAsym.npy")
    LL_PearceInitExtraTransfer = np.load(newPath + "/PearceInitExtraTransfer/BIC_PearceInitExtraTransfer.npy")
    LL_PearceInitAsymTransfer = np.load(newPath + "/PearceInitAsymTransfer/BIC_PearceInitAsymTransfer.npy")
    LL_PearceExtraAsymTransfer = np.load(newPath + "/PearceExtraAsymTransfer/BIC_PearceExtraAsymTransfer.npy")
    LL_PearceInitExtraAsymTransfer = np.load(
        newPath + "/PearceInitExtraAsymTransfer/BIC_PearceInitExtraAsymTransfer.npy"
    )

    BIC_Basic[IDnr] = np.mean([2 * np.log(60) - 2 * np.log(LL_Basic[run]) for run in range(6)])
    BIC_Init[IDnr] = np.mean([4 * np.log(60) - 2 * np.log(LL_Init[run]) for run in range(6)])
    BIC_Extra[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_Extra[run]) for run in range(6)])
    BIC_Asym[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_Asym[run]) for run in range(6)])
    BIC_Transfer[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_Transfer[run]) for run in range(6)])
    BIC_InitExtra[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitExtra[run]) for run in range(6)])
    BIC_InitAsym[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitAsym[run]) for run in range(6)])
    BIC_InitTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_InitTransfer[run]) for run in range(6)])
    BIC_ExtraAsym[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_ExtraAsym[run]) for run in range(6)])
    BIC_ExtraTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_ExtraTransfer[run]) for run in range(6)])
    BIC_AsymTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_AsymTransfer[run]) for run in range(6)])
    BIC_InitExtraAsym[IDnr] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitExtraAsym[run]) for run in range(6)])
    BIC_InitExtraTransfer[IDnr] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitExtraTransfer[run]) for run in range(6)])
    BIC_InitAsymTransfer[IDnr] = np.mean([7 * np.log(60) - 2 * np.log(LL_InitAsymTransfer[run]) for run in range(6)])
    BIC_ExtraAsymTransfer[IDnr] = np.mean([9 * np.log(60) - 2 * np.log(LL_ExtraAsymTransfer[run]) for run in range(6)])
    BIC_InitExtraAsymTransfer[IDnr] = np.mean([11 * np.log(60) - 2 * np.log(LL_InitExtraAsymTransfer[run]) for run in range(6)])

    BIC_Pearce[IDnr] = np.mean([2 * np.log(60) - 2 * np.log(LL_Pearce[run]) for run in range(6)])
    BIC_PearceInit[IDnr] = np.mean([4 * np.log(60) - 2 * np.log(LL_PearceInit[run]) for run in range(6)])
    BIC_PearceExtra[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceExtra[run]) for run in range(6)])
    BIC_PearceAsym[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceAsym[run]) for run in range(6)])
    BIC_PearceTransfer[IDnr] = np.mean([3 * np.log(60) - 2 * np.log(LL_PearceTransfer[run]) for run in range(6)])
    BIC_PearceInitExtra[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitExtra[run]) for run in range(6)])
    BIC_PearceInitAsym[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitAsym[run]) for run in range(6)])
    BIC_PearceInitTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceInitTransfer[run]) for run in range(6)])
    BIC_PearceExtraAsym[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceExtraAsym[run]) for run in range(6)])
    BIC_PearceExtraTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceExtraTransfer[run]) for run in range(6)])
    BIC_PearceAsymTransfer[IDnr] = np.mean([5 * np.log(60) - 2 * np.log(LL_PearceAsymTransfer[run]) for run in range(6)])
    BIC_PearceInitExtraAsym[IDnr] = np.mean([7 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsym[run]) for run in range(6)])
    BIC_PearceInitExtraTransfer[IDnr] = np.mean([
        7 * np.log(60) - 2 * np.log(LL_PearceInitExtraTransfer[run]) for run in range(6)
    ])
    BIC_PearceInitAsymTransfer[IDnr] = np.mean([
        7 * np.log(60) - 2 * np.log(LL_PearceInitAsymTransfer[run]) for run in range(6)
    ])
    BIC_PearceExtraAsymTransfer[IDnr] = np.mean([
        9 * np.log(60) - 2 * np.log(LL_PearceExtraAsymTransfer[run]) for run in range(6)
    ])
    BIC_PearceInitExtraAsymTransfer[IDnr] = np.mean([
        11 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsymTransfer[run]) for run in range(6)
    ])

    best_fitting = np.argmin(
        [
            BIC_Basic[IDnr],
            BIC_Init[IDnr],
            BIC_Extra[IDnr],
            BIC_Asym[IDnr],
            BIC_Transfer[IDnr],
            BIC_InitExtra[IDnr],
            BIC_InitAsym[IDnr],
            BIC_InitTransfer[IDnr],
            BIC_ExtraAsym[IDnr],
            BIC_ExtraTransfer[IDnr],
            BIC_AsymTransfer[IDnr],
            BIC_InitExtraAsym[IDnr],
            BIC_InitExtraTransfer[IDnr],
            BIC_InitAsymTransfer[IDnr],
            BIC_ExtraAsymTransfer[IDnr],
            BIC_InitExtraAsymTransfer[IDnr],
            BIC_Pearce[IDnr],
            BIC_PearceInit[IDnr],
            BIC_PearceExtra[IDnr],
            BIC_PearceAsym[IDnr],
            BIC_PearceTransfer[IDnr],
            BIC_PearceInitExtra[IDnr],
            BIC_PearceInitAsym[IDnr],
            BIC_PearceInitTransfer[IDnr],
            BIC_PearceExtraAsym[IDnr],
            BIC_PearceExtraTransfer[IDnr],
            BIC_PearceAsymTransfer[IDnr],
            BIC_PearceInitExtraAsym[IDnr],
            BIC_PearceInitExtraTransfer[IDnr],
            BIC_PearceInitAsymTransfer[IDnr],
            BIC_PearceExtraAsymTransfer[IDnr],
            BIC_PearceInitExtraAsymTransfer[IDnr],
        ]
    )


    rpeBest = scipy.io.loadmat(
        newPath + "/" + models[best_fitting] + "/" + models[best_fitting] + ".mat"
    )
    scipy.io.savemat(
            newPath + "/rpeBest/rpeBest.mat",
            mdict={"rpe": rpeBest},
        )

    winning_model[models[best_fitting]] += 1
    listBestFits.append([IDnr, models[best_fitting]])

with open("BestFitting.tsv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(listBestFits)

print(winning_model)
