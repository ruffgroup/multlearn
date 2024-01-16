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
    #"PearceTransfer",
    "PearceInitExtra",
    "PearceInitAsym",
    #"PearceInitTransfer",
    "PearceAsymExtra",
    #"PearceExtraTransfer",
    #"PearceAsymTransfer",
    "PearceInitExtraAsym",
    #"PearceInitExtraTransfer",
    #"PearceInitAsymTransfer",
    #"PearceExtraAsymTransfer",
    #"PearceInitExtraAsymTransfer"
]

listBestFits = list()

BIC_Basic = np.zeros((58,6))
BIC_Init = np.zeros((58,6))
BIC_Extra = np.zeros((58,6))
BIC_Asym = np.zeros((58,6))
BIC_Transfer = np.zeros((58,6))
BIC_InitExtra = np.zeros((58,6))
BIC_InitAsym = np.zeros((58,6))
BIC_InitTransfer = np.zeros((58,6))
BIC_ExtraAsym = np.zeros((58,6))
BIC_ExtraTransfer = np.zeros((58,6))
BIC_AsymTransfer = np.zeros((58,6))
BIC_InitExtraAsym = np.zeros((58,6))
BIC_InitExtraTransfer = np.zeros((58,6))
BIC_InitAsymTransfer = np.zeros((58,6))
BIC_ExtraAsymTransfer = np.zeros((58,6))
BIC_InitAsymExtraTransfer = np.zeros((58,6))

BIC_Pearce = np.zeros((58,6))
BIC_PearceInit = np.zeros((58,6))
BIC_PearceExtra = np.zeros((58,6))
BIC_PearceAsym = np.zeros((58,6))
#BIC_PearceTransfer = np.zeros((58,6))
BIC_PearceInitExtra = np.zeros((58,6))
BIC_PearceInitAsym = np.zeros((58,6))
#BIC_PearceInitTransfer = np.zeros((58,6))
BIC_PearceAsymExtra = np.zeros((58,6))
#BIC_PearceExtraTransfer = np.zeros((58,6))
#BIC_PearceAsymTransfer = np.zeros((58,6))
BIC_PearceInitExtraAsym = np.zeros((58,6))
#BIC_PearceInitExtraTransfer = np.zeros((58,6))
#BIC_PearceInitAsymTransfer = np.zeros((58,6))
#BIC_PearceExtraAsymTransfer = np.zeros((58,6))
#BIC_PearceInitExtraAsymTransfer = np.zeros((58,6))

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
    #LL_PearceTransfer = np.load(newPath + "/PearceTransfer/BIC_PearceTransfer.npy")
    LL_PearceInitExtra = np.load(newPath + "/PearceInitExtra/BIC_PearceInitExtra.npy")
    LL_PearceInitAsym = np.load(newPath + "/PearceInitAsym/BIC_PearceInitAsym.npy")
    #LL_PearceInitTransfer = np.load(newPath + "/PearceInitTransfer/BIC_PearceInitTransfer.npy")
    LL_PearceAsymExtra = np.load(newPath + "/PearceAsymExtra/BIC_PearceAsymExtra.npy")
    #LL_PearceExtraTransfer = np.load(newPath + "/PearceExtraTransfer/BIC_PearceExtraTransfer.npy")
    #LL_PearceAsymTransfer = np.load(newPath + "/PearceAsymTransfer/BIC_PearceAsymTransfer.npy")
    LL_PearceInitExtraAsym = np.load(newPath + "/PearceInitExtraAsym/BIC_PearceInitExtraAsym.npy")
    #LL_PearceInitExtraTransfer = np.load(newPath + "/PearceInitExtraTransfer/BIC_PearceInitExtraTransfer.npy")
    #LL_PearceInitAsymTransfer = np.load(newPath + "/PearceInitAsymTransfer/BIC_PearceInitAsymTransfer.npy")
    #LL_PearceExtraAsymTransfer = np.load(newPath + "/PearceExtraAsymTransfer/BIC_PearceExtraAsymTransfer.npy")
    #LL_PearceInitExtraAsymTransfer = np.load(newPath + "/PearceInitExtraAsymTransfer/BIC_PearceInitExtraAsymTransfer.npy")
    best_fitting_run = list()
    best_fitting_run_val = list()
    for run in range(6):
        BIC_Basic[idx,run] = 2 * np.log(60) - 2 * np.log(LL_Basic[run])
        BIC_Init[idx,run] = 4 * np.log(60) - 2 * np.log(LL_Init[run]) 
        BIC_Extra[idx,run] = 3 * np.log(60) - 2 * np.log(LL_Extra[run]) 
        BIC_Asym[idx,run] = 3 * np.log(60) - 2 * np.log(LL_Asym[run]) 
        BIC_Transfer[idx,run] = 3 * np.log(60) - 2 * np.log(LL_Transfer[run]) 
        BIC_InitExtra[idx,run] = 5 * np.log(60) - 2 * np.log(LL_InitExtra[run]) 
        BIC_InitAsym[idx,run] = 5 * np.log(60) - 2 * np.log(LL_InitAsym[run]) 
        BIC_InitTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_InitTransfer[run]) 
        BIC_ExtraAsym[idx,run] = 5 * np.log(60) - 2 * np.log(LL_ExtraAsym[run]) 
        BIC_ExtraTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_ExtraTransfer[run]) 
        BIC_AsymTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_AsymTransfer[run]) 
        BIC_InitExtraAsym[idx,run] = 7 * np.log(60) - 2 * np.log(LL_InitExtraAsym[run]) 
        BIC_InitExtraTransfer[idx,run] = 7 * np.log(60) - 2 * np.log(LL_InitExtraTransfer[run]) 
        BIC_InitAsymTransfer[idx,run] = 7 * np.log(60) - 2 * np.log(LL_InitAsymTransfer[run]) 
        BIC_ExtraAsymTransfer[idx,run] = 9 * np.log(60) - 2 * np.log(LL_ExtraAsymTransfer[run]) 
        BIC_InitAsymExtraTransfer[idx,run] = 11 * np.log(60) - 2 * np.log(LL_InitAsymExtraTransfer[run]) 

        BIC_Pearce[idx,run] = 2 * np.log(60) - 2 * np.log(LL_Pearce[run]) 
        BIC_PearceInit[idx,run] = 4 * np.log(60) - 2 * np.log(LL_PearceInit[run]) 
        BIC_PearceExtra[idx,run] = 3 * np.log(60) - 2 * np.log(LL_PearceExtra[run]) 
        BIC_PearceAsym[idx,run] = 3 * np.log(60) - 2 * np.log(LL_PearceAsym[run]) 
        #BIC_PearceTransfer[idx,run] = 3 * np.log(60) - 2 * np.log(LL_PearceTransfer[run]) 
        BIC_PearceInitExtra[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceInitExtra[run]) 
        BIC_PearceInitAsym[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceInitAsym[run]) 
        #BIC_PearceInitTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceInitTransfer[run]) 
        BIC_PearceAsymExtra[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceAsymExtra[run]) 
        #BIC_PearceExtraTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceExtraTransfer[run]) 
        #BIC_PearceAsymTransfer[idx,run] = 5 * np.log(60) - 2 * np.log(LL_PearceAsymTransfer[run]) 
        BIC_PearceInitExtraAsym[idx,run] = 7 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsym[run]) 
        #BIC_PearceInitExtraTransfer[idx,run] = 
        #    7 * np.log(60) - 2 * np.log(LL_PearceInitExtraTransfer[run]) for run in range(6)
        #])
        #BIC_PearceInitAsymTransfer[idx,run] = 
        #    7 * np.log(60) - 2 * np.log(LL_PearceInitAsymTransfer[run]) for run in range(6)
        #])
        #BIC_PearceExtraAsymTransfer[idx,run] = 
        #    9 * np.log(60) - 2 * np.log(LL_PearceExtraAsymTransfer[run]) for run in range(6)
        #])
        #BIC_PearceInitExtraAsymTransfer[idx,run] = 
        #    11 * np.log(60) - 2 * np.log(LL_PearceInitExtraAsymTransfer[run]) for run in range(6)
        #])

        best_fitting_arr = [
                BIC_Basic[idx,run],
                BIC_Init[idx,run],
                BIC_Extra[idx,run],
                BIC_Asym[idx,run],
                BIC_Transfer[idx,run],
                BIC_InitExtra[idx,run],
                BIC_InitAsym[idx,run],
                BIC_InitTransfer[idx,run],
                BIC_ExtraAsym[idx,run],
                BIC_ExtraTransfer[idx,run],
                BIC_AsymTransfer[idx,run],
                BIC_InitExtraAsym[idx,run],
                BIC_InitExtraTransfer[idx,run],
                BIC_InitAsymTransfer[idx,run],
                BIC_ExtraAsymTransfer[idx,run],
                BIC_InitAsymExtraTransfer[idx,run],
                BIC_Pearce[idx,run],
                BIC_PearceInit[idx,run],
                BIC_PearceExtra[idx,run],
                BIC_PearceAsym[idx,run],
            #    BIC_PearceTransfer[idx,run],
                BIC_PearceInitExtra[idx,run],
                BIC_PearceInitAsym[idx,run],
            #    BIC_PearceInitTransfer[idx,run],
                BIC_PearceAsymExtra[idx,run],
            #    BIC_PearceExtraTransfer[idx,run],
            #    BIC_PearceAsymTransfer[idx,run],
                BIC_PearceInitExtraAsym[idx,run],
            #    BIC_PearceInitExtraTransfer[idx,run],
            #    BIC_PearceInitAsymTransfer[idx,run],
            #    BIC_PearceExtraAsymTransfer[idx,run],
            #    BIC_PearceInitExtraAsymTransfer[idx,run],
            ]
        
        best_fitting_run.append(np.argmin(best_fitting_arr))
        best_fitting_run_val.append(best_fitting_arr[np.argmin(best_fitting_arr)])
    best_fitting = max(set(best_fitting_run), key = best_fitting_run.count)
    #best_fitting_val = np.mean([best_fitting_arr[best_fitting, run] for run in range(6)])


    rpeBest = scipy.io.loadmat(
        newPath + "/" + models[best_fitting] + "/rpe" + models[best_fitting] + ".mat"
    )

    scipy.io.savemat(
            newPath + "/rpeBestCommon.mat",
            mdict={"rpe": rpeBest},
        )

    winning_model[models[best_fitting]][0] += 1
    #winning_model[models[best_fitting]][1] += best_fitting_val
    listBestFits.append([IDnr, models[best_fitting]])

with open("BestFittingCommon.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(listBestFits)

print(dict(sorted(winning_model.items(), key=lambda x: -x[1][0])))
