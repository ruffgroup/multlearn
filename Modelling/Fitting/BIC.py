import numpy as np
import pandas as pd
import os
import pathlib


dataPath = os.path.join(
            pathlib.Path(__file__).resolve().parents[3],
            "/data/fittedParameters"
        )
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


BIC_simple = np.zeros((6,4))
BIC_init = np.zeros((6,4))
BIC_up = np.zeros((6,4))
BIC_upInit = np.zeros((6,4))
BIC_up2 = np.zeros((6,4))
BIC_up4 = np.zeros((6,4))
BIC_upInit2 = np.zeros((6,4))
BIC_upInit4 = np.zeros((6,4))

AIC_simple = np.zeros((6,4))
AIC_init = np.zeros((6,4))
AIC_up = np.zeros((6,4))
AIC_upInit = np.zeros((6,4))
AIC_up2 = np.zeros((6,4))
AIC_up4 = np.zeros((6,4))
AIC_upInit2 = np.zeros((6,4))
AIC_upInit4 = np.zeros((6,4))

keyList = list(range(8))
winning_model = {key: 0 for key in keyList}
winning_model_AIC = {key: 0 for key in keyList}

LL_simple = np.load("BIC_new_simple.npy")
LL_init = np.load("BIC_new_init.npy")
LL_up = np.load("BIC_new_up.npy")
LL_upInit = np.load("BIC_new_upInit.npy")
LL_up2 = np.load("BIC_new_up2.npy")
LL_up4 = np.load("BIC_new_up4.npy")
LL_upInit2 = np.load("BIC_new_upInit2.npy")
LL_upInit4 = np.load("BIC_new_upInit4.npy")



for IDnr in IDs:
    for run in range(6):
        BIC_simple[IDnr, run] = 2*np.log(60) - 2*np.log(LL_simple[run])
        BIC_init[IDnr, run] = 4*np.log(60) - 2*np.log(LL_init[run])
        BIC_up[IDnr, run] = 3*np.log(60) - 2*np.log(LL_up[run])
        BIC_upInit[IDnr, run] = 5*np.log(60) - 2*np.log(LL_upInit[run])
        BIC_up2[IDnr, run] = 4*np.log(60) - 2*np.log(LL_up2[run])
        BIC_up4[IDnr, run] = 6*np.log(60) - 2*np.log(LL_up4[run])
        BIC_upInit2[IDnr, run] = 6*np.log(60) - 2*np.log(LL_upInit2[run])
        BIC_upInit4[IDnr, run] = 8*np.log(60) - 2*np.log(LL_upInit4[run])

        AIC_simple[IDnr, run] = 2 * 2 - 2 * np.log(LL_simple[id, run])
        AIC_init[IDnr, run] = 2 * 4 - 2 * np.log(LL_init[id, run])
        AIC_up[IDnr, run] = 2 * 3 - 2 * np.log(LL_up[id, run])
        AIC_upInit[IDnr, run] = 2 * 5 - 2 * np.log(LL_upInit[id, run])
        AIC_up2[IDnr, run] = 2 * 4 - 2 * np.log(LL_up2[id, run])
        AIC_up4[IDnr, run] = 2 * 6 - 2 * np.log(LL_up4[id, run])
        AIC_upInit2[IDnr, run] = 2 * 6 - 2 * np.log(LL_upInit2[id, run])
        AIC_upInit4[IDnr, run] = 2 * 8 - 2 * np.log(LL_upInit4[id, run])


        best_fitting = np.argmin([BIC_simple[id, run], BIC_init[id, run], BIC_up[id, run], BIC_upInit[id, run],
                                  BIC_up2[id, run], BIC_up4[id, run], BIC_upInit2[id, run], BIC_upInit4[id, run]])

        best_fitting_AIC = np.argmin([AIC_simple[id, run], AIC_init[id, run], AIC_up[id, run], AIC_upInit[id, run],
                                  AIC_up2[id, run], AIC_up4[id, run], AIC_upInit2[id, run], AIC_upInit4[id, run]])

        winning_model[best_fitting] += 1
        winning_model_AIC[best_fitting_AIC] += 1

        print([BIC_simple[id, run], BIC_init[id, run], BIC_up[id, run], BIC_upInit[id, run],
                                  BIC_up2[id, run], BIC_up4[id, run], BIC_upInit2[id, run], BIC_upInit4[id, run]])



print(winning_model)
print(winning_model_AIC)