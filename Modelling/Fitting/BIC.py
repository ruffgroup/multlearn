import numpy as np
import pandas as pd

LL_simple = np.load("BIC_new_simple.npy")
LL_init = np.load("BIC_new_init.npy")
LL_up = np.load("BIC_new_up.npy")
LL_upInit = np.load("BIC_new_upInit.npy")
LL_up2 = np.load("BIC_new_up2.npy")
LL_up4 = np.load("BIC_new_up4.npy")
LL_upInit2 = np.load("BIC_new_upInit2.npy")
LL_upInit4 = np.load("BIC_new_upInit4.npy")

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

for id in range(LL_simple.shape[0]):
    for run in range(LL_simple.shape[1]):
        BIC_simple[id, run] = 2*np.log(60) - 2*np.log(LL_simple[id, run])
        BIC_init[id, run] = 4*np.log(60) - 2*np.log(LL_init[id, run])
        BIC_up[id, run] = 3*np.log(60) - 2*np.log(LL_up[id, run])
        BIC_upInit[id, run] = 5*np.log(60) - 2*np.log(LL_upInit[id, run])
        BIC_up2[id, run] = 4*np.log(60) - 2*np.log(LL_up2[id, run])
        BIC_up4[id, run] = 6*np.log(60) - 2*np.log(LL_up4[id, run])
        BIC_upInit2[id, run] = 6*np.log(60) - 2*np.log(LL_upInit2[id, run])
        BIC_upInit4[id, run] = 8*np.log(60) - 2*np.log(LL_upInit4[id, run])

        AIC_simple[id, run] = 2 * 2 - 2 * np.log(LL_simple[id, run])
        AIC_init[id, run] = 2 * 4 - 2 * np.log(LL_init[id, run])
        AIC_up[id, run] = 2 * 3 - 2 * np.log(LL_up[id, run])
        AIC_upInit[id, run] = 2 * 5 - 2 * np.log(LL_upInit[id, run])
        AIC_up2[id, run] = 2 * 4 - 2 * np.log(LL_up2[id, run])
        AIC_up4[id, run] = 2 * 6 - 2 * np.log(LL_up4[id, run])
        AIC_upInit2[id, run] = 2 * 6 - 2 * np.log(LL_upInit2[id, run])
        AIC_upInit4[id, run] = 2 * 8 - 2 * np.log(LL_upInit4[id, run])


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