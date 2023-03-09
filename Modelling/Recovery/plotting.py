from random import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns

from scipy.io import loadmat
# annots = loadmat('/Users/sbedi/Desktop/multisensory-project-rl/Human task design/Modelling/Recovery/Recovery Data and plots/200main trials, 50 additional/200,50.mat')

# corr = scipy.stats.pearsonr(alphaRecovery['simulated alpha'], alphaRecovery['recovered alpha'])
# ax = sns.lmplot(x="simulated alpha", y="recovered alpha", data=alphaRecovery);
# ax = plt.gca()
# ax.set_title("correlation {:.2f}, p-value{:.2f}".format(corr[0], corr[1]))


pearsonCorr = np.empty((100,1))
pearsonCorr[:] = np.nan

for i in range(0, 100):
    pearsonCorr[i] = scipy.stats.pearsonr(taskSimulationList[i].statSurprise[49:, :, :][~np.isnan(taskSimulationList[i].statSurprise[49:, :, :])], taskSimulationList[i].rewardPE[:, :, :][~np.isnan(taskSimulationList[i].rewardPE[:, :, :])])