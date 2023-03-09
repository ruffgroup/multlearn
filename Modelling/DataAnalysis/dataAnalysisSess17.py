from random import *
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns

path = os.path.join('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Data/mult_Sens_Sess17-Nov-2021', 'fullDataSess17.csv')
fullDataSess17 = pd.read_csv(path)


## Accuracy plots
finalTotalAccuracies = fullDataSess17.totalAccuracy[fullDataSess17.trialNumber == np.unique(fullDataSess17.trialNumber)[-1]]
plt.title('Bar Plot of Sessionwise Total Accuracy')
plt.xlabel('Sessionwise Accuracy')
plt.ylabel('Frequency')
plt.xlim(0.0,1.0)
sns.distplot(finalTotalAccuracies, kde = 0)
plt.savefig('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Modelling/Fitting/analysisPlots/Sess17/accuracy.png', dpi = 1200)
plt.show()

# Reaction time for when correct response is right arrow (presented 50 percent of the time)
RT_RightCorrect = fullDataSess17.responseTime[fullDataSess17.correctResponse == "RightArrow"]
RT_LeftCorrect = fullDataSess17.responseTime[fullDataSess17.correctResponse == "LeftArrow"]
fasterRightMean = np.mean(RT_RightCorrect)- np.mean(RT_LeftCorrect)
plt.title('Bar Plot of RTs for correct response as Right and Left')
plt.xlabel('Time')
plt.ylabel('Frequency')
sns.distplot(RT_RightCorrect, kde = 0,color = "b")
sns.distplot(RT_LeftCorrect,  kde = 0, color = "y")
plt.axvline(np.mean(RT_RightCorrect), color = "b", label = "RT_rightCorrect")
plt.axvline(np.mean(RT_LeftCorrect), color = "y", label = "RT_leftCorrect")
plt.text(2, 160, 'difference between means = \n' + str(round(fasterRightMean,3)))
plt.legend(loc="upper right")
plt.savefig('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Modelling/Fitting/analysisPlots/Sess17/RTright_vs_left.png', dpi = 1200)
plt.show()












# ## RTs acc to how much each modality occured in data
# for i in range(len(np.unique(fullDataSess17.participantID))):
#     RTv0a0 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 0) & (fullDataSess17.audio == 0) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv0a1 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 0) & (fullDataSess17.audio == 1) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv0a2 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 0) & (fullDataSess17.audio == 2) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv1a0 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 1) & (fullDataSess17.audio == 0) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv1a1 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 1) & (fullDataSess17.audio == 1) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv1a2 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 1) & (fullDataSess17.audio == 2) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv2a0 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 2) & (fullDataSess17.audio == 0) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv2a1 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 2) & (fullDataSess17.audio == 1) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTv2a2 = np.mean(fullDataSess17.responseTime[(fullDataSess17.visual == 2) & (fullDataSess17.audio == 2) & (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i])])
#     RTpoints = np.array([RTv0a0, RTv0a1, RTv0a2, RTv1a0, RTv1a1, RTv1a2, RTv2a0, RTv2a1, RTv2a2])

#     v0a0 = sum((fullDataSess17.visual == 0) & (fullDataSess17.audio == 0)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v0a1 = sum((fullDataSess17.visual == 0) & (fullDataSess17.audio == 1)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v0a2 = sum((fullDataSess17.visual == 0) & (fullDataSess17.audio == 2)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v1a0 = sum((fullDataSess17.visual == 1) & (fullDataSess17.audio == 0)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v1a1 = sum((fullDataSess17.visual == 1) & (fullDataSess17.audio == 1)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v1a2 = sum((fullDataSess17.visual == 1) & (fullDataSess17.audio == 2)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v2a0 = sum((fullDataSess17.visual == 2) & (fullDataSess17.audio == 0)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v2a1 = sum((fullDataSess17.visual == 2) & (fullDataSess17.audio == 1)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     v2a2 = sum((fullDataSess17.visual == 2) & (fullDataSess17.audio == 2)& (fullDataSess17.participantID == np.unique(fullDataSess17.participantID)[i]))
#     points = np.array([v0a0, v0a1, v0a2, v1a0, v1a1, v1a2, v2a0, v2a1, v2a2])
    
#     plt.title('RT vs freq subject ' + str(i))
#     plt.xlabel('number of times the audiovisual stimulus occured')
#     plt.ylabel('mean reaction time for given audiovisual modality')
#     plt.scatter(points, RTpoints)
#     m, b = np.polyfit(points, RTpoints, 1)
#     plt.plot(points, m*points + b, color = "black")
#     plt.savefig('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Modelling/Fitting/analysisPlots/Sess17/RTscatter'+str(i)+'.png', dpi = 1200)
#     plt.show()



