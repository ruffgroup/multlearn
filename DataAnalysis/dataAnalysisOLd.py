from random import *
import os
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
import seaborn as sns
from collections import Counter
import matplotlib.patches as mpatches

sess = "all"

if sess == "final":
    path = os.path.join('/Users/sbedi/git/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files', 'fullDataLatestVersion.csv')
else:
    path = os.path.join('/Users/sbedi/git/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files', 'fullData.csv')



fullData = pd.read_csv(path)


# ## Accuracy plots
# finalTotalAccuracies = fullData.totalAccuracy[fullData.trialNumber == fullData.nrTrials]
# plt.title('Bar Plot of Sessionwise Total Accuracy')
# plt.xlabel('Sessionwise Accuracy')
# plt.ylabel('Frequency')
# plt.xlim(0.4, 1.0)
# sns.displot(finalTotalAccuracies, kde=0)
# plt.savefig(
#     '/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files/accuracy.png',
#     dpi=1200)
# plt.show()

# # Reaction time for when correct response is right arrow (presented 50 percent of the time)
# RT_RightCorrect = fullData.responseTime[fullData.correctResponse == "RightArrow"]
# RT_LeftCorrect = fullData.responseTime[fullData.correctResponse == "LeftArrow"]
# fasterRightMean = np.mean(RT_RightCorrect) - np.mean(RT_LeftCorrect)
# plt.title('Bar Plot of RTs for correct response as Right and Left')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# sns.displot(RT_RightCorrect, kde=0, color="b")
# sns.displot(RT_LeftCorrect, kde=0, color="y")
# plt.axvline(np.mean(RT_RightCorrect), color="b")
# plt.axvline(np.mean(RT_LeftCorrect), color="y")
# plt.text(2, 160, 'difference between means = \n' + str(round(fasterRightMean, 3)))
# plt.savefig(
#     '/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files/RTright_vs_left.png',
#     dpi=1200)
# plt.show()

## RTs acc to how much each modality occured in data

RT_common_list = []
RT_rare_list = []
RT_mid_list = []
for i in range(len(np.unique(fullData.participantID))):
    print(i)
    RTv0a0 = np.mean(fullData.responseTime[(fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv0a1 = np.mean(fullData.responseTime[(fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv0a2 = np.mean(fullData.responseTime[(fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv1a0 = np.mean(fullData.responseTime[(fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv1a1 = np.mean(fullData.responseTime[(fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv1a2 = np.mean(fullData.responseTime[(fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv2a0 = np.mean(fullData.responseTime[(fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv2a1 = np.mean(fullData.responseTime[(fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTv2a2 = np.mean(fullData.responseTime[(fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]]])
    RTpoints = np.array([RTv0a0, RTv0a1, RTv0a2, RTv1a0, RTv1a1, RTv1a2, RTv2a0, RTv2a1, RTv2a2])

    v0a0 = sum((fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v0a1 = sum((fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v0a2 = sum((fullData.visual == 0) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v1a0 = sum((fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v1a1 = sum((fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v1a2 = sum((fullData.visual == 1) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v2a0 = sum((fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_1) | (
            fullData.audio==fullData.audioPair_1))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v2a1 = sum((fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_2) | (
            fullData.audio==fullData.audioPair_2))[fullData.participantID == np.unique(fullData.participantID)[i]])
    v2a2 = sum((fullData.visual == 2) & ((fullData.tactile==fullData.tactilePair_3) | (
            fullData.audio==fullData.audioPair_3))[fullData.participantID == np.unique(fullData.participantID)[i]])
    points = np.array([v0a0, v0a1, v0a2, v1a0, v1a1, v1a2, v2a0, v2a1, v2a2])
    print(points)

    RT_common = np.mean(RTpoints[points==max(points)])
    RT_common_list.append(RT_common)
    RT_rare = np.mean(RTpoints[points==min(points)])
    RT_rare_list.append(RT_rare)
    RT_mid = np.mean(RTpoints[(points!=min(points))&(points!=max(points))])
    RT_mid_list.append(RT_mid)


    plt.scatter(['Common', 'Mid', 'Rare'], [RT_common, RT_mid, RT_rare], alpha = 0.5)
    plt.plot(['Common', 'Mid', 'Rare'], [RT_common, RT_mid, RT_rare], alpha = 0.5, linestyle='dashed')

plt.scatter(['Common', 'Mid', 'Rare'], [np.mean(RT_common_list), np.mean(RT_mid_list), np.mean(RT_rare_list)], color = 'black')
plt.plot(['Common', 'Mid', 'Rare'], [np.mean(RT_common_list), np.mean(RT_mid_list), np.mean(RT_rare_list)], 'k')
plt.ylabel("RT")
plt.title("Within participant RT effect all sessions")
plt.show()




df = pd.DataFrame({'type':list(Counter(dict(zip(['Common', 'Mid', 'Rare'],[len(RT_common_list),len(RT_mid_list), len(RT_rare_list)]))).elements()), 'RT': RT_common_list+RT_mid_list+RT_rare_list})


## Violin plots
# blue_patch = mpatches.Patch(color='blue')
# yellow_patch = mpatches.Patch(color='yellow')
# purple_patch = mpatches.Patch(color='purple')
# label = ["commonMeanRT: "+str(round(np.mean(RT_common_list), 3)), "midMeanRT: "+str(round(np.mean(RT_mid_list),3)), "rareMeanRT: "+str(round(np.mean(RT_rare_list),3))]
# fake_handles = [blue_patch, yellow_patch, purple_patch]#repeat(red_patch, len(label))
# sns.violinplot(data=df, x="type", y="RT", inner="box", palette="Set3", cut=2, linewidth=3)
# if sess == "final":
#     plt.title("Reaction time effect for final data (8 participants)")
# else:
#     plt.title("Reaction time effect for all data")
# ax = plt.subplot(111)
# ax.legend(fake_handles, label)
# plt.show()




    # plt.title('RT vs freq subject ' + str(i))
    # plt.xlabel('number of times the audiovisual stimulus occured')
    # plt.ylabel('mean reaction time for given audiovisual modality')
    # plt.scatter(points, RTpoints)
    # m, b = np.polyfit(points, RTpoints, 1)
    # plt.plot(points, m * points + b, color="black")
    # plt.savefig('/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files/RTscatter' + str(i) + '.png', dpi=1200)
    # plt.show()


