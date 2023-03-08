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
import sys
import pathlib
import re
import platform

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

sys.path.append(sys.path[0] + '/..')

sess = "all"



subjects = []
Individual_expInfo = []
individual_savedValues = []
individual_fullData = []


if sess == "final":

    subjectFiles1 = listdir_nohidden('/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/mult_Sens_Sess10-Aug-2022')
    subjectFiles2 = listdir_nohidden('/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/mult_Sens_Sess11-Aug-2022')

    subjectFiles = subjectFiles1+subjectFiles2

    # Combining the savedVal file with the experimentInfo
    for i in range(len(subjectFiles)):
        subjects.append(subjectFiles[i][-2:len(subjectFiles[i])])
        df = pd.read_csv(
            subjectFiles[i] + '/participant' + str(format(np.double(subjects[i]), '.6f')) + '_savedValues.csv')
        df['participantID'] = subjectFiles[i][-2:len(subjectFiles[i])]
        individual_savedValues.append(df)
        individual_fullData.append(individual_savedValues[i])
        # Individual_expInfo.append(pd.read_csv(subjectFiles[i]+'/participant'+str(format(np.double(subjects[i]),'.6f'))+'_expInfo.csv'))
        #  Individual_expInfo[i] = np.transpose(pd.concat([Individual_expInfo[i].iloc[0]] * individual_savedValues[i].shape[0], axis=1))
        #  Individual_expInfo[i] = Individual_expInfo[i].reset_index()
        #  individual_fullData.append(pd.concat([individual_savedValues[i], Individual_expInfo[i]], axis=1))
        if i == 0:
            fullData = individual_fullData[i]
        else:
            fullData = pd.concat([fullData, individual_fullData[i]], axis=0)

    locals()["fullDataLatestVersion"] = fullData

    del i

    locals()["fullData"].to_csv(
        '/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files/fullDataLatestVersion.csv')




else:
    if platform.system() == 'Windows':
        dir_file = pathlib.Path().absolute().parent.parent.resolve()
        wanted_dir = os.path.join(dir_file, 'Data/Pilot/beepStimuli')
    else:
        dir_file = os.path.dirname(os.path.dirname(os.getcwd()))
        wanted_dir = os.path.join(dir_file, 'Data/Pilot/beepStimuli')
        # Get all expInfo and savedVals files
    subjectFiles = [os.path.join(root, name) for root, dirs, files in os.walk(wanted_dir) for name in files if
             name.endswith('savedValues.csv') if not name.startswith('modified')]

    # Combining the savedVal file with the experimentInfo
    for i in range(len(subjectFiles)):
        subjects.append(subjectFiles[i][-25:-23])
        df = pd.read_csv(subjectFiles[i])
        df['participantID'] = subjectFiles[i][-25:-23]
        individual_savedValues.append(df)
        individual_fullData.append(individual_savedValues[i])
       # Individual_expInfo.append(pd.read_csv(subjectFiles[i]+'/participant'+str(format(np.double(subjects[i]),'.6f'))+'_expInfo.csv'))
       #  Individual_expInfo[i] = np.transpose(pd.concat([Individual_expInfo[i].iloc[0]] * individual_savedValues[i].shape[0], axis=1))
       #  Individual_expInfo[i] = Individual_expInfo[i].reset_index()
       #  individual_fullData.append(pd.concat([individual_savedValues[i], Individual_expInfo[i]], axis=1))
        if i == 0:
            fullData = individual_fullData[i]
        else:
            fullData = pd.concat([fullData, individual_fullData[i]], axis=0)

    locals()["fullData"] = fullData

    del i

    locals()["fullData"].to_csv(
        '/Users/sbedi/Desktop/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/Pilot/beepStimuli/modified_files/fullData.csv')



