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



def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

subjectFiles = listdir_nohidden('/Users/sbedi/Library/CloudStorage/OneDrive-UniversitätZürichUZH/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/mult_Sens_Sess11-Nov-2021')

subjects = []
Individual_expInfo = []
individual_savedValues = []
individual_fullData = []

# Adding path of Data
sys.path.insert(1, '/Users/sbedi/Library/CloudStorage/OneDrive-UniversitätZürichUZH/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/mult_Sens_Sess10-Nov-2021')
sys.path.insert(1, '/Users/sbedi/Library/CloudStorage/OneDrive-UniversitätZürichUZH/multisensory-learning/Human_audioTactile_version-intermixed-beeps-noBepps/Data/mult_Sens_Sess11-Nov-2021')


# Combining the savedVal file with the experimentInfo
for i in range(len(subjectFiles)):
    subjects.append(subjectFiles[i][-2:len(subjectFiles[i])])
    individual_savedValues.append(pd.read_csv(subjectFiles[i]+'/participant'+str(subjects[i])+'_savedValues.csv'))
    individual_savedValues[i]['participantID'] = subjects[i]
    Individual_expInfo.append(pd.read_csv(subjectFiles[i]+'/participant'+str(subjects[i])+'_expInfo.csv'))
    Individual_expInfo[i] = np.transpose(pd.concat([Individual_expInfo[i].iloc[0]]*individual_savedValues[i].shape[0], axis = 1))
    Individual_expInfo[i] = Individual_expInfo[i].reset_index()
    individual_fullData.append(pd.concat([individual_savedValues[i], Individual_expInfo[i]], axis = 1))
    if i == 0:
       fullData = individual_fullData[i]  
    else:
        fullData = pd.concat([fullData,individual_fullData[i]], axis = 0)
    

locals()["fullData" + subjectFiles[0][-18:-12]] = fullData

del i

locals()["fullData" + subjectFiles[0][-18:-12]].to_csv('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Data/mult_Sens_Sess11-Nov-2021/fullData'+subjectFiles[0][-18:-12]+'.csv')