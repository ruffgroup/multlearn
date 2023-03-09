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

subjectFiles = listdir_nohidden('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Data/mult_Sens_Sess15-Dec-2021')

subjects = []
individual_savedValues = []

# Adding path of Data
sys.path.insert(1, '/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Data/mult_Sens_Sess15-Dec-2021')

# Combining the savedVal file with the experimentInfo
for i in range(len(subjectFiles)):
    subjects.append(subjectFiles[i][-2:len(subjectFiles[i])])
    individual_savedValues.append(pd.read_csv(subjectFiles[i]+'/participant'+str(subjects[i])+'_savedValues.csv'))
    individual_savedValues[i]['participantID'] = subjects[i]
    if i == 0:
       fullData = individual_savedValues[i]  
    else:
       fullData = pd.concat([fullData,individual_savedValues[i]], axis = 0)

locals()["fullData" + subjectFiles[0][-18:-12]] = fullData

del i

locals()["fullData" + subjectFiles[0][-18:-12]].to_csv('/Users/sbedi/Desktop/multisensory-project-rl/Human_task_design/Data/mult_Sens_Sess15-Dec-2021/fullData'+subjectFiles[0][-18:-12]+'.csv')
