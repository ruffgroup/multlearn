from random import *
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy.ma as ma
import pickle
# import seaborn as sns
import sys
import copy
import pathlib
import re
import platform

if platform.system() == 'Windows':
    dir_file = '/Volumes/g_econ_department$/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri'
    wanted_dir = os.path.join(dir_file, 'data/sourcedata/behavior')

else:
    dir_file = '/Volumes/g_econ_department$/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri'
    wanted_dir = os.path.join(dir_file, 'data/sourcedata/behavior')

# Get all expInfo and savedVals files
files = [os.path.join(root, name) for root, dirs, files in os.walk(wanted_dir) for name in files if name.endswith('savedValues.csv')]

for file in files:
    print(file)
    parID = re.findall(r'\d+', file)[2]
    dataset = pd.read_csv(file)
    modifiedDataset = copy.deepcopy(dataset)

    if int(parID) % 2 == 0:
        modifiedDataset.loc[modifiedDataset.chosenKey == '4$', 'action'] = 0.0
        modifiedDataset.loc[modifiedDataset.chosenKey == '2@', 'action'] = 1.0
        resp = {'4$':'0', '2@':'1'}
        modifiedDataset['correctResponse'] = modifiedDataset['correctResponse'].astype(str).map(resp).astype(float)
    else:
        modifiedDataset.loc[modifiedDataset.chosenKey == '2@', 'action'] = 0.0
        modifiedDataset.loc[modifiedDataset.chosenKey == '4$', 'action'] = 1.0
        resp = {'4$':'1', '2@':'0'}
        modifiedDataset['correctResponse'] = modifiedDataset['correctResponse'].astype(str).map(resp).astype(float)

    modifiedDataset["stimulusPair"] = np.where(~np.isnan(modifiedDataset['audio']),modifiedDataset[["visual", "audio"]].apply(tuple, axis=1), modifiedDataset[["visual", "tactile"]].apply(tuple, axis=1))
    modifiedDataset["stimulusPair"] = modifiedDataset["stimulusPair"].apply(lambda row: tuple(map(int, row)))

    # if int(parID) < 25:
    #     modifiedDataset = modifiedDataset[(modifiedDataset.runNumber == 3) | (modifiedDataset.runNumber == 4)]
    #     resp = {3: 1, 4: 2}
    #     modifiedDataset['runNumber'] = modifiedDataset['runNumber'].map(resp)

    # if max(modifiedDataset.runNumber) == 5:
    #     modifiedDataset = modifiedDataset[(modifiedDataset.runNumber == 2) | (modifiedDataset.runNumber == 3) | (modifiedDataset.runNumber == 4) | (modifiedDataset.runNumber == 5)]
    #     resp = {2: 1, 3: 2, 4: 3, 5: 4}
    #     modifiedDataset['runNumber'] = modifiedDataset['runNumber'].map(resp)

    # elif max(modifiedDataset.runNumber) == 6:
    #     modifiedDataset = modifiedDataset[(modifiedDataset.runNumber == 3) | (modifiedDataset.runNumber == 4) | (modifiedDataset.runNumber == 5) | (modifiedDataset.runNumber == 6)]
    #     resp = {3: 1, 4: 2, 5: 3, 6: 4}
    #     modifiedDataset['runNumber'] = modifiedDataset['runNumber'].map(resp)

    #savePath = os.path.join(os.path.dirname(os.path.dirname(scriptDirectory))+'/Data/mult_Sens_Sess15-Dec-2021')
    os.makedirs(wanted_dir+'/modified_files', exist_ok=True)

    if platform.system() == 'Windows':
        modifiedDataset.to_csv(wanted_dir+'/modified_files/modified_{}'.format(file.split('\\')[-1]), index = False)
    else:
        modifiedDataset.to_csv(wanted_dir + '/modified_files/modified_{}'.format(file.split('/')[-1]), index=False)


