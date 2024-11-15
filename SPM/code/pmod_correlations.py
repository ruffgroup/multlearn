import nipype_helpers
from nilearn import plotting
from os.path import join as opj
import os.path as op
import json
import os
from glob import glob
from scipy import io, stats
from itertools import chain
import pandas as pd
import numpy as np
import pytest as pt
import nibabel as nb
import nipype
import argparse
from bids.layout import BIDSLayout
from scipy import stats

data_folder = "/shares/zne.uzh/multlearn"
bestfitting_path = "/shares/zne.uzh/multlearn/bestFittingVals"
subject_list = [f'{sub:02d}' for sub in range(1,65) if sub not in [8, 13, 16, 31, 32, 44]]
for sub in subject_list:
    _, _, V_df, rpe, surprise = nipype_helpers.get_subject_info_model2(sub, data_folder, bestfitting_path)

    #reward = V_df["reward"]
    rpe = rpe["rpe"]
    V = V_df["V"]
    spe = surprise["spe"]

    #corr = stats.pearsonr(reward,rpe)
    #corr = stats.pearsonr(rpe, V)
    corr = stats.pearsonr(rpe, spe)
    print(corr)