# Importing the files with reaction time and with the fitted parameters
import platform
import glob
import os
import os.path as op
import numpy as np
import pandas as pd

def get_all_subject_ids():
    subjects = list(range(1, 65))

    subjects.pop(subjects.index(31))
    subjects.pop(subjects.index(32))

    return subjects

def guess_root_folder():
    if platform.system() == "Windows":
        root_dir   = "/data"
    else:
        root_dir = "/Volumes/SDrive"
    return root_dir


def get_run_index():

    return pd.MultiIndex.from_product([range(1, 61), range(1, 7)], names=['trialNumber', 'runNumber'])

def get_behavior(root_dir=None):
    
    subjects = get_all_subject_ids()

    if root_dir is None:
        root_dir = guess_root_folder()

    behavior = []
    for subject in subjects:
        fn = op.join(root_dir, 'data', 'sourcedata', 'behavior', 'modified_files',  f'modified_participant{subject:02d}_savedValues.csv')

        if not op.exists(fn):
            print(subject)

        behavior.append(pd.read_csv(fn, index_col=[0, 1]))

    behavior = pd.concat(behavior, keys=subjects, names=['subject'])

    return behavior

def get_values_spes(root_dir=None):

    subjects = get_all_subject_ids()

    if root_dir is None:
        root_dir = guess_root_folder()

    parameters = []

    par_dir = op.join(root_dir, 'multlearn-sns', 'Modelling', 'Fitting', 'bestFittingVals')

    pars = []
    
    par_index = []

    for subject in subjects:
        d = {}
        for key in ['V0', 'V1', 'spe']:
            template = op.join(par_dir, f'sub-{subject:02d}', f'{key}*.npy')
            fns = glob.glob(template)

            if len(fns) == 0:
                print(f'{template} has no corresponding filenames')
            else:
                fn = fns[0]
                values = np.load(fn)[:, :60].ravel()

                par_index.append((subject, key))

            pars.append(pd.Series(values, index=get_run_index()))

    return pd.concat(pars, keys=par_index, names=['subject', 'parameter']).unstack('parameter')

