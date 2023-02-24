import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
from nilearn import image
import numpy as np


def main(subject, bids_folder='T:/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri/data'):
    sourcedata = op.join(bids_folder, 'sourcedata')

    if int(subject) < 10:
        target_dir = op.join("/shares/zne.uzh/ds-mlearn/derivatives/fmriprep", f'sub-0{subject}', 'func')
    else:
        target_dir = op.join("/shares/zne.uzh/ds-mlearn/derivatives/fmriprep", f'sub-{subject}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    behavior = pd.read_table(op.join(sourcedata,
                                         f'behavior/{subject}/participant{subject}_savedValues.csv'), sep=",")
    

    for run in range(1, 7):
        print(subject, run)
        if int(subject) < 10:
            nii = op.join(target_dir, f'sub-0{subject}_task-learn_run-{run}_bold.nii')
           
        else:
            nii = op.join(target_dir, f'sub-{subject}_task-learn_run-{run}_bold.nii')
        print(nii)
        
        if op.exists(nii):
            n_volumes = image.load_img(nii).shape[-1]
        else:
            n_volumes = 135

        run_behavior = behavior[behavior.runNumber == run].reset_index()
        run_behavior['trial_nr'] = run_behavior['trialNumber'].astype(int)
        runType = "tactile" if ~np.isnan(run_behavior['tactile'][0]) else "audio"

        #print(run_behavior)

        stim = pd.DataFrame()
        feedback = pd.DataFrame()

        stim['onset'] = run_behavior.stimulusOnsetTime
        stim['trial_type'] = 'stimulus'
        stim['duration'] = run_behavior.stimulusOffsetTime - run_behavior.stimulusOnsetTime
        stim['trial_nr'] = run_behavior['trial_nr']
        stim["runType"] = runType

        feedback['onset'] = run_behavior.feedbackOnsetTime
        feedback['trial_type'] = 'choice'
        feedback['duration'] = run_behavior.feedbackOffsetTime - run_behavior.feedbackOnsetTime

        events = pd.concat((stim, feedback)).sort_index().reset_index(drop=True)
        # result['choice'] = result['choice'].astype(int)
        events = events[['trial_nr','runType','onset', 'duration', 'trial_type']]


        if int(subject) < 10:
            fn = op.join(target_dir, f'sub-0{subject}_task-learn_run-{run}_events.tsv')
        else:
            fn = op.join(target_dir, f'sub-0{subject}_task-learn_run-{run}_events.tsv')
        events.to_csv(fn, index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='T:/projects/2022/bedi_casimiro_ruff_multisensorylearningfmri/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)