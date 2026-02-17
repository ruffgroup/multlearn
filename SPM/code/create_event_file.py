import pandas as pd
import numpy as np
from pathlib import Path

from nipype_helpers import get_subject_info_model2
import argparse


def main(subject_id, bids_folder='/shares/zne.uzh/multlearn', best_fitting_folder='/shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals'):

    if type(subject_id) == int:
        subject_id = f'{subject_id:02d}'

    subject_info, _ = get_subject_info_model2(subject_id, bids_folder, best_fitting_folder)

    for ix, b in enumerate(subject_info):
        run = ix+1

        conditions = b.conditions

        onsets = []

        for j, condition in enumerate(b.conditions):

            o = pd.DataFrame({'onset': b.onsets[j], 'duration': b.durations[j], 'trial_type': condition, })

            pmod = b.pmod[j]

            for k, name in enumerate(pmod.name):
                o['param_' +  name] = np.squeeze(pmod.param[k])

            onsets.append(o)

        onsets = pd.concat(onsets).sort_values('onset')

        fn = Path(bids_folder) / f'sub-{subject_id}' / 'func' / f'sub-{subject_id}_task-learn_run-{run}_events.tsv'

        onsets.to_csv(fn, sep='\t', index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create event files for SPM analysis')
    parser.add_argument('subject_id', type=int, help='Subject ID to create event files for')
    parser.add_argument('--bids_folder', type=str, default='/shares/zne.uzh/multlearn', help='Path to BIDS folder')
    parser.add_argument('--best_fitting_folder', type=str, default='/shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals', help='Path to best fitting parameters folder')

    args = parser.parse_args()

    main(args.subject_id, args.bids_folder, args.best_fitting_folder)