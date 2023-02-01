import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
from nilearn import image


def main(subject, bids_folder='/data', max_rt=1.0):
    sourcedata = op.join(bids_folder, 'sourcedata')

    target_dir = op.join(bids_folder, 'ds-mlearn', f'sub-{subject}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run in range(1, 7):
        print(subject, run)
        nii = op.join(target_dir, f'sub-{subject}_task-learn_run-{run}_bold.nii')
        print(nii)

        if op.exists(nii):
            n_volumes = image.load_img(nii).shape[-1]
        else:
            n_volumes = 135

        behavior = pd.read_table(op.join(sourcedata,
                                         f'behavior/{subject}/sub-{subject}_task-task_run-{run}_events.tsv'))
        behavior['trial_nr'] = behavior['trial_nr'].astype(np.int)

        print(behavior)

        pulses = behavior[behavior.event_type == 'pulse'][['trial_nr', 'onset']]

        pulses['ipi'] = pulses['onset'].diff()
        pulses = pulses[((pulses['ipi'] > 1.) & (pulses['ipi'] < 5.)) | pulses.ipi.isnull()]
        print(pulses.sort_values('ipi'))

        if n_volumes != pulses.shape[0]:
            pulses = pulses.set_index(np.arange(1, pulses.shape[0] + 1))[['trial_nr', 'onset']]
            t0 = pulses.loc[1, 'onset'] - (n_volumes - pulses.shape[0]) * 2.3
            print(f'Pulse missing: {pulses.loc[1, "onset"]}, {t0}')
        else:
            pulses = pulses.set_index(np.arange(1, n_volumes + 1))[['trial_nr', 'onset']]
            t0 = pulses.loc[1, 'onset']

        stim1 = behavior[(behavior['event_type'] == 'stim') & (behavior['phase'] == 4)]
        stim1['n'] = stim1['n1']
        stim1['onset'] -= t0
        stim1['trial_type'] = 'stimulus 1'

        stim2 = behavior[(behavior['event_type'] == 'stim') & (behavior['phase'] == 8)]
        stim2['n'] = stim2['n2']
        stim2['onset'] -= t0
        stim2['trial_type'] = 'stimulus 2'

        choice = behavior[(behavior['event_type'] == 'choice')]
        choice['onset'] -= t0
        choice['trial_type'] = 'choice'

        events = pd.concat((stim1, stim2, choice)).sort_index().reset_index(drop=True)
        # result['choice'] = result['choice'].astype(int)
        events = events[['trial_nr', 'onset', 'trial_type', 'prob1', 'prob2', 'n1', 'n2', 'choice']]

        fn = op.join(target_dir, f'sub-{subject}_task-task_run-{run}_events.tsv')
        events.to_csv(fn, index=False, sep='\t')


def get_hazard(x, s=1.0, loc=0.0, scale=10, cut=30, use_cut=False):
    import scipy.stats as ss

    x = x / .7

    dist = ss.lognorm(s, loc, scale)

    if use_cut:
        sf = lambda x: 1 - (dist.cdf(x) / dist.cdf(cut))
    else:
        sf = dist.sf

    return np.clip(dist.pdf(x) / sf(x), 0, np.inf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)