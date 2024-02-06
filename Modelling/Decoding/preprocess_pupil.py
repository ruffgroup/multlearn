import re
import pandas as pd
import hedfpy
import os.path as op
import os
from glob import glob
import numpy as np
import scipy as sp

analysis_params = {
                'sample_rate' : 500.0,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'zscore',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : True,
                }

def main(subject, bids_folder):

    print('starting subject: ', subject)
    source_folder = op.join(bids_folder, 'sourcedata', 'behavior', f'{subject:02d}', 'ETdata')
    target_folder = op.join(bids_folder, 'ds-mlearn', 'derivatives', 'pupil_preproc', f'sub-{subject:02d}', 'func')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    timing_df = pd.read_csv(op.join(bids_folder, 'sourcedata', 'behavior', f'{subject:02d}', f'participant{subject:02d}_savedValues.csv'))

    hdf5_file = op.join(target_folder, f'sub-{subject:02d}_pupil.hdf5')
    if op.exists(hdf5_file):
        os.remove(hdf5_file)

    ho = hedfpy.HDFEyeOperator(hdf5_file)

    # if subject in [32]:
    #     analysis_params['sample_rate'] = 1000

    for run in range(1, 7):

        timing_run = timing_df[timing_df['runNumber'] == run]
        hedf_key = f'sub-{subject:02d}_run-{run}'
        fn = op.join(source_folder, f'M{subject}R{run}.edf')
        if not op.exists(fn):
            continue
        ho.add_edf_file(fn)
        ho.edf_message_data_to_hdf(hedf_key)

#        if (subject == '32') & (run == 1):
#            analysis_params['sample_rate'] = 1000
#        else:
#            analysis_params['sample_rate'] = 500

        ho.edf_gaze_data_to_hdf(alias=hedf_key,
                                sample_rate=analysis_params['sample_rate'],
                                pupil_lp=analysis_params['lp'],
                                pupil_hp=analysis_params['hp'],
                                normalization=analysis_params['normalization'],
                                regress_blinks=analysis_params['regress_blinks'],
                                regress_sacs=analysis_params['regress_sacs'],
                                use_standard_blinksac_kernels=analysis_params['use_standard_blinksac_kernels'],
                                )


        properties = ho.block_properties(hedf_key)
        assert(analysis_params['sample_rate'] == properties.loc[0, 'sample_rate']), print(analysis_params['sample_rate'], properties.loc[0, 'sample_rate'])

        # # Detect behavioral messages
        messages = pd.DataFrame(ho.edf_operator.read_generic_events())
        mapping = {'Stimulus Onset': 'stimOn','Stimulus Offset': 'stimOff',
                   'Feedback Onset': 'feedbackOn','Feedback Offset': 'feedbackOff'}
        messages['type'] = messages.message.apply(lambda x: mapping[x] if x in mapping else x)
        timing_messages = (messages.iloc[7::4]
            .assign(EL_timestamp = lambda x: x['EL_timestamp'], type = 'response', message = 'response')
            .rename(lambda x: x + .5))
        messages = pd.concat([messages, timing_messages], sort=False).sort_index().reset_index(drop=True)
        messages['trial'] = np.where(messages.type.str.contains('stimOn|stimOff|response|feedbackOn|feedbackOff'),messages.groupby('type').cumcount()+1, 0)
        response_times = np.nan_to_num(timing_run['responseTime'].values * 1000, 3000)
        messages.loc[messages['type'] == 'response', 'EL_timestamp'] += response_times
        messages.sort_values('EL_timestamp', ignore_index=True, inplace=True)
        
        start_ix = messages[messages.type == 'stimOn'].index[0]
        start_ts = messages[messages.type == 'stimOn'].iloc[0]['EL_timestamp'] - (timing_run['stimulusOnsetTime'].values[0] * 1000)
        last_ts = messages.EL_timestamp.max()

        events = messages.loc[start_ix:]

        events['onset'] = (events['EL_timestamp'] - start_ts) / 1000

        # # detect saccades
        saccades = ho.detect_saccades_during_period([start_ts, last_ts+5000], hedf_key)
        saccades['onset'] = (saccades['start_timestamp'] - start_ts) / 1000.
        print(saccades)
        
        # saccades['duration'] /=  1000.
        # saccades = saccades[['duration', 'onset']]

        saccades_eyelink = ho.saccades_from_message_file_during_period([start_ts, last_ts+5000], hedf_key)
        saccades_eyelink['onset'] = (saccades_eyelink['start_timestamp'] - start_ts) / 1000.

        # # Detect blinks
        blinks = ho.blinks_during_period([start_ts, last_ts + 5000], hedf_key)
        blinks['onset'] = (blinks['start_timestamp'] - start_ts) / 1000.
        blinks['duration'] /=  1000.
        blinks = blinks[['onset', 'duration']]

        eye = ho.block_properties(hedf_key).loc[0, 'eye_recorded']


        # # Get data
        # resample to TR, remove after 222
        # Make GLM
        d = ho.data_from_time_period([start_ts, last_ts+5000], hedf_key)
        d['time'] = (d['time'] - start_ts) / 1000. - 1./analysis_params['sample_rate']

        TR = 2.3  # TR in seconds
        #original_scale = np.arange(d['time'].iloc[0], d['time'].iloc[-1] + 1./analysis_params['sample_rate'], 1./analysis_params['sample_rate'])
        #desired_scale = np.arange(0, d['time'].iloc[-1] + 1./analysis_params['sample_rate'], TR)
        desired_scale = np.arange(TR/2.0, d['time'].iloc[-1], TR) # Set to half of TR to match fmriprep
        
        d = d.set_index(pd.Index(d['time'], name='time'))
        d['interpolated'] = d[f'{eye}_interpolated_timepoints'].astype(bool)
        d['pupil'] = d[f'{eye}_pupil_bp']
        d = d[['interpolated', 'pupil']]

        d_resampled = pd.DataFrame()
        resampler = sp.interpolate.interp1d(d.index, d['pupil'], kind='linear')
        d_resampled['pupil'] = resampler(desired_scale)
        d_resampled = d_resampled.set_index(pd.Index(desired_scale, name='onset'))
        d_resampled = d_resampled.iloc[:222,:]

        # # Save everything
        saccades.to_csv(op.join(target_folder, f'sub-{subject:02d}_run-{run}_saccades.tsv'), sep='\t', index=False)
        saccades_eyelink.to_csv(op.join(target_folder, f'sub-{subject:02d}_run-{run}_saccadesel.tsv'), sep='\t', index=False)
        blinks.to_csv(op.join(target_folder, f'sub-{subject:02d}_run-{run}_blinks.tsv'), sep='\t', index=False)
        d.to_csv(op.join(target_folder, f'sub-{subject:02d}_run-{run}_pupil.tsv.gz'), sep='\t')
        d_resampled.to_csv(op.join(target_folder, f'sub-{subject:02d}_run-{run}_pupil_resampled.tsv'), sep='\t')
        


if __name__ == '__main__':
    subjects = range(1,65)
    bids_folder = '/mnt/d/data'

    for sub in subjects:
        if sub not in [8, 13, 16, 31, 32, 44]:
            main(sub, bids_folder)