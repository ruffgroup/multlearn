from nilearn import image
import numpy as np
import os.path as op
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix, compute_regressor, FirstLevelModel
import glob
from scipy import io
import os
from glmsingle import GLM_single
import argparse
import nibabel as nib
from nilearn import plotting
from nilearn.image import resample_to_img

def get_fmri_data(subject, nruns=6, bids_folder="/mnt/d/data/ds-mlearn", space="T1w"):
    runs = np.arange(1, nruns + 1)

    data = []

    for run in runs:
        fn = op.join(
            bids_folder,
            "derivatives",
            "fmriprep",
            f"sub-{subject:02d}",
            "func",
            f"sub-{subject:02d}_task-learn_run-{run}_space-{space}_desc-preproc_bold.nii",
        )

        data.append(nib.load(fn))

    return data

def get_conf_regressors(subject, bids_folder, nruns=6):
    regressors_all = []

    for run in range(1,nruns+1):

        confounds = pd.read_csv(
            f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject:02d}/func/sub-{subject:02d}_task-learn_run-{run}_desc-confounds_timeseries.tsv",
            delimiter="\t",
        )

        confounds = confounds.loc[
            :,
            [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
            ],
        ]

        physio_path = f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject:02d}/beh/physio/RegPhysio_sub-{subject:02d}_run_{run}.mat"
        fn3 = glob.glob(physio_path)
        assert len(fn3) == 1
        fn3 = fn3[0]

        physio = io.loadmat(fn3, simplify_cells=True)["physio"]["model"]
        physio = pd.DataFrame(
            data=physio["R"],
            columns=physio["R_column_names"],
        )
        physio.columns = pd.MultiIndex.from_arrays([physio.columns, physio.columns.to_series().groupby(physio.columns).cumcount().astype(str)]).map('_'.join)
        regressors = pd.concat([confounds, physio], axis=1)
        
        regressors_all.append(regressors)
    
    return regressors_all

def get_pupil_data(subject, bids_folder, nruns=6, tr=2.3, n=222):
    conditions_all = []
    onsets_all = []
    pupil_all = []

    frametimes = np.linspace(tr / 2.0, (n - 0.5) * tr, n)

    for run in range(1, nruns+1):

        blinks = pd.read_csv(op.join(bids_folder, "derivatives", "pupil_preproc", f"sub-{subject:02d}", "func", f"sub-{subject:02d}_run-{run}_blinks.tsv"), delimiter="\t")
        blinks['trial_type'] = "blink"
        conditions = blinks['trial_type'].values.tolist()
        onsets = blinks['onset'].values.tolist()

        saccades = pd.read_csv(op.join(bids_folder, "derivatives", "pupil_preproc", f"sub-{subject:02d}", "func", f"sub-{subject:02d}_run-{run}_saccades.tsv"), delimiter="\t")
        saccades['trial_type'] = "saccade"
        conditions += saccades['trial_type'].values.tolist()
        onsets += saccades['onset'].values.tolist()

        conditions_all.append(conditions)
        onsets_all.append(onsets)

        pupil = pd.read_csv(op.join(bids_folder, "derivatives", "pupil_preproc", f"sub-{subject:02d}", "func", f"sub-{subject:02d}_run-{run}_pupil_resampled.tsv"), delimiter="\t")
        pupil['duration'] = 0
        pupil_arr = pupil.loc[:, ["onset", "duration", "pupil"]]
        missing_rows = frametimes.shape[0] - pupil_arr.shape[0]
        missing_rows_arr = pd.DataFrame(np.zeros((int(missing_rows), 3)), columns=["onset", "duration", "pupil"])
        pupil_arr = pd.concat([pupil_arr, missing_rows_arr], ignore_index=True).to_numpy().T

        pupil_reg = compute_regressor(pupil_arr, frame_times=frametimes, hrf_model="spm")
        pupil_all.append(pupil_reg)

    return conditions_all, onsets_all, pupil_all
    
def create_dm(subject, bids_folder, nruns=6, tr=2.3, n=222):

    conditions_all, onsets_all, pupil_all = get_pupil_data(subject, bids_folder, nruns, tr, n)
    regressors_all = get_conf_regressors(subject, bids_folder, nruns)
    frametimes = np.linspace(tr / 2.0, (n - 0.5) * tr, n)
    all_design_matrices = []
    for run in range(1, nruns+1):
        events = pd.DataFrame({'trial_type': conditions_all[run-1], 'onset': onsets_all[run-1], 'duration': 0})
        dm = make_first_level_design_matrix(frame_times=frametimes, events=events,add_regs=pd.concat([pd.DataFrame(pupil_all[run-1][0], columns=['pupil']),regressors_all[run-1]], axis=1), add_reg_names = ['pupil']+regressors_all[0].iloc[:,:-1].columns.values.tolist(), hrf_model='spm')
        all_design_matrices.append(dm)

    return all_design_matrices

def main(subject, bids_folder, nruns=6, tr=2.3, n=222):
    
    all_design_matrices = create_dm(subject, bids_folder, nruns, tr, n)
    data = get_fmri_data(subject, nruns, bids_folder)
    brain_mask = nib.load(
            op.join(
                bids_folder,
                "derivatives",
                "fmriprep",
                f"sub-{subject:02d}",
                "func",
                f"sub-{subject:02d}_task-learn_run-1_space-T1w_desc-brain_mask.nii.gz",
            )
        )
    fmri_glm = FirstLevelModel(signal_scaling=False, t_r=tr, mask_img=brain_mask)
    fmri_glm_multirun = fmri_glm.fit(data, design_matrices=all_design_matrices)

    contrast_pupil = np.array([[1 if 'pupil' in col else 0 for col in all_design_matrices[0].columns]])
    contrast_blinks = np.array([[1 if 'blink' in col else 0 for col in all_design_matrices[0].columns]])
    contrast_saccades = np.array([[1 if 'saccade' in col else 0 for col in all_design_matrices[0].columns]])

    z_map_pupil = fmri_glm_multirun.compute_contrast(contrast_pupil, stat_type='t', output_type='stat')
    z_map_blinks = fmri_glm_multirun.compute_contrast(contrast_blinks, stat_type='t', output_type='stat')
    z_map_saccades = fmri_glm_multirun.compute_contrast(contrast_saccades, stat_type='t', output_type='stat')

    plotting_folder = op.join(bids_folder, "derivatives", "pupil_preproc", f"sub-{subject:02d}", "plots")
    if not op.exists(plotting_folder):
        os.makedirs(plotting_folder)

    T1w = op.join(bids_folder, "derivatives", "fmriprep", f"sub-{subject:02d}", "anat", f"sub-{subject:02d}_desc-preproc_T1w.nii.gz")

    plotting.plot_stat_map(
    z_map_pupil,
    bg_img=T1w,
    threshold=3.1,
    output_file=op.join(plotting_folder, "pupil.png"),
    title='Pupil stat map')

    plotting.plot_stat_map(
    z_map_blinks,
    bg_img=T1w,
    threshold=3.1,
    output_file=op.join(plotting_folder, "blinks.png"),
    title='Blinks stat map')

    plotting.plot_stat_map(
    z_map_saccades,
    bg_img=T1w,
    threshold=3.1,
    output_file=op.join(plotting_folder, "saccades.png"),
    title='Saccades stat map')



if __name__ == "__main__":
    subjects = range(6,65)
    bids_folder = '/mnt/d/data/ds-mlearn'

    for sub in subjects:
        if sub not in [5, 8, 13, 16, 31, 32, 44]:
            main(sub, bids_folder)

    #parser = argparse.ArgumentParser(description="fit GLM to pupil data")
    #parser.add_argument("subject", type=int, help="The subject id")
    #parser.add_argument("bids_folder", type=str, help="The path to the BIDS folder")
    #parser.add_argument("--nruns", type=int, default=6, help="The number of runs")
    #parser.add_argument("--tr", type=float, default=2.3, help="The TR")
    #parser.add_argument("--n", type=int, default=222, help="The number of volumes")
    #args = parser.parse_args()

    #main(args.subject, args.bids_folder, args.nruns, args.tr, args.n)