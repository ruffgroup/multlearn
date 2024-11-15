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
         
        data.append(nib.load(fn).get_fdata()) 

    return data


def get_template_image(subject, bids_folder="/mnt/d/data/ds-mlearn", space="T1w"):
    run = 1
    fn = op.join(
        bids_folder,
        "derivatives",
        "fmriprep",
        f"sub-{subject:02d}",
        "func",
        f"sub-{subject:02d}_task-learn_run-{run}_space-{space}_desc-preproc_bold.nii",
    )

    return image.load_img(fn)


def get_rpe(
    subject,
    nruns=6,
    ntrials=60,
    rpe_folder="/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals",
):
    template = op.join(rpe_folder, f"sub-{subject:02d}", "rpe*.mat")
    fn = glob.glob(template)
    print(fn)
    assert len(fn) == 1
    fn = fn[0]

    return (
        pd.DataFrame(
            io.loadmat(fn)["rpe"],
            index=pd.Index(np.arange(1, nruns + 1), name="run"),
            columns=pd.Index(np.arange(1, ntrials + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("rpe")
    )


def get_events(
    subject,
    nruns=6,
    ntrials=60,
    data_folder="/mnt/d/data",
    rpe_folder="/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals",
):
    runs = np.arange(1, nruns + 1)

    events = []

    for run in runs:
        fn = op.join(
            data_folder,
            "ds-mlearn",
            "derivatives",
            "fmriprep",
            f"sub-{subject:02d}",
            "func",
            f"sub-{subject:02d}_task-learn_run-{run}_events.tsv",
        )
        e = pd.read_csv(fn, sep="\t").sort_values("onset")
        e["trial_nr"] = e["trial_nr"].ffill().astype(int)
        e = e.set_index(["trial_nr", "trial_type"])

        events.append(e)

    fn2 = op.join(
        data_folder,
        "sourcedata",
        "behavior",
        f"{subject:02d}",
        f"participant{subject:02d}_savedValues.csv",
    )

    events = pd.concat(events, keys=runs, names=["run"])
    stimulus_info = (
        pd.read_csv(
            fn2,
            usecols=["trialNumber", "runNumber", "visual"],
        )
        .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
        .set_index(["run", "trial_nr"])
        .sort_index()
    )

    events["visual_stimulus"] = np.repeat(
        stimulus_info["visual"].map(lambda x: f"stimulus {x+1}").values, 2
    )

    events = events.join(get_rpe(subject, nruns, ntrials, rpe_folder))

    return events

def get_conf_regressors(subject, bids_folder, nruns=6):
    regressors_all = []

    for run in range(1,nruns+1):

        confounds = pd.read_csv(op.join(bids_folder,
            f"derivatives/fmriprep/sub-{subject:02d}/func/sub-{subject:02d}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(bids_folder, f"derivatives/fmriprep/sub-{subject:02d}/beh/physio/RegPhysio_sub-{subject:02d}_run_{run}.mat")
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
        dm = make_first_level_design_matrix(frame_times=frametimes, events=events,add_regs=pd.concat([pd.DataFrame(pupil_all[run-1][0], columns=['pupil']),regressors_all[run-1]], axis=1), add_reg_names = ['pupil']+regressors_all[0].iloc[:,:-1].columns.values.tolist(), hrf_model='spm', drift_model=None, oversampling=100)
        all_design_matrices.append(dm)

    return all_design_matrices



