from nilearn import image
import numpy as np
import os.path as op
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
import glob
from scipy import io
import os
from glmsingle import GLM_single
import argparse


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

        data.append(image.load_img(fn).get_fdata())

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
    data_folder="/mnt/ddata",
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


def main(subject, bids_folder, data_folder, rpe_folder, nruns, ntrials, tr, n, space):
    runs = np.arange(1, nruns + 1)
    events = get_events(subject, nruns, ntrials, data_folder, rpe_folder)
    data = get_fmri_data(subject, nruns, bids_folder, space)

    feedback_onsets = events.xs("feedback", 0, "trial_type")

    rpe_onsets = feedback_onsets.copy()
    rpe_onsets["trial_type"] = "rpe"

    rpe_onsets["modulation"] = (
        rpe_onsets["rpe"].groupby("run").transform(lambda x: x - x.mean())
    )

    feedback_onsets["trial_type"] = "feedback"
    feedback_onsets["modulation"] = 1.0

    glm_onsets = pd.concat((rpe_onsets, feedback_onsets))
    frametimes = np.linspace(tr / 2.0, (n - 0.5) * tr, n)

    glm_dm = [
        make_first_level_design_matrix(
            frametimes, glm_onsets.loc[run, slice(None)], oversampling=100.0, drift_order=0, drift_model=None
        ).drop("constant", axis=1)
        for run in runs
    ]

    events["onset"] = ((events["onset"] + tr / 2.0) // 2.3) * 2.3

    events["duration"] = 0.0
    events["trial_type"] = events["visual_stimulus"]

    dm = [
        make_first_level_design_matrix(
            frametimes,
            events.loc[run, slice(None), "choice"],
            hrf_model="fir",
            oversampling=100.0,
            drift_order=0,
            drift_model=None,
        ).drop("constant", axis=1)
        for run in runs
    ]

    dm = pd.concat(dm, keys=runs, names=["run"]).fillna(0)
    dm.columns = [c.replace("_delay_0", "") for c in dm.columns]
    dm /= dm.max()

    derivatives = op.join(bids_folder, "derivatives")
    base_dir = "glmsingle"
    base_dir = op.join(derivatives, base_dir, f"sub-{subject:02d}", "func", space)

    if not op.exists(base_dir):
        os.makedirs(base_dir)

    X = [dm.loc[run].values for run in runs]

    # create a directory for saving GLMsingle outputs

    opt = dict()

    # set important fields for completeness (but these would be enabled by default)
    opt["wantlibrary"] = 1
    opt["wantglmdenoise"] = 1
    opt["wantfracridge"] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt["wantfileoutputs"] = [0, 0, 0, 1]

    opt["extra_regressors"] = [cf.values for cf in glm_dm]

    print(opt)
    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    try:
        results_glmsingle = glmsingle_obj.fit(X, data, 0.6, 2.3, outputdir=base_dir)

        print("Keys in results_glmsingle:", results_glmsingle.keys())
        print("Keys in results_glmsingle['typed']:", results_glmsingle['typed'].keys())

        betas = results_glmsingle["typed"]["betasmd"]
        betas = image.new_img_like(get_template_image(subject, space=space), betas)

        output_path = op.join(base_dir, f"sub-{subject:02d}_task-task_space-{space}_desc-visualstim.nii.gz")

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        betas.to_filename(output_path)
        print(f"Saved betas to {output_path}")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int)
    parser.add_argument("--bids_folder", default="/mnt/d/data/ds-mlearn")
    parser.add_argument("--data_folder", default="/mnt/d/data")
    parser.add_argument("--rpe_folder", default="/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals")
    parser.add_argument("--tr", type=float, default=2.3)
    parser.add_argument("--n", type=int, default=222)
    parser.add_argument("--nruns", type=int, default=6)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument('--space', default="T1w")
    args = parser.parse_args()

main(
    args.subject,
    bids_folder=args.bids_folder,
    data_folder=args.data_folder,
    rpe_folder=args.rpe_folder,
    nruns=args.nruns,
    ntrials=args.trials,
    tr=args.tr,
    n=args.n,
    space=args.space,
)
