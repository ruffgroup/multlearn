from nilearn import image
import numpy as np
import os.path as op
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import glob
from scipy import io
import os
from glmsingle import GLM_single
import argparse
import nibabel as nib
from glm_helpers import get_fmri_data, get_events, get_template_image, create_dm


def main(subject, bids_folder, data_folder, rpe_folder, nruns, ntrials, tr, n, space):
    
    runs = np.arange(1, nruns + 1)
    events = get_events(subject, nruns, ntrials, data_folder, rpe_folder)
    data = get_fmri_data(subject, nruns, bids_folder, space)
    
    all_pupil_dms = create_dm(subject, bids_folder, nruns, tr, n)

    feedback_onsets = events.xs("feedback", 0, "trial_type")

    rpe_onsets = feedback_onsets.copy()
    rpe_onsets["trial_type"] = "rpe"

    rpe_onsets["modulation"] = (
        rpe_onsets["rpe"].groupby("run").transform(lambda x: x - x.mean())
    )

    feedback_onsets["trial_type"] = "feedback"
    feedback_onsets["modulation"] = 1.0

    glm_onsets = pd.concat([rpe_onsets, feedback_onsets])
    frametimes = np.linspace(tr / 2.0, (n - 0.5) * tr, n)

    glm_dm = [
        make_first_level_design_matrix(
            frametimes, glm_onsets.loc[run, slice(None)], oversampling=100.0, drift_order=0, drift_model=None
        ).drop("constant", axis=1)
        for run in runs
    ]

    full_glm_dm = [pd.concat([glm_dm[run-1], all_pupil_dms[run-1]], axis=1) for run in runs]

    #plot_design_matrix(full_glm_dm[1], output_file=op.join('/home/ecasim',f"sub-{subject:02d}"+'_glm_1.pdf'))

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
    base_dir = op.join(derivatives, base_dir, f"sub-{subject:02d}", "func", "pupil")

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

    opt["extra_regressors"] = [cf.values for cf in full_glm_dm]

    print(opt)
    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    try:
        results_glmsingle = glmsingle_obj.fit(X, data, 0.6, 2.3, outputdir=base_dir)
        

        print("Keys in results_glmsingle:", results_glmsingle.keys())
        print("Keys in results_glmsingle['typed']:", results_glmsingle['typed'].keys())

        betas = results_glmsingle["typed"]["betasmd"]
        
        betas = image.new_img_like(get_template_image(subject, bids_folder=bids_folder, space=space), betas)

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
    # subjects = range(1,2)
    # space = "MNI152NLin2009cAsym"
    # bids_folder = "/mnt/d/data/ds-mlearn"
    # data_folder = "/mnt/d/data"
    # rpe_folder = "/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals"
    # nruns = 6
    # ntrials = 60
    # tr = 2.3
    # n = 222


    # for sub in subjects:
    #     if sub not in [5, 8, 13, 16, 31, 32, 44]:
    #         main(sub, space=space, bids_folder=bids_folder, data_folder=data_folder, rpe_folder=rpe_folder, nruns=nruns, ntrials=ntrials, tr=tr, n=n)
