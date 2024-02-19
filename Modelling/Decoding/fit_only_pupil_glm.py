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
from glm_helpers import get_fmri_data, get_conf_regressors, get_pupil_data, create_dm


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