import numpy as np
import pandas as pd
import os.path as op
from nilearn import plotting, datasets
import nibabel as nib
import os

def wrap_angle(x):
    return np.mod(x + np.pi, 2*np.pi) - np.pi

def to_complex(x):
    return np.exp(1j*x)

def from_complex(x):
    x = np.angle(x)
    return np.where(x < 0, x + 2*np.pi, x)

def get_posterior_stats(posterior, ground_truth=None):
    posterior = posterior.copy()
    complex_grid = np.asarray(to_complex(posterior.columns))

    # Take integral over the posterior to get to the expectation (mean posterior)
    # In this case a complex number that we convert back to an angle between 0 and 2pi
    E = from_complex(np.trapz(posterior*complex_grid[np.newaxis,:], axis=1))

    # Take the integral over the posterior to get the expectation of the distance to the
    # mean posterior (i.e., standard deviation)
    relative_error = E[:, np.newaxis] - posterior.columns.values[np.newaxis,:]

    # Wrap the angle to be between 0 and pi, the error can never be larger than pi (180 degrees)
    relative_error = wrap_angle(relative_error)
    absolute_error = np.abs(relative_error)
    sd = np.trapz(absolute_error * posterior, posterior.columns, axis=1)

    stats = pd.DataFrame({'E':E, 'sd':sd}, index=posterior.index)

    if ground_truth is not None:
        stats['E_error'] = wrap_angle(stats['E'] - ground_truth)
        stats['E_error_abs'] = np.abs(stats['E_error'])
        stats['ground_truth'] = ground_truth

    return stats

def plot_encoding_ind(subject, r2_img, space="T1w", bids_folder="/mnt/d/data/ds-mlearn"):
    print("Plotting subject", subject)
    derivatives = op.join(bids_folder, "derivatives")

    if space == "MNI152NLin2009cAsym":
        space_img = datasets.load_mni152_template()
    else:
        space_img = nib.load(
            op.join(
                derivatives,
                "fmriprep",
                f"sub-{subject:02d}",
                "anat",
                f"sub-{subject:02d}_desc-preproc_{space}.nii.gz",
            )
        )
    plotting.plot_stat_map(r2_img, space_img, threshold=0.01, output_file=f"sub-{subject:02d}_{space}_pupil_r2.png")


def plot_encoding_group(encoding_dir, bids_folder):
    derivatives = op.join(bids_folder, "derivatives")
    brain_mask = nib.load(
        op.join(
            derivatives,
            "fmriprep",
            f"sub-01",
            "func",
            f"sub-01_task-learn_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        )
    )
    _,_,r2_files = os.walk(encoding_dir)
    r2_files = [file for file in r2_files if file.endswith('.npy')]
    loaded_r2s = list()
    for file in r2_files:
        if file.endswith('.nii.gz'):
            r2 = nib.load(op.join(encoding_dir, file)).get_fdata()
            loaded_r2s.append(r2)
    mean_r2 = np.mean(loaded_r2s)
    anatomical_image = datasets.load_mni152_template()
    plotting.plot_stat_map(mean_r2, anatomical_image)