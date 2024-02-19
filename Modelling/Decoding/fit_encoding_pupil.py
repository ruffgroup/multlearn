import sys
sys.path.insert(0, 'braincoder/')
import braincoder
from braincoder.models import VonMisesPRF
from braincoder.optimize import WeightFitter, ResidualFitter
from braincoder.utils import get_rsq
import os
import os.path as op
import pandas as pd
import nibabel as nib
import numpy as np
import seaborn as sns
from nilearn.maskers import NiftiMasker
from nilearn import plotting, datasets
from nilearn.image import math_img
from sklearn.model_selection import KFold
import glob
import argparse

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



def main(
    subject,
    data_folder="/mnt/d/data",
    bids_folder="/mnt/d/data/ds-mlearn",
    space="T1w",
    mask=None,
    mu=[0.0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi],
    amplitude=1,
    kappa=1.0,
    baseline=0.0,
    alpha=1.0,
    plotting=0,
):
    
    fn = op.join(
        data_folder,
        "sourcedata",
        "behavior",
        f"{subject:02d}",
        f"participant{subject:02d}_savedValues.csv",
    )
    stimulus_info = (
        pd.read_csv(
            fn,
            usecols=["trialNumber", "runNumber", "visual"],
        )
        .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
        .set_index(["run", "trial_nr"])
        .sort_index()
    )
    mapping = [22.5 * np.pi / 180, 67.5 * np.pi / 180, 112.5 * np.pi / 180]
    mapping = np.array(mapping) * 2
    stimulus_info["visual"] = stimulus_info["visual"].map(lambda x: mapping[x])

    derivatives = op.join(bids_folder, "derivatives")
    base_dir = "glmsingle"
    base_dir = op.join(derivatives, base_dir, f"sub-{subject:02d}", "func", "pupil")
    betas = nib.load(
        op.join(
            base_dir, f"sub-{subject:02d}_task-task_space-{space}_desc-visualstim.nii.gz"
        )
    )
    if mask == 'visual':
        print('Using visual mask')
        brain_mask = nib.load(op.join(derivatives,'encoding_model',f'sub-{subject:02d}','anat', f'sub-{subject:02d}_surf2img_v1.nii.gz'))
        brain_mask = math_img('img == 1', img=brain_mask)
    else:
        brain_mask = nib.load(
            op.join(
                derivatives,
                "fmriprep",
                f"sub-{subject:02d}",
                "func",
                f"sub-{subject:02d}_task-learn_run-1_space-{space}_desc-brain_mask.nii.gz",
            )
        )
    masker = NiftiMasker(brain_mask)
    data = masker.fit_transform(betas)

    parameters = pd.DataFrame({"mu": mu})
    parameters["amplitude"] = amplitude
    parameters["kappa"] = kappa
    parameters["baseline"] = baseline

    paradigm = stimulus_info["visual"].astype(np.float32)

    kfold = KFold(n_splits=6)
    rsq = []
    posteriors = []
    posterior_stats = []
    for i, (train,test) in enumerate(kfold.split(data)):
        train_data = data[train,:].astype(np.float32)
        test_data = data[test,:].astype(np.float32)
        model = VonMisesPRF(parameters=parameters, paradigm=paradigm.values[train])
        fitter = WeightFitter(model=model, paradigm=paradigm.values[train], parameters=parameters, data=train_data)
        weights = fitter.fit(1.0)
        pred = model.predict(paradigm=paradigm.values[test], weights=weights)
        r2 = get_rsq(test_data, pred.values)
        rsq.append(r2)
        best_100 = np.sort(np.argsort(r2)[-100:])
        train_best100 = train_data[:,best_100]
        test_best100 = test_data[:,best_100]
        truth = stimulus_info['visual'].values[test]
        resid_fitter = ResidualFitter(model, train_best100, paradigm=paradigm.values[train], parameters=parameters, weights=weights.numpy()[:,best_100])
        omega, dof = resid_fitter.fit()
        stimulus_range = np.linspace(0,2*np.pi,100) # posterior + posterior_stats
        posterior = model.get_stimulus_pdf(test_best100.astype(np.float32), mapping.astype(np.float32), model.parameters, omega=omega, dof=dof)
        posterior_stat = get_posterior_stats(posterior, ground_truth=truth)
        posterior_full_range = model.get_stimulus_pdf(test_best100.astype(np.float32), stimulus_range.astype(np.float32), model.parameters, omega=omega, dof=dof)
        posterior_full_range_stat = get_posterior_stats(posterior_full_range, ground_truth=truth)
        posterior['prediction'] = mapping[np.argmax(posterior,axis=1)]
        posterior['truth'] = truth
        posterior_stat = posterior_stat.join(posterior_full_range_stat, lsuffix='', rsuffix='_full_range')
        posteriors.append(posterior)
        posterior_stats.append(posterior_stat)

    mean_r2 = np.mean(np.array(rsq),axis=0)
    r2_img = masker.inverse_transform(mean_r2)

    posteriors = pd.concat(posteriors, keys=np.arange(1,7), names=['run'])
    posterior_stats = pd.concat(posterior_stats, keys=np.arange(1,7), names=['run'])

    uncertainty = posterior_stats[['E_error_abs', 'sd']].corr().values[0,1]
    posteriors['correct'] = posteriors['prediction'] == posteriors['truth']
    accuracy = posteriors['correct'].mean()
    
    print('Accuracy:', accuracy)
    print('Decoded uncertainty:', uncertainty)

    postcorrelation_matrix = pd.crosstab(posteriors['prediction'], posteriors['truth'])
    
    if plotting == 1:
        plot_encoding_ind(subject, r2_img, space)

    target_dir = op.join(
        bids_folder, "derivatives", "encoding_model", f"sub-{subject:02d}", "func", "pupil"
    )

    stats_dir = op.join(bids_folder, "derivatives", "encoding_model", f"sub-{subject:02d}", "stats")

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    if not op.exists(stats_dir):
        os.makedirs(stats_dir)

    if space == "MNI152NLin2009cAsym":
        with open(op.join(target_dir,'MNI152NLin2009cAsym_r2_pars_pupil.npy'), 'wb') as f:
            np.save(f, r2)

    if mask == 'visual':
        r2_img.to_filename(op.join(target_dir, f"sub-{subject:02d}_space-{space}_desc-r2_visual_pars_pupil.nii.gz"))
        posteriors.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_visual_posterior_pupil.csv'),index=True)
        posterior_stats.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_visual_posterior_stats_pupil.csv'),index=True)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_visual_postcorrelation_matrix_pupil.npy'), postcorrelation_matrix.values)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_visual_summary_stats_pupil.npy'), np.array([accuracy, uncertainty]))
    else:
        r2_img.to_filename(op.join(target_dir, f"sub-{subject:02d}_space-{space}_desc-r2_pars_pupil.nii.gz"))
        posteriors.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_posterior_pupil.csv'),index=True)
        posterior_stats.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_posterior_stats_pupil.csv'),index=True)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_postcorrelation_matrix_pupil.npy'), postcorrelation_matrix.values)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_summary_stats_pupil.npy'), np.array([accuracy, uncertainty]))
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int)
    parser.add_argument("--bids_folder", default="/mnt/d/data/ds-mlearn")
    parser.add_argument("--data_folder", default="/mnt/d/data")
    parser.add_argument("--mu", default=[0.0, 0.5 * np.pi, 1 * np.pi, 1.5 * np.pi]) #[0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]
    parser.add_argument("--amplitude", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=1.)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--plotting", default=0, type=int)
    parser.add_argument('--space', default="T1w")
    parser.add_argument('--mask', default=None)
    args = parser.parse_args()

main(
    args.subject,
    bids_folder=args.bids_folder,
    data_folder=args.data_folder,
    mu=args.mu,
    amplitude=args.amplitude,
    kappa=args.kappa,
    baseline=args.baseline,
    alpha=args.alpha,
    space=args.space,
    plotting=args.plotting,
    mask=args.mask,
)
