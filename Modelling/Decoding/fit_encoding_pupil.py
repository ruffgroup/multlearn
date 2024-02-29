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
from encoding_helpers import wrap_angle, to_complex, from_complex, get_posterior_stats, plot_encoding_ind, plot_encoding_group

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

    parameters = pd.DataFrame({"mu": mu}).astype(np.float32)
    parameters["amplitude"] = amplitude
    parameters["kappa"] = kappa
    parameters["baseline"] = baseline
    parameters = parameters.astype(np.float32)
    
    data_columns = ['voxel_'+str(idx) for idx in range(1,data.shape[1]+1)]
    data = pd.DataFrame(data, columns=data_columns, index=pd.Index(np.arange(len(data)), name='frame')).astype(np.float32)
    paradigm = stimulus_info["visual"].astype(np.float32).to_frame()
    paradigm.index = data.index

    kfold = KFold(n_splits=6)
    rsq = []
    posteriors = []
    posterior_stats = []
    for i, (train,test) in enumerate(kfold.split(data)):
        train_data = data.loc[train,:]
        test_data = data.loc[test,:]
        model = VonMisesPRF(parameters=parameters, paradigm=paradigm.loc[train].astype(np.float32))
        fitter = WeightFitter(model=model, paradigm=paradigm.loc[train].astype(np.float32), parameters=parameters, data=train_data.astype(np.float32))
        weights = pd.DataFrame(fitter.fit(1.0).numpy(), columns=data_columns, index=parameters.index)
        pred = model.predict(paradigm=paradigm.loc[test].astype(np.float32), weights=weights).astype(np.float32)
        r2 = get_rsq(test_data, pred.values)
        r2.dropna(inplace=True)
        rsq.append(r2)
        best_100 = list(np.sort(np.array(r2.sort_values(ascending=False).index[:100])))
        train_best100 = train_data.loc[:,best_100].astype(np.float32)
        test_best100 = test_data.loc[:,best_100].astype(np.float32)
        truth = stimulus_info['visual'].values[test].astype(np.float32)
        resid_fitter = ResidualFitter(model, train_best100, paradigm=paradigm.loc[train],
                                      parameters=parameters, weights=weights.loc[:,best_100].astype(np.float32))
        omega, dof = resid_fitter.fit()
        stimulus_range = np.linspace(0,2*np.pi,100) # posterior + posterior_stats
        posterior = model.get_stimulus_pdf(test_best100, mapping.astype(np.float32), model.parameters, omega=omega, dof=dof)
        posterior_stat = get_posterior_stats(posterior, ground_truth=truth)
        posterior_pupil_range = model.get_stimulus_pdf(test_best100, stimulus_range.astype(np.float32), model.parameters, omega=omega, dof=dof)
        posterior_pupil_range_stat = get_posterior_stats(posterior_pupil_range, ground_truth=truth)
        posterior['prediction'] = mapping[np.argmax(posterior,axis=1)]
        posterior['truth'] = truth
        posterior_stat = posterior_stat.join(posterior_pupil_range_stat, lsuffix='', rsuffix='_pupil_range')
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
