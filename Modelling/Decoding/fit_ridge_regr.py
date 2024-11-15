import sys
import os
import os.path as op
import pandas as pd
import nibabel as nib
import numpy as np
import seaborn as sns
from nilearn.maskers import NiftiMasker
from nilearn import plotting, datasets
from nilearn.image import math_img, threshold_img
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import glob
import argparse
from encoding_helpers import plot_encoding_ind, plot_encoding_group

def main(
    subject,
    data_folder="/mnt/d/data",
    bids_folder="/mnt/d/data/ds-mlearn",
    space="T1w",
    mask=None,
    plotting=0,
    nvoxels=100,
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

    stimulus_info["visual"] = stimulus_info["visual"]

    derivatives = op.join(bids_folder, "derivatives")
    base_dir = "glmsingle"
    base_dir = op.join(derivatives, base_dir, f"sub-{subject:02d}", "func", "pupil")
    betas = nib.load(
        op.join(
            base_dir, f"sub-{subject:02d}_task-task_space-{space}_desc-visualstim.nii.gz"
        )
    )
    if mask == 'v1':
        print('Using v1 mask')
        brain_mask = nib.load(op.join(derivatives,'encoding_model',f'sub-{subject:02d}','anat', f'sub-{subject:02d}_surf2img_v1.nii.gz'))
        brain_mask = math_img('img == 1', img=brain_mask)
    elif mask == 'v123':
        print('Using v123 mask')
        brain_mask = nib.load(op.join(derivatives,'encoding_model',f'sub-{subject:02d}','anat', f'sub-{subject:02d}_surf2img_v1.nii.gz'))
        brain_mask = math_img('(img == 1) | (img == 2) | (img == 3)', img=brain_mask)
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
    print(data)
    
    data_columns = ['voxel_'+str(idx) for idx in range(1,data.shape[1]+1)]
    data = pd.DataFrame(data, columns=data_columns, index=pd.Index(np.arange(len(data)), name='frame')).astype(np.float32)
    paradigm = stimulus_info["visual"].to_frame()
    paradigm.index = data.index

    estimator = SVC(kernel='poly', degree=5,random_state=42)
    kfold = KFold(n_splits=6)
    accuracies = []
    posteriors = []
    for i, (train,test) in enumerate(kfold.split(data)):
        train_data = np.array(data.loc[train,:])
        test_data = np.array(data.loc[test,:])
     
        pred = estimator.fit(train_data, paradigm.loc[train].to_numpy().ravel()).predict(test_data)
        print(pred)
        acc = accuracy_score(paradigm.loc[test].to_numpy().ravel(), pred)
        print(acc)
        accuracies.append(acc)
        posterior = pd.DataFrame()
        posterior['prediction'] = pred
        posterior['truth'] = paradigm.loc[test].to_numpy().ravel()
        posteriors.append(posterior)


    posteriors = pd.concat(posteriors, keys=np.arange(1,7), names=['run'])
    posteriors['correct'] = posteriors['prediction'] == posteriors['truth']
    accuracy = posteriors['correct'].mean()
    
    print('Accuracy:', accuracy)

    postcorrelation_matrix = pd.crosstab(posteriors['prediction'], posteriors['truth'])
    

    target_dir = op.join(
        bids_folder, "derivatives", "ridge_regr", f"sub-{subject:02d}", "func", "pupil"
    )

    stats_dir = op.join(bids_folder, "derivatives", "ridge_regr", f"sub-{subject:02d}", "stats")

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    if not op.exists(stats_dir):
        os.makedirs(stats_dir)

    if space == "MNI152NLin2009cAsym":
        with open(op.join(target_dir,'MNI152NLin2009cAsym_r2_pars_pupil.npy'), 'wb') as f:
            np.save(f, r2)

    if mask == 'v1':
        posteriors.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v1_posterior_pupil.csv'),index=True)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v1_postcorrelation_matrix_pupil.npy'), postcorrelation_matrix.values)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v1_summary_stats_pupil.npy'), np.array(accuracy))
    elif mask == 'v123':
        posteriors.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v123_posterior_pupil.csv'),index=True)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v123_postcorrelation_matrix_pupil.npy'), postcorrelation_matrix.values)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_v123_summary_stats_pupil.npy'), np.array(accuracy))  
    else:
        posteriors.to_csv(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_posterior_pupil.csv'),index=True)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_postcorrelation_matrix_pupil.npy'), postcorrelation_matrix.values)
        np.save(op.join(stats_dir, f'sub-{subject:02d}_space-{space}_summary_stats_pupil.npy'), np.array(accuracy))
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int)
    parser.add_argument("--bids_folder", default="/mnt/d/data/ds-mlearn")
    parser.add_argument("--data_folder", default="/mnt/d/data")
    parser.add_argument("--plotting", default=0, type=int)
    parser.add_argument('--space', default="T1w")
    parser.add_argument('--mask', default=None)
    args = parser.parse_args()

main(
    args.subject,
    bids_folder=args.bids_folder,
    data_folder=args.data_folder,
    space=args.space,
    plotting=args.plotting,
    mask=args.mask,
)
