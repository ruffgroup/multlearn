import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import argparse
import pandas as pd
import glob
from scipy import io
import os

def create_stats_df(bids_folder="/shares/zne.uzh/multlearn/ds-mlearn", paramregr_folder="/shares/zne.uzh/multlearn", space='T1w', mask='v1', pupil=True, nvoxels=100):
    print(pupil)
    '''
    inputs:
    bids folder
    space T1w or MNI
    mask None, v1 or v123
    pupil True or False
    '''
    subjects = range(1,65)
    accuracy = list()
    nruns = 6
    ntrials = 60
    combined_df = list()
    for subject in subjects:
        if subject not in [5, 8, 13, 16, 31, 32, 44]:
            stats_dir = op.join(bids_folder, "derivatives", "encoding_model", f"sub-{subject:02d}", "stats")
            if pupil == True:
                summary_stats = np.load(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_summary_stats_pupil.npy")) if mask is not None else np.load(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_visual_summary_stats_pupil.npy"))
                posterior_stats = pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_posterior_stats_pupil.csv")) if mask is not None else pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_posterior_stats_pupil.csv"))
                posterior = pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_posterior_pupil.csv")) if mask is not None else pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_posterior_pupil.csv"))
            else:
                summary_stats = np.load(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_summary_stats.npy")) if mask is not None else np.load(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_visual_summary_stats.npy"))
                posterior_stats = pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_posterior_stats.csv")) if mask is not None else pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_posterior_stats.csv"))
                posterior = pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_{mask}_posterior.csv")) if mask is not None else pd.read_csv(op.join(stats_dir, f"sub-{subject:02d}_space-{space}_n-{nvoxels}_posterior.csv"))
            print(f"Subject {subject:02d}: accuray = {summary_stats[0]:.2f}, uncertainty = {summary_stats[1]:.2f}")
            accuracy.append(summary_stats[0])

            template = op.join(paramregr_folder, f'bestFittingVals/sub-{subject:02d}/spe*.mat')
            fn = glob.glob(template)
            print(fn)
            assert len(fn) == 1
            fn = fn[0]
            spe = pd.DataFrame(
                    io.loadmat(fn)["spe"],
                    index=pd.Index(np.arange(1, nruns + 1), name="run"),
                    columns=pd.Index(np.arange(1, ntrials + 1), name="trial_nr"),
                    ).stack().to_frame("spe")

            template2 = op.join(paramregr_folder, f'bestFittingVals/sub-{subject:02d}/rpe*.mat')
            fn2 = glob.glob(template2)
            print(fn2)
            assert len(fn2) == 1
            fn2 = fn2[0]
            rpe = pd.DataFrame(
                    io.loadmat(fn2)["rpe"],
                    index=pd.Index(np.arange(1, nruns + 1), name="run"),
                    columns=pd.Index(np.arange(1, ntrials + 1), name="trial_nr"),
                    ).stack().to_frame("rpe")

            template3 = op.join(paramregr_folder, f'bestFittingVals/sub-{subject:02d}/V0*.npy')
            fn3 = glob.glob(template3)
            print(fn3)
            assert len(fn3) == 1
            fn3 = fn3[0]
            V0_values = np.load(fn3)
            V0_values = pd.DataFrame(
                V0_values, 
                index=pd.Index(np.arange(1, nruns + 1), name="run"),
                columns=pd.Index(np.arange(1, ntrials + 1), name="trial_nr"),
            ).stack().to_frame("V0_values")

            template4 = op.join(paramregr_folder, f'bestFittingVals/sub-{subject:02d}/V1*.npy')
            fn4 = glob.glob(template4)
            print(fn4)
            assert len(fn4) == 1
            fn4 = fn4[0]
            V1_values = np.load(fn4)
            V1_values = pd.DataFrame(
                V1_values, 
                index=pd.Index(np.arange(1, nruns + 1), name="run"),
                columns=pd.Index(np.arange(1, ntrials + 1), name="trial_nr"),
            ).stack().to_frame("V1_values")

            fn5 = op.join(
            paramregr_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject:02d}_savedValues.csv",
            )
            stimulus_info = (
            pd.read_csv(
                fn5,
                usecols=["trialNumber", "runNumber", "accurate", "action", "responseTime"],
            )
            .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
            .set_index(["run", "trial_nr"])
            .sort_index()
            )


            posterior_stats = pd.concat([posterior_stats, spe, rpe, V0_values, V1_values], axis=1)
            posterior_stats['subject'] = subject
            posterior_stats['correct'] = posterior['correct']
            combined_df.append(posterior_stats)
    return pd.concat(combined_df)

def main(bids_folder="/shares/zne.uzh/multlearn/ds-mlearn", paramregr_folder="/shares/zne.uzh/multlearn", space='T1w', mask='v1', pupil=True, nvoxels=100):
    print(pupil)
    plots_dir = op.join(bids_folder, "derivatives", "encoding_model", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    combined_df = create_stats_df(bids_folder, paramregr_folder, space, mask, pupil, nvoxels)
    ax = sns.swarmplot(y='correct', data=combined_df.groupby('subject')[['correct']].mean())
    plt.axhline(0.33, c='k', ls='--')
    if pupil == True:
        print("pupil true")
        plt.title(f"Decoding accuracy for each individual in {mask} with {nvoxels} voxels (GLM incl. pupil data)") if mask is not None else plt.title(f"Whole brain decoding accuracy for each individual with {nvoxels} voxels (GLM incl. pupil data)")
        plt.savefig(op.join(plots_dir, f"decoding_swarmplot_{space}_n-{nvoxels}_{mask}_pupil.png") if mask is not None else plt.savefig(f"decoding_swarmplot_{space}_n-{nvoxels}_pupil.png"))
        np.save(op.join(plots_dir, f"decoding_accuracies_{space}_n-{nvoxels}_{mask}_pupil.npy"), combined_df.groupby('subject')[['correct']].mean()) if mask is not None else np.save(op.join(plots_dir, f"decoding_accuracies_{space}_n-{nvoxels}_pupil.npy"), combined_df.groupby('subject')[['correct']].mean())
    else:
        print("pupil false")
        plt.title(f"Decoding accuracy for each individual in {mask} with {nvoxels} voxels (GLM excl. pupil data)") if mask is not None else plt.title(f"Whole brain decoding accuracy for each individual with {nvoxels} voxels (GLM excl. pupil data)")
        plt.savefig(op.join(plots_dir, f"decoding_swarmplot_{space}_n-{nvoxels}_{mask}.png") if mask is not None else plt.savefig(f"decoding_swarmplot_{space}_n-{nvoxels}.png"))
        np.save(op.join(plots_dir, f"decoding_accuracies_{space}_n-{nvoxels}_{mask}.npy"), combined_df.groupby('subject')[['correct']].mean()) if mask is not None else np.save(op.join(plots_dir, f"decoding_accuracies_{space}_n-{nvoxels}.npy"), combined_df.groupby('subject')[['correct']].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_folder", default="/shares/zne.uzh/multlearn/ds-mlearn")
    parser.add_argument("--paramregr_folder", default="/shares/zne.uzh/multlearn")
    parser.add_argument("--space", default='T1w')
    parser.add_argument("--mask", default='v1')
    parser.add_argument("--pupil", action='store_true')
    parser.add_argument("--nvoxels", type=int, default=100)
    args = parser.parse_args()

main(bids_folder=args.bids_folder, paramregr_folder=args.paramregr_folder, space=args.space, mask=args.mask, pupil=args.pupil, nvoxels=args.nvoxels)


