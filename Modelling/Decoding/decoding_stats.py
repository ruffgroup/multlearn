import numpy as np
import os.path as op
import argparse
import pandas as pd
import glob
from scipy import io, stats
import os

def main(bids_folder, nvoxels, mask):
    plots_dir = op.join(bids_folder, "derivatives", "encoding_model", "plots")
    accuracies_pupil = np.load(op.join(plots_dir, f"decoding_accuracies_T1w_n-{nvoxels}_{mask}_pupil.npy"), allow_pickle=True)
    accuracies_basic = np.load(op.join(plots_dir, f"decoding_accuracies_T1w_n-{nvoxels}_{mask}.npy"), allow_pickle=True)

    stat_vals = stats.ttest_rel(accuracies_basic, accuracies_pupil)
    print(stat_vals)

if __name__ == "__main__":
    main("/shares/zne.uzh/multlearn/ds-mlearn", 400, "v1")