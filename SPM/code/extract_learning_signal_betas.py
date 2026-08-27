"""Extract per-subject ROI betas for the three learning-signal modulators of model7.

For every previously reported learning-signal region we average, within the ROI,
the first-level model7 beta of each parametric modulator:
    surprise (ChoicexAudio/Tactile x surprise^1)
    uRPE     (FeedbackAudio/Tactile x urpe^1)
    RPE      (FeedbackAudio/Tactile x rpe^1, signed)

ROI definitions
    surprise : /shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_*_con5_3_1_pos.nii
    rpe      : /shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_*_con1_8_0_pos.nii
    urpe     : connected components (18-connectivity) of the model7 con1
               SnPM_filtered_t3_1_pos map (uRPE main effect, cluster-FWE survivors)

Outputs (long format, one row per subject x ROI x modulator x run):
    <out-dir>/learning_signal_betas.tsv
    <out-dir>/learning_signal_roi_info.tsv   (per-ROI provenance: voxel counts, CoM, peak)

Run on the cluster (reads ~2000 small NIfTIs from /shares):
    srun -c 2 --mem 8G --time 45 --account=zne.uzh \
        $HOME/data/conda/envs/multlearn/bin/python extract_learning_signal_betas.py
"""

import argparse
import os
import os.path as op
import re
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy import ndimage

MODEL2_ROI = "/shares/zne.uzh/multlearn/nipype/model2/ROI"
MODEL7 = "/shares/zne.uzh/multlearn/nipype/model7"
# t4.0: the manuscript's uRPE map threshold (Fig. urpe_pos); at t3.1 the
# lateral-frontal clusters merge into one component
URPE_MAP = op.join(MODEL7, "2ndLevel/cluster_SnPM_SecondLevel_con1/SnPM_filtered_t4_0_pos.nii")

# GLM subject set: 62 behavioural subjects minus 8, 13, 16, 44 (see submit_glm_1st.sh)
SUBJECTS = [f"{s:02d}" for s in range(1, 65) if s not in (8, 13, 16, 31, 32, 44)]

SIGNALS = {"xsurprise^1": "surprise", "xurpe^1": "urpe", "xrpe^1": "rpe"}
MIN_URPE_CLUSTER_VOX = 10  # drop residual blobs, as in the reported cluster tables


def com_mni(mask_data, affine):
    com_ijk = ndimage.center_of_mass(mask_data > 0)
    return nib.affines.apply_affine(affine, com_ijk)


def collect_rois():
    """Return list of dicts: roi_set, roi_id, mask_img (native grid), provenance."""
    rois = []
    for roi_set, pattern in [("surprise", "cluster_*_con5_3_1_pos.nii"),
                             ("rpe", "cluster_*_con1_8_0_pos.nii")]:
        for fn in sorted(glob(op.join(MODEL2_ROI, pattern))):
            cluster_nr = int(re.match(r"cluster_(\d+)_", op.basename(fn)).group(1))
            img = nib.load(fn)
            data = np.squeeze(np.nan_to_num(img.get_fdata()))  # some masks are stored 4D
            img3d = nib.Nifti1Image((data > 0).astype(np.uint8), img.affine)
            com = com_mni(data, img.affine)
            rois.append(dict(roi_set=roi_set, roi_id=f"{roi_set}_{cluster_nr}",
                             src=fn, mask_img=img3d, n_vox_native=int((data > 0).sum()),
                             com_x=com[0], com_y=com[1], com_z=com[2],
                             peak_x=np.nan, peak_y=np.nan, peak_z=np.nan, peak_t=np.nan))

    tmap = nib.load(URPE_MAP)
    tdata = np.nan_to_num(tmap.get_fdata())
    labels, n = ndimage.label(tdata > 0, structure=ndimage.generate_binary_structure(3, 2))
    print(f"uRPE main-effect map: {int((tdata > 0).sum())} voxels in {n} components")
    for lab in range(1, n + 1):
        mask = labels == lab
        if mask.sum() < MIN_URPE_CLUSTER_VOX:
            continue
        peak_ijk = np.unravel_index(np.argmax(np.where(mask, tdata, 0)), tdata.shape)
        peak = nib.affines.apply_affine(tmap.affine, peak_ijk)
        com = com_mni(mask, tmap.affine)
        rois.append(dict(roi_set="urpe", roi_id=f"urpe_{lab}", src=URPE_MAP,
                         mask_img=nib.Nifti1Image(mask.astype(np.uint8), tmap.affine),
                         n_vox_native=int(mask.sum()),
                         com_x=com[0], com_y=com[1], com_z=com[2],
                         peak_x=peak[0], peak_y=peak[1], peak_z=peak[2],
                         peak_t=float(tdata[peak_ijk])))
    return rois


def beta_columns(subject_dir):
    """Map beta filename -> (signal, modality, run) from the SPM descrip header."""
    cols = []
    for fn in sorted(glob(op.join(subject_dir, "beta_*.nii"))):
        descrip = nib.load(fn).header["descrip"].tobytes().decode(errors="ignore")
        for key, signal in SIGNALS.items():
            if key in descrip:
                run = int(re.search(r"Sn\((\d+)\)", descrip).group(1))
                modality = "audio" if "Audio" in descrip else "tactile"
                cols.append((fn, signal, modality, run))
                break
    return cols


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rois = collect_rois()

    # Resample all masks once onto the model7 beta grid (nearest neighbour)
    ref = nib.load(op.join(MODEL7, "1stLevel", f"sub-{SUBJECTS[0]}", "beta_0001.nii"))
    for roi in rois:
        res = resample_to_img(roi["mask_img"], ref, interpolation="nearest",
                              force_resample=True, copy_header=True)
        roi["mask"] = np.nan_to_num(res.get_fdata()) > 0
        roi["n_vox_beta_grid"] = int(roi["mask"].sum())

    pd.DataFrame([{k: v for k, v in roi.items() if k not in ("mask_img", "mask")}
                  for roi in rois]).to_csv(
        op.join(out_dir, "learning_signal_roi_info.tsv"), sep="\t", index=False)

    rows = []
    for sub in SUBJECTS:
        sub_dir = op.join(MODEL7, "1stLevel", f"sub-{sub}")
        cols = beta_columns(sub_dir)
        assert len(cols) == 18, f"sub-{sub}: expected 18 modulator betas, got {len(cols)}"
        for fn, signal, modality, run in cols:
            img = nib.load(fn)
            assert np.allclose(img.affine, ref.affine), f"{fn}: grid differs from reference"
            data = img.get_fdata()
            for roi in rois:
                rows.append(dict(subject=sub, roi_set=roi["roi_set"], roi_id=roi["roi_id"],
                                 signal=signal, modality=modality, run=run,
                                 beta=float(np.nanmean(data[roi["mask"]]))))
        print(f"sub-{sub} done ({len(rows)} rows)", flush=True)

    out_tsv = op.join(out_dir, "learning_signal_betas.tsv")
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print("wrote", out_tsv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=op.join(MODEL7, "roi_betas"))
    args = parser.parse_args()
    main(args.out_dir)
