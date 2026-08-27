"""Anatomical overview: for every relevant region, does it code positively or
negatively for RPE, uRPE and surprise?

The whole-brain sweep answers "how much does the picture move when you move the
threshold".  This answers the complementary question: take a fixed, *threshold-
independent* anatomical parcellation, keep the parcels that any of the three
learning signals actually touches, and report the sign and strength of all three
signals in every one of them.  Because the parcels are anatomical rather than
defined by one of the contrasts, no signal gets a home-field advantage -- which
is exactly the selection bias that makes separately-thresholded maps look
specific.

Parcellation : Harvard-Oxford maxprob-thr25-2mm, cortical (split into left and
               right) plus subcortical, resampled to the contrast grid.
Relevance    : a parcel is kept if at least --min-sig-vox of its voxels are
               inside a cluster-FWE surviving cluster of ANY signal in EITHER
               sign, at the reference cluster-forming threshold.
Signals      : the per-subject contrast images that the 2nd level itself tests
               (already averaged over runs and modalities), one-sample t-tested
               within each parcel.

Outputs to $SWEEP_ROOT/roi_matrix/<source>/ :
    roi_sign_matrix.tsv        one row per parcel x signal
    roi_parcels.nii.gz         the kept parcels, for the figure's slice thumbnails
    roi_parcel_info.tsv        parcel id, name, volume, centre of mass, peak slice

Run on the cluster:
    srun -c 2 --mem 16G --time 60 --account=zne.uzh \
        $HOME/data/conda/envs/multlearn/bin/python roi_sign_matrix.py --source model7
"""

import argparse
import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import resample_to_img
from scipy import ndimage, stats

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from sweep_config import (ANALYSES, SUBJECTS, SWEEP_ROOT, first_level,  # noqa: E402
                          snpm_dir, t_str)

REF_THRESHOLD = 3.2395  # t at one-tailed p = .001, df = 57


def build_parcellation(ref_img):
    """Harvard-Oxford cortical (L/R split) + subcortical on the contrast grid."""
    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

    cort_img = resample_to_img(cort.maps, ref_img, interpolation="nearest",
                               force_resample=True, copy_header=True)
    sub_img = resample_to_img(sub.maps, ref_img, interpolation="nearest",
                              force_resample=True, copy_header=True)
    cort_d = np.round(np.nan_to_num(cort_img.get_fdata())).astype(int)
    sub_d = np.round(np.nan_to_num(sub_img.get_fdata())).astype(int)

    # x coordinate of every voxel, to split the bilateral cortical labels
    i, j, k = np.indices(cort_d.shape)
    x = nib.affines.apply_affine(ref_img.affine, np.stack([i, j, k], -1).reshape(-1, 3))
    x = x[:, 0].reshape(cort_d.shape)

    parcels = np.zeros(cort_d.shape, int)
    names = {}
    next_id = 1
    for lab, name in enumerate(cort.labels):
        if lab == 0:
            continue
        for side, sel in (("L", x < 0), ("R", x >= 0)):
            mask = (cort_d == lab) & sel
            if mask.sum() == 0:
                continue
            parcels[mask] = next_id
            names[next_id] = f"{name} {side}"
            next_id += 1
    # subcortical labels are already lateralised; they win over cortex on overlap
    for lab, name in enumerate(sub.labels):
        if lab == 0 or any(skip in name for skip in
                           ("Cerebral Cortex", "Cerebral White Matter",
                            "Lateral Ventricle")):
            continue
        mask = sub_d == lab
        if mask.sum() == 0:
            continue
        parcels[mask] = next_id
        names[next_id] = name.replace("Left ", "").replace("Right ", "") + \
            (" L" if "Left" in name else (" R" if "Right" in name else ""))
        next_id += 1
    return parcels, names


def load_survivors(source, signal, thr):
    """SnPM cluster-FWE survivor masks (pos, neg) for one signal at one CFT."""
    key = f"{source}_{signal}"
    if not any(a["key"] == key for a in ANALYSES):
        key = f"model7_{signal}"  # uRPE only exists in model7
    d = op.join(SWEEP_ROOT, "snpm", key)
    out = {}
    for tag in ("pos", "neg"):
        fn = op.join(d, f"t{t_str(thr)}_{tag}.nii")
        if op.exists(fn):
            data = np.nan_to_num(np.squeeze(nib.load(fn).get_fdata()))
            out[tag] = np.abs(data) > 0
        else:
            out[tag] = None
    return out


def main(source, min_sig_vox, thr, out_root):
    signals = ["rpe", "urpe", "surprise"]
    con_of = {}
    for signal in signals:
        key = f"{source}_{signal}"
        match = [a for a in ANALYSES if a["key"] == key]
        if not match:  # uRPE is not in model2's design; borrow model7's
            match = [a for a in ANALYSES if a["key"] == f"model7_{signal}"]
        con_of[signal] = (match[0]["model"], match[0]["con"])

    ref = nib.load(first_level(con_of["rpe"][0], SUBJECTS[0], con_of["rpe"][1]))
    parcels, names = build_parcellation(ref)

    # analysis mask, as used by SnPM
    tmap = nib.load(op.join(snpm_dir(*con_of["rpe"]), "snpmT+.img"))
    tdata = np.asarray(tmap.dataobj, dtype=np.float64)
    analysis_mask = np.isfinite(tdata) & (tdata != 0)
    parcels = np.where(analysis_mask, parcels, 0)

    # which parcels does any signal touch?
    surv = {s: load_survivors(source, s, thr) for s in signals}
    touched = np.zeros(parcels.shape, bool)
    for s in signals:
        for tag in ("pos", "neg"):
            if surv[s][tag] is not None:
                touched |= surv[s][tag]

    keep = []
    for pid in sorted(set(np.unique(parcels)) - {0}):
        mask = parcels == pid
        if mask.sum() < 30:          # too small in-mask to average meaningfully
            continue
        n_sig = int((mask & touched).sum())
        if n_sig >= min_sig_vox:
            keep.append((pid, mask, n_sig))
    print(f"[{source}] {len(keep)} of {len(names)} parcels touched by >= "
          f"{min_sig_vox} surviving voxels at t = {thr}", flush=True)

    # per-subject parcel means of each signal's contrast image
    rows, info = [], []
    subj_means = {s: np.zeros((len(SUBJECTS), len(keep))) for s in signals}
    for signal in signals:
        model, con = con_of[signal]
        for si, sub in enumerate(SUBJECTS):
            data = np.nan_to_num(nib.load(first_level(model, sub, con)).get_fdata())
            for pi, (_, mask, _) in enumerate(keep):
                subj_means[signal][si, pi] = data[mask & analysis_mask].mean()
        print(f"[{source}] extracted {signal} ({model} con{con})", flush=True)

    for pi, (pid, mask, n_sig) in enumerate(keep):
        com = nib.affines.apply_affine(ref.affine, ndimage.center_of_mass(mask))
        info.append(dict(parcel_id=pid, name=names[pid], n_vox=int(mask.sum()),
                         n_sig_vox=n_sig, com_x=com[0], com_y=com[1], com_z=com[2]))
        for signal in signals:
            v = subj_means[signal][:, pi]
            t, p = stats.ttest_1samp(v, 0)
            frac = {}
            for tag in ("pos", "neg"):
                m = surv[signal][tag]
                frac[tag] = float((mask & m).sum() / mask.sum()) if m is not None else np.nan
            rows.append(dict(source=source, parcel_id=pid, name=names[pid],
                             signal=signal, mean=float(v.mean()),
                             sem=float(stats.sem(v)), t=float(t), p=float(p),
                             dz=float(v.mean() / v.std(ddof=1)),
                             frac_sig_pos=frac["pos"], frac_sig_neg=frac["neg"]))

    df = pd.DataFrame(rows)
    # Holm across all parcel x signal tests
    order = np.argsort(df["p"].values)
    m = len(df)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * df["p"].values[idx])
        adj[idx] = min(1.0, running)
    df["p_holm"] = adj

    out_dir = op.join(out_root, source)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(op.join(out_dir, "roi_sign_matrix.tsv"), sep="\t", index=False)
    pd.DataFrame(info).to_csv(op.join(out_dir, "roi_parcel_info.tsv"), sep="\t", index=False)

    kept_img = np.zeros(parcels.shape, np.float32)
    for pid, mask, _ in keep:
        kept_img[mask] = pid
    img = nib.Nifti1Image(kept_img, ref.affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(op.join(out_dir, "roi_parcels.nii.gz"))
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["model7", "model2"], default="model7")
    parser.add_argument("--min-sig-vox", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=REF_THRESHOLD)
    parser.add_argument("--out-root", default=op.join(SWEEP_ROOT, "roi_matrix"))
    args = parser.parse_args()
    main(args.source, args.min_sig_vox, args.threshold, args.out_root)
