"""ROIs defined by the clusters actually found in the six learning-signal
contrasts, and what every signal does inside each of them.

The paper reports three signals on three separately thresholded maps and reads
their non-overlap as regional specificity.  This script builds the table that
tests that directly:

  1. Take the six contrast maps -- RPE, uRPE and surprise, each positive and
     negative -- at ONE common cluster-forming threshold, so no signal gets a
     more permissive threshold than another.
  2. Every surviving connected component becomes an ROI.  Components whose peaks
     are within MERGE_MM of each other are the same territory found by more than
     one contrast, and are merged into a single row.
  3. For each ROI report (a) how much of it each of the six contrast maps covers
     -- the overlap -- and (b) the group one-sample t of all three signals inside
     it, so a region found by one contrast can be checked against the others.

ROI names come from the paper's own cluster tables (appendices con1 / con5 /
urpe_contrasts) by nearest reported peak; anything with no reported peak nearby
falls back to a Talairach gyrus label, whose vocabulary ("Angular Gyrus",
"Precuneus", "Middle Frontal Gyrus") is the one those tables already speak.
Harvard-Oxford is deliberately not used.  (AAL would have been the closer match
still, but gin.cnrs.fr serves it over a certificate that fails verification.)

Outputs to $SWEEP_ROOT/contrast_rois/<source>/ :
    contrast_roi_overlap.tsv   one row per ROI x signal
    contrast_roi_info.tsv      one row per ROI (name, peak, size, source contrasts)
    contrast_rois.nii.gz       the ROI masks, for the figure's slice thumbnails

Run on the cluster:
    srun -c 2 --mem 16G --time 60 --account=zne.uzh \
        $HOME/data/conda/envs/multlearn/bin/python contrast_roi_overlap.py --source model7
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
from sweep_config import ANALYSES, SUBJECTS, SWEEP_ROOT, first_level, snpm_dir, t_str  # noqa: E402

SIGNALS = ["rpe", "urpe", "surprise"]
CONN18 = ndimage.generate_binary_structure(3, 2)

# One threshold for all six contrasts: t at one-tailed p = 1e-4, df = 57.
# High enough that the RPE map breaks into distinguishable components, low enough
# that the surprise map still has some.
REF_T = 3.9756
MIN_VOX = 25          # drop residual specks
MAX_PER_CONTRAST = 8  # keep the table readable; the count dropped is reported
MERGE_MM = 12.0       # peaks closer than this are the same territory

# The paper's own cluster tables (notes/paper/latex/appendices/*.tex).
PAPER_PEAKS = [
    # con1, RPE, model2, T > 8
    ("Caudate / Putamen R", 18, 14, -12), ("Caudate / Putamen L", -16, 14, -12),
    ("Angular gyrus L", -46, -66, 44), ("Visual cortex (mid/lat occipital) L", -46, -82, 2),
    ("vmPFC", 8, 54, -8), ("Visual cortex (lingual/fusiform) L", -22, -88, -12),
    ("Cerebellum R", 42, -70, -40),
    # con5, Shannon surprise, model2, T > 3.1
    ("Precuneus L", -4, -54, 40), ("DLPFC L", -28, 24, 48),
    ("Angular gyrus R", 48, -72, 37), ("Angular gyrus L (surprise)", -36, -82, 40),
    # model7 con1, uRPE, T > 4
    ("dmPFC R", 2, 32, 37), ("Insula R", 36, 24, -5), ("Insula L", -34, 26, -2),
    ("Frontal middle gyrus R", 32, 60, 2), ("Orbital IFG R", 50, 36, -12),
    ("Superior frontal sulcus R", 48, 24, 37), ("Inferior parietal lobule R", 44, -46, 48),
]
PAPER_MATCH_MM = 20.0


def load_map(source, signal, sign, thr):
    key = f"{source}_{signal}"
    d = op.join(SWEEP_ROOT, "snpm", key)
    if not op.isdir(d):
        d = op.join(SWEEP_ROOT, "snpm", f"model7_{signal}")
    fn = op.join(d, f"t{t_str(thr)}_{sign}.nii")
    if not op.exists(fn):
        return None
    img = nib.load(fn)
    return img, np.nan_to_num(np.squeeze(img.get_fdata()))


def gyrus_labeller(ref):
    """Talairach gyrus label of a mask's modal voxel, plus hemisphere from x."""
    atlas = datasets.fetch_atlas_talairach(level_name="gyrus")
    img = resample_to_img(atlas.maps, ref, interpolation="nearest",
                          force_resample=True, copy_header=True)
    data = np.round(np.nan_to_num(img.get_fdata())).astype(int)
    labels = list(atlas.labels)

    def label_at(ijk, mask, peak_mm):
        vals = data[mask]
        vals = vals[vals != 0]
        if vals.size == 0:
            return "Unlabelled"
        name = labels[int(np.bincount(vals).argmax())]
        if name in ("Sub-Gyral", "Unlabelled", "Background"):
            return f"Unlabelled ({peak_mm[0]:.0f}, {peak_mm[1]:.0f}, {peak_mm[2]:.0f})"
        side = "L" if peak_mm[0] < -4 else ("R" if peak_mm[0] > 4 else "")
        return f"{name} {side}".strip()
    return label_at


def name_for(peak_mm, ijk, mask, gyrus_label):
    d = [(np.linalg.norm(np.array(peak_mm) - np.array(p[1:])), p[0]) for p in PAPER_PEAKS]
    dist, nm = min(d)
    if dist <= PAPER_MATCH_MM:
        return nm, round(float(dist), 1)
    return gyrus_label(ijk, mask, peak_mm), np.nan


def main(source, thr, out_root):
    con_of = {}
    for signal in SIGNALS:
        key = f"{source}_{signal}"
        match = [a for a in ANALYSES if a["key"] == key] or \
                [a for a in ANALYSES if a["key"] == f"model7_{signal}"]
        con_of[signal] = (match[0]["model"], match[0]["con"])

    ref = nib.load(first_level(con_of["rpe"][0], SUBJECTS[0], con_of["rpe"][1]))
    gyrus_label = gyrus_labeller(ref)

    tmap = nib.load(op.join(snpm_dir(*con_of["rpe"]), "snpmT+.img"))
    td = np.asarray(tmap.dataobj, dtype=np.float64)
    analysis_mask = np.isfinite(td) & (td != 0)

    # --- 1. the six maps, and their surviving components -------------------
    maps, rois = {}, []
    for signal in SIGNALS:
        for sign in ("pos", "neg"):
            got = load_map(source, signal, sign, thr)
            if got is None:
                maps[(signal, sign)] = None
                continue
            img, data = got
            maps[(signal, sign)] = data != 0
            labels, n = ndimage.label(data != 0, CONN18)
            comps = []
            for lab in range(1, n + 1):
                m = labels == lab
                if m.sum() < MIN_VOX:
                    continue
                pk = np.unravel_index(np.argmax(np.where(m, np.abs(data), 0)), data.shape)
                comps.append(dict(mask=m, size=int(m.sum()), peak_ijk=pk,
                                  peak_t=float(data[pk]),
                                  peak_mm=nib.affines.apply_affine(img.affine, np.array(pk)),
                                  source=f"{signal} {'+' if sign == 'pos' else '−'}"))
            comps.sort(key=lambda c: -c["size"])
            dropped = max(0, len(comps) - MAX_PER_CONTRAST)
            if dropped:
                print(f"[{source}] {signal} {sign}: showing {MAX_PER_CONTRAST} of "
                      f"{len(comps)} components (dropped {dropped})", flush=True)
            rois.extend(comps[:MAX_PER_CONTRAST])

    # --- 2. merge components from different contrasts that are the same place
    merged = []
    for c in sorted(rois, key=lambda c: -c["size"]):
        hit = None
        for m in merged:
            if np.linalg.norm(np.array(c["peak_mm"]) - np.array(m["peak_mm"])) <= MERGE_MM:
                hit = m
                break
        if hit is None:
            c["sources"] = [c["source"]]
            merged.append(c)
        else:
            hit["sources"].append(c["source"])
            hit["mask"] = hit["mask"] | c["mask"]
    print(f"[{source}] {len(rois)} components -> {len(merged)} distinct territories",
          flush=True)

    for r in merged:
        r["mask"] &= analysis_mask
        nm, dist = name_for(r["peak_mm"], r["peak_ijk"], r["mask"], gyrus_label)
        r["name"], r["paper_dist_mm"] = nm, dist

    # disambiguate repeated names
    seen = {}
    for r in merged:
        seen[r["name"]] = seen.get(r["name"], 0) + 1
        if seen[r["name"]] > 1:
            r["name"] = f"{r['name']} ({seen[r['name']]})"

    # --- 3. per-ROI: overlap with each of the six maps, and each signal's t --
    subj = {}
    for signal in SIGNALS:
        model, con = con_of[signal]
        arr = np.stack([np.nan_to_num(nib.load(first_level(model, s, con)).get_fdata())
                        for s in SUBJECTS])
        subj[signal] = arr
        print(f"[{source}] loaded {signal} ({model} con{con})", flush=True)

    rows, info = [], []
    for r in merged:
        m = r["mask"]
        cov = {}
        for (signal, sign), mp in maps.items():
            tag = f"{signal}_{sign}"
            cov[tag] = float((m & mp).sum() / m.sum()) if mp is not None else 0.0
        info.append(dict(name=r["name"], n_vox=int(m.sum()),
                         peak_x=r["peak_mm"][0], peak_y=r["peak_mm"][1],
                         peak_z=r["peak_mm"][2], peak_t=r["peak_t"],
                         found_by=" | ".join(sorted(set(r["sources"]))),
                         paper_dist_mm=r["paper_dist_mm"], **cov))
        for signal in SIGNALS:
            v = subj[signal][:, m].mean(axis=1)
            t, p = stats.ttest_1samp(v, 0)
            rows.append(dict(source=source, name=r["name"], signal=signal,
                             mean=float(v.mean()), sem=float(stats.sem(v)),
                             t=float(t), p=float(p),
                             dz=float(v.mean() / v.std(ddof=1)),
                             cov_pos=cov[f"{signal}_pos"], cov_neg=cov[f"{signal}_neg"]))

    df = pd.DataFrame(rows)
    order = np.argsort(df["p"].values)
    n = len(df)
    adj, running = np.empty(n), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * df["p"].values[idx])
        adj[idx] = min(1.0, running)
    df["p_holm"] = adj

    out_dir = op.join(out_root, source)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(op.join(out_dir, "contrast_roi_overlap.tsv"), sep="\t", index=False)
    pd.DataFrame(info).to_csv(op.join(out_dir, "contrast_roi_info.tsv"), sep="\t", index=False)

    lab = np.zeros(analysis_mask.shape, np.float32)
    for i, r in enumerate(merged, start=1):
        lab[r["mask"]] = i
    img = nib.Nifti1Image(lab, ref.affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(op.join(out_dir, "contrast_rois.nii.gz"))
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["model7", "model2"], default="model7")
    parser.add_argument("--threshold", type=float, default=REF_T)
    parser.add_argument("--out-root", default=op.join(SWEEP_ROOT, "contrast_rois"))
    args = parser.parse_args()
    main(args.source, args.threshold, args.out_root)
