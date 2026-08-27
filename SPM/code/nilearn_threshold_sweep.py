"""nilearn half of the threshold sweep: permutation inference on the same
first-level contrast images that SnPM used, with cluster-extent, cluster-mass,
voxel-wise FWE and TFCE.

Why a second engine at all: SnPM gives us the paper's own inference, but it
cannot do TFCE and it tests each tail as a separate one-tailed family.
`nilearn.glm.second_level.non_parametric_inference` gives, from one sign-flipping
permutation run, cluster-extent FWE, cluster-mass FWE, voxel-wise (max-t) FWE and
TFCE, with a proper two-tailed family.

Three engine differences are deliberately controllable so the comparison can
attribute the discrepancy:

  1. Cluster connectivity.  SPM/SnPM label clusters with 18-connectivity
     (`spm_clusters`); nilearn hardcodes 6-connectivity
     (`generate_binary_structure(3, 1)`).  `--connectivity 18` monkeypatches the
     nilearn module so it matches SPM.  The patch cannot survive joblib's loky
     workers, so that mode forces `n_jobs = 1` (still only ~minutes per run:
     the expensive part is TFCE, which is run separately).
  2. One- vs two-tailed FWE family.  SnPM's `Tsign` runs each tail as its own
     one-tailed family; `--two-sided` (default) controls FWE over both tails at
     once, which is the more defensible -- and stricter -- choice.
  3. The analysis mask.  We reuse SnPM's own mask (the finite voxels of
     snpmT+.img) so voxel counts are directly comparable between engines.

Outputs (all float32, gzipped) go to $SWEEP_ROOT/nilearn/<key>/ :
    t.nii.gz                     group t map (two-tailed, signed)
    logp_max_t.nii.gz            -log10 p, voxel-wise FWE
    logp_max_size_t<thr>.nii.gz  -log10 p, cluster-extent FWE at that CFT
    logp_max_mass_t<thr>.nii.gz  -log10 p, cluster-mass FWE at that CFT
    tfce.nii.gz / logp_max_tfce.nii.gz

Run on the cluster:
    sbatch --array=0-4 SPM/cluster/submit_threshold_sweep.sh nilearn
"""

import argparse
import json
import os
import os.path as op
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from sweep_config import (ANALYSES, SUBJECTS, SWEEP_ROOT, THRESHOLDS,  # noqa: E402
                          first_level, snpm_dir, t_str)


def patch_connectivity_18():
    """Make nilearn label clusters the way SPM does (18-connectivity)."""
    from scipy.ndimage import generate_binary_structure
    from nilearn.mass_univariate import permuted_least_squares as pls

    pls.generate_binary_structure = lambda rank, conn: generate_binary_structure(3, 2)


def save(img, path):
    """Write float32 with an identity scaling -- see the NIfTI dtype trap: a
    uint8 mask makes inverse_transform quantise everything to 256 levels."""
    data = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float64)).astype(np.float32)
    out = nib.Nifti1Image(data, img.affine)
    out.set_data_dtype(np.float32)
    out.header.set_slope_inter(slope=1, inter=0)
    out.to_filename(path)


def snpm_mask(model, con):
    """SnPM's own analysis mask: the voxels where snpmT+.img is finite."""
    img = nib.load(op.join(snpm_dir(model, con), "snpmT+.img"))
    data = np.asarray(img.dataobj, dtype=np.float64)
    mask = np.isfinite(data) & (data != 0)
    out = nib.Nifti1Image(mask.astype(np.int8), img.affine)
    out.set_data_dtype(np.int8)
    return out, int(mask.sum())


def run_analysis(analysis, n_perm, n_jobs, connectivity, two_sided, out_root,
                 skip_tfce=False, only_tfce=False):
    from nilearn.glm.second_level import non_parametric_inference

    key = analysis["key"]
    out_dir = op.join(out_root, key)
    os.makedirs(out_dir, exist_ok=True)

    imgs = [first_level(analysis["model"], s, analysis["con"]) for s in SUBJECTS]
    missing = [f for f in imgs if not op.exists(f)]
    assert not missing, f"{key}: missing {len(missing)} contrast images, e.g. {missing[0]}"

    mask, n_mask_vox = snpm_mask(analysis["model"], analysis["con"])
    design = pd.DataFrame({"intercept": np.ones(len(imgs))})
    print(f"[{key}] {len(imgs)} subjects, {n_mask_vox} mask voxels, "
          f"n_perm={n_perm}, n_jobs={n_jobs}, connectivity={connectivity}, "
          f"two_sided={two_sided}", flush=True)

    meta = dict(key=key, model=analysis["model"], con=analysis["con"],
                signal=analysis["signal"], n_subjects=len(imgs),
                n_mask_voxels=n_mask_vox, n_perm=n_perm,
                connectivity=connectivity, two_sided=two_sided, timings={})

    common = dict(design_matrix=design, second_level_contrast="intercept",
                  mask=mask, n_perm=n_perm, two_sided_test=two_sided,
                  random_state=42, n_jobs=n_jobs, verbose=0, smoothing_fwhm=None)

    if not only_tfce:
        for thr, p in THRESHOLDS:
            tag = t_str(thr)
            done = op.join(out_dir, f"logp_max_mass_t{tag}.nii.gz")
            if op.exists(done):
                print(f"[{key}] t={thr} exists, skipping", flush=True)
                continue
            t0 = time.time()
            out = non_parametric_inference(imgs, threshold=thr, tfce=False, **common)
            meta["timings"][f"cluster_t{tag}"] = round(time.time() - t0, 1)
            save(out["t"], op.join(out_dir, "t.nii.gz"))
            save(out["logp_max_t"], op.join(out_dir, "logp_max_t.nii.gz"))
            save(out["logp_max_size"], op.join(out_dir, f"logp_max_size_t{tag}.nii.gz"))
            save(out["logp_max_mass"], op.join(out_dir, f"logp_max_mass_t{tag}.nii.gz"))
            print(f"[{key}] t={thr:.4f} (p={p:g}) done in "
                  f"{meta['timings'][f'cluster_t{tag}']}s", flush=True)

    if not skip_tfce and not op.exists(op.join(out_dir, "logp_max_tfce.nii.gz")):
        t0 = time.time()
        out = non_parametric_inference(imgs, threshold=None, tfce=True, **common)
        meta["timings"]["tfce"] = round(time.time() - t0, 1)
        save(out["t"], op.join(out_dir, "t.nii.gz"))
        save(out["logp_max_t"], op.join(out_dir, "logp_max_t.nii.gz"))
        save(out["tfce"], op.join(out_dir, "tfce.nii.gz"))
        save(out["logp_max_tfce"], op.join(out_dir, "logp_max_tfce.nii.gz"))
        print(f"[{key}] TFCE done in {meta['timings']['tfce']}s", flush=True)

    meta_path = op.join(out_dir, "meta.json")
    if op.exists(meta_path):
        old = json.load(open(meta_path))
        old["timings"].update(meta["timings"])
        meta = old
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                        help="SLURM array index into ANALYSES; omit to run all")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--connectivity", type=int, choices=[6, 18], default=18)
    parser.add_argument("--one-sided", action="store_true",
                        help="one-tailed family per tail, as SnPM does (positive tail only)")
    parser.add_argument("--skip-tfce", action="store_true")
    parser.add_argument("--only-tfce", action="store_true")
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    if args.connectivity == 18:
        patch_connectivity_18()
        if args.n_jobs != 1:
            print("connectivity=18 forces n_jobs=1 (the patch does not reach loky workers)")
            args.n_jobs = 1

    out_root = args.out_root or op.join(SWEEP_ROOT, "nilearn",
                                        f"conn{args.connectivity}")
    analyses = ANALYSES
    if args.index is not None:
        analyses = [ANALYSES[args.index]]
    if args.only:
        analyses = [a for a in analyses if a["key"] in args.only]

    for analysis in analyses:
        run_analysis(analysis, n_perm=args.n_perm, n_jobs=args.n_jobs,
                     connectivity=args.connectivity, two_sided=not args.one_sided,
                     out_root=out_root, skip_tfce=args.skip_tfce,
                     only_tfce=args.only_tfce)
