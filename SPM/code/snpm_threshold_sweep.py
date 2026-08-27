"""SnPM half of the threshold sweep: re-run snpm_pp at a range of cluster-forming
thresholds, re-using the permutations that are already on disk.

The 2nd level was computed with SnPM's `ST_later` option (ST_U = -1), so
SnPM_ST.mat holds the suprathreshold "mountain tops" for every permutation down
to ST_Ut = 2.39356756 (= t at one-tailed p = .01, df = 57).  Any cluster-forming
threshold >= ST_Ut can therefore be evaluated by re-running the *inference* step
only -- no 5000-permutation recompute.

For each (analysis, threshold, sign) this writes
    <SnPM dir>/SnPM_filtered_t<t>_<pos|neg>.nii
which is SnPM's cluster-FWE survivor map (only voxels in clusters with
p_FWE < .05 are non-zero), and then copies it to $SWEEP_ROOT/snpm/<key>/.

Side effects on the paper's SnPM directories are limited to *new* SnPM_filtered_*
files (the paper's t2_8 / t3_1 / t4_0 maps use different names and are untouched)
plus the regenerable post-processing files SnPM_pp.mat / SnPM_pp_Neg.mat.
SnPM.mat, SnPM_ST.mat and STCS.mat -- the actual permutation results -- are only
read.

Run on the cluster (needs MATLAB + SPM12/SnPM):
    sbatch --array=0-4 SPM/cluster/submit_threshold_sweep.sh snpm
"""

import argparse
import os
import os.path as op
import shutil
import subprocess
import sys

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from sweep_config import ANALYSES, SWEEP_ROOT, THRESHOLDS, snpm_dir, t_str  # noqa: E402

SPM_PATH = op.expanduser("~/spm12")
MATLAB = os.environ.get("MATLAB_CMD", "matlab")

BATCH = """try
    addpath('{spm}');
    addpath(fullfile('{spm}', 'toolbox', 'snpm'));
    spm('defaults', 'fMRI');
    spm_jobman('initcfg');
    cd('{snpm_dir}');
    clear matlabbatch
    matlabbatch{{1}}.spm.tools.snpm.inference.SnPMmat = cellstr('{snpm_mat}');
    matlabbatch{{1}}.spm.tools.snpm.inference.Thr.Clus.ClusSize.CFth = {thr};
    matlabbatch{{1}}.spm.tools.snpm.inference.Thr.Clus.ClusSize.ClusSig.FWEthC = 0.05;
    matlabbatch{{1}}.spm.tools.snpm.inference.Tsign = {sign};
    matlabbatch{{1}}.spm.tools.snpm.inference.WriteFiltImg.name = '{out}';
    matlabbatch{{1}}.spm.tools.snpm.inference.Report = 'MIPtable';
    spm_jobman('run', matlabbatch);
    fprintf('SWEEP_OK\\n');
catch ME
    fprintf('SWEEP_FAIL %s\\n', ME.message);
end
"""


def run_one(analysis, thr, sign, force=False):
    """Return 'ok' | 'skip' | 'empty' | 'fail:<msg>'."""
    d = snpm_dir(analysis["model"], analysis["con"])
    tag = "pos" if sign == 1 else "neg"
    stem = op.join(d, f"SnPM_filtered_t{t_str(thr)}_{tag}")
    out_nii = stem + ".nii"

    dest_dir = op.join(SWEEP_ROOT, "snpm", analysis["key"])
    os.makedirs(dest_dir, exist_ok=True)
    dest = op.join(dest_dir, f"t{t_str(thr)}_{tag}.nii")

    if op.exists(out_nii) and not force:
        shutil.copyfile(out_nii, dest)
        return "skip"

    script = BATCH.format(spm=SPM_PATH, snpm_dir=d, snpm_mat=op.join(d, "SnPM.mat"),
                          thr=thr, sign=sign, out=stem)
    # MATLAB's -batch chokes on a multi-line argument through the apptainer
    # wrapper, so drop the code in a .m next to the SnPM output (also useful
    # provenance) and run that.
    m_path = op.join(d, f"SnPM_inference_t{t_str(thr)}_{tag}_sweep.m")
    with open(m_path, "w") as fh:
        fh.write(script)
    proc = subprocess.run([MATLAB, "-nodisplay", "-nosplash", "-batch",
                           f"run('{m_path}')"], capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    with open(op.join(dest_dir, f"t{t_str(thr)}_{tag}.log"), "w") as fh:
        fh.write(log)

    if op.exists(out_nii):
        shutil.copyfile(out_nii, dest)
        return "ok"
    if "SWEEP_FAIL" in log:
        msg = log.split("SWEEP_FAIL", 1)[1].strip().splitlines()[0]
        # SnPM's snpm_pp always loops both signs; if a tail has zero voxels above
        # the cluster-forming threshold it dies on an undefined `Locs_vox`.
        if "Locs_vox" in msg:
            return "empty"
        return f"fail:{msg[:120]}"
    return "fail:no output, no error marker"


def main(only=None, force=False, index=None):
    os.makedirs(op.join(SWEEP_ROOT, "snpm"), exist_ok=True)
    analyses = [ANALYSES[index]] if index is not None else ANALYSES
    for analysis in analyses:
        if only and analysis["key"] not in only:
            continue
        for thr, p in THRESHOLDS:
            for sign, tag in ((1, "pos"), (-1, "neg")):
                status = run_one(analysis, thr, sign, force=force)
                print(f"{analysis['key']:18s} t={thr:<7.4f} (p={p:g}) {tag}  ->  {status}",
                      flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these analysis keys")
    parser.add_argument("--force", action="store_true", help="re-run even if the map exists")
    parser.add_argument("--index", type=int, default=None,
                        help="SLURM array index into ANALYSES; omit to run all")
    args = parser.parse_args()
    main(only=args.only, force=args.force, index=args.index)
