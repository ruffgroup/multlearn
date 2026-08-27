"""Create first-level con_0024.nii = uRPE - RPE for every model7 subject.

model7's contrast list (nipype_helpers.get_contrasts_model7) stops at con23 and
contains no direct uRPE-vs-RPE contrast. Because SPM contrast images are linear
in the betas, the difference contrast [1/6, 1/6, -1/6, -1/6] over
(FeedbackAudioxurpe, FeedbackTactilexurpe, FeedbackAudioxrpe, FeedbackTactilexrpe)
is exactly con_0001 (urpe, weights 1/6) minus con_0019 (rpe, weights 1/6) --
no MATLAB/EstimateContrast rerun needed. The resulting con_0024.nii files feed
the standard SnPM second level:  sbatch --array=24 submit_GLM_2ndlevel_nipype.sh

Note: rpe is the second feedback modulator, so SPM's default within-condition
serial orthogonalisation means the rpe betas are orthogonalised w.r.t. urpe
(shared |RPE| variance is credited to urpe). State this caveat wherever the
con24 map is reported.
"""

import os.path as op

import nibabel as nib
import numpy as np

MODEL7 = "/shares/zne.uzh/multlearn/nipype/model7/1stLevel"
SUBJECTS = [f"{s:02d}" for s in range(1, 65) if s not in (8, 13, 16, 31, 32, 44)]

for sub in SUBJECTS:
    d = op.join(MODEL7, f"sub-{sub}")
    urpe = nib.load(op.join(d, "con_0001.nii"))
    rpe = nib.load(op.join(d, "con_0019.nii"))
    assert np.allclose(urpe.affine, rpe.affine)
    diff = urpe.get_fdata() - rpe.get_fdata()
    img = nib.Nifti1Image(diff.astype(np.float32), urpe.affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.header["descrip"] = b"urpe - rpe (con_0001 - con_0019)"
    out = op.join(d, "con_0024.nii")
    img.to_filename(out)
    print(out, flush=True)

# Read-back sanity check on the last subject (dtype trap guard)
chk = nib.load(out)
print("dtype:", chk.get_data_dtype(),
      "unique(3dp):", len(np.unique(np.round(chk.get_fdata()[np.isfinite(chk.get_fdata())], 3))))
