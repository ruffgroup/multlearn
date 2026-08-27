"""Shared configuration for the learning-signal threshold sweep.

The sweep asks how much the reported RPE / uRPE / surprise "specificity" depends
on the (arbitrary) cluster-forming threshold and on the inference engine.

Two model sources are swept, because the manuscript mixes them:

    model2 : RPE and surprise as reported in the paper (surprise + value at
             stimulus onset, signed RPE at feedback; NO uRPE in the design)
    model7 : all three modulators in ONE design, so the three maps are
             apples-to-apples (surprise, value | urpe, rpe)

Cluster-forming thresholds are exact t values for one-tailed p at df = 57
(n = 58 subjects), rather than the FSL z-conventions 2.3 / 2.6 / 3.1.
Conveniently SnPM's stored suprathreshold floor for these analyses is
ST_Ut = 2.39356756 = t(p = .01, df = 57), so the whole sweep can be run by
re-running snpm_pp on the existing permutations -- no re-permutation needed.
"""

import os.path as op

DF = 57
N_SUBJECTS = 58
SUBJECTS = [f"{s:02d}" for s in range(1, 65) if s not in (8, 13, 16, 31, 32, 44)]

BASE = "/shares/zne.uzh/multlearn/nipype"
SWEEP_ROOT = "/shares/zne.uzh/multlearn/threshold_sweep"

# (t, one-tailed p) -- t = scipy.stats.t.isf(p, 57)
THRESHOLDS = [
    (2.3936, 1e-2),
    (2.6649, 5e-3),
    (3.2395, 1e-3),
    (3.9756, 1e-4),
    (5.2929, 1e-6),
]

# label -> (model, contrast number, event, manuscript colour)
# model7 contrast numbers from nipype_helpers.get_contrasts_model7
# model2 contrast numbers: con1 = rpe, con5 = surprise (as reported in the paper)
ANALYSES = [
    dict(key="model7_rpe", model="model7", con=19, signal="rpe"),
    dict(key="model7_urpe", model="model7", con=1, signal="urpe"),
    dict(key="model7_surprise", model="model7", con=5, signal="surprise"),
    dict(key="model2_rpe", model="model2", con=1, signal="rpe"),
    dict(key="model2_surprise", model="model2", con=5, signal="surprise"),
    # uRPE only exists in model7; the "model2" PDF re-uses this panel and says so.
]

SIGNAL_LABEL = {
    "rpe": "Signed RPE",
    "urpe": "Unsigned RPE",
    "surprise": "Shannon surprise",
}

# Manuscript colours (from notes/figures/plot_figureS_dissociation.py)
SIGNAL_COLOR = {
    "rpe": "#C44E52",
    "urpe": "#2E7D32",
    "surprise": "#3B5BA5",
}


def t_str(t):
    """SnPM's filename convention: 3.2395 -> '3_2395'."""
    return f"{t:g}".replace(".", "_")


def snpm_dir(model, con):
    return op.join(BASE, model, "2ndLevel", f"cluster_SnPM_SecondLevel_con{con}")


def first_level(model, sub, con):
    return op.join(BASE, model, "1stLevel", f"sub-{sub}", f"con_{con:04d}.nii")
