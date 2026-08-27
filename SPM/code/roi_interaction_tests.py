"""Interaction and equivalence tests on the contrast-defined ROIs.

Two questions the overlap figure cannot answer on its own:

1. **Is this region's response to signal A actually different from its response
   to signal B?**  A non-significant one-sample t for B is not evidence that B is
   absent -- absence of evidence is not evidence of absence.  So every pair gets
   three verdicts, not one:
       different    -- the paired difference test is significant (Holm)
       equivalent   -- BF01 > 3, moderate evidence for no difference
       undetermined -- neither; the data cannot tell
   TOST against DELTA_DZ is reported alongside for readers who prefer it.
   DELTA_DZ is the smallest standardised difference this design has 80% power to
   detect at n = 58, so "equivalent" means any true difference is smaller than
   what the study could have found.

2. **Is region A more A-signal-ish than region B is?**  That is the crossover
   interaction, ((s1 - s2) in A) - ((s1 - s2) in B), and it is the only test that
   licenses a double dissociation.  Reported for every ROI pair.

Caveats printed with the results: surprise is a stimulus-onset modulator while
RPE and uRPE are feedback modulators, so those comparisons cross events; and
until the non-orthogonalised GLM exists, any RPE-vs-uRPE comparison in model7 is
biased in uRPE's favour, since uRPE is entered first and keeps the shared
variance.

    srun -c 2 --mem 8G --time 20 --account=zne.uzh \
        $HOME/data/conda/envs/multlearn/bin/python roi_interaction_tests.py \
        --source model7 --variant extent_p1e3
"""

import argparse
import itertools
import os.path as op
import sys

import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from sweep_config import SWEEP_ROOT  # noqa: E402

SIGNALS = ["rpe", "urpe", "surprise"]
PAIRS = [("urpe", "surprise"), ("rpe", "urpe"), ("rpe", "surprise")]
# smallest dz this design has 80% power to detect, two-sided alpha = .05, n = 58
DELTA_DZ = 0.368


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def fdr_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def bf10_ttest(t, n, r=0.707):
    """JZS Bayes factor for a one-sample / paired t (Rouder et al. 2009)."""
    nu = n - 1

    def integrand(g):
        return ((1 + n * g) ** -0.5
                * (1 + t ** 2 / ((1 + n * g) * nu)) ** (-(nu + 1) / 2)
                * r / np.sqrt(2 * np.pi) * g ** -1.5 * np.exp(-r ** 2 / (2 * g)))

    num, _ = integrate.quad(integrand, 1e-12, np.inf, limit=200)
    den = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    return num / den


def tost(d, delta_dz):
    """Two one-sided tests for equivalence to zero within +/- delta_dz * SD."""
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    bound = delta_dz * sd
    t_lo = (d.mean() + bound) / se          # H0: diff <= -bound
    t_hi = (d.mean() - bound) / se          # H0: diff >= +bound
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return max(p_lo, p_hi), bound


def verdict(p_diff, bf01):
    """'equivalent' is judged on the Bayes factor, not Holm-corrected TOST.

    Holm across 40-60 equivalence tests is punishing enough that nothing ever
    clears it, which would report every null as 'undetermined' and hide the cells
    where the data really do favour no difference. BF01 > 3 is the conventional
    'moderate evidence for the null'; the TOST p-values are kept in the table so
    a stricter reader can use them instead."""
    if p_diff < 0.05:
        return "different"
    if bf01 > 3:
        return "equivalent"
    return "undetermined"


def main(source, variant, out_root):
    d = op.join(out_root, variant, source)
    long = pd.read_csv(op.join(d, "subject_values.tsv"), sep="\t")
    wide = long.pivot_table(index=["subject"], columns=["name", "signal"],
                            values="value")
    rois = sorted({n for n, _ in wide.columns})
    n_sub = len(wide)

    # ---- 1. within-ROI pairwise signal contrasts ------------------------
    rows = []
    for roi in rois:
        for a, b in PAIRS:
            diff = (wide[(roi, a)] - wide[(roi, b)]).values
            t, p = stats.ttest_1samp(diff, 0)
            p_tost, bound = tost(diff, DELTA_DZ)
            rows.append(dict(source=source, variant=variant, name=roi,
                             signal_a=a, signal_b=b, n=n_sub,
                             mean_diff=float(diff.mean()),
                             dz=float(diff.mean() / diff.std(ddof=1)),
                             t=float(t), p=float(p), p_tost=float(p_tost),
                             equiv_bound=float(bound),
                             bf01=float(1.0 / bf10_ttest(float(t), n_sub))))
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm(pair["p"].values)
    pair["p_tost_holm"] = holm(pair["p_tost"].values)
    pair["verdict"] = [verdict(a, b) for a, b in
                       zip(pair["p_holm"], pair["bf01"])]
    pair.to_csv(op.join(d, "roi_pair_tests.tsv"), sep="\t", index=False)

    # ---- 2. crossover interactions across ROI pairs ---------------------
    rows = []
    for a, b in PAIRS:
        for ra, rb in itertools.combinations(rois, 2):
            da = (wide[(ra, a)] - wide[(ra, b)]).values
            db = (wide[(rb, a)] - wide[(rb, b)]).values
            inter = da - db
            t, p = stats.ttest_1samp(inter, 0)
            rows.append(dict(source=source, variant=variant, signal_a=a,
                             signal_b=b, roi_a=ra, roi_b=rb, n=n_sub,
                             mean=float(inter.mean()),
                             dz=float(inter.mean() / inter.std(ddof=1)),
                             t=float(t), p=float(p)))
    cross = pd.DataFrame(rows)
    cross["p_fdr"] = fdr_bh(cross["p"].values)
    cross.to_csv(op.join(d, "roi_crossover_tests.tsv"), sep="\t", index=False)

    n_eq = int((pair["verdict"] == "equivalent").sum())
    n_diff = int((pair["verdict"] == "different").sum())
    print(f"[{source}/{variant}] {len(rois)} ROIs, {len(pair)} pairwise tests: "
          f"{n_diff} different, {n_eq} equivalent, "
          f"{len(pair) - n_diff - n_eq} undetermined", flush=True)
    print(f"[{source}/{variant}] {len(cross)} crossover interactions, "
          f"{int((cross['p_fdr'] < 0.05).sum())} significant at FDR .05", flush=True)
    print("wrote", d, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model7")
    ap.add_argument("--variant", default="extent_p1e3")
    ap.add_argument("--out-root", default=op.join(SWEEP_ROOT, "contrast_rois"))
    a = ap.parse_args()
    main(a.source, a.variant, a.out_root)
