"""Figure: does it matter whether the FWE correction comes from SPM or nilearn?

Both engines run the same test -- a sign-flipping permutation test on the same 58
first-level contrast images, with the same analysis mask (SnPM's own), 5000
permutations each.  Neither uses a parametric random-field approximation.  So the
question is not "permutation or not" but what the remaining choices do:

  * SnPM tests each tail as its own one-tailed family (`Tsign`), which is what the
    paper reports; nilearn's `two_sided_test=True` controls the familywise error
    over both tails at once, and is therefore about twice as strict in the tail.
  * SnPM's cluster test uses extent only.  nilearn also offers cluster mass, and
    two threshold-free corrections -- voxel-wise max-t, and TFCE.
  * SPM labels clusters with 18-connectivity; nilearn hardcodes 6.  These runs
    patch nilearn to 18 so that difference is not in play here.

    ~/mambaforge/bin/python notes/figures/fig_engines.py
"""

import argparse
import os.path as op
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, op.dirname(op.abspath(__file__)))
import plot_threshold_sweep as P  # noqa: E402

NOTES = op.dirname(op.dirname(op.abspath(__file__)))
MASK_VOX = 59838

SIGNALS = ["rpe", "urpe", "surprise"]
LABEL = {"rpe": "Signed RPE", "urpe": "Unsigned RPE", "surprise": "Shannon surprise"}
COLOR = {"rpe": "#C44E52", "urpe": "#2E7D32", "surprise": "#3B5BA5"}

METHODS = [
    ("SnPM, cluster extent", "snpm", "size", "-", "o"),
    ("nilearn, cluster extent", "nilearn", "size", "--", "s"),
    ("nilearn, cluster mass", "nilearn", "mass", ":", "^"),
]
FREE = [("TFCE", "tfce", "D"), ("Voxel max-t", "t", "P")]

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})


def n_vox(img):
    if img is None:
        return np.nan
    return int((np.nan_to_num(img.get_fdata()) != 0).sum())


def pct(img):
    n = n_vox(img)
    return np.nan if np.isnan(n) else 100 * n / MASK_VOX


def collect(source):
    rows = {}
    for signal in SIGNALS:
        for sign in ("pos", "neg"):
            for _, engine, stat, _, _ in METHODS:
                vals = []
                for thr, p in P.THRESHOLDS:
                    img = (P.load_snpm(source, signal, thr, sign) if engine == "snpm"
                           else P.load_nilearn(source, signal, thr, sign, stat))
                    vals.append(pct(img))
                rows[(signal, sign, engine, stat)] = vals
            for _, stat, _ in FREE:
                img = P.load_nilearn(source, signal, 1e-6, sign, stat)
                rows[(signal, sign, "nilearn", stat)] = pct(img)
    return rows


def dice(source):
    out = []
    for signal in SIGNALS:
        for sign in ("pos", "neg"):
            for thr, p in P.THRESHOLDS:
                a = P.load_snpm(source, signal, thr, sign)
                b = P.load_nilearn(source, signal, thr, sign, "size")
                if a is None or b is None:
                    continue
                A = np.nan_to_num(a.get_fdata()) != 0
                B = np.nan_to_num(b.get_fdata()) != 0
                if A.sum() + B.sum() == 0:
                    continue
                out.append(dict(signal=signal, sign=sign, p=p,
                                dice=2 * (A & B).sum() / (A.sum() + B.sum()),
                                snpm=int(A.sum()), nil=int(B.sum())))
    return out


def main(source, out_stem):
    rows = collect(source)
    x = np.array([-np.log10(p) for _, p in P.THRESHOLDS])
    xfree = x.max() + np.array([0.9, 1.7])

    fig = plt.figure(figsize=(7.25, 6.4))
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.985, top=0.865, bottom=0.30,
                          hspace=0.55, wspace=0.16)
    fig.text(0.005, 0.985,
             "Both engines are permutation tests — what actually differs is the "
             "family and the statistic",
             fontsize=10.5, fontweight="bold", va="top", color="0.08")
    fig.text(0.005, 0.951,
             f"{source} · n = 58 · 5000 permutations each · same contrast images, "
             f"same mask ({MASK_VOX:,} voxels), nilearn patched to 18-connectivity",
             fontsize=7.5, va="top", color="0.42")

    for i, signal in enumerate(SIGNALS):
        for j, sign in enumerate(("pos", "neg")):
            ax = fig.add_subplot(gs[i, j])
            for name, engine, stat, ls, mk in METHODS:
                y = rows[(signal, sign, engine, stat)]
                ax.plot(x, y, ls=ls, marker=mk, ms=3.4, lw=1.1,
                        color=COLOR[signal],
                        mfc=COLOR[signal] if engine == "snpm" else "white",
                        mec=COLOR[signal], mew=0.9, clip_on=False, zorder=3)
            for (name, stat, mk), xf in zip(FREE, xfree):
                ax.plot([xf], [rows[(signal, sign, "nilearn", stat)]], marker=mk,
                        ms=4.2, color=COLOR[signal], mfc="white",
                        mec=COLOR[signal], mew=0.9, clip_on=False, zorder=3)
            ax.axvline(x.max() + 0.45, color="0.85", lw=0.6, zorder=0)
            ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 0.1, 1, 10, 100])
            ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
            ax.set_xticks(list(x) + list(xfree))
            # p<.01 and p<.005 are only 0.3 apart on a -log10 axis, so tilt
            ax.set_xticklabels(["1e-2", "5e-3", "1e-3", "1e-4", "1e-6",
                                "TFCE", "Vox"], fontsize=6.2, rotation=35,
                               ha="right", rotation_mode="anchor")
            ax.set_title(f"{LABEL[signal]} — {'positive' if sign == 'pos' else 'negative'}",
                         fontsize=8, color=COLOR[signal], pad=3)
            if j == 0:
                ax.set_ylabel("% of mask", fontsize=7.5)
            if i == 2:
                ax.set_xlabel("Cluster-forming threshold (one-tailed p)", fontsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    # a compact key, drawn once, off the data
    ax0 = fig.axes[0]
    for k, (name, engine, stat, ls, mk) in enumerate(METHODS):
        ax0.plot([], [], ls=ls, marker=mk, ms=3.4, lw=1.1, color="0.35",
                 mfc="0.35" if engine == "snpm" else "white", mec="0.35",
                 label=name)
    for name, stat, mk in FREE:
        ax0.plot([], [], ls="none", marker=mk, ms=4.2, color="0.35", mfc="white",
                 mec="0.35", label=name + " (threshold-free)")
    h, lab = ax0.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.928), ncol=5,
               fontsize=6.5, handlelength=2.2, columnspacing=1.6, frameon=False)

    d = dice(source)
    exact = sum(1 for r in d if r["dice"] > 0.995)
    zero = [r for r in d if r["dice"] == 0]
    lines = [
        f"Across the {len(d)} signal × sign × threshold panels where both engines",
        f"return a map, {exact} agree to the voxel (Dice > 0.995). Where they",
        "disagree it is always a marginal effect, and the cause is the family,",
        "not the software:",
        "",
    ]
    for r in sorted(d, key=lambda r: r["dice"])[:3]:
        if r["dice"] > 0.995:
            continue
        lines.append(f"   {LABEL[r['signal']]} {r['sign']}, p < {r['p']:g}:  "
                     f"SnPM {r['snpm']:,} vox vs nilearn {r['nil']:,}")
    right = [
        "SnPM's one-tailed-per-tail family is the more permissive of the two.",
        "",
        "The Shannon-surprise clusters sit right on the boundary: their nilearn",
        "two-tailed cluster-EXTENT p is .071 and .054 — not significant — while",
        "cluster MASS gives .048 and .021, and SnPM's one-tailed family puts",
        "them under .05. Same data, three defensible procedures, opposite",
        "verdicts.",
        "",
        "The RPE and uRPE maps are indifferent to all of this. The surprise map",
        "is not, and neither is the small uRPE-positive set.",
    ]
    fig.text(0.52, 0.235, "\n".join(right), fontsize=7.4, va="top", color="0.18",
             linespacing=1.75)
    fig.text(0.005, 0.235, "\n".join(lines), fontsize=7.4, va="top", color="0.18",
             linespacing=1.75)


    # no bbox_inches="tight": the text blocks would otherwise stretch the page
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".svg")
    plt.close(fig)
    print("wrote", out_stem + ".pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model7")
    ap.add_argument("--outdir", default=op.join(NOTES, "figures"))
    a = ap.parse_args()
    main(a.source, op.join(a.outdir, f"fig_engines_{a.source}"))
