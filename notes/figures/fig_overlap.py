"""Figure: where the three learning-signal maps overlap.

One story: the regions attributed to Shannon surprise, and every region where
unsigned RPE goes negative, sit INSIDE the signed-RPE map.  The only clusters
that fall outside it are the uRPE-positive frontal and insular regions.  So the
reported dissociation survives in one place and dissolves everywhere else.

ROIs are the clusters actually found in the six contrasts (RPE, uRPE and
surprise, each tail) at one common cluster-forming threshold, so no signal gets
a more permissive threshold than another.  Names follow the paper's own cluster
tables; anything with no reported peak nearby falls back to a Talairach gyrus
label (`SPM/code/contrast_roi_overlap.py`).

Target medium: journal double column, 180 mm.  Regenerate at a different width
rather than scaling the PDF.

    ~/mambaforge/bin/python notes/figures/fig_overlap.py
"""

import argparse
import os.path as op

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from nilearn import plotting  # noqa: E402
from nilearn.image import resample_img  # noqa: E402

NOTES = op.dirname(op.dirname(op.abspath(__file__)))
SWEEP = op.join(NOTES, "data", "threshold_sweep")
BG_1MM = op.join(NOTES, "data", "surprise_vs_rpe", "MNI152_T1_1mm.nii.gz")

SIGNALS = ["rpe", "urpe", "surprise"]
LABEL = {"rpe": "RPE", "urpe": "uRPE", "surprise": "Surprise"}
COLOR = {"rpe": "#C44E52", "urpe": "#2E7D32", "surprise": "#3B5BA5"}
REF_T = 3.9756          # one-tailed p < 1e-4, df = 57 -- the same for all six maps
GROUP_ORDER = ["urpe +", "rpe +", "urpe −", "surprise +", "rpe −", "surprise −"]

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})


def t_str(t):
    return f"{t:g}".replace(".", "_")


ACRONYMS = {"dmpfc": "dmPFC", "vmpfc": "vmPFC", "dlpfc": "DLPFC", "dacc": "dACC",
            "ifg": "IFG", "mfg": "MFG", "sfs": "SFS", "pfc": "PFC", "r": "R", "l": "L"}


def tidy_name(name):
    """Sentence case, but leave the acronyms the paper uses alone.

    Talairach hands back Title Case and the paper's tables use sentence case; the
    two collide on rows like 'Inferior Parietal Lobule R' vs 'Inferior parietal
    lobule R', which are the same region found by two different contrasts and
    read as a typo unless harmonised."""
    out, first = [], True
    for w in name.split():
        key = w.strip("()/,").lower()
        if key in ACRONYMS:
            out.append(w.replace(w.strip("()/,"), ACRONYMS[key]))
        elif w.startswith("(") or not w[:1].isalpha():
            out.append(w)
        else:
            out.append(w.capitalize() if first else w.lower())
        if w[:1].isalpha():
            first = False
    return " ".join(out)


def load(source):
    d = op.join(SWEEP, "contrast_rois", source)
    info = pd.read_csv(op.join(d, "contrast_roi_info.tsv"), sep="\t")
    long = pd.read_csv(op.join(d, "contrast_roi_overlap.tsv"), sep="\t")
    # tidy, then re-disambiguate: harmonising the casing can collapse two
    # genuinely different ROIs (the paper's "Inferior parietal lobule R" and
    # Talairach's "Inferior Parietal Lobule R") onto one label
    tidy, seen = {}, {}
    for n in info["name"]:
        base = tidy_name(n)
        seen[base] = seen.get(base, 0) + 1
        tidy[n] = base if seen[base] == 1 else f"{base} ({seen[base]})"
    info["name"] = info["name"].map(tidy)
    long["name"] = long["name"].map(tidy)
    # The organising split IS the finding: does the signed-RPE map cover this ROI?
    info["outside"] = info["rpe_pos"] < 0.05
    # a territory can be found by more than one contrast (that is the point);
    # order on the first-listed source, but keep the full list for the label
    info["primary"] = info["found_by"].str.split(" | ", regex=False).str[0]
    info["group"] = pd.Categorical(info["primary"], categories=GROUP_ORDER, ordered=True)
    info = info.sort_values(["outside", "group", "n_vox"],
                            ascending=[False, True, False]).reset_index(drop=True)
    info["found_by"] = info["found_by"].astype(str)
    info["roi_id"] = np.arange(1, len(info) + 1)   # keys the brain maps to the rows
    return info, long


# --------------------------------------------------------------------------
# Geometry: x in axes-like units 0-1, y one unit per row with gaps between
# groups. Every marker is sized in POINTS, so nothing is distorted by the
# panel's aspect ratio -- drawing these as data-space rectangles makes squares
# come out as wide rectangles.

X_LABEL = 0.055          # left edge of the ROI name
X_COV0 = 0.475           # first "covered by map" column
X_STEP = 0.052
X_EFF0 = 0.815           # first "effect in ROI" column
COV_MS = 9.5             # side of a 100%-coverage square, in points


def row_layout(info):
    """y for every row, plus (label, y0, y1) for each found-by group and each
    covered/not-covered super-group."""
    ys, groups, supers = [], [], []
    y = 0.0
    prev_found, prev_outside = None, None
    for rec in info.itertuples():
        if prev_outside is not None and rec.outside != prev_outside:
            y -= 1.5
        if rec.found_by != prev_found:
            if prev_found is not None:
                y -= 1.32
            groups.append([rec.found_by, y, y])  # label carries every source
        else:
            groups[-1][2] = y
        if prev_outside is None or rec.outside != prev_outside:
            supers.append([rec.outside, y, y])
        else:
            supers[-1][2] = y
        ys.append(y)
        prev_found, prev_outside = rec.found_by, rec.outside
        y -= 1.0
    return np.array(ys), groups, supers


def panel_matrix(fig, gs, info, long):
    ax = fig.add_subplot(gs)
    ax.set_axis_off()
    ys, groups, supers = row_layout(info)
    ax.set_xlim(0, 1)
    ax.set_ylim(ys.min() - 1.4, 3.6)

    cov_cols = [(s, sg) for s in SIGNALS for sg in ("pos", "neg")]
    x_cov = X_COV0 + np.arange(len(cov_cols)) * X_STEP
    x_eff = X_EFF0 + np.arange(3) * X_STEP * 1.35

    # --- headers
    for xi, (s, sg) in zip(x_cov, cov_cols):
        ax.text(xi, 0.75, "+" if sg == "pos" else "\u2212", ha="center", va="bottom",
                fontsize=8, color=COLOR[s], fontweight="bold")
    for s, i in zip(SIGNALS, (0, 2, 4)):
        ax.text(x_cov[i] + X_STEP / 2, 1.65, LABEL[s], ha="center", va="bottom",
                fontsize=7.5, color=COLOR[s])
    ax.text(np.mean(x_cov), 2.7, "Covered by map", ha="center", va="bottom",
            fontsize=8, color="0.15")
    for s, xi in zip(SIGNALS, x_eff):
        ax.text(xi, 0.75, LABEL[s], ha="center", va="bottom", fontsize=7.5,
                color=COLOR[s])
    ax.text(np.mean(x_eff), 2.7, "Effect in ROI", ha="center", va="bottom",
            fontsize=8, color="0.15")

    # --- rows
    eff = long.set_index(["name", "signal"])
    for yi, rec in zip(ys, info.itertuples()):
        ax.text(X_LABEL, yi, f"{rec.roi_id}", ha="right", va="center", fontsize=6.5,
                color="0.45")
        ax.text(X_LABEL + 0.018, yi, rec.name, ha="left", va="center", fontsize=7,
                color="0.1")
        for xi, (s, sg) in zip(x_cov, cov_cols):
            c = getattr(rec, f"{s}_{sg}")
            if c <= 0.01:
                ax.plot([xi], [yi], marker=".", ms=1.5, color="0.78")
                continue
            ms = COV_MS * np.sqrt(c)
            if sg == "pos":
                ax.plot([xi], [yi], marker="s", ms=ms, mfc=COLOR[s], mec="none")
            else:
                ax.plot([xi], [yi], marker="s", ms=ms, mfc="white", mec=COLOR[s],
                        mew=1.0)
        for s, xi in zip(SIGNALS, x_eff):
            r = eff.loc[(rec.name, s)]
            ms = 3.2 + 5.6 * min(abs(r["t"]) / 10.0, 1.0)
            filled = r["p_holm"] < 0.05
            ax.plot([xi], [yi], marker="^" if r["t"] > 0 else "v", ms=ms,
                    mfc=COLOR[s] if filled else "white", mec=COLOR[s], mew=0.9)

    # --- found-by group labels, in the gutter above each block
    for found, y0, y1 in groups:
        parts = [p.split() for p in found.split(" | ")]
        text = "Found by " + " and ".join(f"{LABEL[a]} {b}" for a, b in parts)
        ax.text(X_LABEL + 0.018, y0 + 0.74, text, ha="left", va="center", fontsize=6.8,
                color=COLOR[parts[0][0]], fontstyle="italic")

    # --- the finding, as brackets rather than arrows
    for outside, y0, y1 in supers:
        xb = 0.022
        ax.plot([xb, xb], [y0 + 1.05, y1 - 0.45], lw=1.4, color="0.25",
                solid_capstyle="butt")
        for yy in (y0 + 1.05, y1 - 0.45):
            ax.plot([xb, xb + 0.012], [yy, yy], lw=1.4, color="0.25")
        ax.text(xb - 0.012, (y0 + y1) / 2, "Outside the RPE map" if outside
                else "Inside the RPE map", rotation=90, ha="center", va="center",
                fontsize=7.5, color="0.15", fontweight="bold")

    ax.plot([x_cov[-1] + X_STEP * 0.62] * 2, [ys.min() - 0.6, 1.2], lw=0.6,
            color="0.78")
    return ax


def panel_key(fig, gs):
    """A glyph matrix needs a real key; this one names every mark on panel a."""
    ax = fig.add_subplot(gs)
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    grey = "#4A4A4A"

    def head(y, txt):
        ax.text(0.0, y, txt, fontsize=7.5, color="0.1", va="center",
                fontweight="bold")

    def entry(y, txt):
        ax.text(0.26, y, txt, fontsize=6.6, color="0.28", va="center",
                linespacing=1.55)

    head(0.985, "Rows")
    ax.text(0.0, 0.945, "One cluster found by one of the six\ncontrasts. The bracket "
            "says whether the\nsigned-RPE map covers it (\u2265 5% of its\nvoxels). "
            "The number keys the row to\nits peak in b.",
            fontsize=6.6, color="0.28", va="top", linespacing=1.6)

    head(0.815, "Covered by map")
    ax.text(0.0, 0.775, "Share of the ROI's voxels lying inside\nthat map's surviving "
            "clusters.", fontsize=6.6, color="0.28", va="top", linespacing=1.6)
    for i, c in enumerate((1.0, 0.5, 0.1)):
        yy = 0.700 - i * 0.055
        ax.plot([0.12], [yy], marker="s", ms=COV_MS * np.sqrt(c), mfc=grey, mec="none")
        entry(yy, f"{c:.0%} of the ROI")
    ax.plot([0.12], [0.535], marker=".", ms=1.5, color="0.78")
    entry(0.535, "Below 1% — no overlap")
    ax.plot([0.12], [0.480], marker="s", ms=COV_MS * 0.8, mfc=grey, mec="none")
    entry(0.480, "Filled = positive tail")
    ax.plot([0.12], [0.425], marker="s", ms=COV_MS * 0.8, mfc="white", mec=grey, mew=1.0)
    entry(0.425, "Open = negative tail")

    head(0.335, "Effect in ROI")
    ax.text(0.0, 0.295, "Group one-sample t of that signal's\ncontrast value, averaged "
            "over the ROI.", fontsize=6.6, color="0.28", va="top", linespacing=1.6)
    ax.plot([0.12], [0.215], marker="^", ms=8.8, mfc=grey, mec=grey)
    entry(0.215, "Positive, and significant")
    ax.plot([0.12], [0.160], marker="v", ms=8.8, mfc=grey, mec=grey)
    entry(0.160, "Negative, and significant")
    ax.plot([0.12], [0.105], marker="^", ms=5.2, mfc="white", mec=grey, mew=0.9)
    entry(0.105, "Open = not significant")
    ax.text(0.0, 0.055, "Bigger marker = a larger effect. Significance is\nHolm-corrected "
            "over all ROI \u00d7 signal tests.", fontsize=6.6, color="0.28", va="top",
            linespacing=1.6)
    ax.text(0.0, -0.015, "All six maps share one cluster-forming\nthreshold "
            "(t > 3.98, one-tailed p < 1e-4).", fontsize=6.6, color="0.45",
            va="top", linespacing=1.6, fontstyle="italic")
    return ax


def load_map(source, signal, sign, thr=REF_T):
    d = op.join(SWEEP, "snpm", f"{source}_{signal}")
    if not op.isdir(d):
        d = op.join(SWEEP, "snpm", f"model7_{signal}")
    fn = op.join(d, f"t{t_str(thr)}_{sign}.nii")
    if not op.exists(fn):
        return None
    img = nib.load(fn)
    data = np.abs(np.nan_to_num(np.squeeze(img.get_fdata())))
    return nib.Nifti1Image((data > 0).astype(np.float32), img.affine)


def panel_slices(fig, gs, source, cuts, info):
    """One row of cuts carrying every tail at once.

    Signed RPE is always shown in its positive direction, as a red fill, because
    it is the reference footprint the other maps are being compared against.
    Everything else is an outline on top: solid for the positive tail, dashed for
    the negative one. Numbers are the ROIs of panel a, placed at their peaks."""
    bg = resample_img(nib.load(BG_1MM), target_affine=np.diag([2.0, 2.0, 2.0]),
                      interpolation="continuous", force_resample=True, copy_header=True)
    ax = fig.add_subplot(gs)
    disp = plotting.plot_roi(load_map(source, "rpe", "pos"), bg_img=bg,
                             display_mode="z", cut_coords=cuts, axes=ax,
                             cmap=mcolors.ListedColormap([COLOR["rpe"]]), alpha=0.38,
                             colorbar=False, annotate=False, draw_cross=False,
                             black_bg=False, resampling_interpolation="nearest")
    for signal, sg, ls in (("urpe", "pos", "solid"), ("urpe", "neg", "dashed"),
                           ("surprise", "pos", "solid"), ("surprise", "neg", "dashed"),
                           ("rpe", "neg", "dashed")):
        m = load_map(source, signal, sg)
        if m is None or np.asarray(m.dataobj).sum() == 0:
            continue
        disp.add_contours(m, levels=[0.5], colors=[COLOR[signal]], linewidths=1.0,
                          linestyles=ls)
    disp.annotate(size=6.5)

    # ROI numbers, on whichever cut is nearest that ROI's peak
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for slicer_coord, slicer in disp.axes.items():
        for rec in info.itertuples():
            if abs(rec.peak_z - slicer_coord) > 8:
                continue
            sig = rec.found_by.split(" | ")[0].split()[0]
            slicer.ax.text(rec.peak_x, rec.peak_y, f"{rec.roi_id}", fontsize=5.8,
                           ha="center", va="center", color=COLOR[sig],
                           fontweight="bold", path_effects=halo, zorder=100,
                           clip_on=False)
    return ax


def main(source, out_stem):
    info, long = load(source)

    fig = plt.figure(figsize=(7.25, 7.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.30], width_ratios=[1, 0.245],
                          left=0.005, right=0.995, top=0.958, bottom=0.045,
                          hspace=0.16, wspace=0.01)

    ax_m = panel_matrix(fig, gs[0, 0], info, long)
    panel_key(fig, gs[0, 1])
    ax_b = panel_slices(fig, gs[1, :], source, [-12, 0, 12, 24, 36, 48], info)

    ax_m.text(-0.005, 1.008, "a", transform=ax_m.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left")
    ax_b.text(-0.005, 1.055, "b", transform=ax_b.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left")

    ax_b.text(0.030, 1.055, "Every map, both tails, on the same slices",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.5,
              color="0.1", fontweight="bold")
    ax_b.text(1.0, 1.055,
              "Red fill: signed RPE, positive · outlines: uRPE (green) and surprise "
              "(blue) · solid = positive tail, dashed = negative",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.8,
              color="0.35")
    ax_b.text(0.5, -0.10,
              "Surprise has no surviving negative clusters at any threshold tested. "
              "The dashed green outlines are uRPE\ngoing negative — they fall inside "
              "the red map, while the solid green ones sit outside it.",
              transform=ax_b.transAxes, ha="center", va="top", fontsize=7,
              color="0.35", linespacing=1.7)

    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out_stem + ".pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.source, a.out or op.join(NOTES, "figures", f"fig_overlap_{a.source}"))
