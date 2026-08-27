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


ALPHA = {"rpe": 0.68, "urpe": 0.55, "surprise": 0.55}   # RPE is the reference, so a bit stronger
MASK_VOX = 59838          # SnPM's analysis mask
AXIAL = [-16, -6, 4, 14, 26, 38, 50]
CORONAL = [-80, -60, -40, -20, 0, 20, 40]


def mix(rgb, other, w):
    return tuple((1 - w) * np.array(rgb) + w * np.array(other))


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


def load(source, variant):
    d = op.join(SWEEP, "contrast_rois", variant, source)
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
    label = "Inference variant"
    lp = op.join(d, "variant.txt")
    if op.exists(lp):
        label = open(lp).read().strip()
    return info, long, label


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
            "says whether\nthe signed-RPE map covers it (\u2265 5%). The\nnumber keys the row to its peak in b.",
            fontsize=6.6, color="0.28", va="top", linespacing=1.6)

    head(0.800, "Covered by map")
    ax.text(0.0, 0.762, "Share of the ROI's voxels lying inside\nthat map's surviving "
            "clusters.", fontsize=6.6, color="0.28", va="top", linespacing=1.6)
    for i, c in enumerate((1.0, 0.5, 0.1)):
        yy = 0.688 - i * 0.055
        ax.plot([0.12], [yy], marker="s", ms=COV_MS * np.sqrt(c), mfc=grey, mec="none")
        entry(yy, f"{c:.0%} of the ROI")
    ax.plot([0.12], [0.523], marker=".", ms=1.5, color="0.78")
    entry(0.523, "Below 1% — no overlap")
    ax.plot([0.12], [0.468], marker="s", ms=COV_MS * 0.8, mfc=grey, mec="none")
    entry(0.468, "Filled = positive tail")
    ax.plot([0.12], [0.413], marker="s", ms=COV_MS * 0.8, mfc="white", mec=grey, mew=1.0)
    entry(0.413, "Open = negative tail")

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


def load_map(source, signal, sign, variant):
    fn = op.join(SWEEP, "contrast_rois", variant, source,
                 f"map_{signal}_{sign}.nii.gz")
    if not op.exists(fn):
        return None
    img = nib.load(fn)
    data = np.abs(np.nan_to_num(img.get_fdata()))
    return nib.Nifti1Image((data > 0).astype(np.float32), img.affine)


_BG = {}


def background():
    if "img" not in _BG:
        _BG["img"] = resample_img(nib.load(BG_1MM), target_affine=np.diag([2.0, 2.0, 2.0]),
                                  interpolation="continuous", force_resample=True,
                                  copy_header=True)
    return _BG["img"]


def slice_row(fig, cell, source, variant, mode, cuts, which, info=None):
    """One row of cuts, maps drawn as filled semi-transparent overlays.

    Sign is carried by the ROW, not by line style: solid-vs-dashed is a second
    encoding the reader has to hold in mind, and at 1 pt it is hard to read at
    print size. With one row per tail no sign key is needed at all, and a row can
    still put both tails together -- no voxel is ever in both tails of the same
    signal, so overlapping fills stay unambiguous within a hue."""
    ax = fig.add_subplot(cell)
    loaded = []
    empty = []
    for signal, tail in which:
        m = load_map(source, signal, tail, variant)
        if m is None or np.asarray(m.dataobj).sum() == 0:
            empty.append((signal, tail))
        else:
            loaded.append((signal, m, int((np.asarray(m.dataobj) > 0).sum())))
    # Paint the biggest map first and the smallest last. Drawn in list order, a
    # 211-voxel surprise map disappears under an 8.6%-of-brain uRPE-negative one.
    loaded.sort(key=lambda t: -t[2])
    loaded = [(sig, m) for sig, m, _ in loaded]
    if not loaded:                      # nothing survives: draw the anatomy alone
        disp = plotting.plot_roi(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32),
                                                 np.eye(4)),
                                 bg_img=background(), display_mode=mode,
                                 cut_coords=cuts, axes=ax, colorbar=False,
                                 annotate=False, draw_cross=False, black_bg=False)
    else:
        sig0, m0 = loaded[0]
        disp = plotting.plot_roi(m0, bg_img=background(), display_mode=mode,
                                 cut_coords=cuts, axes=ax, colorbar=False,
                                 cmap=mcolors.ListedColormap([COLOR[sig0]]),
                                 alpha=ALPHA[sig0], annotate=False, draw_cross=False,
                                 black_bg=False, resampling_interpolation="nearest")
        for signal, m in loaded[1:]:
            disp.add_overlay(m, threshold=0.5, transparency=ALPHA[signal],
                             cmap=mcolors.ListedColormap([COLOR[signal]]))
    # outline everything, small maps last and heavier so they stay findable
    for signal, m in loaded[::-1]:
        n = int((np.asarray(m.dataobj) > 0).sum())
        lw = 0.45 if n > 3000 else (0.6 if n > 800 else 0.85)
        disp.add_contours(m, levels=[0.5], linewidths=lw,
                          colors=[mix(mcolors.to_rgb(COLOR[signal]), (0, 0, 0), 0.35)])
    disp.annotate(size=6.2)

    if info is not None:
        halo = [pe.withStroke(linewidth=1.9, foreground="white")]
        axis = {"z": 2, "y": 1, "x": 0}[mode]
        plane = {"z": ("peak_x", "peak_y"), "y": ("peak_x", "peak_z"),
                 "x": ("peak_y", "peak_z")}[mode]
        for coord, slicer in disp.axes.items():
            for rec in info.itertuples():
                pk = (rec.peak_x, rec.peak_y, rec.peak_z)[axis]
                if abs(pk - coord) > 9:
                    continue
                sig = rec.found_by.split(" | ")[0].split()[0]
                slicer.ax.text(getattr(rec, plane[0]), getattr(rec, plane[1]),
                               f"{rec.roi_id}", fontsize=5.6, ha="center",
                               va="center", color=COLOR[sig], fontweight="bold",
                               path_effects=halo, zorder=100, clip_on=False)
    return ax, empty


def map_block(fig, gs_rows, source, variant, which, title, note, info=None):
    """Two rows of cuts -- axial then coronal -- for one selection of maps."""
    ax_a, empty = slice_row(fig, gs_rows[0], source, variant, "z", AXIAL, which, info=info)
    ax_c, _ = slice_row(fig, gs_rows[1], source, variant, "y", CORONAL, which, info=info)
    ax_a.text(0.030, 1.075, title, transform=ax_a.transAxes, ha="left", va="bottom",
              fontsize=7.5, color="0.1", fontweight="bold")
    if note:
        ax_a.text(1.0, 1.075, note, transform=ax_a.transAxes, ha="right", va="bottom",
                  fontsize=6.8, color="0.35")
    return ax_a, ax_c, empty


POS = [("rpe", "pos"), ("urpe", "pos"), ("surprise", "pos")]
NEG = [("rpe", "neg"), ("urpe", "neg"), ("surprise", "neg")]
BOTH = POS + [("rpe", "neg"), ("urpe", "neg")]


def map_extents(source, variant):
    """Surviving extent of each of the six maps, as a share of the analysis mask.

    Stated on the figure so nothing is hidden by the ROI decomposition: a map can
    be large and still yield no named ROI (one confluent blob), or nearly empty."""
    out = []
    for sig in SIGNALS:
        for tail in ("pos", "neg"):
            m = load_map(source, sig, tail, variant)
            n = 0 if m is None else int((np.asarray(m.dataobj) > 0).sum())
            out.append((sig, tail, n, 100 * n / MASK_VOX))
    return out


def banner(fig, label, source, variant, y_top=0.995):
    """The inference choice, stated large -- it is the thing under test."""
    fig.text(0.005, y_top, label, fontsize=10.5, fontweight="bold", va="top",
             color="0.08")
    ext = map_extents(source, variant)
    parts = []
    for sig in SIGNALS:
        vals = {t: p for s, t, n, p in ext if s == sig for t in [t]}
        parts.append(f"{LABEL[sig]} +{vals['pos']:.1f} / −{vals['neg']:.1f}")
    fig.text(0.005, y_top - 0.0295,
             f"{source}  ·  n = 58  ·  surviving extent, % of mask:  "
             + "   ".join(parts),
             fontsize=7.5, va="top", color="0.42")


def figure_overlap(source, variant, info, long, label, out_stem):
    n = len(info)
    h_matrix = max(1.5, 0.235 * n + 1.15)
    fig = plt.figure(figsize=(7.25, h_matrix + 2.85))
    hr = [h_matrix, 1.05, 1.05]
    gs = fig.add_gridspec(3, 2, height_ratios=hr, width_ratios=[1, 0.245],
                          left=0.005, right=0.995, top=1 - 0.62 / (h_matrix + 2.85),
                          bottom=0.045, hspace=0.22, wspace=0.01)
    banner(fig, label, source, variant)
    ax_m = panel_matrix(fig, gs[0, 0], info, long)
    panel_key(fig, gs[0, 1])
    ax_b, ax_b2, _ = map_block(
        fig, [gs[1, :], gs[2, :]], source, variant, BOTH, "Every map, both tails",
        "Signed RPE (red) · uRPE (green) · surprise (blue) · numbers mark the ROIs of a",
        info=info)
    ax_m.text(-0.005, 1.008, "a", transform=ax_m.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left")
    ax_b.text(-0.005, 1.075, "b", transform=ax_b.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left")
    ax_b2.text(0.5, -0.13,
               "Clusters smaller than 25 voxels are not given a row in a; the extents in "
               "the header are the full maps, so a signal can\nbe large here and absent "
               "from a (one confluent blob), or present as a map and too small to name.",
               transform=ax_b2.transAxes, ha="center", va="top", fontsize=7,
               color="0.35", linespacing=1.7)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out_stem + ".pdf")


def figure_maps(source, variant, info, label, out_stem):
    fig = plt.figure(figsize=(7.25, 9.3))
    gs = fig.add_gridspec(6, 1, left=0.005, right=0.995, top=0.938, bottom=0.055,
                          hspace=0.22)
    banner(fig, label, source, variant)
    ax1, ax1b, _ = map_block(fig, [gs[0], gs[1]], source, variant, POS,
                             "Positive tails",
                             "Signed RPE (red) · uRPE (green) · surprise (blue)")
    ax2, ax2b, empty = map_block(fig, [gs[2], gs[3]], source, variant, NEG,
                                 "Negative tails", "Same three colours")
    ax3, ax3b, _ = map_block(fig, [gs[4], gs[5]], source, variant, BOTH,
                             "Both tails together",
                             "Numbers mark the ROIs of the companion figure",
                             info=info)
    for ax, letter in ((ax1, "a"), (ax2, "b"), (ax3, "c")):
        ax.text(-0.005, 1.075, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
    if any(sg == "surprise" for sg, _ in empty):
        ax2.text(0.030, -0.16, "Surprise has no surviving negative clusters here.",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=6.8,
                 color="0.35")
    ax3b.text(0.5, -0.15,
              "The green of b sits on the red of a — the same territory carries a "
              "positive signed-RPE response and a negative unsigned one.",
              transform=ax3b.transAxes, ha="center", va="top", fontsize=7,
              color="0.35", linespacing=1.7)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out_stem + ".pdf")


def main(source, variant, outdir):
    info, long, label = load(source, variant)
    figure_overlap(source, variant, info, long, label,
                   op.join(outdir, f"fig_overlap_{source}_{variant}"))
    figure_maps(source, variant, info, label,
                op.join(outdir, f"fig_maps_{source}_{variant}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model7")
    ap.add_argument("--variant", default="extent_p1e4")
    ap.add_argument("--outdir", default=op.join(NOTES, "figures"))
    a = ap.parse_args()
    main(a.source, a.variant, a.outdir)
