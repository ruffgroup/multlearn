"""Multi-page PDF: how the RPE / uRPE / surprise maps depend on the
cluster-forming threshold and on the inference engine.

One page per (engine, inference method, cluster-forming threshold).  Every page
is the same 3 x 2 grid -- row = learning signal, column = direction of the
effect -- and every panel shows one 2D slice per surviving cluster, cut through
that cluster's peak in whichever plane shows most of it.  No glass brains: a
glass brain hides exactly the thing at issue here, which is how much of the
brain a map actually covers.

Colour follows the manuscript (RPE red, uRPE green, surprise blue).  Negative
panels are rendered as a photographic negative -- dark background, the same hue
now emitting rather than absorbing, dashed cluster outlines, a downward marker
in the header -- so sign is legible at a glance without spending a second hue.

Two PDFs are produced, one per model source:
    threshold_sweep_model7.pdf   all three modulators in one design
    threshold_sweep_model2.pdf   RPE and surprise as reported in the paper
                                 (uRPE does not exist in model2 and is shown
                                 from model7, flagged on the page)

Run locally, after SPM/code/fetch_threshold_sweep.sh has pulled the maps:
    ~/mambaforge/bin/python notes/figures/plot_threshold_sweep.py
"""

import argparse
import os.path as op
from collections import OrderedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from nilearn import plotting  # noqa: E402
from nilearn.image import resample_img  # noqa: E402
from scipy import ndimage  # noqa: E402

NOTES = op.dirname(op.dirname(op.abspath(__file__)))
SWEEP = op.join(NOTES, "data", "threshold_sweep")
FIGDIR = op.join(NOTES, "figures")
BG_1MM = op.join(NOTES, "data", "surprise_vs_rpe", "MNI152_T1_1mm.nii.gz")

SIGNALS = ["rpe", "urpe", "surprise"]
SIGNAL_LABEL = {"rpe": "Signed RPE", "urpe": "Unsigned RPE",
                "surprise": "Shannon surprise"}
SIGNAL_EVENT = {"rpe": "feedback", "urpe": "feedback", "surprise": "stimulus onset"}
SIGNAL_COLOR = {"rpe": "#C44E52", "urpe": "#2E7D32", "surprise": "#3B5BA5"}

THRESHOLDS = [(2.3936, 1e-2), (2.6649, 5e-3), (3.2395, 1e-3),
              (3.9756, 1e-4), (5.2929, 1e-6)]
FWE_ALPHA = 0.05
LOGP_ALPHA = -np.log10(FWE_ALPHA)
MAX_SLOTS = 8          # cluster slices shown per panel
# how to arrange n cluster slices inside one panel (the panel cell is ~2:1)
SLOT_GRID = {0: (1, 1), 1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2),
             5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4)}
CONNECTIVITY = ndimage.generate_binary_structure(3, 2)  # 18, as in spm_clusters

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def t_str(t):
    return f"{t:g}".replace(".", "_")


def mix(rgb, other, w):
    return tuple((1 - w) * np.array(rgb) + w * np.array(other))


def hue_ramp(hex_color, dark_bg):
    """Low t -> high t.  On white the ramp darkens; on black it lights up, so in
    both cases the strongest voxels have the most contrast against the brain."""
    base = mcolors.to_rgb(hex_color)
    if dark_bg:
        stops = [mix(base, (0, 0, 0), 0.60), base, mix(base, (1, 1, 1), 0.80)]
    else:
        stops = [mix(base, (1, 1, 1), 0.72), base, mix(base, (0, 0, 0), 0.45)]
    return mcolors.LinearSegmentedColormap.from_list("ramp", stops)


_BG_CACHE = {}


def background():
    if "img" not in _BG_CACHE:
        img = nib.load(BG_1MM)
        # 1 mm is needlessly slow for ~1500 small panels
        # a 3x3 target_affine lets nilearn recompute the offset and keep the
        # field of view; a 4x4 diagonal one silently moves the origin to (0,0,0)
        # and chops off every negative coordinate
        _BG_CACHE["img"] = resample_img(
            img, target_affine=np.diag([2.0, 2.0, 2.0]),
            interpolation="continuous", force_resample=True, copy_header=True)
    return _BG_CACHE["img"]


# --------------------------------------------------------------------------
# loading


def load_snpm(source, signal, thr, sign):
    """SnPM survivor map -> signed t restricted to cluster-FWE surviving voxels."""
    key = f"{source}_{signal}"
    d = op.join(SWEEP, "snpm", key)
    if not op.isdir(d):
        d = op.join(SWEEP, "snpm", f"model7_{signal}")
    fn = op.join(d, f"t{t_str(thr)}_{sign}.nii")
    if not op.exists(fn):
        fn += ".gz"
    if not op.exists(fn):
        return None
    img = nib.load(fn)
    data = np.nan_to_num(np.squeeze(img.get_fdata()))
    if sign == "neg":
        data = -np.abs(data)
    else:
        data = np.abs(data)
    return nib.Nifti1Image(data.astype(np.float32), img.affine)


def load_nilearn(source, signal, thr, sign, kind="size"):
    """nilearn -> signed t restricted to voxels whose corrected p < .05."""
    key = f"{source}_{signal}"
    d = op.join(SWEEP, "nilearn", "conn18", key)
    if not op.isdir(d):
        d = op.join(SWEEP, "nilearn", "conn18", f"model7_{signal}")
    t_fn = op.join(d, "t.nii.gz")
    if kind in ("size", "mass"):
        p_fn = op.join(d, f"logp_max_{kind}_t{t_str(thr)}.nii.gz")
    else:
        p_fn = op.join(d, f"logp_max_{kind}.nii.gz")
    if not (op.exists(t_fn) and op.exists(p_fn)):
        return None
    t_img = nib.load(t_fn)
    t = np.nan_to_num(t_img.get_fdata())
    logp = np.nan_to_num(nib.load(p_fn).get_fdata())
    keep = logp > LOGP_ALPHA
    keep &= (t > 0) if sign == "pos" else (t < 0)
    out = np.where(keep, t, 0.0)
    return nib.Nifti1Image(out.astype(np.float32), t_img.affine)


def analysis_mask_size(source, signal):
    d = op.join(SWEEP, "nilearn", "conn18", f"{source}_{signal}")
    if not op.isdir(d):
        d = op.join(SWEEP, "nilearn", "conn18", f"model7_{signal}")
    meta = op.join(d, "meta.json")
    if op.exists(meta):
        import json
        return json.load(open(meta))["n_mask_voxels"]
    t_fn = op.join(d, "t.nii.gz")           # the t map is already masked
    if op.exists(t_fn):
        return int((np.nan_to_num(nib.load(t_fn).get_fdata()) != 0).sum())
    return None


# --------------------------------------------------------------------------
# clusters


def cluster_table(img, min_vox=1):
    """(size, peak_ijk, peak_t, mask) per connected component, largest first."""
    data = np.nan_to_num(img.get_fdata())
    labels, n = ndimage.label(data != 0, CONNECTIVITY)
    out = []
    for lab in range(1, n + 1):
        mask = labels == lab
        size = int(mask.sum())
        if size < min_vox:
            continue
        vals = np.where(mask, np.abs(data), 0)
        peak = np.unravel_index(np.argmax(vals), vals.shape)
        out.append(dict(size=size, peak_ijk=peak, peak_t=float(data[peak]), mask=mask))
    out.sort(key=lambda c: -c["size"])
    return out


def best_plane(mask, peak_ijk):
    """The cut through the peak that shows the most of this cluster."""
    counts = [mask[peak_ijk[0], :, :].sum(), mask[:, peak_ijk[1], :].sum(),
              mask[:, :, peak_ijk[2]].sum()]
    return int(np.argmax(counts))


# --------------------------------------------------------------------------
# drawing


def draw_panel(fig, cell, img, signal, sign, cft, vmax, mask_vox, voxel_mm3,
               note=None):
    dark = sign == "neg"
    hue = SIGNAL_COLOR[signal]
    face = "#000000" if dark else "#FFFFFF"  # match nilearn's black_bg exactly
    fg = "#E8E8E8" if dark else "#1A1A1A"

    bg_ax = fig.add_subplot(cell)
    bg_ax.set_facecolor(face)
    bg_ax.set_xticks([]); bg_ax.set_yticks([])
    for spine in bg_ax.spines.values():
        spine.set_color(hue); spine.set_linewidth(1.2)

    head_cell, body_cell = cell.subgridspec(
        2, 1, height_ratios=[0.30, 1], hspace=0.10)
    head = fig.add_subplot(head_cell); head.set_axis_off()

    direction = "Negative" if dark else "Positive"
    title = f"{SIGNAL_LABEL[signal]} — {direction}"
    # a drawn triangle rather than a glyph: Helvetica has no U+25B2/U+25BC
    head.plot([0.009], [0.62], marker="v" if dark else "^", ms=7, color=hue,
              transform=head.transAxes, clip_on=False)

    if img is None:
        head.text(0.028, 0.62, title, transform=head.transAxes, fontsize=10.5,
                  fontweight="bold", color=hue, va="center")
        head.text(0.028, 0.12, note or "Map not available", transform=head.transAxes,
                  fontsize=8, color="0.5" if not dark else "0.6", style="italic")
        return dict(n_clusters=0, n_vox=0)

    clusters = cluster_table(img)
    n_vox = sum(c["size"] for c in clusters)
    pct = 100 * n_vox / mask_vox if mask_vox else np.nan
    peak = max((abs(c["peak_t"]) for c in clusters), default=0.0)

    head.text(0.028, 0.62, title, transform=head.transAxes, fontsize=10.5,
              fontweight="bold", color=hue, va="center")
    summary = (f"{len(clusters)} cluster{'s' if len(clusters) != 1 else ''} · "
               f"{n_vox:,} voxels ({pct:.1f}% of mask) · "
               f"{n_vox * voxel_mm3 / 1000:,.1f} cm³ · peak |t| = {peak:.1f}")
    head.text(0.028, 0.18, summary, transform=head.transAxes, fontsize=7.6,
              color=fg, va="center")

    if clusters and 100 * clusters[0]["size"] / max(n_vox, 1) > 80 and pct > 15:
        head.text(0.98, 0.18, "One cluster covers the map — "
                  "cluster-extent inference is uninformative here",
                  transform=head.transAxes, fontsize=6.8,
                  color="#F0B429" if dark else "#9A6700",
                  ha="right", va="center", style="italic")

    if not clusters:
        ax = fig.add_subplot(body_cell); ax.set_axis_off()
        ax.text(0.5, 0.5, "No suprathreshold clusters", transform=ax.transAxes,
                ha="center", va="center", fontsize=9,
                color="0.5" if not dark else "0.6", style="italic")
        return dict(n_clusters=0, n_vox=0)

    cmap = hue_ramp(hue, dark_bg=dark)
    edge = "#FFFFFF" if dark else mix(mcolors.to_rgb(hue), (0, 0, 0), 0.55)
    abs_img = nib.Nifti1Image(np.abs(np.nan_to_num(img.get_fdata())).astype(np.float32),
                              img.affine)
    bg = background()

    shown = clusters[:MAX_SLOTS]
    nrows, ncols = SLOT_GRID[len(shown)]
    inner = body_cell.subgridspec(nrows, ncols, hspace=0.34, wspace=0.04)

    # a cluster spanning a large part of the brain cannot be shown by one cut,
    # so give it a short montage inside its own slot instead
    giant_vox = 0.06 * mask_vox if mask_vox else 4000

    for slot, clu in enumerate(shown):
        ax = fig.add_subplot(inner[slot // ncols, slot % ncols])
        axis = best_plane(clu["mask"], clu["peak_ijk"])
        coord = nib.affines.apply_affine(img.affine, np.array(clu["peak_ijk"]))
        mode = "xyz"[axis]
        if clu["size"] > giant_vox:
            mode = "z"
            zs = np.nonzero(clu["mask"].any(axis=0).any(axis=0))[0]
            cuts = np.linspace(zs[0], zs[-1], min(6, len(zs))).round().astype(int)
            world = [nib.affines.apply_affine(img.affine, [0, 0, z])[2] for z in cuts]
            cut_coords = sorted(set(np.round(world, 1)))
        else:
            cut_coords = [coord[axis]]
        disp = plotting.plot_stat_map(
            abs_img, bg_img=bg, display_mode=mode, cut_coords=cut_coords,
            threshold=cft, vmax=vmax, cmap=cmap, colorbar=False, axes=ax,
            annotate=False, draw_cross=False, black_bg=dark,
            resampling_interpolation="nearest")
        clu_img = nib.Nifti1Image(clu["mask"].astype(np.float32), img.affine)
        disp.add_contours(clu_img, levels=[0.5], colors=[edge], linewidths=0.7,
                          linestyles="dashed" if dark else "solid")
        where = (f"{len(cut_coords)} axial cuts" if len(cut_coords) > 1
                 else f"{mode}={coord[axis]:.0f}")
        label = (f"#{slot + 1}  {clu['size'] * voxel_mm3:,.0f} mm³  "
                 f"t={clu['peak_t']:+.1f}\n{where}  peak "
                 f"({coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f})")
        ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.0, color=fg, linespacing=1.3)

    if len(clusters) > MAX_SLOTS:
        rest = sum(c["size"] for c in clusters[MAX_SLOTS:])
        head.text(0.98, 0.62, f"+{len(clusters) - MAX_SLOTS} more clusters "
                  f"({100 * (n_vox - rest) / n_vox:.0f}% of volume shown)",
                  transform=head.transAxes, fontsize=6.8, color=fg, ha="right",
                  va="center", style="italic")
    return dict(n_clusters=len(clusters), n_vox=n_vox)


def render_page(pdf, source, loader, title, subtitle, cft, voxel_mm3, urpe_note):
    fig = plt.figure(figsize=(16.5, 11.7))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(3, 2, left=0.012, right=0.988, top=0.925, bottom=0.012,
                             hspace=0.10, wspace=0.03)
    fig.text(0.012, 0.972, title, fontsize=15, fontweight="bold", va="center")
    fig.text(0.012, 0.945, subtitle, fontsize=8.8, color="0.35", va="center")

    stats = {}
    for row, signal in enumerate(SIGNALS):
        imgs = {s: loader(source, signal, cft, s) for s in ("pos", "neg")}
        vals = [np.abs(np.nan_to_num(i.get_fdata())) for i in imgs.values() if i is not None]
        vals = np.concatenate([v[v > 0] for v in vals]) if vals else np.array([])
        vmax = float(np.percentile(vals, 99.5)) if vals.size else cft * 2
        vmax = max(vmax, cft * 1.35)
        mask_vox = analysis_mask_size(source, signal)
        for col, sign in enumerate(("pos", "neg")):
            note = urpe_note if (signal == "urpe" and source == "model2") else None
            stats[(signal, sign)] = draw_panel(
                fig, outer[row, col], imgs[sign], signal, sign, cft, vmax,
                mask_vox, voxel_mm3, note=note)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
# summary across engines / methods


def build_summary(source, voxel_mm3):
    rows = []
    for signal in SIGNALS:
        mask_vox = analysis_mask_size(source, signal)
        for cft, p in THRESHOLDS:
            for sign in ("pos", "neg"):
                maps = OrderedDict(
                    snpm_size=load_snpm(source, signal, cft, sign),
                    nilearn_size=load_nilearn(source, signal, cft, sign, "size"),
                    nilearn_mass=load_nilearn(source, signal, cft, sign, "mass"),
                )
                masks = {}
                for name, img in maps.items():
                    if img is None:
                        continue
                    data = np.nan_to_num(img.get_fdata())
                    m = data != 0
                    masks[name] = m
                    clusters = cluster_table(img)
                    rows.append(dict(
                        source=source, signal=signal, sign=sign, cft=cft, p=p,
                        method=name, n_vox=int(m.sum()),
                        pct_mask=100 * m.sum() / mask_vox if mask_vox else np.nan,
                        cm3=m.sum() * voxel_mm3 / 1000,
                        n_clusters=len(clusters),
                        max_abs_t=float(np.abs(data[m]).max()) if m.any() else 0.0))
                if "snpm_size" in masks and "nilearn_size" in masks:
                    a, b = masks["snpm_size"], masks["nilearn_size"]
                    denom = a.sum() + b.sum()
                    rows.append(dict(
                        source=source, signal=signal, sign=sign, cft=cft, p=p,
                        method="dice_snpm_vs_nilearn",
                        n_vox=int((a & b).sum()),
                        pct_mask=np.nan, cm3=np.nan, n_clusters=np.nan,
                        max_abs_t=2 * (a & b).sum() / denom if denom else np.nan))
    return pd.DataFrame(rows)


def render_summary_page(pdf, source, summary, voxel_mm3, notes_text):
    fig = plt.figure(figsize=(16.5, 11.7))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.97, top=0.90, bottom=0.30,
                          hspace=0.45, wspace=0.18)
    fig.text(0.012, 0.965, f"How much of the brain survives — {source}",
             fontsize=15, fontweight="bold", va="center")
    fig.text(0.012, 0.938,
             "Extent of the surviving map as a function of the cluster-forming "
             "threshold. If a signal's advantage is sensitivity rather than "
             "anatomy, it shows up here as a curve that sits above the others "
             "at every threshold.", fontsize=8.8, color="0.35", va="center")

    x = [-np.log10(p) for _, p in THRESHOLDS]
    xticklabels = [f"p<{p:g}\nt>{t:.2f}" for t, p in THRESHOLDS]

    for col, sign in enumerate(("pos", "neg")):
        for row, (metric, ylabel, logy) in enumerate([
                ("pct_mask", "Surviving voxels (% of mask)", True),
                ("n_clusters", "Number of surviving clusters", False),
                ("max_abs_t", "Peak |t|", False)]):
            ax = fig.add_subplot(gs[row, col])
            for signal in SIGNALS:
                for method, ls, marker in [("snpm_size", "-", "o"),
                                           ("nilearn_size", "--", "s")]:
                    sub = summary[(summary.signal == signal) & (summary.sign == sign)
                                  & (summary.method == method)].sort_values("cft")
                    if sub.empty:
                        continue
                    ax.plot([-np.log10(v) for v in sub.p], sub[metric],
                            ls=ls, marker=marker, ms=3.5, lw=1.3,
                            color=SIGNAL_COLOR[signal],
                            label=f"{SIGNAL_LABEL[signal]} ({'SnPM' if 'snpm' in method else 'nilearn'})")
            if logy:
                ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels, fontsize=6.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{'Positive' if sign == 'pos' else 'Negative'} effects",
                         fontsize=9.5, fontweight="bold",
                         color="#333333" if sign == "pos" else "#333333")
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, frameon=False, ncol=1, loc="upper right")

    fig.text(0.012, 0.255, "Engine comparison and caveats", fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.012, 0.235, notes_text, fontsize=7.8, color="0.2", va="top",
             linespacing=1.55, wrap=True)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# ROI sign overview


def render_roi_pages(pdf, source, per_page=22):
    tsv = op.join(SWEEP, "roi_matrix", source, "roi_sign_matrix.tsv")
    if not op.exists(tsv):
        return
    df = pd.read_csv(tsv, sep="\t")
    wide_t = df.pivot(index="name", columns="signal", values="t")
    wide_p = df.pivot(index="name", columns="signal", values="p_holm")

    # order: by which signal dominates, then by that signal's |t|
    dom = wide_t.abs().idxmax(axis=1)
    strength = wide_t.abs().max(axis=1)
    order = pd.DataFrame({"dom": dom, "strength": strength})
    order["dom_rank"] = order["dom"].map({s: i for i, s in enumerate(SIGNALS)})
    order = order.sort_values(["dom_rank", "strength"], ascending=[True, False])
    names = list(order.index)

    tmax = float(np.nanmax(np.abs(wide_t.values))) * 1.08

    for start in range(0, len(names), per_page):
        chunk = names[start:start + per_page]
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, left=0.30, right=0.965, top=0.885, bottom=0.06,
                              wspace=0.10)
        page = start // per_page + 1
        n_pages = int(np.ceil(len(names) / per_page))
        fig.text(0.012, 0.955,
                 f"Which regions code which signal, and with which sign — "
                 f"{source} ({page}/{n_pages})",
                 fontsize=15, fontweight="bold", va="center")
        fig.text(0.012, 0.925,
                 "Harvard-Oxford parcels touched by any signal's surviving clusters "
                 "(cluster-forming t > 3.24, p < .001). Bars are the group one-sample "
                 "t of that parcel's mean contrast value; anatomical parcels give no "
                 "signal a home-field advantage. Filled = Holm-significant across all "
                 "parcel × signal tests, open = not.",
                 fontsize=8.6, color="0.35", va="center")

        y = np.arange(len(chunk))[::-1]
        for ci, signal in enumerate(SIGNALS):
            ax = fig.add_subplot(gs[0, ci])
            vals = wide_t.loc[chunk, signal].values
            pvals = wide_p.loc[chunk, signal].values
            sig = pvals < 0.05
            ax.barh(y, vals, height=0.66, color=SIGNAL_COLOR[signal],
                    alpha=1.0, edgecolor=SIGNAL_COLOR[signal], lw=0.9,
                    zorder=3)
            ax.barh(y[~sig], vals[~sig], height=0.66, color="white",
                    edgecolor=SIGNAL_COLOR[signal], lw=0.9, zorder=4)
            ax.axvline(0, color="0.25", lw=0.8, zorder=5)
            ax.set_xlim(-tmax, tmax)
            ax.set_ylim(-0.7, len(chunk) - 0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(chunk if ci == 0 else [], fontsize=7.2)
            ax.set_title(f"{SIGNAL_LABEL[signal]}\n({SIGNAL_EVENT[signal]})",
                         fontsize=10, fontweight="bold", color=SIGNAL_COLOR[signal])
            ax.set_xlabel("Group t", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis="x", alpha=0.25, lw=0.5, zorder=0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if ci > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)


# --------------------------------------------------------------------------

ENGINE_NOTES = """SnPM and nilearn run the same sign-flipping permutation test on the same 58 contrast images and the same analysis mask (SnPM's own),
but they are not identical, and three differences matter for reading the pages that follow:

  1.  Family.  SnPM's `Tsign` treats each tail as its own one-tailed family, which is what the paper reports; nilearn's `two_sided_test=True`
      controls the family-wise error over both tails at once.  nilearn is therefore the stricter test by roughly a factor of two in the tail,
      and its maps should be slightly smaller wherever a threshold is near the edge of significance.
  2.  Connectivity.  SPM labels clusters with 18-connectivity (`spm_clusters`); nilearn hardcodes 6-connectivity.  These runs monkeypatch
      nilearn to 18 so the two engines parcellate identically -- otherwise nilearn splits a single SPM cluster into several smaller ones,
      each of which is then easier to kill on extent.
  3.  Statistic.  SnPM's cluster-extent test uses cluster size only.  nilearn additionally provides cluster *mass* (the summed excess of t
      over the threshold) and TFCE, both of which reward tall-and-narrow clusters that pure extent penalises.

TFCE and voxel-wise FWE are threshold-free and appear once, not once per cluster-forming threshold.  Where the two engines disagree, the Dice
overlap on the summary panels is the place to look first; large disagreement almost always means the map is sitting on the edge of the
extent criterion, which is itself the point of this document."""


def make_pdf(source, out_path, voxel_mm3=31.5, mass_thresholds=(1e-2, 1e-3)):
    urpe_note = ("uRPE is not a regressor in model2; this row is model7's uRPE, "
                 "shown for reference only.")
    summary = build_summary(source, voxel_mm3)
    summary.to_csv(op.join(SWEEP, f"summary_{source}.tsv"), sep="\t", index=False)

    dice = summary[summary.method == "dice_snpm_vs_nilearn"]["max_abs_t"].dropna()
    extra = ""
    if len(dice):
        extra = (f"\n\nAcross all {len(dice)} signal × sign × threshold panels in this "
                 f"document, the SnPM and nilearn survivor maps overlap with a median "
                 f"Dice of {dice.median():.2f} (range {dice.min():.2f}–{dice.max():.2f}).")

    with PdfPages(out_path) as pdf:
        render_summary_page(pdf, source, summary, voxel_mm3, ENGINE_NOTES + extra)

        for cft, p in THRESHOLDS:
            render_page(pdf, source, load_snpm,
                        f"SnPM cluster-extent FWE — {source}",
                        f"Cluster-forming t > {cft:.4f} (one-tailed p < {p:g}, df = 57); "
                        f"clusters with p_FWE < .05, 5000 permutations, one-tailed family "
                        f"per tail. Each slice is cut through one cluster's peak.",
                        cft, voxel_mm3, urpe_note)

        for cft, p in THRESHOLDS:
            render_page(pdf, source,
                        lambda s, sig, t, sn: load_nilearn(s, sig, t, sn, "size"),
                        f"nilearn cluster-extent FWE — {source}",
                        f"Cluster-forming t > {cft:.4f} (one-tailed p < {p:g}); clusters with "
                        f"two-tailed p_FWE < .05, 5000 sign flips, 18-connectivity.",
                        cft, voxel_mm3, urpe_note)

        for cft, p in THRESHOLDS:
            if p not in mass_thresholds:
                continue
            render_page(pdf, source,
                        lambda s, sig, t, sn: load_nilearn(s, sig, t, sn, "mass"),
                        f"nilearn cluster-MASS FWE — {source}",
                        f"Cluster-forming t > {cft:.4f} (one-tailed p < {p:g}); clusters with "
                        f"two-tailed p_FWE < .05 on summed excess t rather than extent.",
                        cft, voxel_mm3, urpe_note)

        render_page(pdf, source,
                    lambda s, sig, t, sn: load_nilearn(s, sig, t, sn, "tfce"),
                    f"TFCE — {source}",
                    "Threshold-free cluster enhancement, two-tailed p_FWE < .05, "
                    "5000 sign flips. No cluster-forming threshold is chosen at all, so "
                    "this page is the one that does not depend on the arbitrary decision "
                    "the rest of the document sweeps over.",
                    1e-6, voxel_mm3, urpe_note)

        render_page(pdf, source,
                    lambda s, sig, t, sn: load_nilearn(s, sig, t, sn, "t"),
                    f"Voxel-wise (max-t) FWE — {source}",
                    "Voxel-level familywise error, two-tailed p_FWE < .05, 5000 sign flips. "
                    "The strictest and least assumption-laden test here: no spatial "
                    "criterion of any kind.",
                    1e-6, voxel_mm3, urpe_note)

        render_roi_pages(pdf, source)
    print("wrote", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=["model7", "model2"])
    parser.add_argument("--out-dir", default=FIGDIR)
    args = parser.parse_args()
    for source in args.sources:
        make_pdf(source, op.join(args.out_dir, f"threshold_sweep_{source}.pdf"))
