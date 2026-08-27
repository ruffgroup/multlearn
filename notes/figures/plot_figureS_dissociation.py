"""Figure S: dissociation matrix of the three learning signals.

3 x 3 grid of glass brains. Row = learning signal (manuscript colors:
RPE red, uRPE green, surprise blue). Column 1 = the signal's main effect
(its manuscript map/threshold); columns 2-3 = the direct SnPM contrasts
against the other two signals, shown in the direction favoring the row's
signal. All maps are SnPM cluster-FWE survivor maps (p < .05).

Maps live in notes/data/figureS_maps/ (rsynced from the cluster; see names
below for provenance). The uRPE-vs-RPE panels come from model7 con24
(con_0001 - con_0019; SPM/code/make_con24_urpe_vs_rpe.py) -- if those files
are not present yet the cells render as "Pending".

Run locally:
    ~/mambaforge/bin/python notes/figures/plot_figureS_dissociation.py
"""

import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors as mcolors
from nilearn import plotting

ROOT = op.dirname(op.dirname(op.abspath(__file__)))  # notes/
MAPS = op.join(ROOT, "data", "figureS_maps")
FIGDIR = op.join(ROOT, "figures")

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def truncated(cmap_name, lo=0.35, hi=0.95):
    base = mpl.colormaps[cmap_name]
    return mcolors.LinearSegmentedColormap.from_list(
        cmap_name + "_trunc", base(np.linspace(lo, hi, 256)))


# 5 panels per row: main effect (light layer = t>3.1 extent, saturated = the
# manuscript's cluster-forming threshold), then each direct contrast at BOTH
# t>3.1 and the row's main-contrast threshold (t>8 RPE, t>4 uRPE). Surprise's
# manuscript threshold already is 3.1, so those cells carry a note instead.
# Cell = (title, filename-or-None, threshold-label, base_layer-or-None).
ROWS = [
    dict(signal="Signed RPE", event="feedback", cmap=truncated("Reds"),
         color="#C44E52", light="#F5C6C2",
         panels=[
             ("RPE > 0", "rpe_gt0_model2_con1_t8_0.nii", "t > 8.0 on t > 3.1",
              "rpe_gt0_model2_con1_t3_1.nii"),
             ("RPE > uRPE", "rpe_gt_urpe_model7_con24_t3_1.nii", "t > 3.1", None),
             ("RPE > uRPE", "rpe_gt_urpe_model7_con24_t8_0.nii", "t > 8.0", None),
             ("RPE > surprise", "rpe_gt_surprise_model2_con13_t3_1.nii", "t > 3.1", None),
             ("RPE > surprise", "rpe_gt_surprise_model2_con13_t8_0.nii", "t > 8.0", None),
         ]),
    dict(signal="Unsigned RPE", event="feedback", cmap=truncated("Greens"),
         color="#2E7D32", light="#C3E0C5",
         panels=[
             ("uRPE > 0", "urpe_gt0_model7_con1_t4_0.nii", "t > 4.0 on t > 3.1",
              "urpe_gt0_model7_con1_t3_1.nii"),
             ("uRPE > RPE", "urpe_gt_rpe_model7_con24_t3_1.nii", "t > 3.1", None),
             ("uRPE > RPE", "urpe_gt_rpe_model7_con24_t4_0.nii", "t > 4.0", None),
             ("uRPE > surprise", "urpe_gt_surprise_model7_con13_t3_1.nii.gz", "t > 3.1", None),
             ("uRPE > surprise", "urpe_gt_surprise_model7_con13_t4_0.nii", "t > 4.0", None),
         ]),
    dict(signal="Shannon surprise", event="stimulus", cmap=truncated("Blues"),
         color="#3B5BA5", light="#C9D4EC",
         panels=[
             ("Surprise > 0", "surprise_gt0_model2_con5_t3_1.nii", "t > 3.1", None),
             ("Surprise > RPE", "surprise_gt_rpe_model2_con13_t3_1.nii", "t > 3.1", None),
             (None, None, "Main-contrast threshold\nalready t > 3.1 (see left)", None),
             ("Surprise > uRPE", "surprise_gt_urpe_model7_con13_t3_1.nii.gz", "t > 3.1", None),
             (None, None, "Main-contrast threshold\nalready t > 3.1 (see left)", None),
         ]),
]


def load_survivors(path):
    """Load an SnPM filtered map as positive survivor t-values (or None)."""
    if not op.exists(path):
        return None
    img = nib.load(path)
    data = np.abs(np.nan_to_num(np.squeeze(img.get_fdata())))  # neg maps store signed t
    if (data > 0).sum() == 0:
        return "empty"
    return nib.Nifti1Image(data.astype(np.float32), img.affine)


def main():
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(3, 5, left=0.065, right=0.995, top=0.91, bottom=0.01,
                          hspace=0.25, wspace=0.05)

    for i, row in enumerate(ROWS):
        for j, (title, fn, thr, base_fn) in enumerate(row["panels"]):
            ax = fig.add_subplot(gs[i, j])
            if title is None:  # surprise row: no higher main threshold exists
                ax.set_axis_off()
                ax.text(0.5, 0.5, thr, transform=ax.transAxes, ha="center",
                        va="center", fontsize=7.5, color="0.55", style="italic")
                continue
            img = load_survivors(op.join(MAPS, fn))
            if img is None or img == "empty":
                ax.set_axis_off()
                note = "No suprathreshold clusters" if img == "empty" \
                    else "Pending (SnPM running)"
                ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color="0.45", style="italic")
            else:
                vmax = np.percentile(img.get_fdata()[img.get_fdata() > 0], 99.5)
                base = load_survivors(op.join(MAPS, base_fn)) if base_fn else None
                if base is not None and base != "empty":
                    # Light layer: full t>3.1 survivor extent (flat tint)
                    disp = plotting.plot_glass_brain(
                        base, axes=ax, display_mode="lzr", threshold=1e-6,
                        cmap=mcolors.ListedColormap([row["light"]]),
                        colorbar=False, plot_abs=True, annotate=False,
                        black_bg=False)
                    disp.add_overlay(img, threshold=1e-6, cmap=row["cmap"],
                                     vmax=vmax)
                else:
                    plotting.plot_glass_brain(
                        img, axes=ax, display_mode="lzr", threshold=1e-6,
                        cmap=row["cmap"], vmax=vmax, colorbar=False,
                        plot_abs=True, annotate=False, black_bg=False)
            ax.set_title(f"{title}   ({thr})", fontsize=9, color=row["color"],
                         pad=2)
        # Row label on the left margin
        fig.text(0.012, gs[i, 0].get_position(fig).y1 - 0.02,
                 f"{row['signal']}\n({row['event']}-locked)",
                 rotation=90, va="top", ha="left", fontsize=9,
                 fontweight="bold", color=row["color"])

    fig.text(0.10, 0.965,
             "Each map: SnPM cluster-extent FWE survivors (cluster p < .05); "
             "columns 2–3 are direct contrasts between modulators.\n"
             "Main-effect panels: light shade = t > 3.1 extent, saturated = "
             "the manuscript's cluster-forming threshold.",
             fontsize=8, color="0.35", va="bottom")

    for ext in ["pdf", "png"]:
        fig.savefig(op.join(FIGDIR, f"figureS_dissociation_matrix.{ext}"))
    print("wrote", op.join(FIGDIR, "figureS_dissociation_matrix.pdf"))


if __name__ == "__main__":
    main()
