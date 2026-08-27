"""ROI-level specificity of the three learning signals (RPE, uRPE, surprise).

Reads the per-subject ROI betas extracted from model7 on the cluster
(SPM/code/extract_learning_signal_betas.py), then:
  1. names each ROI by matching its coordinates to the reported cluster peaks,
  2. runs paired t-tests between all three learning-signal betas within each ROI
     (Holm-corrected across all pairwise tests) plus one-sample tests vs 0,
  3. makes the supplementary figure: betas per ROI x regressor, subject dots
     colored by best-fitting RL model (Modelling/Fitting/BestFitting.tsv),
  4. writes a drop-in LaTeX appendix table.

Run locally:
    ~/mambaforge/bin/python notes/figures/plot_learning_signal_roi_betas.py
"""

import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = op.dirname(op.dirname(op.abspath(__file__)))          # notes/
REPO = op.dirname(ROOT)
DATA = op.join(ROOT, "data", "specificity_betas")
FIGDIR = op.join(ROOT, "figures")

# Reported cluster peaks (MNI) from the manuscript / dissociation note
KNOWN = {
    "rpe": [
        ("Caudate/putamen R", (18, 14, -12)),
        ("Caudate/putamen L", (-16, 14, -12)),
        ("Angular gyrus L", (-46, -66, 44)),
        ("Visual cortex L (mid. occ.)", (-46, -82, 2)),
        ("vmPFC", (8, 54, -8)),
        ("Visual cortex L (lingual)", (-22, -88, -12)),
        ("Cerebellum R", (42, -70, -40)),
    ],
    "surprise": [
        ("Precuneus L", (-4, -54, 40)),
        ("dlPFC L", (-28, 24, 48)),
        ("Angular gyrus R", (48, -72, 37)),
        ("Angular gyrus L", (-36, -82, 40)),
    ],
    "urpe": [
        ("dmPFC R", (2, 32, 37)),
        ("Anterior insula R", (36, 24, -5)),
        ("Insula L", (-34, 26, -2)),
        ("Middle frontal gyrus R", (32, 60, 2)),
        ("Orbital IFG R", (50, 36, -12)),
        ("Superior frontal sulcus R", (48, 24, 37)),
        ("Inferior parietal lobule R", (44, -46, 48)),
    ],
}
MATCH_RADIUS = 25.0  # mm; ROIs whose coordinates match no reported peak are dropped

NETWORK_LABEL = {"rpe": "RPE regions", "surprise": "Surprise regions", "urpe": "uRPE regions"}
SIGNAL_LABEL = {"rpe": "RPE", "urpe": "uRPE", "surprise": "Surprise"}
SIGNAL_ORDER = ["rpe", "urpe", "surprise"]
MODEL_PALETTE = {"basic": "#3B5BA5", "transfer": "#C44E52", "asym": "#5D8C3F"}
MODEL_LABEL = {"basic": "Basic", "transfer": "Transfer", "asym": "Asym"}

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def name_rois(roi_info):
    """Match each extracted ROI to a reported peak; drop unmatched (unreported) clusters."""
    rows = []
    for _, r in roi_info.iterrows():
        # uRPE clusters carry a t-map peak; model2 masks only a centre of mass
        coord = np.array([r.peak_x, r.peak_y, r.peak_z]) if np.isfinite(r.peak_x) \
            else np.array([r.com_x, r.com_y, r.com_z])
        cands = KNOWN[r.roi_set]
        dists = [np.linalg.norm(coord - np.array(xyz)) for _, xyz in cands]
        i = int(np.argmin(dists))
        rows.append(dict(roi_id=r.roi_id, roi_set=r.roi_set, n_vox=r.n_vox_native,
                         name=cands[i][0], match_mm=dists[i],
                         x=coord[0], y=coord[1], z=coord[2]))
    named = pd.DataFrame(rows)
    dropped = named[named.match_mm > MATCH_RADIUS]
    if len(dropped):
        print("Dropped unreported clusters (no peak within "
              f"{MATCH_RADIUS:g} mm):\n{dropped.to_string(index=False)}\n")
    named = named[named.match_mm <= MATCH_RADIUS].copy()
    # If two clusters map to the same reported region, keep the closer/larger one
    named = (named.sort_values(["match_mm"])
                  .drop_duplicates(subset=["roi_set", "name"], keep="first"))
    return named


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    betas = pd.read_csv(op.join(DATA, "learning_signal_betas.tsv"), sep="\t",
                        dtype={"subject": str})
    roi_info = pd.read_csv(op.join(DATA, "learning_signal_roi_info.tsv"), sep="\t")
    named = name_rois(roi_info)

    df = betas.merge(named[["roi_id", "roi_set", "name"]], on=["roi_id", "roi_set"])
    # One value per subject x ROI x signal (mean over the 6 runs, both modalities)
    subj = (df.groupby(["roi_set", "name", "signal", "subject"], as_index=False)
              .beta.mean())

    fits = pd.read_csv(op.join(REPO, "Modelling", "Fitting", "BestFitting.tsv"),
                       sep="\t", header=None, names=["subject", "variant", "bic"],
                       dtype={"subject": str})
    subj = subj.merge(fits[["subject", "variant"]], on="subject")
    assert not subj.variant.isna().any()

    # ---------- t-tests ----------
    wide = subj.pivot_table(index=["roi_set", "name", "subject"],
                            columns="signal", values="beta").reset_index()
    # neg_urpe = sign-flipped uRPE ("outcome expectedness"). Note: the paired
    # test urpe vs neg_urpe is mathematically identical to urpe vs 0.
    wide["neg_urpe"] = -wide["urpe"]
    pairs = [("urpe", "rpe"), ("urpe", "surprise"), ("rpe", "surprise"),
             ("rpe", "neg_urpe"), ("surprise", "neg_urpe"),
             ("urpe", "neg_urpe")]
    recs = []
    for (roi_set, name), g in wide.groupby(["roi_set", "name"], sort=False):
        for s in SIGNAL_ORDER:  # one-sample vs 0
            t, p = stats.ttest_1samp(g[s], 0)
            recs.append(dict(roi_set=roi_set, roi=name, test=f"{s} vs 0",
                             kind="onesample", t=t, p=p, dz=g[s].mean() / g[s].std(),
                             n=len(g)))
        for a, b in pairs:  # paired between signals
            diff = g[a] - g[b]
            t, p = stats.ttest_rel(g[a], g[b])
            recs.append(dict(roi_set=roi_set, roi=name, test=f"{a} vs {b}",
                             kind="paired", t=t, p=p, dz=diff.mean() / diff.std(),
                             n=len(g)))
    res = pd.DataFrame(recs)
    for kind in ["onesample", "paired"]:
        m = res.kind == kind
        res.loc[m, "p_holm"] = holm(res.loc[m, "p"].values)
    res.to_csv(op.join(DATA, "learning_signal_roi_ttests.tsv"), sep="\t", index=False)

    # ---------- supplementary figure: one page per network ----------
    # Dots dodged by best-fitting RL model; brackets = Holm-corrected paired
    # t-tests between the three modulators (the "interaction" tests).
    SHORT = {"Visual cortex L (mid. occ.)": "Visual L (mid. occ.)",
             "Visual cortex L (lingual)": "Visual L (lingual)",
             "Superior frontal sulcus R": "Sup. frontal sulcus R",
             "Middle frontal gyrus R": "Mid. frontal gyrus R",
             "Inferior parietal lobule R": "Inf. parietal lobule R"}
    subj["signal_label"] = subj.signal.map(SIGNAL_LABEL)
    subj["variant_label"] = subj.variant.map(MODEL_LABEL)
    subj["short_name"] = subj["name"].map(lambda n: SHORT.get(n, n))
    # 4th x-position: uRPE sign-flipped ("outcome expectedness") — makes the
    # negative uRPE loading in RPE/surprise regions explicit. Same betas, not
    # an additional regressor.
    flipped = subj[subj.signal == "urpe"].copy()
    flipped["beta"] *= -1
    flipped["signal_label"] = "−uRPE"
    subj = pd.concat([subj, flipped], ignore_index=True)
    # Per page, the network's own signal comes first (bolded x tick)
    PAGE_ORDER = {"rpe": ["rpe", "urpe", "surprise"],
                  "surprise": ["surprise", "rpe", "urpe"],
                  "urpe": ["urpe", "rpe", "surprise"]}
    hue_order = [MODEL_LABEL[k] for k in ["basic", "transfer", "asym"]]
    palette = {MODEL_LABEL[k]: v for k, v in MODEL_PALETTE.items()}

    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

    def bracket(ax, x1, x2, y, txt, color="0.35"):
        h = 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.6, c=color,
                clip_on=False, solid_capstyle="butt")
        ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom",
                fontsize=6.5 if txt == "ns" else 8, color=color)

    # The interaction tests: paired t-tests between the network's own signal and
    # each of the other two; x positions resolved from the page's tick order
    BRACKET_TESTS = {
        "rpe": [("urpe vs rpe", "urpe", "rpe"),
                ("rpe vs surprise", "rpe", "surprise"),
                ("rpe vs neg_urpe", "rpe", "neg_urpe")],
        "surprise": [("urpe vs surprise", "urpe", "surprise"),
                     ("rpe vs surprise", "rpe", "surprise"),
                     ("surprise vs neg_urpe", "surprise", "neg_urpe")],
        "urpe": [("urpe vs rpe", "urpe", "rpe"),
                 ("urpe vs surprise", "urpe", "surprise"),
                 ("urpe vs neg_urpe", "urpe", "neg_urpe")],
    }
    paired = res[res.kind == "paired"].set_index(["roi_set", "roi", "test"])

    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(op.join(FIGDIR, "learning_signal_roi_betas.pdf"))
    ymin, ymax = subj.beta.min(), subj.beta.max()
    NCOL = 3
    for rs in ["rpe", "surprise", "urpe"]:
        dat = subj[subj.roi_set == rs]
        coords = dict(KNOWN[rs])  # reported peak MNI per region name
        names_sorted = list(named[named.roi_set == rs]
                            .sort_values("n_vox", ascending=False).name)
        panel_order = [SHORT.get(n, n) for n in names_sorted]
        order = [SIGNAL_LABEL[s] for s in PAGE_ORDER[rs]] + ["−uRPE"]
        xpos = {s: k for k, s in enumerate(PAGE_ORDER[rs])}
        xpos["neg_urpe"] = 3
        brackets = [(test, *sorted((xpos[a], xpos[b])))
                    for test, a, b in BRACKET_TESTS[rs]]
        g = sns.FacetGrid(dat, col="short_name", col_order=panel_order,
                          col_wrap=NCOL, height=2.0, aspect=1.25, sharey=True)

        def _panel(data, **kws):
            ax = plt.gca()
            ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
            sns.stripplot(data=data, x="signal_label", y="beta", order=order,
                          hue="variant_label", hue_order=hue_order,
                          palette=palette, dodge=True, size=2.2, alpha=0.55,
                          jitter=0.09, legend=False, ax=ax, zorder=1)
            # Manuscript style: widest horizontal line = group mean, narrower
            # lines = ±1 SEM (no vertical bars/caps)
            for xi, sig in enumerate(order):
                v = data.loc[data.signal_label == sig, "beta"]
                m, se = v.mean(), v.std() / np.sqrt(len(v))
                ax.hlines(m, xi - 0.24, xi + 0.24, color="black", lw=2.0,
                          zorder=100)
                ax.hlines([m - se, m + se], xi - 0.13, xi + 0.13, color="black",
                          lw=0.9, zorder=100)
            roi = data["name"].iloc[0]
            for k, (test, x1, x2) in enumerate(brackets):
                p = paired.loc[(rs, roi, test), "p_holm"]
                bracket(ax, x1, x2, ymax + 0.12 + 0.32 * k, stars(p))

        g.map_dataframe(_panel)
        for ax, name in zip(g.axes.flat, names_sorted):
            x, y, z = coords[name]
            ax.set_title(f"{SHORT.get(name, name)}\n"
                         f"MNI {x:g}, {y:g}, {z:g}", fontsize=8, linespacing=1.3)
        g.set_xlabels("")
        g.set_ylabels("Beta (a.u.)")
        g.set(ylim=(ymin - 0.05, ymax + 1.1))
        for k, ax in enumerate(g.axes.flat):
            if k % NCOL != 0:
                ax.tick_params(labelleft=False)

        handles = [mpl.lines.Line2D([], [], marker="o", linestyle="none",
                                    markersize=4, color=MODEL_PALETTE[k],
                                    label=MODEL_LABEL[k])
                   for k in ["basic", "transfer", "asym"]]
        n_rows = int(np.ceil(len(panel_order) / NCOL))
        legend_free_cell = len(panel_order) < n_rows * NCOL
        g.figure.legend(handles=handles, title="Best-fitting RL model",
                        ncol=1 if legend_free_cell else 3,
                        loc="lower right" if legend_free_cell else "lower center",
                        bbox_to_anchor=(0.97, 0.05) if legend_free_cell
                        else (0.5, -0.01))
        g.figure.suptitle(NETWORK_LABEL[rs], fontsize=11, fontweight="bold",
                          fontfamily="Arial", x=0.02, y=0.99, ha="left")
        g.figure.text(0.02, 0.005,
                      "−uRPE: the uRPE betas sign-flipped (outcome "
                      "expectedness) — same data, not an additional regressor. "
                      "The uRPE-vs-−uRPE bracket is equivalent to testing uRPE "
                      "against zero.", fontsize=7, color="0.4")
        sns.despine(fig=g.figure, offset=3, trim=False)
        g.figure.set_size_inches(7.25, 1.2 + 2.1 * n_rows)
        g.figure.subplots_adjust(top=0.76 if n_rows == 1
                                 else 0.87 if n_rows == 2 else 0.91)
        g.figure.canvas.draw()  # materialize tick labels before styling them
        for ax in g.axes.flat:  # network's own signal first, in bold
            texts = [t.get_text() for t in ax.get_xticklabels()]
            if texts and texts[0] == order[0]:
                # set_xticklabels pins the labels so the weight survives redraws.
                # macOS Helvetica.ttc exposes no bold face to matplotlib, so use
                # metric-compatible Arial Bold for the emphasized tick.
                new = ax.set_xticklabels(texts)
                new[0].set_fontweight("bold")
                new[0].set_fontfamily("Arial")
        pdf.savefig(g.figure)
        g.figure.savefig(op.join(FIGDIR, f"learning_signal_roi_betas_{rs}.png"))
        plt.close(g.figure)
    pdf.close()
    print("wrote figure + ", op.join(DATA, "learning_signal_roi_ttests.tsv"))

    # ---------- LaTeX appendix table (paired tests only) ----------
    def fmt_p(p):
        return "$<.001$" if p < .001 else f"${p:.3f}$"

    # Build a wide stats table: one row per ROI, columns per comparison
    piv = res[res.kind == "paired"].pivot_table(
        index=["roi_set", "roi"], columns="test", values=["t", "p_holm"], sort=False)
    lines = [
        r"\subsection{\rev Pairwise comparisons of learning-signal betas per region}",
        "",
        r"\begin{table}[htbp]", r"    \centering", r"    \rev",
        r"    \begin{threeparttable}",
        r"    \caption{Paired $t$-tests between the three learning-signal parametric",
        r"    modulators of model7 (signed RPE, unsigned RPE (uRPE), Shannon surprise),",
        r"    computed on subject-level mean betas within each previously reported",
        r"    region ($n=58$). $p$-values are Holm-corrected across all",
        f"    {int((res.kind == 'paired').sum())} pairwise tests.}}",
        r"    \label{tab:roi_signal_ttests}",
        r"    \begin{tabularx}{\textwidth}{X rr rr rr}", r"        \toprule",
        r"        & \multicolumn{2}{c}{uRPE $-$ RPE} & \multicolumn{2}{c}{uRPE $-$ surprise} & \multicolumn{2}{c}{RPE $-$ surprise} \\",
        r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"        \textbf{Region} & $t$ & $p$ & $t$ & $p$ & $t$ & $p$ \\",
        r"        \midrule",
    ]
    for rs in ["rpe", "surprise", "urpe"]:
        block = piv.loc[rs] if rs in piv.index.get_level_values(0) else None
        if block is None:
            continue
        lines.append(rf"        \multicolumn{{7}}{{l}}{{\textit{{{NETWORK_LABEL[rs]}}}}} \\")
        for roi, row in block.iterrows():
            cells = []
            for test in ["urpe vs rpe", "urpe vs surprise", "rpe vs surprise"]:
                cells.append(f"{row[('t', test)]:.2f}")
                cells.append(fmt_p(row[("p_holm", test)]))
            lines.append(f"        {roi} & " + " & ".join(cells) + r" \\")
        lines.append(r"        \addlinespace")
    lines += [
        r"        \bottomrule", r"    \end{tabularx}",
        r"    \begin{tablenotes}\footnotesize",
        r"    \item Betas are averaged over the six runs; all modulators were",
        r"    z-scored per run before entering the GLM, so betas share a common",
        r"    scale. Positive $t$: the first-named signal has the larger beta.",
        r"    \end{tablenotes}",
        r"    \end{threeparttable}", r"\end{table}%",
    ]
    tex = op.join(ROOT, "paper", "latex", "appendices", "roi_signal_ttests.tex")
    with open(tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", tex)

    # Console summary
    with pd.option_context("display.width", 200):
        print(res[res.kind == "paired"]
              .assign(sig=lambda d: np.where(d.p_holm < .05, "*", ""))
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
