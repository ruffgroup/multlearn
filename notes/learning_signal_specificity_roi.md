# ROI-level specificity: pairwise t-tests between RPE, uRPE & surprise + supplementary figures

Follow-up on `surprise_vs_rpe_dissociation.md` (whole-brain surprise-vs-uRPE contrast).
Here: (1) all three learning signals tested **against each other** with paired t-tests in
every previously reported region, (2) a supplementary betas-per-ROI figure colored by
best-fitting RL model, (3) a whole-brain 3×3 "dissociation matrix" figure (figure S).

## Method

- **Betas**: model7 first level, which contains all three modulators in one GLM —
  surprise (stimulus onset), uRPE (feedback, 1st modulator), signed RPE (feedback, 2nd
  modulator). All modulators z-scored per run → betas on a common scale. Per subject
  (n = 58, GLM set), each modulator's betas averaged over the 6 runs and over voxels
  within each ROI (`SPM/code/extract_learning_signal_betas.py`, run on cluster).
- **ROIs** (18 extracted, 15 named/reported):
  - 7 RPE regions: model2 con1 cluster masks (`cluster_*_con1_8_0_pos.nii`, t > 8.0).
    Unreported small clusters (mid-cingulate, caudate body R, 2 visual) dropped.
  - 4 surprise regions: model2 con5 masks (t > 3.1).
  - 7 uRPE regions: connected components of model7 con1 `SnPM_filtered_t4_0_pos`
    (**t > 4.0, the manuscript's uRPE threshold** — at 3.1 the right-frontal clusters
    merge). Peaks match the paper's table exactly (dmPFC 2/32/37, ant. insula R,
    insula L, MFG R, orbital IFG R, SFS R, IPL R).
- **Tests**: paired t-tests per ROI for uRPE−RPE, uRPE−surprise, RPE−surprise; Holm
  correction across all 45 pairwise tests. One-sample tests vs 0 as well (in the TSV).

## Results (Holm-corrected)

- **RPE regions: complete dissociation, 7/7.** Every RPE region shows RPE > uRPE
  (t = 8.3–15.6, all p < 10⁻⁹) *and* RPE > surprise (t = 7.6–15.6). Striatal effects
  are enormous (dz ≈ 2).
- **Surprise regions: dissociated vs uRPE (4/4), but NOT vs RPE.** All four show
  surprise > uRPE (t = 4.3–6.7). However, the significant RPE-vs-surprise
  differences (t = 3.0–4.9) go the **other way**: the RPE beta (mean 0.24, all
  four one-sample p_holm < .001) exceeds the surprise beta (mean 0.07, all
  significant > 0) in every surprise region. Consistent at the map level:
  surprise > RPE survives in only 446 voxels whole-brain (model2 con13 neg),
  while RPE > surprise covers ~28k. So surprise regions carry a genuine surprise
  signal at stimulus onset *and* a larger feedback-locked RPE signal —
  "surprise-specific vs RPE" is NOT supported; the specific claim that holds is
  surprise-vs-uRPE. (Caveat: comparing a stimulus-locked to a feedback-locked
  modulator crosses events; the RPE main effect is near-global, so this mostly
  restates that RPE modulates almost everything at feedback.)
- **uRPE regions: 7/7 vs surprise; 3–5/7 vs RPE (correction-dependent).** All
  seven show uRPE > surprise. uRPE > RPE: dmPFC R, anterior insula R, SFS R are
  solid; insula L and orbital IFG R are borderline (uncorrected p = .005/.002;
  p_holm = .047/.025 when correcting over the 45 core pairwise tests, but
  .098/.052 once the vs-−uRPE tests join the family, 90 tests). Clearly NOT
  uRPE-specific vs RPE: middle frontal gyrus R (p ≈ .02 unc.) and IPL R
  (t ≈ −0.2) — shared between the feedback signals.
- **Each signal vs −uRPE (sign-flipped uRPE)** — "is the own-signal response
  bigger than the outcome-expectedness response?":
  - RPE regions: RPE > −uRPE everywhere (t = 4.2–11.2, all Holm-sig).
  - **Surprise regions: surprise vs −uRPE is n.s. in all four regions**
    (uncorrected p = .006–.894; none Holm-sig). The stimulus-locked surprise
    response is statistically indistinguishable in size from the feedback-locked
    uRPE-deactivation — a strong, quantitative version of the earlier map-level
    caveat that "surprise > uRPE" is roughly half surprise, half uRPE
    deactivation.
  - uRPE regions: uRPE vs −uRPE significant everywhere (equivalent to the
    one-sample uRPE test by construction).
  Holm correction in the TSV now spans all 90 paired tests; the borderline
  effects above straddle .05 depending on family — report both if used.
- Side observation: in RPE regions uRPE betas are consistently *negative*
  (uRPE < surprise there too), consistent with the note's earlier point that reward
  regions deactivate with outcome surprise.

## Caveats to state in the paper

- uRPE-vs-RPE comparisons (t-tests **and** the con24 map): both are feedback
  modulators, uRPE entered first, so SPM's serial orthogonalisation credits shared
  variance to uRPE (Mumford et al. 2015). Surprise comparisons are immune (different
  events). The MFG/IPL "shared" result is conservative *against* uRPE-specificity, so
  it is not an artifact of ordering; the strong uRPE > RPE effects could in principle
  be inflated by it.
- RPE > surprise at t 3.1 covers ~28k voxels; as with surprise > uRPE, difference
  maps are also driven by the other regressor going negative — figure caption should
  say the region claims rest on the main-effect-defined ROIs.

## Model-type coloring (Saurabh's numbers are already in the repo)

`Modelling/Fitting/BestFitting.tsv` = per-subject winner (subject, variant, BIC),
exactly the 58 GLM subjects: **basic 31, asym 16, transfer 11**.
⚠️ The manuscript (main.tex:110 & :328) says "Basic 31, **Transfer 16, Asym 11**" —
the Transfer/Asym counts are swapped relative to the TSV. One of the two is wrong;
check with Saurabh which labels the fitting actually produced.
(`BestFittingCommon.tsv` is a different model space — Pearce variants — not used.)

## Whole-brain pairwise maps (figure S)

- surprise vs uRPE: model7 con13 (existing).
- surprise vs RPE: model2 con13 (existing).
- **uRPE vs RPE: new — model7 con24** = per-subject `con_0001 − con_0019`
  (`SPM/code/make_con24_urpe_vs_rpe.py`; identical to the [±1/6] contrast by
  linearity), then the standard SnPM 2nd level
  (`sbatch --array=24 submit_GLM_2ndlevel_nipype.sh`; must be submitted from a
  **login shell** or `module load matlab` silently fails → matlab exit 127).
  Results (cluster-forming t = 3.1, cluster p < .05 FWE, 5000 perms):

  **uRPE > RPE (493 vox, 3 clusters)** — the focal uRPE core:
  | Region | Peak MNI | Max t | mm³ |
  |---|---|---|---|
  | Anterior insula R | 44, 24, −2 | 6.96 | 6961 |
  | dmPFC / dACC (ext. pre-SMA) | 6, 36, 37 | 6.68 | 4756 |
  | Lateral PFC R (SFS/MFG) | 48, 18, 40 | 4.55 | 3811 |

  **RPE > uRPE (31,320 vox)** — one merged mega-cluster (985k mm³) peaking in
  bilateral striatum (18, 14, −12, t = 15.97; −12, 18, −16, t = 14.43) and
  bilateral occipital cortex (t ≈ 11), plus two small clusters (temporal pole R,
  cerebellar vermis). All reward-network regions sit inside it — map-level
  confirmation of the 7/7 ROI result.

  ⚠️ **SnPM bug at high thresholds**: `snpm_pp.m` always loops both signs for
  T-stats; when the *positive* side has zero observed voxels above the
  cluster-forming threshold (uRPE > RPE peaks at t = 6.96 < 8), `Locs_vox` is
  undefined → crash ("Unrecognized function or variable 'Locs_vox'", line 902).
  So a genuine SnPM inference at CF t = 8 on con24 is impossible. The figure's
  "RPE > uRPE (t > 8.0)" panel is therefore the **t3.1 cluster-FWE survivor map
  display-thresholded at |t| > 8** (1361 vox; near-equivalent, since with peak
  t = 16 any t8 cluster would trivially exceed the t8 FWE cluster-size
  criterion). All other higher-threshold panels are genuine SnPM inference
  outputs (model2 con13 t8, model7 con13 t4, con24 t4).
- ⚠️ File gotcha: `notes/data/surprise_vs_rpe/SnPM_filtered_t3_1_{pos,neg}.nii` are
  **model2** con13 files; the model7 con13 survivors are the
  `con13_uRPE_gt_surprise/-surprise_gt_uRPE.nii.gz` files (807 / 8833 vox).

## Files

- Data: `notes/data/specificity_betas/learning_signal_{betas,roi_info}.tsv`,
  `learning_signal_roi_ttests.tsv`; maps in `notes/data/figureS_maps/`.
- Figures: `notes/figures/learning_signal_roi_betas.pdf` (3 pages, one per
  network; dots dodged by best-fitting model; brackets = Holm-corrected paired
  t-tests of the network's signal vs the other two; 4th x-position "−uRPE" =
  the uRPE betas sign-flipped, making the negative feedback-locked uRPE loading
  in RPE/surprise regions explicit as "outcome expectedness"; per-page PNGs
  `learning_signal_roi_betas_{rpe,surprise,urpe}.png`),
  `notes/figures/figureS_dissociation_matrix.{pdf,png}` (3×5 matrix: main effect
  with t3.1 extent light + manuscript threshold saturated, then each direct
  contrast at t3.1 AND at the row's main-contrast threshold).
- LaTeX: `notes/paper/latex/appendices/roi_signal_ttests.tex` (drop-in, `\rev`-red).
- Scripts: `SPM/code/extract_learning_signal_betas.py`,
  `SPM/code/make_con24_urpe_vs_rpe.py`,
  `notes/figures/plot_learning_signal_roi_betas.py`,
  `notes/figures/plot_figureS_dissociation.py`.
