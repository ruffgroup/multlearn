# Threshold sweep: how much of the RPE / uRPE / surprise "specificity" is a thresholding artefact?

Follow-up to `surprise_vs_rpe_dissociation.md` and `learning_signal_specificity_roi.md`.
The question here is Gilles's: the paper's specificity claims rest on three maps
thresholded separately, and if RPE is significant nearly everywhere while uRPE is always
smaller and often *opposite in sign*, the non-overlap may be a statement about statistical
power rather than about anatomy.

Everything below is whole-brain, n = 58 (the GLM subject set), 5000 permutations.

## What was run

Two engines, on the **same** 58 first-level contrast images and the **same** analysis mask
(SnPM's own, 59,838 voxels):

- **SnPM** — the paper's own inference. The 2nd level was computed with `ST_later`
  (`ST_U = -1`), and `SnPM.mat` stores `ST_Ut = 2.39356756`, which happens to be exactly
  t at one-tailed p = .01 for df = 57. So the whole sweep re-runs `snpm_pp` on the existing
  permutations — no re-permutation was needed.
  (`SPM/code/snpm_threshold_sweep.py`)
- **nilearn** `non_parametric_inference` — adds cluster-*mass* FWE, voxel-wise (max-t) FWE
  and TFCE, which SnPM cannot do. Patched to 18-connectivity to match `spm_clusters`.
  (`SPM/code/nilearn_threshold_sweep.py`)

Cluster-forming thresholds are the exact t for one-tailed p at df = 57, not the FSL z
conventions: **2.3936** (p<.01), **2.6649** (.005), **3.2395** (.001), **3.9756** (1e-4),
**5.2929** (1e-6).

Two model sources, because the manuscript mixes them:

| Source | Choice modulators | Feedback modulators | Note |
|---|---|---|---|
| **model2** | surprise, V | rpe | RPE and surprise as reported in the paper |
| **model7** | surprise, V | **urpe, rpe** | all three signals in one design |

`model3` is `model2` with two changes only: choice events get `duration = 0`, and it passes
`orth=["No"]` — which is a **no-op** (see below). There is no model3 output on the cluster.
`model6` is model7 without signed RPE.

## Orthogonalisation: yes, and not the way the Methods section says

`nipype_helpers.py` passes `orth=["No"] * len(conditions)` inside the `Bunch`, but nipype
never forwards it: there is no `orth` handling in `nipype/algorithms/modelgen.py` (its only
`orth` is an internal helper for temporal derivatives) or in `nipype/interfaces/spm/model.py`.
**SPM therefore applied its default within-condition serial orthogonalisation in every model.**
This confirms the suspicion recorded in `paper/specificity_edits.md` EDIT 5.

In model7 the feedback pmod is `Bunch(name=["urpe", "rpe"], ...)` — **uRPE first**. So signed
RPE is residualised against uRPE, and uRPE keeps all shared variance. This matters twice:

- The model7 RPE map is the *unique* signed-RPE variance; the model2 RPE map is the full
  effect. The two PDFs are not interchangeable for RPE.
- The ordering biases uRPE *positive*. The negative uRPE loadings reported below therefore
  cannot be an artefact of modulator order — they survive a bias pointing the other way.

## Result 1 — RPE really is significant nearly everywhere

Surviving extent as a percentage of the analysis mask (SnPM, cluster-extent FWE p<.05):

| Signal (model2 / model7) | p<.01 | p<.005 | p<.001 | p<1e-4 | p<1e-6 |
|---|---|---|---|---|---|
| RPE positive (model2) | **65.9%** | 61.7% | 52.5% | 41.1% | **22.5%** |
| RPE positive (model7) | 64.9% | 60.3% | 50.1% | — | — |
| uRPE negative | 30.3% | 25.4% | 16.7% | 8.6% | 1.6% |
| uRPE positive | 2.2% | 2.4% | 2.0% | 1.0% | 0.2% |
| Surprise positive (model2) | 1.1% | 1.6% | 0.9% | 0.2% | 0.0% |
| Surprise negative | 0 at every threshold |

RPE still covers **a fifth of the brain at p<1e-6**. At any threshold at which RPE forms a
map you could describe anatomically, it is a single connected component covering half the
mask — at which point cluster-extent inference is not telling you anything about location.
The panels flag this automatically.

The uRPE **negative** tail is an order of magnitude larger than the positive tail at every
threshold. The uRPE effect the paper reports is the small half of a mostly-negative map.

## Result 2 — the surprise effect sits exactly on p = .05, and which side depends on the procedure

For the surprise clusters at cluster-forming t > 2.39 / 2.66, nilearn's two-tailed
cluster-**extent** p is **.071** and **.054** — not significant — while SnPM's one-tailed
family puts the same clusters under .05, and nilearn cluster-**mass** gives **.048** and
**.021**. Same data, three defensible choices, opposite verdicts.

## Result 3 — cluster extent is non-monotonic here, which is a warning sign in itself

Raising the cluster-forming threshold sometimes *increases* the surviving volume, because at
a low threshold real clusters merge with noise and the permuted max-cluster-size distribution
runs away:

- Surprise (model7): 616 → **894** → 590 → 211 → 6 voxels across the five thresholds.
- Surprise (model2): 668 → **977** → 510 → 129 → 5.
- RPE negative (model2): — → 253 → **330** → 196 → 61.

## Result 4 — the ROI overview

93 Harvard-Oxford parcels are touched by at least one signal's surviving clusters at
cluster-forming t > 3.24. Anatomical parcels are used deliberately: they are defined
independently of all three contrasts, so no signal gets a home-field advantage. Each parcel's
mean contrast value is one-sample t-tested per signal, Holm-corrected over all parcel × signal
tests (`SPM/code/roi_sign_matrix.py`).

- RPE significantly **positive** in **64** parcels.
- uRPE significantly **negative** in **28**; positive in few.
- The two co-occur in **22** parcels.
- Shannon surprise reaches significance in **0** parcels, in either direction.
- Most common pattern: RPE+ / uRPE n.s. / surprise n.s. (42 of 93 parcels).

Typical rows (group t): Putamen R **+9.3 / −3.9 / −2.0**; Frontal Medial Cortex R
**+8.6 / −6.5 / +1.0**; Amygdala L **+7.9 / −5.3 / +0.5**; Lateral Occipital inf. L
**+9.1 / −5.6 / +0.0** (RPE / uRPE / surprise).

Wherever RPE is strongly positive, uRPE is negative and surprise is flat. Combined with the
orthogonalisation ordering (which favours uRPE), the natural reading is that across most of
the brain the "uRPE" regressor is tracking the inverse of the RPE response rather than an
independent unsigned-prediction-error signal — and that any uRPE-vs-RPE difference contrast
will therefore be significant almost everywhere for reasons that have nothing to do with
regional specialisation.

## Engine comparison: does SPM vs nilearn matter?

Mostly no, and where it does, it is the marginal effects that move.

- **Median Dice between the SnPM and nilearn survivor maps = 1.00** across the panels. For
  RPE and uRPE the two engines usually agree to the voxel (e.g. RPE p<.01: 38,852 vs 38,852).
- The disagreements are all in the surprise maps and in small uRPE clusters, i.e. exactly
  where the cluster p is near .05. Three engine differences drive them:
  1. **Family.** SnPM's `Tsign` runs each tail as its own one-tailed family (what the paper
     reports); nilearn's `two_sided_test=True` controls FWE over both tails at once and is
     roughly twice as strict in the tail.
  2. **Statistic.** SnPM offers cluster size only; cluster mass rewards tall-narrow clusters
     that extent penalises, and it rescues the surprise clusters that extent kills.
  3. **Connectivity.** SPM uses 18-connectivity, nilearn hardcodes 6. At t>2.39 that alone
     turns the model7 RPE map from 8 components into 1. The runs here are patched to 18.

### One nilearn trap, recorded so nobody repeats it

`non_parametric_inference(threshold=...)` takes the cluster-forming threshold **in p-scale**,
and with `two_sided_test=True` converts it internally as `t.isf(threshold / 2, df)`. Passing
t values does not error — it silently returns cluster maps in which every in-mask voxel
carries p ≈ 0.9998. The sweep script now passes twice the one-tailed p and asserts that the
resulting t matches the SnPM run.

## Why the cluster counts here are far below the paper's tables

The manuscript enumerates regions with `nilearn.reporting.get_clusters_table`, which lists
**local maxima ≥ 8 mm apart**. One sprawling connected component becomes a dozen rows. On the
same maps (model7, SnPM survivors):

| Map | Voxels | Components (18-conn) | Components (6-conn) | `get_clusters_table` rows |
|---|---|---|---|---|
| RPE+ t>3.24 | 29,988 | 1 | 18 | 21 |
| uRPE+ t>3.24 | 1,215 | 4 | 7 | 13 |
| Surprise+ t>3.24 | 590 | 4 | 4 | 13 |

Both numbers are printed in every panel header so the pages can be read against the paper's
tables.

## Files

- Figures: `notes/figures/threshold_sweep_model7.pdf`, `notes/figures/threshold_sweep_model2.pdf`
  (21 pages each: summary curves; SnPM cluster-extent at 5 thresholds; nilearn cluster-extent
  at 5; nilearn cluster-mass at 2; TFCE; voxel-wise FWE; ROI sign patterns; ROI bars).
- Maps and tables: `notes/data/threshold_sweep/` (`snpm/`, `nilearn/conn18/`, `roi_matrix/`,
  `summary_model{2,7}.tsv`).
- Scripts: `SPM/code/{sweep_config,snpm_threshold_sweep,nilearn_threshold_sweep,roi_sign_matrix}.py`,
  `SPM/code/fetch_threshold_sweep.sh`, `SPM/cluster/submit_threshold_sweep.sh`,
  `notes/figures/plot_threshold_sweep.py`.

**Pending:** the five TFCE runs (~2 h each, threshold-free, so one page not five). The TFCE
page is the one that does not depend on the arbitrary choice everything else sweeps over, and
is the most important single page for settling the question.

## Figure: where the maps overlap

`notes/figures/fig_overlap_model7.pdf` (and `_model2`), from
`notes/figures/fig_overlap.py`. ROIs are the clusters actually found in the six
contrasts at **one common cluster-forming threshold** (t > 3.9756, one-tailed p < 1e-4),
so no signal gets a more permissive threshold than another; names follow the paper's own
cluster tables, falling back to Talairach gyrus labels for territories the paper never
reported (`SPM/code/contrast_roi_overlap.py`). AAL would have been the closer match to the
paper's vocabulary but gin.cnrs.fr serves it over a certificate that fails verification.

The organising split is the finding: **7 of 21 territories fall outside the signed-RPE map,
14 inside.**

- **Outside**: five of the six uRPE-positive clusters — dmPFC R, superior frontal sulcus R,
  frontal middle gyrus R, insula R, insula L — plus inferior parietal lobule R (found by
  uRPE−) and a small right insula cluster (found by RPE−). These are the regions where the
  reported uRPE specificity holds up.
- **Inside**: everything else. The surprise clusters are on average **75%** covered by the
  RPE-positive map, and the uRPE-negative clusters **61%**. In every one of them the RPE
  effect is positive and Holm-significant, and the uRPE effect is negative.
- The one uRPE-positive cluster that lands inside the RPE map is **inferior parietal lobule R**
  — the same region the earlier ROI analysis (`learning_signal_specificity_roi.md`) had already
  flagged as shared rather than uRPE-specific. Two independent routes to the same exception.

So the claim that survives is narrower than the paper's: a right frontal / anterior insular
uRPE network is genuinely separable from the RPE response, and the surprise network is not.

## Robustness across inference rules

Three variants of the same figure, each generated end-to-end under one rule
(`SPM/code/contrast_roi_overlap.py --variant`, `notes/figures/fig_overlap.py --variant`).
The rule is printed large at the top of every figure, together with the surviving extent of
all six maps, so nothing is hidden by the ROI decomposition.

| Variant | Rule | ROIs | Outside the RPE map |
|---|---|---|---|
| `extent_p1e4` | cluster-forming t > 3.98 (p < 1e-4), cluster-extent FWE | 21 | 7 |
| `extent_p1e2` | cluster-forming t > 2.39 (p < .01), cluster-extent FWE | 4 | 0 |
| `tfce` | no cluster-forming threshold, TFCE, two-tailed FWE | 2 | 0 |

Surviving extent as a share of the analysis mask:

| Map | t > 3.98 | t > 2.39 | TFCE |
|---|---|---|---|
| RPE + | 37.5% | 64.9% | 42.9% |
| RPE − | 0.1% | 0 | 0 |
| uRPE + | 1.0% | 2.2% | **0.04%** (27 voxels) |
| uRPE − | 8.6% | 30.3% | 26.6% |
| Surprise + | 0.4% | 1.0% | **0** |
| Surprise − | 0 | 0 | 0 |

Two things follow.

- **The regional decomposition is a product of the threshold.** At p < .01 every map is a
  single confluent blob and there are only four nameable territories; under TFCE, two. The
  set of "regions" the paper reports exists only in a window of cluster-forming thresholds
  strict enough to fragment the maps.
- **Only the RPE-positive / uRPE-negative pair is robust.** It is large under every rule.
  Under TFCE the surprise map is **empty** and the uRPE-positive map is **27 voxels**.
  State the caveat honestly, though: TFCE integrates over cluster-forming thresholds in a way
  that rewards broad smooth effects and penalises small focal ones, and the uRPE-positive
  clusters are exactly that (largest 143 voxels at t > 3.98). So the right conclusion is that
  the frontal/insular uRPE-positive effect is **method-dependent** — clear under cluster-extent
  inference at a strict threshold, near-absent under TFCE — not that it is spurious.

## Engine overview figure

`notes/figures/fig_engines_model7.pdf` (and `_model2`), from `notes/figures/fig_engines.py`.

**Both engines are permutation tests.** SnPM and nilearn each run a sign-flipping
permutation test with 5000 permutations, on the same 58 first-level contrast images and the
same analysis mask. Neither uses a parametric random-field approximation. Cluster-extent FWE
in nilearn is the max-cluster-size null built from those sign flips, exactly as in SnPM.

What actually differs is (i) the **family** — SnPM's `Tsign` tests each tail as its own
one-tailed family, nilearn's `two_sided_test=True` controls both tails at once and is about
twice as strict in the tail; (ii) the **statistic** — SnPM offers extent only, nilearn adds
cluster mass, voxel-wise max-t and TFCE; (iii) **connectivity** — SPM uses 18, nilearn
hardcodes 6, patched to 18 here so it is not in play.

Result: **16 of 23 panels agree to the voxel** (Dice > 0.995). The disagreements are all
marginal effects and all in the same direction, SnPM being the more permissive:

| Panel | SnPM | nilearn extent |
|---|---|---|
| Signed RPE −, p < .001 | 88 vox | 0 |
| Surprise +, p < .01 | 616 vox | 0 |
| Surprise +, p < .005 | 894 vox | 0 |

The figure also puts the two threshold-free corrections on the same axis, which is the
compact version of the whole document: RPE + survives everything (≈ 40% of the mask under
TFCE, ≈ 20% under voxel max-t), uRPE − survives everything (≈ 27% under TFCE), and
surprise + survives cluster-extent inference at a permissive family and nothing else.

## The non-orthogonalised GLM was never run — verified in SPM.mat

Read straight out of `/shares/zne.uzh/multlearn/nipype/model7/1stLevel/sub-01/SPM.mat`:

```
model7  cond=ChoiceTactile     orth=1   pmods=['surprise', 'V']
model7  cond=FeedbackTactile   orth=1   pmods=['urpe', 'rpe']
model2  cond=ChoiceTactile     orth=1   pmods=['surprise', 'V']
model2  cond=FeedbackTactile   orth=1   pmods=['rpe']
```

`SPM.Sess(k).U(j).orth = 1` in every condition of both models: SPM's serial
orthogonalisation was applied. The design matrix confirms it — the correlation between the
uRPE and signed-RPE modulator columns is **exactly 0.0000 in all six runs**, which is the
signature of `spm_orth` having residualised the second modulator against the first.

Why the intent was lost:

- `orth=["No"] * len(conditions)` is passed in the `Bunch` for models **1, 3, 4, 5 and the
  PPI** — but **not** for model2, model6 or model7, i.e. not for the one model where two
  modulators share an event and it would matter.
- Even where it is passed it is a no-op: nipype has no handling of `Bunch.orth` in
  `nipype/algorithms/modelgen.py` (its only `orth` is an internal helper for temporal
  derivatives) or anywhere in `nipype/interfaces/spm/`.

So the Methods sentence "No orthogonalization was applied to the regressors of interest"
(main.tex:371) is wrong, as `paper/specificity_edits.md` EDIT 5 suspected — and the run that
would have made it true does not exist.

**Why it matters here.** In model7 uRPE is entered first, so it keeps every scrap of variance
it shares with signed RPE and the RPE map is the residual. The large negative uRPE map is
therefore *not* an artefact of ordering — ordering biases uRPE the other way — but the
positive uRPE effect could in principle be inflated by it. Only the non-orthogonalised run
settles that.

**How to run it.** SPM12's batch exposes `spm.stats.fmri_spec.sess.cond.orth`, but nipype
never writes it. Two workable routes:

1. Insert a MATLAB step between `Level1Design` and `EstimateModel` that loads `SPM.mat`, sets
   `SPM.Sess(k).U(j).orth = 0` for every condition, and calls `spm_fMRI_design(SPM)` to
   rebuild `SPM.xX.X` without orthogonalisation. Smallest change to the existing pipeline.
2. Emit the `matlabbatch` for the design directly with `cond.orth = 0`, bypassing nipype's
   Bunch translation for this model only.

Cost: first level for 58 subjects × 6 runs, then a 5000-permutation second level — a few
hours of cluster time. Everything downstream (sweep, ROIs, figures) is already parameterised
by model source, so a `model7_noorth` would drop straight in.
