# Direct contrast: unsigned RPE vs. surprise — cluster enumeration & cross-reference

**Question (Christian Ruff):** does the paper contain a *direct* statistical contrast
β(surprise) vs. β(RPE)? It does not (Figs report the three effects on separate maps).
This note runs that direct contrast and cross-references it against the clusters we
already reported, in both directions.

- **Contrast:** model7, `con13` = `[surprise < urpe]` → positive tail = **uRPE > surprise**,
  negative tail = **surprise > uRPE**.
- **Inference:** SnPM cluster-extent FWE (cluster-forming T = 3.1, cluster p_FWE < 0.05).
  The `SnPM_filtered` maps contain **only surviving voxels** (verified earlier with the
  all-or-nothing test: 0 partial clusters; survivors ≥163 vox, dropped ≤83 vox).
- **Scale comparability:** both modulators are z-scored per run in our Python code
  (`nipype_helpers.py`), so the β's are on a common scale before SPM. SPM itself only
  mean-centers parametric modulators (via default `spm_orth`); it does **not** rescale to
  unit variance or to range. So the larger-scale worry is handled upstream.
- **Method = same as before:** `nilearn.reporting.get_clusters_table` on the SnPM-filtered
  map, exactly as in `SPM/code/extract_betas.ipynb` / `extract_network_ROIs.ipynb`. Note
  voxel volume here is **31.5 mm³** (3.0×3.0×3.5), not the 15.5 used in the older 2.5 mm maps.

## New direct-contrast clusters (con13)

### uRPE > surprise (807 voxels total)
| Region | Voxels | Max T | MNI (x,y,z) |
|---|---|---|---|
| dmPFC / dACC R | 171 | 6.76 | 2, 32, 37 |
| Anterior insula R | 163 | 6.43 | 36, 24, −5 |
| Lateral PFC R (MFG / orbital IFG / SFS) | 473 | 5.77 | 42, 54, 20 |

### surprise > uRPE (8833 voxels total)
| Region | Voxels | Max T | MNI (x,y,z) |
|---|---|---|---|
| vmPFC / mPFC (extended, midline DMN) | 6740 | 7.77 | −6, 50, −8 |
| Angular gyrus / lateral occipital R | 1514 | 7.16 | 42, −78, 26 |
| Insula / frontal operculum R | 225 | 5.03 | 42, 2, 9 |
| Middle temporal gyrus L | 140 | 5.54 | −52, −4, −19 |
| Putamen / insula R | 113 | 5.46 | 38, −16, −2 |
| Supramarginal / angular gyrus L | 91 | 4.49 | −46, −36, 26 |

(Residual <5-voxel blobs omitted.)

## Cross-reference: are the previously-reported clusters confirmed by the direct contrast?

For each previously-reported peak we read the **signed direct-contrast t** at the peak
voxel (and the most extreme value within ±2 voxels). `t@peak` is the value exactly at the
reported coordinate; `t(±2vx)` is the strongest survivor in its immediate neighborhood.

### Unsigned-RPE clusters → do they survive **uRPE > surprise**?
| uRPE region | MNI peak | t@peak | t(±2vx) | Stronger than surprise? |
|---|---|---|---|---|
| dmPFC R | 2, 32, 37 | 6.76 | 6.76 | **YES** |
| Insula R | 36, 24, −5 | 6.43 | 6.43 | **YES** |
| Insula L | −34, 26, −2 | 0.00 | 0.00 | no (n.s.) |
| Frontal middle gyrus R | 32, 60, 2 | 5.24 | 5.35 | **YES** |
| Orbital IFG R | 50, 36, −12 | 3.71 | 4.37 | **YES** |
| Superior frontal sulcus R | 48, 24, 37 | 4.19 | 4.56 | **YES** |
| Inferior parietal lobule R | 44, −46, 48 | 0.00 | 0.00 | no (n.s.) |

→ **5 of 7** uRPE clusters are significantly stronger for uRPE than surprise. The two
exceptions (left insula, right IPL) correlate with uRPE but are **not** significantly
*more* uRPE-driven than surprise (i.e. shared, not uRPE-specific).

### Surprise clusters → do they survive **surprise > uRPE**?
| Surprise region | MNI peak | t@peak | t(±2vx) | Stronger than uRPE? |
|---|---|---|---|---|
| Precuneus L | −4, −54, 40 | −3.91 | −5.57 | **YES** |
| DLPFC L | −28, 24, 48 | −5.06 | −6.09 | **YES** |
| Angular gyrus R | 48, −72, 37 | −4.00 | −6.08 | **YES** |
| Angular gyrus L | −36, −82, 40 | −5.48 | −5.88 | **YES** |

→ **All 4** surprise clusters are significantly stronger for surprise than uRPE. Clean,
complete dissociation on the surprise side.

### Signed-RPE clusters (con1, reward/value network) → vs **uRPE > surprise** *(separate comparison)*
The paper's "RPE" (con1) map is **signed** RPE (caudate/putamen, vmPFC, visual, cerebellum —
classic reward network). It is **not** directly testable against a uRPE-vs-surprise contrast,
but for completeness, sampling the same con13 map shows these regions go the *opposite* way
(several fall inside surprise > uRPE) or are n.s. — i.e. signed-RPE/reward regions are not
uRPE-specific.

| Signed-RPE region | MNI peak | t@peak | t(±2vx) | con13 direction |
|---|---|---|---|---|
| Caudate/putamen R | 18, 14, −12 | 0.00 | −3.13 | (opp.) surp>uRPE |
| Caudate/putamen L | −16, 14, −12 | 0.00 | −3.93 | (opp.) surp>uRPE |
| Angular gyrus L | −46, −66, 44 | 0.00 | −4.84 | (opp.) surp>uRPE |
| Visual ctx mid/lat occ L | −46, −82, 2 | −5.04 | −5.54 | (opp.) surp>uRPE |
| vmPFC | 8, 54, −8 | −5.23 | −7.30 | (opp.) surp>uRPE |
| Visual ctx lingual/fusiform L | −22, −88, −12 | −3.77 | −4.03 | (opp.) surp>uRPE |
| Cerebellum R | 42, −70, −40 | 0.00 | 0.00 | n.s. |

## Reverse direction: are the interaction peaks NOT in the main effects?

A difference/interaction can be significant where *neither* main effect is positive — e.g.
driven by the other regressor going **negative**. Distance from each con13 peak to the
nearest *relevant* main-effect peak (<8 mm ≈ same peak; 8–18 mm = edge/extends; >18 mm = new):

**uRPE > surprise peaks vs uRPE main-effect peaks**
| con13 peak | MNI | vox | nearest uRPE peak | dist | verdict |
|---|---|---|---|---|---|
| dmPFC/dACC R | 2,32,37 | 171 | dmPFC R | 0 mm | in main effect |
| Ant insula R | 36,24,−5 | 163 | Insula R | 0 mm | in main effect |
| Lateral PFC R | 42,54,20 | 473 | FMG R | 21 mm | partly new (same R-PFC territory, peak shifted) |

**surprise > uRPE peaks vs surprise main-effect peaks**
| con13 peak | MNI | vox | nearest surprise peak | dist | verdict |
|---|---|---|---|---|---|
| vmPFC/mPFC | −6,50,−8 | 6740 | DLPFC L | 66 mm | **NEW — not in surprise main effect** |
| Angular/LOC R | 42,−78,26 | 1514 | Angular R | 14 mm | edge/extends |
| Insula/operc R | 42,2,9 | 225 | (all far) | 79 mm | **NEW** |
| MTG L | −52,−4,−19 | 140 | (all far) | 76 mm | **NEW** |
| Putamen/insula R | 38,−16,−2 | 113 | (all far) | 69 mm | **NEW** |
| SMG/Angular L | −46,−36,26 | 91 | (all far) | 48 mm | **NEW** |

→ On the **uRPE>surprise** side the contrast essentially re-finds the uRPE main-effect regions
(2/3 exact; the lateral-PFC peak shifts ~21 mm but stays in right PFC).
→ On the **surprise>uRPE** side only the angular/LOC cluster matches a surprise main-effect
peak. The dominant cluster — vmPFC/mPFC, 6740 vox — and four others (right insula/operculum,
putamen, left MTG, left SMG) are **NOT** in the surprise main effect. They are almost
certainly driven by **uRPE going negative** there (feedback-related DMN/value deactivation),
not by positive surprise. **Caveat:** confirming the uRPE<0 driver needs the uRPE main-effect
map (currently only on the cluster, temporarily offline). This is an important interpretational
point: a large part of "surprise>uRPE" is a uRPE *de*activation effect, not a surprise effect.

## Bottom line
- The direct contrast **does** exist and is well-powered in both directions.
- **Surprise dissociation is complete**: every surprise cluster is significantly stronger
  for surprise than for uRPE.
- **uRPE dissociation is near-complete**: 5/7 uRPE clusters (dmPFC, right insula, right
  lateral PFC) are significantly stronger for uRPE than surprise; left insula and right IPL
  are shared rather than uRPE-specific.
- The paper's signed-"RPE" (reward) network is a different system and is not uRPE-specific.

## Files
- New contrast maps (survivors): `notes/data/surprise_vs_rpe/signed_t_uRPEvsSurprise_model7_con13.nii.gz`
- Latex appendix table (drop-in, matches con1/con5 style): `notes/paper/latex/appendices/con13_clusters.tex`
- Figure (manuscript colors): `notes/figures/uRPE_vs_surprise_model7_con13_mscolors.{png,pdf}`

*Caveat:* exact per-cluster p_FWE values for con13 are marked `<0.05` (the filtered maps are
by construction the cluster-FWE survivor set). Exact per-cluster p's live in the SnPM output
on the cluster (`/shares/zne.uzh/multlearn/nipype/model7/2ndLevel/...con13/`) — I can pull
them if you want them in the table.
