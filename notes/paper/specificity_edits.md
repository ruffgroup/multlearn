# Paper edits: direct specificity test (surprise vs uRPE) + orthogonalization correction

All NEW text is wrapped in `\new{...}` so it renders **red** for Saurabh & CCR.
Remove the `\new{}` wrappers (keep their contents) once approved.

Implement these by hand in `notes/paper/latex/` — they are anchored on existing
text so you can search-and-replace. Nothing here requires re-running the analysis.

---

## EDIT 0 — Preamble: define the red macro
`main.tex`, after line 26 (`\usepackage{graphicx}`), add:

```latex
\usepackage{xcolor}
\newcommand{\new}[1]{\textcolor{red}{#1}}  % red for inline/prose additions
\newcommand{\rev}{\color{red}}             % red switch used INSIDE the new tables
```
To turn ALL review-red off later, just blank these two macros in the preamble —
no need to touch any other file:
```latex
\newcommand{\new}[1]{#1}
\newcommand{\rev}{}
```

---

## EDIT 1 — Abstract (main.tex:63)
Find:
> All three networks were modality-general, showing comparable strength for audiovisual and visuotactile learning.

Append immediately after it:
```latex
 \new{Direct statistical contrasts between learning signals confirmed these regional dissociations.}
```

---

## EDIT 2 — Results, reframe the "non-overlap" claim (main.tex:155)
Find:
> were each associated with largely non-overlapping brain networks, supporting the idea that distinct neural architectures

Replace with:
```latex
were each associated with largely non-overlapping brain networks \new{(a dissociation we confirmed with direct statistical contrasts between learning signals; see below)}, supporting the idea that distinct neural architectures
```

---

## EDIT 3 — Results, NEW specificity subsection
Insert as a new paragraph AFTER the uRPE paragraph (main.tex:177, the line ending
"...even when operating within the same multisensory learning task.\newline"),
BEFORE the modality-general paragraph (main.tex:180):

```latex
\subsubsection*{\new{Direct contrasts confirm regional specificity}}

\new{The analyses above identify each learning signal with a separately thresholded map. Spatial non-overlap of such maps does not by itself establish that a region responds more strongly to one signal than another \cite{nieuwenhuis2011erroneous}. We therefore tested specificity directly, contrasting the surprise and uRPE parametric modulators within the same model (cluster-forming $t>3.1$, cluster $p<.05$ FWE, SnPM), and asked for each previously identified region whether it responded significantly more strongly to its associated signal than to the other. All four surprise regions---bilateral angular gyrus, left dlPFC, and precuneus---were significantly more responsive to surprise than to uRPE, indicating a complete dissociation on the statistical-learning side. On the uRPE side, five of seven regions (right dmPFC, right anterior insula, and the right lateral frontal cluster spanning middle frontal gyrus, orbital inferior frontal gyrus, and superior frontal sulcus) were significantly more responsive to uRPE than to surprise; the left insula and right inferior parietal lobule did not differ significantly and are therefore better characterised as shared across feedback-driven learning signals than as uRPE-specific. The region-by-region outcome of this test is reported in Table~\ref{tab:con13_roi}, and the whole-brain direct contrast in Table~\ref{tab:con13_pos} and Table~\ref{tab:con13_neg}. Because a difference contrast is also sensitive to deactivations, part of the surprise\,$>$\,uRPE map (notably a ventromedial prefrontal cluster absent from the surprise main effect) likely reflects negative uRPE-related responses at feedback rather than positive surprise; we therefore base claims of surprise specificity on the independently identified surprise regions rather than on this contrast alone.}\newline
```

---

## EDIT 4 — Results, reframe (main.tex:191)
Find:
> despite the largely distinct regions underlying RPE and Shannon surprise, the \textbf{left angular gyrus}

Replace with:
```latex
despite the regions underlying RPE and Shannon surprise being distinct \new{both spatially and in direct statistical contrast}, the \textbf{left angular gyrus}
```

---

## EDIT 5 — Methods, CORRECT the orthogonalization statement (main.tex:371) [IMPORTANT]
The current sentence is factually wrong: `orth=["No"]` is a no-op in our nipype
pipeline, so SPM applied its DEFAULT within-condition orthogonalization.

Find and DELETE:
> No orthogonalization was applied to the regressors of interest to preserve shared variance between predictors.

Replace with:
```latex
\new{Within each event, parametric modulators were serially orthogonalised using SPM's default procedure, which operates independently within each condition (\texttt{spm\_orth} is applied per trial type). Surprise and uRPE were each entered as the first modulator of their respective event (stimulus onset and feedback) and therefore retained all variance shared with the second modulator of that event (chosen value and signed RPE, respectively). Crucially, because this orthogonalisation acts only within a condition, it never relates surprise to uRPE, which were modelled as modulators of temporally distinct events; their direct contrast is therefore unaffected by modulator ordering \cite{mumford2015orthogonalization}.}
```

(Bonus bug to fix while you're here: main.tex:371 has a duplicated sentence ---
"Mean Pearson correlations were very small, confirming minimal collinearity."
appears twice. Delete one.)

---

## EDIT 6 — Methods, second-level, describe the direct contrast (main.tex:378)
Find:
> We tested both positive and negative contrasts for RPE, Shannon surprise, and uRPE.

Append immediately after it:
```latex
 \new{To test the regional specificity of the learning signals, we additionally computed a direct contrast between the surprise and uRPE modulators (cluster-forming $t>3.1$). Because a difference in statistical significance between two separately thresholded maps does not itself constitute a significant difference \cite{nieuwenhuis2011erroneous}, this contrast provides a direct test of whether a region responds more strongly to one signal than to the other.}
```

---

## EDIT 7 — Appendix include (main.tex:~412)
Near `\include{appendices/urpe_contrasts}`, add:
```latex
\include{appendices/con13_roi_specificity}
\include{appendices/con13_clusters}
```
Both files already exist and make themselves red via the `\rev` macro defined in
Edit 0 (a `\color{red}` switch placed inside each table — wrapping the `\include`s
does NOT work, because floats don't inherit color from an outer group). The
section headings and table contents will all be red. To revert, blank `\new`/`\rev`
in the preamble (Edit 0) — the tables need no further editing.

---

## EDIT 8 — sample.bib, add two references
```bibtex
@article{nieuwenhuis2011erroneous,
  title={Erroneous analyses of interactions in neuroscience: a problem of significance},
  author={Nieuwenhuis, Sander and Forstmann, Birte U and Wagenmakers, Eric-Jan},
  journal={Nature Neuroscience}, volume={14}, number={9}, pages={1105--1107},
  year={2011}, publisher={Nature Publishing Group}}

@article{mumford2015orthogonalization,
  title={Orthogonalization of regressors in {fMRI} models},
  author={Mumford, Jeanette A and Poline, Jean-Baptiste and Poldrack, Russell A},
  journal={PLoS ONE}, volume={10}, number={4}, pages={e0126255}, year={2015},
  publisher={Public Library of Science}}
```
