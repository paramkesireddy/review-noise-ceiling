# Pre-registration — noise ceiling and evaluation artifacts in app-review intent classification

Frozen before the annotation sample is drawn. Later changes are logged as dated amendments in data/processed/manifest.json.

## Already known at pre-registration (measured, results/tables/summary_DS3.csv, summary_DS4.csv, distilbert_cv_summary.csv)
- Reference metric recovered: one-vs-rest P/R/F1 on one-hot gold, macro-averaged; reproduces the reference study's Table 2 to 0.1.
- Fine-tuned DistilBERT (faithful re-implementation, attention masks at inference): DS3 78.6 ± 0.5 (3 seeds), DS4 85.8 ± 0.4 (OVR);
  single-label argmax 79.8 / 87.2. Reference paper reports 63.8 (DS3) and 59.2 (DS4, original 7-class scheme).
- Qwen3-4B zero-shot, prompt P2, greedy, strict parser: DS3 79.4, DS4 87.5. GPT-4o on the same rows (reference package): 79.55 / 87.62.
- Prompt wording moves Qwen3-4B by 3.4 (DS3) / 3.2 (DS4) points; the reference study's substring parser lowers GPT-4o by 5.5 (DS3) when
  its outputs are re-parsed first-mention; Qwen3-1.7B is below parity (ratio 0.84–0.93).
- Feature request is the weakest class for every system (62–78 F1 vs 86–92 for bug/other).

## Hypotheses and decision rules (fixed)
- H1 (artifacts, RQ1): with a correctly trained baseline, one parser and one prompt, the GPT-4o minus DistilBERT gap is < 2 macro-F1
  points on DS3 and DS4 (OVR view). Measured: 1.0 and 1.8. Supported.
- H2 (convergence, RQ2): pairwise error overlap (Jaccard) between any two of {DistilBERT seed 1 argmax, Qwen3-4B P2-G, GPT-4o}
  exceeds the overlap expected under independent errors at the observed error rates. Permutation test, 10,000 draws, alpha = 0.01,
  Holm correction over the 3 pairs per dataset.
- H3 (noise ceiling, RQ3): population gold-label error rate epsilon >= 8% on DS3 (point estimate); the convergence F1 lies within the
  95% CI of the estimated noise ceiling. Refuted if epsilon < 4% with upper CI < 8%. Indeterminate otherwise.
- H4 (attribution, RQ4): for each of the three systems, errors are >= 3x enriched on items adjudicated wrong-or-contested (risk ratio,
  95% CI excluding 1; Holm over 3 systems). CI excluding 1 but not 3 is reported as "enriched, below threshold".
- H5 (shift, RQ5, if time): cross-dataset F1 of fine-tuned systems (DS3->DS4, DS4->DS3) drops >= 10 points vs in-distribution.

## Strata (computed by script before sampling; stratum weights are exact, not assumed)
Systems: f1 = DistilBERT seed 1 out-of-fold argmax; f2 = Qwen3-4B, P2, mode G, strict parser; f3 = GPT-4o, reference-parser label.
"Agrees with gold" = prediction is a single label contained in the gold set (gold may be multi-label).
h1: all three agree with gold. h2: all three output the same single label and it is not in gold. h3: everything else.

## Sample
Per dataset (DS3, DS4): n_h = 100 per stratum, stratum taken whole if smaller; seed 20261027; opaque item keys; two per-annotator
queues in different seeded orders containing only the review text and the P1 class definitions. External annotator: 100 items
(50 per dataset, stratified), seed 20261028.

## Annotation and adjudication
Two authors, independent, blind to gold, model outputs and strata. Label = one of bug report / feature request / other; optional
multi-intent flag; optional note. Dry run on 20 DS1 items first (not in the sample). Reconciliation: one discussion per
disagreement with the guidelines open, still blind. Statuses assigned by script: gold correct; gold wrong (both annotators agree on
one other label after reconciliation); contested (still disagree, or both flag multi-intent). epsilon counts gold wrong only.
Agreement reported before and after reconciliation (Cohen's kappa, percent agreement), and vs the external annotator.

## Estimators and tests
epsilon_hat = sum_h W_h m_h/n_h with finite-population variance; per-stratum Clopper–Pearson 95% intervals. Ceiling = macro-F1 of the
adjudicated labels scored against original gold with stratum weights, bootstrap CI. Risk ratio with log-normal CI and Fisher exact.
Overlap: Jaccard with permutation null. Parity: paired bootstrap B = 2000 (seed 1), McNemar exact, Cohen's kappa. Differences under
2 F1 points are pre-declared practically negligible.

## Seeds
k-fold 1234; training seeds 1,2,3; bootstrap 1; annotation sample 20261027; external subset 20261028. Inference is greedy.

## Scope cuts if behind schedule (in order)
Drop RQ5 -> drop external annotator -> drop DS4 annotation (keep DS3) -> drop ablation A2. Never drop: blinding, DS3 stratified
estimate, artifact decomposition, bootstrap intervals.
