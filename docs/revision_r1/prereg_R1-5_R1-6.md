# Revision round 1 — analysis plan for R1-5 (calibration) and R1-6 (multiplicity)

Written and committed before the analyses below were run. No model is trained or
refitted: every number comes from the submitted out-of-fold (OOF) and held-out
predictions (pass B, the predictions behind the frozen results). Eyes, masks and folds are
those of `fusion_eval_common.load_all` — 240 OOF eyes, 37 held-out eyes from 19 patients,
52 locations.

## Branches

| Branch | Definition |
|---|---|
| Summary | XGBoost on the 26 printed parameters |
| Image, per backbone | Inception-ResNet-v2, Inception-v3, VGG16, Xception, DenseNet121 |
| Image ensemble | mean of the five backbones |
| Centred image ensemble | ensemble minus one constant. OOF fold k: the mean residual (prediction − measurement) of the ensemble over the masked locations of the other four folds, as in `centred_ensemble.py`. Held-out: the mean residual over all five OOF folds (no held-out data used) |
| Fusion | Inception-ResNet-v2 + summary. OOF: nested w (fitted on the other four folds); held-out: w fitted on all OOF folds (0.47) |
| Ensemble fusion | image ensemble + summary, same weight convention |

The weight conventions are those of `bias_by_bin.py` and `fusion_sensitivity_trend.py`.
Held-out predictions are the mean of the five fold models.

## R1-5 — calibration

For every branch, split (OOF, held-out) and stratum, pooled over masked locations:

* **Mean bias** = mean(ŷ − y), dB.
* **Calibration slope and intercept**: ordinary least squares of measured on predicted,
  y = a + b·ŷ. Perfect calibration is b = 1, a = 0.

Strata:

* All locations.
* Mean bias by **measured**-sensitivity bin [0,10), [10,20), [20,30), [30,∞) dB, the bins
  of the submitted analysis.
* Slope, intercept and mean bias by **predicted**-sensitivity bin, same edges. Slope and
  intercept are not computed within bins of measured sensitivity: selecting on the
  outcome pulls the within-bin slope toward 0 for any predictor with error, including a
  perfectly calibrated one, so those numbers would not describe calibration. Bias by
  measured bin carries the same caveat (regression toward the mean) and is reported with
  it, as the submitted figure caption already does.

Uncertainty: 95% percentile intervals from a patient-clustered bootstrap (5000 resamples,
seed 42), resampling patients and keeping all their eyes and locations.

**Centred ensemble as a main comparator.** Per-eye RMSE and MAE, paired across eyes:

* fusion vs centred ensemble;
* ensemble fusion vs centred ensemble;

on OOF and held-out. Two-sided Wilcoxon signed-rank test and the patient-clustered
bootstrap interval of the mean difference. Holm correction within each split across the
four tests (two fusions × two metrics); raw and adjusted p values are both reported.

## R1-6 — multiplicity for the sensitivity-bin comparisons

Within each eye and measured-sensitivity bin with at least three locations, per-eye RMSE
of fusion minus image (the quantity of the submitted `per_bin` rows). Comparisons:

* fusion vs Inception-ResNet-v2 image branch;
* ensemble fusion vs image ensemble;

in each of the four bins, on OOF and held-out. Per comparison: mean and median
difference, number of eyes, two-sided Wilcoxon signed-rank p, patient-clustered bootstrap
interval. **Family = the 8 bin-level tests of one split** (4 bins × 2 comparators); Holm
adjustment within that family, OOF and held-out separately. The submitted trend tests
(slope across bins) are not re-run; they are cited from `fusion_sensitivity_trend.json`.

## Outputs

`experiments/revision_r1/calibration_multiplicity.py` → `calibration_multiplicity.json`
(producer), frozen through `freeze_paper_results.py` and checked by `check_numbers.py`
before any number enters the response or the manuscript.

## Decision log

* 2026-10-13, authors. The sensitivity-bin analyses are reported in the revision as
  exploratory, with the Holm adjustment above, and are not described as pre-specified.
  Reason: the bin edges, the minimum of three locations per bin and the primary trend cell
  are written into `fusion_sensitivity_trend.py` from its first saved version and never
  changed afterwards, but that first saved version (2026-08-14 16:32, from the editor's
  local history; the working tree was not under version control then) is the same moment
  at which the memo reporting the bin-level results was first saved. No record shows the
  definitions fixed before the results were seen.
* 2026-10-14, authors. On the authors' confirmation that the sensitivity bins were fixed
  before the analysis, the manuscript keeps the statement that they were fixed before the
  analysis. Because the repository record does not demonstrate this, neither the manuscript
  nor the response to the reviewers cites the repository record as evidence. The choice of
  the primary cell is described neutrally. This decision supersedes the 2026-10-13
  decision above to reclassify the analyses as exploratory.

## Addendum 2026-10-16 — centred single-backbone image branch (R1-5)

Written and committed before the analysis below was run, but after the R1-5 results above
had been seen. The reviewer asks that the centred image branch be included as a main
comparator; the analyses above centre only the image ensemble. This addendum gives the
primary backbone the same treatment. No model is trained or refitted; the inputs are the
pass-B predictions above.

* **Centred image branch (Inception-ResNet-v2)**: Inception-ResNet-v2 minus one constant,
  computed by the rule of the centred ensemble. OOF fold k: the mean residual (prediction −
  measurement) of Inception-ResNet-v2 over the masked locations of the other four folds.
  Held-out: the mean residual over all five OOF folds (no held-out data used).
* **Calibration**: mean bias, slope and intercept of the new branch in every stratum of the
  R1-5 section, OOF and held-out, with the patient-clustered bootstrap of that section
  (5000 resamples, seed 42) on the same resamples as the frozen branches. Before adding the
  branch, the producer recomputes `calibration_multiplicity.json` and stops unless every
  calibration and comparison value is reproduced exactly.
* **Comparison**: fusion (Inception-ResNet-v2 + summary, weights as above) vs the centred
  Inception-ResNet-v2 branch, per-eye RMSE and MAE, OOF and held-out; two-sided Wilcoxon
  signed-rank test and the patient-clustered bootstrap interval of the mean difference, as
  above.
* **Multiplicity**: the two new tests join the R1-5 family of their split, which becomes six
  tests (three fusion–centred pairs × two metrics). Holm-adjusted p values are reported over
  the six. The four-test adjusted values in `calibration_multiplicity.json` stay frozen and
  unchanged; enlarging the family changes the adjustment of the existing four tests, so
  their six-test values are reported as well.
* **Output**: `experiments/revision_r1/centred_irv2.py` → `centred_irv2.json`, frozen through
  `freeze_paper_results.py` and checked by `check_numbers.py`.
