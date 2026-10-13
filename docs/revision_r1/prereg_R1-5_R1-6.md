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
