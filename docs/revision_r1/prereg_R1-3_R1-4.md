# Revision round 1 — analysis plan for R1-3 (common coordinate frame) and R1-4 (floor labels)

Written and committed before any run below was started. The only runs made before this
commit are the guard probes in §1, which change nothing about the submitted configuration.
Tag `submitted-v1` marks the repository as submitted. The submitted manuscript and the frozen
results in `paper/results_frozen/` are not changed by this work; new results go through
producer script → `freeze_paper_results.py` → `check_numbers.py`, as before.

## 0. Fixed for every run in this plan

* Data: the label file and fold assignment (`cv_fold`) of the submitted analysis; 240
  out-of-fold (OOF) eyes, 37 held-out eyes from 19 patients; 52 locations (blind spot
  excluded).
* CNN: `train.py`, `--input_geometry paper` (two thickness maps, 161 × 322; deviation maps
  not used), RMSprop, lr 1e-4, momentum 0.9, weight decay 1e-5, batch 64, at most 300
  epochs, patience 100 on validation MAE, training transform with brightness/contrast
  jitter 0.10, `num_workers 0`, seed 42, five folds. The held-out set is evaluated once,
  at the selected checkpoint; held-out predictions are the mean of the five fold models.
* XGBoost: the settings of `export_xgb_oof.py` (300 trees, learning rate 0.05, depth 4,
  subsample 0.8, column subsample 0.8, alpha 0.1, lambda 1.0, seed 42, one model per field
  location, no early stopping), 26 printed summary parameters.
* Late fusion: `w · XGB + (1 − w) · CNN`. Each endpoint uses the weight convention of the
  submitted analysis it is compared with: Table 3 convention (w fitted on all OOF folds,
  grid 0.00–1.00 step 0.01, pooled RMSE) for single-backbone fusion; nested w (fitted on
  the other four folds) for the ensemble ceiling and the sensitivity-bin analysis.
* Statistics: unit = eye. Paired two-sided Wilcoxon signed-rank test on per-eye RMSE / MAE,
  and a patient-clustered bootstrap 95% percentile interval of the mean per-eye difference
  (5000 resamples, seed 42), as in the submitted analysis.
* Outputs are written outside the submitted artefact directories.

## 1. Guard (run before this commit; contains no manipulated variable)

For each CNN that will be retrained, fold 0 is run for three epochs with the submitted
configuration and the current code. The per-epoch history (train loss, train MAE,
validation MSE, RMSE, MAE) must equal the submitted run's first three epochs exactly
(`experiments/revision_r1/probe_compare.py`). Probes: Inception-v3, VGG16, Xception,
DenseNet121, and the R1-4 wrapper with `--floor_rule none` on Inception-ResNet-v2 (shows
that the wrapper alone changes nothing). A backbone that fails is not launched and the
failure is reported.

## 2. R1-3 — the image branch in the right-eye frame

**Question.** Do the submitted conclusions hold when left-eye thickness maps are mirrored
so that images, labels and summary parameters share the right-eye frame?

**Manipulated variable.** Left-eye maps are mirrored left–right after cropping and before
the transform (`--flip_os_images`). Nothing else changes; XGBoost is not refitted (its
inputs and targets are already in the right-eye frame).

**Runs.** Inception-v3, VGG16, Xception, DenseNet121: five folds each, seed 42. The
mirrored Inception-ResNet-v2 run already exists (identical settings, trained before this
plan; summarised in `osflip_compare.json`) and is reused, not retrained.

**Endpoints** (each reported next to the submitted, unmirrored value):

1. Per backbone, fusion vs summary parameters (XGBoost): per-eye RMSE and MAE, OOF and
   held-out. **Primary for R1-3: Inception-ResNet-v2, OOF, per-eye RMSE** — the submitted
   headline comparison. The other four backbones and MAE are secondary.
2. Five-backbone mirrored ensemble (mean of the five CNN predictions) and its fusion with
   nested w: ensemble fusion vs ensemble, OOF and held-out (the "ceiling").
3. Sensitivity-bin trend as in `fusion_sensitivity_trend.py`: bins of measured sensitivity
   [0,10), [10,20), [20,30), [30,∞) dB, at least 3 locations per eye and bin, per-eye OLS
   slope of (fusion − CNN) RMSE on bin index, one-sample Wilcoxon on the slopes; five
   backbones and the ensemble, OOF and held-out.
4. Descriptive: mirrored vs unmirrored CNN, per backbone, per-eye paired difference.

**Reporting rule.** All results are reported whatever their direction. Whether the
revised manuscript adopts the mirrored frame as its primary analysis is an author decision,
recorded in §5 before the first R1-3 endpoint is computed.

## 3. R1-4 — the floor-label rule

**Floor cells.** Locations recorded as "<0 dB" (−1 in the source report) that the label
file carries as 0 dB: 1,792 of 14,560 cells (12.31%). They are identified from the
pre-conversion copy of the label file, matched row by row (274 rows on patient, eye and
test date; the 6 rows whose test date was later corrected on patient and eye, unique
candidate). Two checks are asserted: no non-floor value differs between the files, and
every floor cell is 0 in the current file (`experiments/revision_r1/floor_cells.py`).

**Manipulated variable.** The treatment of floor cells in training. Checkpoint selection
uses the same rule as the training loss. Image branch: Inception-ResNet-v2, unmirrored
frame (as submitted).

| Arm | CNN (IR-v2) | XGBoost |
|---|---|---|
| 0 submitted | floor cells are 0 dB targets | floor cells are 0 dB targets |
| 1 exclusion | floor cells get m_j = 0 in loss and validation MAE | floor cells dropped from that location's training samples |
| 2 censored (CNN only) | floor cells penalised only above 0 dB: max(ŷ,0)² in the loss, max(ŷ,0) in the validation MAE; other cells unchanged | — |

Before arm 1 of XGBoost is fitted, arm 0 is refitted through the same code path and must
reproduce the frozen pooled OOF RMSE / MAE 8.6603 / 6.3975 dB.

**Endpoints.**

1. **Primary for R1-4:** pooled RMSE on non-floor cells, OOF, for fusion arm 1 (CNN arm 1 +
   XGBoost arm 1) vs fusion arm 0, with the per-eye paired test (Wilcoxon + clustered
   bootstrap) on non-floor cells. Fusion weight fixed at the submitted w = 0.47; the
   refitted OOF-optimal w is reported as secondary.
2. Secondary: the same comparison for CNN arm 1 vs 0 and XGBoost arm 1 vs 0; censored arm
   2 vs arms 0 and 1 (CNN, and fusion with XGBoost arm 1); held-out versions of all.
3. Deep end and calibration, every arm: mean bias (ŷ − y) in the lowest sensitivity bin
   [0,10) dB computed (a) with the rule applied (floor cells included as 0 dB) and (b)
   with floor cells excluded; calibration slope and intercept (measured regressed on
   predicted) on non-floor cells, overall and per bin; mean prediction at floor cells.

## 4. Expected run time (one RTX A5000 per run, measured from the submitted runs)

| Run | Submitted epochs (5 folds) | s / epoch alone | Set |
|---|---|---|---|
| Inception-v3 mirrored | 1,133 | 9.0 | 2.8 h |
| VGG16 mirrored | 1,189 | 10.5 | 3.7 h |
| Xception mirrored | 907 | 6.1 | 1.5 h |
| DenseNet121 mirrored | 905 | 7.7 | 1.9 h |
| IR-v2 exclusion | 1,076 | 11.4 | 3.4 h |
| IR-v2 censored | 1,076 | 11.4 | 3.4 h |

Two runs at once (one per GPU) were 1.5–1.8× slower per run in an earlier pilot; data
loading runs in the main process.

## 5. Decision log

* 2026-10-13, guard result (§1). Exact match over three epochs: Inception-v3, and the R1-4
  wrapper with `--floor_rule none` on Inception-ResNet-v2. VGG16, Xception and DenseNet121
  differ from the submitted runs from epoch 1 on. Per §1 those three are not launched. A
  repeat probe tests whether the current code is deterministic run to run; the outcome and
  any change to R1-3 that follows will be recorded here before those runs start.
* 2026-10-13, repeat probe. Xception and DenseNet121: two probes of the current code agree
  exactly with each other and both differ from the submitted run, so the current code is
  deterministic but does not reproduce those runs. VGG16: the two probes differ from each
  other, so VGG16 training is not deterministic run to run. The July training script
  (recovered from the editor's local history) differs from the current one only in how the
  held-out set is handled, which cannot affect epoch 1; the cause is not established.
  Decision:
  - Endpoints 1-3 of R1-3 are computed among mirrored runs only, so they do not depend on
    reproducing the submitted unmirrored runs. The mirrored VGG16, Xception and DenseNet121
    runs are launched as planned.
  - Endpoint 4 (mirrored vs unmirrored) would mix mirroring with the reproduction gap for
    these three. Each of them is therefore also retrained unmirrored with the current code,
    and endpoint 4 compares mirrored with this matched rerun. The rerun vs the submitted run
    is reported separately, as the size of the reproduction gap. These reruns are queued last.
  - VGG16 endpoint-4 differences include run-to-run noise, which is reported as such.
* (pending, before the first R1-3 endpoint is computed) Primary frame of the revised
  manuscript: mirrored or unmirrored.
