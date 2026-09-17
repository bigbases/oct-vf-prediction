# Journal manuscript canonical specification

> **Public excerpt, 2026-09-17.** This file is §1–§4 of the specification that
> accompanies the manuscript. `scripts/check_numbers.py` reads the protocol
> constants and the §2.1 / §2.2 table anchors from here, which is what lets the
> numbers gate run from a clone.
>
> §5 *Evidence sources* and §6 *Open items before submission* are not included.
> They index internal audit records and paths outside this tree. Cutting them
> did not cut the reasoning: §2.1 carries the correction that mattered most
> (280 pairs → 277 pairs) in full, §2.2 carries the cohort arithmetic and the
> coincidence warning that went with it, and the per-artefact denominators are
> written down in the `SOURCES` comments of `scripts/freeze_paper_results.py`.

> The final format will be decided once a target journal is fixed. The MDPI
> wrapper in `main.tex` is a temporary adapter for editing and compilation; it
> does not indicate the journal being submitted to.
>
> Precedence when content and numbers conflict:
> 1. verification results recomputed from the raw CSV/NPZ
> 2. this document
> 3. the English submission manuscript in `sections/*.tex`
> 4. `skeleton.md`
> 5. `manuscript_ko.md` and `skeleton_notes.md`
>
> `skeleton_notes.md` is an audit trail and `manuscript_ko.md` is a Korean
> working draft; neither is cited as canonical on its own. Those three and
> `sections/*.tex` live in the manuscript tree and are not in this repository.
> The precedence rule itself is kept because it is what decides which value is
> canonical when two sources disagree.

## 1. Research question and claim boundaries

- Research question: compare the predictive information that two
  representations derived from the same SD-OCT acquisition — the thickness
  maps and the device summary parameters — provide for pointwise visual-field
  prediction, in the same cohort and under the same protocol.
- Primary claim (C1): adding the thickness-map prediction by late fusion
  reduces error relative to prediction from the 26 summary parameters alone.
- Honest negative (C4): the mean gain from adding summary parameters to a
  strong 5-backbone image ensemble is not significant.
- Permitted reading: the results are consistent with the thickness maps
  retaining predictive information that is not present in the summary
  parameters, and the candidate mechanism is spatial detail lost to sector
  averaging.
- Forbidden readings: that loss of spatial information has been demonstrated
  causally; that fusion always beat both unimodal branches; that visual field
  testing can be replaced; that a new fusion algorithm is proposed.
- The confounding between input representation and model family (CNN vs
  XGBoost), which vary together, remains a limitation. Loss of spatial
  information is not stated as established until raw maps and sector-averaged
  maps are compared within the same model.

## 2. Primary data and protocol

- Primary cohort CSV: `ml_final_90d_excl_empty_flip.csv`
- Inclusion: OCT–VF interval within 90 days, 280 eyes of 145 patients
- split-first: patient-level folds 0–4 = 47/50/47/47/51 eyes, held-out = 38 eyes
- filter-second: 240 OOF eyes and 37 held-out eyes have both modalities
- held-out prediction: the five fold models of each branch are averaged first,
  then evaluated
- Target: Humphrey 24-2 total threshold in raw dB, 52 locations after removing
  the two blind-spot points
- An extracted value of `-1`, meaning `<0`, is mapped to 0 dB. 1,792 of 14,560
  cells (12.31%) come from that rule, which is 93.1% of all 0 dB labels.
- Labels and tabular features use the CSV in which OS is normalised to OD
  orientation.
- Images keep their acquisition orientation (`flip_os_images=False`). The
  sensitivity of that choice was evaluated by separate retraining, and the
  reading is confined to the present GAP architecture.
- The reliability filter is not applied to the primary cohort; it is reported
  as a sensitivity analysis.

### 2.1 Cohort MD statistics — on the 277-pair basis (corrected 2026-08-27)

⚠️ The `cohort_severity_EMR_MD` block of `runs/skeleton_numbers.json` is on the
**pre-filter 280-pair** basis (`labels.n_eyes = 280`). What the manuscript
reports is the **277-pair** analysis cohort, so the values below are canonical.
An earlier version of the manuscript carried the 280-pair values, and they
passed verification because `274` reads as `277 - 3`.

| Quantity | 280 pairs (JSON, superseded) | **277 pairs (canonical)** |
|---|---|---|
| pairs with MD | 274 | **271** |
| mean | -7.42 | **-7.39** |
| SD | 8.67 | **8.70** |
| median | -4.00 | **-3.97** |
| range | -32.16 to +2.34 | **-32.16 to +2.34** (unchanged) |
| normal (MD > -3) | 40.1% | **40.6%** |
| early (-6 < MD <= -3) | 22.6% | **22.5%** |
| moderate (-12 < MD <= -6) | 12.8% | **12.5%** |
| advanced (MD <= -12) | 24.5% | **24.4%** |

Derivation: take the 277 pairs with both modalities from
`ml_final_90d_excl_empty_flip.csv`, then join `cohort_md.csv` on
`(patient_id, eye, vf_date)`. The 6 join failures (all OD, because
`cohort_md.csv` records both eyes under the OS test date) are a key mismatch
rather than missing data, and are not counted in the 271.

The OCT–VF interval is negligibly different between the two bases (277 pairs:
6.55 ± 18.07 days, 80.1% same-day; 280 pairs: 6.66 ± 18.18 days, 80.0%). The
manuscript sentence "6.7 ± 18.2 days, 80.0%" sits **before** the point where
280 pairs are introduced, so reading it on the 280 basis is correct and it is
left unchanged.

### 2.2 Analysis cohort structure — before and after the filter

| Stage | Pairs | Patients | Eyes | CV/OOF | held-out |
|---|---|---|---|---|---|
| Before filter (split-first) | 280 | 145 | 273 | 242 | 38 |
| After filter (analysis cohort) | 277 | 144 | 272 | 240 | 37 |

The filter keeps only pairs that have both modalities, removing the 3 pairs
without RNFL (2 in CV, 1 in held-out). `runs/prediction_artifact_audit.json`
fixes the pre-filter split: `xgb.oof_rows = 242` and
`xgb.held_out_rows_per_fold = [38 x5]`.

⚠️ **`277`, `272` and `144` do not exist as counts anywhere in `runs/*.json`.**
The analysis cohort sizes the manuscript uses as headline numbers are derived
by hand from 240 + 37 and the 6 join failures; no machine artefact supports
them. The table above is their canonical source, and
`scripts/check_numbers.py` uses **only table rows** of this specification as
evidence — prose is excluded, because prose here is transcribed from the
manuscript and verifying the manuscript against it would be circular.

Meeting the same integers in an artefact does not make them cohort counts:
`best_epochs` in `bbcmp_*_summary.json` contains 272, 273 and 277; `best_epoch`
in `repro_90d_5seed_summary.json` is 272; and `fusion_vs_cnn.wins` in
`reliability_sensitivity.json` is 144. Those are epoch indices and win counts.
The gate pins these values as **table anchors** so that such coincidences
cannot be mistaken for evidence.

**If the 88-patient cohort is merged, re-derive this table before touching the
manuscript.** `runs/*.json` does not update itself. The MD statistics error of
2026-08-27 — 280-pair values printed in a 277-pair context — came from doing
those two steps in the opposite order.

## 3. Inputs and models

- Primary image input: the GCA thickness map and the RNFL thickness map, two
  images.
- The two deviation maps are read by the dataset loader but are not used in the
  primary model forward pass (`use_deviation=False`).
- Each thickness map is processed at 161 x 161 and the two are concatenated
  horizontally into a 3 x 161 x 322 input.
- Image model: ImageNet-pretrained Inception-ResNet-v2, GAP, and a
  1024–512–256–52 linear regression head.
- Training: RMSprop, learning rate 1e-4, weight decay 1e-5, momentum 0.9,
  batch 64, up to 300 epochs, patience 100, early stopping on validation MAE,
  gradient clipping 1.0, seed 42.
- No spatial flip or rotation augmentation is used. The current code does apply
  a brightness/contrast ColorJitter of 0.10 as a training transform, so the
  Methods disclose it as photometric augmentation.
- Summary model: the 26 device summary parameters of the target eye, with 52
  pointwise XGBoost models.
- Late fusion: `y_hat = w * XGB + (1-w) * CNN`, with `w=0.47` for IR-v2.
- Held-out performance is not used to select the model or the weight for either
  the primary or the held-out results.
- The OOF value 8.064 is a development estimate: `w` was chosen on the full OOF
  set and reported on that same set. The nested sensitivity analysis, which
  also excludes the evaluated fold from the weight selection, gives 8.097. The
  held-out 8.401 uses the `w` fixed on OOF and is therefore unaffected.

## 4. Verified primary numbers

⚠️ **This section was rewritten on the pass B (submitted) basis, 2026-09-17.**
The original was written at pass A, and six of its values — 8.653, 6.384,
5.887, 8.935, 6.872 and 6.293 — do not exist in the pass B evidence pool.
Every value below was taken from the frozen artefacts in
`paper/results_frozen/`, with its path given in parentheses. The model is the
single Inception-ResNet-v2 backbone unless stated otherwise.

- OOF, 240 eyes, pooled RMSE/MAE in dB: XGB **8.660/6.398**,
  CNN **8.540/6.123**, fusion **8.069/5.894**.
  (`skeleton_numbers.json`, `backbones.inception_resnet_v2.oof.{xgb,cnn,fusion}`)
- Held-out, 37 eyes, pooled RMSE/MAE in dB: XGB **8.919/6.832**,
  CNN **8.732/6.033**, fusion **8.411/6.272**.
  (same file, `...test.{xgb,cnn,fusion}`)
- Primary effect, the pooled RMSE by which fusion improves on the summary
  branch: **0.591 dB** out-of-fold and **0.509 dB** held-out.
  (`...oof.delta_fus_minus_xgb_rmse` = -0.591,
  `...test.delta_fus_minus_xgb_rmse` = -0.509. The stored sign is
  fusion − XGB, hence negative; the manuscript reports the magnitude of the
  gain and so flips it.)
- OOF eye-level fusion vs XGB: Wilcoxon **p=1.30e-10**, fusion better in
  **160/240** eyes. (`patient_cluster_stats.json`,
  `OOF.eye_level.fusion_vs_XGB.{wilcoxon_p, fusion_better}`)
- Held-out eye-level fusion vs XGB: Wilcoxon **p=0.026**, fusion better in
  **27/37** eyes. (same file, `TEST.eye_level.fusion_vs_XGB`)
- Patient-level fusion vs XGB: OOF **p=3.89e-8** (125 patients, 94/125),
  held-out **p=0.104** (19 patients, 14/19). (same file,
  `{OOF,TEST}.patient_level.fusion_vs_XGB`) The held-out evidence is therefore
  an independent confirmation of direction, and is not described as independent
  corroboration at the patient level.
- For the five-backbone ensemble the canonical body-text result is the nested
  weight, refitted on OOF to `w=0.33`: image **8.115**, fusion **7.933**,
  **p=0.42** against the ensemble CNN and **p=2.50e-13** against the summary
  branch. (`ensemble_decisive.json`, `OOF.ensemble_cnn_rmse` and `OOF.w_refit.*`)
  The fixed `w=0.47` result (**7.967**, p=0.59 and p=1.64e-16) belongs in a
  footnote or the supplement. (`OOF.w0.47.*`)
- The primary effect of 0.591 dB is smaller than the pointwise residual SD of
  2.0–5.5 dB reported by Rabiolo et al. It is not read as a clinically
  substitutive effect.
