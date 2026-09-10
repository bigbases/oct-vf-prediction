# Frozen results

The JSON files the manuscript's tables, figures and prose numbers are read from,
copied here so a value in the paper can be checked without the withheld data.
They are produced by the scripts named below; `scripts/freeze_paper_results.py`
copies them out of a working tree and `python scripts/freeze_paper_results.py
--source <tree> --check` reports any drift.

Metrics are pooled RMSE and MAE in decibels over the 52 non-blind-spot points,
unless a file's own `note` says otherwise. `oof` is the 240-eye out-of-fold
estimate, `test` the 37-eye held-out set.

| File | Producer | Reported in |
|---|---|---|
| `trivial_baselines.json` | `scripts/build_trivial_baselines.py` | Table 1 |
| `final_model_comparison.json` | `scripts/build_final_model_comparison.py` | Table 2 |
| `backbone_matrix_table.json` | `scripts/build_backbone_matrix_table.py` | Table 3 |
| `aggregation_comparability.json` | `scripts/aggregation_comparability.py` | Table 4 |
| `ensemble_decisive.json` | `scripts/ensemble_decisive_test.py` | Table 5 |
| `severity_region.json` | `scripts/build_severity_region.py` | Table 6 |
| `skeleton_numbers.json` | `scripts/verify_skeleton_numbers.py` | every quoted number, recomputed from raw predictions |
| `fusion_noninferiority.json` | `scripts/fusion_noninferiority.py` | Section 4 non-inferiority margins |
| `fusion_consistency_matrix.json` | `scripts/fusion_consistency_matrix.py` | Section 4, the 5 × 2 × 2 win matrix |
| `patient_cluster_stats.json` | `scripts/patient_cluster_test.py` | patient-clustered bootstrap and Wilcoxon tests |
| `robustness_gains.json` | `scripts/export_robustness_gains.py` | Section 4 robustness prose |
| `multiseed_fusion.json` | `scripts/multiseed_fusion.py` | seed 42/43/44 spread |
| `heldout_mae_reversal.json` | `scripts/verify_heldout_mae_reversal.py` | Section 5, the held-out MAE reversal |
| `forest_rows.json` | `experiments/forest_robustness/compute_rows.py` | Figure 2 |
| `bias_by_bin.json` | `experiments/bias_structure/bias_by_bin.py` | Figure 4 |
| `bias_by_bin_refit.json` | `experiments/bias_structure/bias_by_bin.py --refit` | Figure 4, centring sensitivity |
| `case_profile.json` | `experiments/case_study/case_profile.py` | Figure 6, and the representative-case paragraph |
| `resnet50_late_fusion.json` | `experiments/reviewer_round3/resnet50_late_fusion.py` | Section 5, the sixth backbone |

## Which pass these are

The laterality correction described in `docs/coordinate_frame_convention.md` was
applied upstream and every analysis re-run afterwards. **These files are the
post-correction (pass B) values, and so is the manuscript.** A working tree may
still carry pre-correction copies at the top level of `runs/`; those differ in
the second decimal (for example out-of-fold XGB 8.65 rather than 8.66) and are
not what the paper reports. The freezer takes the pass-B tree explicitly for
exactly this reason.

## What was removed

Two transforms are applied on the way in, and no number is ever rewritten:

* absolute paths become repository-relative, with any home directory written as
  `/home/<user>`;
* in `case_profile.json`, the pseudonym, the laterality and the examination date
  of the representative eye are dropped. Each alone is harmless; together they
  identify a study eye. The metrics and the eye's position in the cohort
  distribution — which is what the manuscript cites — are unchanged.

## Notes worth reading before quoting a number

* `aggregation_comparability.json` carries a caution on its literature column:
  those comparator values came from a secondary source and were not re-checked
  against the primary papers.
* `forest_rows.json` uses the sign convention `delta = mean over eyes of
  RMSE(fusion) − RMSE(summary)`, so **negative favours fusion**. The manuscript
  states gains as positive in one section and differences as negative in
  another; check the direction before quoting. Its `note` field still calls the
  forest plot "Figure 6" — the figure was renumbered to 2 after that file was
  written, and the note is kept as produced rather than edited after the fact.
* `case_profile.json` records that the representative eye sits in the favourable
  tail of the cohort (roughly the 10th percentile of the fusion-minus-summary
  difference) and that no selection rule was recorded at the time.
