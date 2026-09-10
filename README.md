# Predicting the 24-2 visual field from SD-OCT: code

Code accompanying the manuscript on predicting 52 pointwise 24-2 sensitivities
from a single SD-OCT session by late fusion of a thickness-map CNN and a
gradient-boosted model on the report's summary parameters.

## What is and is not in this repository

**Included.** Every script that trained a model, evaluated it, fitted the fusion
weight, produced a number printed in the paper, or drew a figure.

**Not included.**

| Excluded | Why |
|---|---|
| OCT images, visual-field reports, and every derived `.csv` | Patient data. Cannot be redistributed under the study's IRB approval. |
| Trained weights (`runs/**/*.pt`) | Fitted on patient data; release is subject to institutional review. |
| `runs/` result artefacts (`.json`, `.npz`) | Contain per-eye predictions keyed to study eyes. |
| The manuscript source and figure files | Held separately. |

**Consequence: nothing here runs end to end as published.** The scripts are the
executable record of the method, not a runnable demonstration. Reproduction
requires an equivalently structured dataset; the CSV schema each script expects
is documented in its own module docstring.

## Layout

```
train.py                  image branch: 5-fold CNN on 161x322 thickness maps
train_paper_track.py      image branch: single random hold-out track
image_preprocessing.py    cube -> cropped GCA/RNFL thickness panels, concat, transforms
baseline_xgb.py           summary branch: 52 pointwise XGBoost regressors
paths.py                  every filesystem root, resolved from environment variables

extract_cirrus.py         Cirrus/HFA report pages -> panel crops (needs poppler)
ocr_*.py                  summary parameters and 24-2 thresholds off the reports
cirrus_regions.json       report region coordinates used by the above
build_ml_final.py         assemble the modelling table
merge_rnfl_into_ml_final.py, make_rnfl_flip.py, phase_b_apply.py

run_*.sh                  launch configurations, including the 42/43/44 seed sweeps
scripts/                  evaluation, fusion, statistics, tables, figures
experiments/              the ablations and robustness analyses reported in the paper
```

`paths.py` resolves all roots from the environment, so no absolute path is baked
into the code:

```bash
export HVF_ROOT=/path/to/this/repo        # default: the repo directory
export HVF_DATA_ROOT=/path/to/private     # parent of cirrus_out/
```

## Environment

```bash
conda env create -f environment.yml && conda activate hvf
# or
pip install -r requirements.txt
```

Python 3.10.18, PyTorch 1.12.1 + CUDA 11.3, timm 1.0.27, XGBoost 3.2.0,
NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2. `extract_cirrus.py` and the
`ocr_*.py` scripts additionally need the `tesseract` and `poppler-utils` system
binaries.

## Training

```bash
# image branch, 5-fold, the paper's configuration
python train.py --csv ml_final_90d_excl_empty_flip.csv \
    --backbone inception_resnet_v2 --input_geometry paper \
    --epochs 300 --batch_size 64 --optimizer rmsprop --lr 1e-4 \
    --patience 100 --early_stop_metric mae --seed 42 \
    --out_dir runs/phasec_b0_inception_resnet_v2_5fold --dump_val_preds

# summary branch, out-of-fold predictions on the same folds
python scripts/export_xgb_oof.py --csv ml_final_90d_excl_empty_flip.csv

# late fusion: one scalar weight, fitted inside each fold
python scripts/recompute_late_fusion.py
```

Launch scripts for every reported configuration are in `run_*.sh` and
`scripts/run_*.sh`. The seed sweeps are `scripts/run_repro_90d_3seed.sh`
(seeds 42/43/44), `run_repro_90d_5seed.sh`, and `run_repro_180d_5seed.sh`.

## Tables

| Table | Contents | Command |
|---|---|---|
| 1 | Baseline ladder | `python scripts/build_trivial_baselines.py` |
| 2 | Four ways of using the two representations | `python scripts/build_final_model_comparison.py` |
| 3 | Fusion against each branch, five backbones | `python scripts/build_backbone_matrix_table.py` |
| 4 | Both aggregation axes | `python scripts/aggregation_comparability.py` |
| 5 | The fusion ceiling | `python scripts/ensemble_decisive_test.py` |
| 6 | Severity and field region | `python scripts/build_severity_region.py` |

## Figures

| Figure | Command |
|---|---|
| 1 Study pipeline | `python scripts/make_fig_pipeline.py <patient_id> <eye>` |
| 2 Robustness forest | `python experiments/forest_robustness/compute_rows.py && python experiments/forest_robustness/make_fig_forest.py` |
| 3 Where the gain is, by sensitivity bin | `python experiments/bias_structure/bias_by_bin.py && python scripts/make_fig_where_gain_2panel.py` |
| 4 Representative eye | `python scripts/make_case_heatmap.py <patient_id> <eye>` |
| 5 Held-out error quantiles | `python scripts/make_fig_heldout_quantiles.py` |
| 6 Bland-Altman | `python scripts/make_fig_bland_altman_single.py` |

Shared figure style (serif, 8/7 pt, 180 mm, 600 dpi) is in `scripts/fig_style.py`.
Figures 1 and 4 embed thickness maps and a measured field from one study eye, so
they take that eye's identifiers as arguments; no identifier is stored in the code.

## The two laterality passes

Several analyses under `experiments/` read predictions from two parallel trees:

```
experiments/laterality_qfix/step4_work/A/   # pass A, before the laterality fix
experiments/laterality_qfix/step4_work/B/   # pass B, after it -- the paper's numbers
```

Each tree is a copy of this repository plus that pass's `runs/` directory. Neither
is distributed, because both contain the modelling tables and per-eye predictions.
To reproduce those analyses, copy this repository into both paths and place the
corresponding predictions under `<tree>/runs/`. Scripts that need this assert on
the tree root at import, so they fail loudly rather than silently reading the
wrong pass.

## Verifying the reported numbers

```bash
python scripts/verify_skeleton_numbers.py   # recompute every quoted number from raw predictions
python scripts/verify_fusion_integrity.py   # fold hygiene of the fusion weight
python scripts/verify_heldout_mae_reversal.py
python experiments/metric_sanity.py         # independent reimplementation of every metric
python scripts/patient_cluster_test.py      # patient-clustered bootstrap and Wilcoxon tests
```

`scripts/check_numbers.py` cross-checks the manuscript source against these
artefacts. It needs the manuscript tree, which is not distributed here, so it
will not run from this repository alone.

## Ablations and robustness

`experiments/` holds one directory per analysis: `aggregation_axis`,
`bias_structure`, `case_study`, `forest_robustness`, `gca_crop`,
`image_vs_summary`, `metadata_audit`, `resolution_ablation`, `reviewer_round3`,
`sector_mask`, `weight_theory`. Each script writes its own `.json` and a `.md`
report beside itself.

## License

MIT, see `LICENSE`. Release of the code is separate from any release of data or
trained weights.
