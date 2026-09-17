# Predicting the 24-2 visual field from a single SD-OCT session

Research code accompanying a study that predicts 52 pointwise Humphrey 24-2
threshold sensitivities from one SD-OCT session. A convolutional branch reads a
pair of Cirrus thickness maps, a gradient-boosted branch reads the report's
summary parameters, and the two are combined by late fusion with a single scalar
weight fitted inside each cross-validation fold.

**Research code, provided as-is.** It was written for one cohort on one device
and is not a general-purpose tool.

## What this repository contains

| Path | Contents |
|---|---|
| `train.py`, `train_paper_track.py` | Image branch — 5-fold CNN and the single hold-out track, on 161 × 322 thickness maps |
| `image_preprocessing.py` | Cube → cropped GCA and RNFL panels → concatenation → transforms. The Section 3.2 geometry |
| `baseline_xgb.py` | Summary branch — 52 independent pointwise XGBoost regressors |
| `paths.py` | Every filesystem root, resolved from environment variables |
| `config/params.yaml` | The constants this study reports: seeds, geometry, XGBoost settings, fold count, fusion grid, backbone set |
| `scripts/` | Fusion, evaluation, statistics, the six tables and the six figures |
| `experiments/` | One directory per ablation and robustness analysis, each writing its own JSON and a report beside it |
| `extract_cirrus.py`, `ocr_*.py`, `cirrus_regions.json` | Report extraction: page regions, summary parameters, 24-2 thresholds |
| `build_ml_final.py`, `merge_rnfl_into_ml_final.py`, `make_rnfl_flip.py` | Assembly of the modelling table, including the OS → OD mirroring |
| `phase_b_build.py`, `phase_b_apply.py` | Manual review pass over the extracted thresholds |
| `run_*.sh`, `scripts/run_*.sh` | Launch configurations for every reported run, including the 42/43/44 seed sweeps |
| `paper/results_frozen/` | The 18 JSON files the manuscript's tables, figures and prose numbers are read from |
| `docs/coordinate_frame_convention.md` | On-screen vs anatomical frame, and what follows for the OS mirroring step |
| `tests/` | Config-consistency and frozen-result checks |
| `scripts/verify_no_phi.py` | The identifier scan described under Verification |

Module docstrings, identifiers and this README are in English; inline comments
are in Korean.

## What this repository does not contain

- **Raw imaging data.** No report pages, no crops, no per-eye tables. Withheld
  under the governing IRB approval. Everything in `paper/results_frozen/` is
  aggregate over 240 or 37 eyes.
- **Any patient identifier.** No registration numbers, names, dates of birth or
  examination dates. The two figures that embed a study eye take that eye's
  identifiers as command-line arguments; none is stored in the code.
- **Trained weights.** `runs/**/*.pt` were fitted on patient data; their release
  is subject to institutional review and is not part of this repository.
- **Per-eye predictions.** The `.npz` and `.json` files under `runs/` are keyed
  to study eyes. Only the frozen aggregates are here.
- **The manuscript.** LaTeX sources, the compiled PDF and the figure files are
  held separately. This repository exists to make the computation checkable, not
  to redistribute the paper.
- **Internal working documents.** One record stays internal: the figure
  registry binds each pseudonym printed in the paper to a study eye, so it is
  withheld, and `experiments/case_study/case_profile.py` says so at the point
  where it would otherwise have cited it. The manuscript's companion
  specification is partly here — `docs/CANONICAL_SPEC.md` carries §1–§4, which
  is what `scripts/check_numbers.py` needs in order to run from a clone. Its
  §5 and §6 are omitted because they index internal audit records; the excerpt
  says so at the top and keeps the reasoning that depended on them. Every other
  citation in the source resolves inside this repository: the laterality and
  coordinate-frame arguments point at `docs/coordinate_frame_convention.md`, and
  the rest name a numbered section of the manuscript rather than an audit note.
- **Development history.** The repository begins at a small number of commits;
  the day-to-day working history is not part of the release.

### Expected behaviour on a fresh clone

`make check` passes: the configuration validates against the constants in the
code, the frozen results parse, and the identifier scan runs.

Everything else stops immediately. The table, figure and verification scripts
raise `FileNotFoundError` on a path under `runs/` or on a `ml_final_*.csv` — for
example `runs/phasec_b0_inception_resnet_v2_5fold/`. **This is expected.** Those
inputs hold per-eye rows and are withheld, and the scripts fail rather than fall
back to a default, so no number can be produced from absent data. To see what
they would have written, read `paper/results_frozen/`.

Reproduction on new data requires an equivalently structured dataset. The CSV
schema each script expects is documented in its own module docstring.

## Reproduced values and their producers

| Reported quantity | Script | Frozen output |
|---|---|---|
| Table 1 — baseline ladder | `scripts/build_trivial_baselines.py` | `trivial_baselines.json` |
| Table 2 — four ways of using the two representations | `scripts/build_final_model_comparison.py` | `final_model_comparison.json` |
| Table 3 — fusion against each branch, five backbones | `scripts/build_backbone_matrix_table.py` | `backbone_matrix_table.json` |
| Table 4 — both aggregation axes | `scripts/aggregation_comparability.py` | `aggregation_comparability.json` |
| Table 5 — the fusion ceiling | `scripts/ensemble_decisive_test.py` | `ensemble_decisive.json` |
| Table 6 — severity and field region | `scripts/build_severity_region.py` | `severity_region.json` |
| Figure 1 — study pipeline | `scripts/make_fig_pipeline.py <patient_id> <eye>` | figure file (not distributed) |
| Figure 2 — robustness forest | `experiments/forest_robustness/compute_rows.py`, then `make_fig_forest.py` | `forest_rows.json` |
| Figure 3 — held-out error quantiles | `scripts/make_fig_heldout_quantiles.py` | — |
| Figure 4 — where the gain is, by sensitivity bin | `experiments/bias_structure/bias_by_bin.py`, then `scripts/make_fig_where_gain_2panel.py` | `bias_by_bin.json`, `bias_by_bin_refit.json` |
| Figure 5 — Bland–Altman | `scripts/make_fig_bland_altman_single.py` | — |
| Figure 6 — representative eye | `scripts/make_case_heatmap.py <patient_id> <eye>` | `case_profile.json` |
| Non-inferiority margins | `scripts/fusion_noninferiority.py` | `fusion_noninferiority.json` |
| The 5 × 2 × 2 win matrix | `scripts/fusion_consistency_matrix.py` | `fusion_consistency_matrix.json` |
| Patient-clustered bootstrap and Wilcoxon tests | `scripts/patient_cluster_test.py` | `patient_cluster_stats.json` |
| Robustness prose | `scripts/export_robustness_gains.py` | `robustness_gains.json` |
| Seed 42/43/44 spread | `scripts/multiseed_fusion.py` | `multiseed_fusion.json` |
| Held-out MAE reversal | `scripts/verify_heldout_mae_reversal.py` | `heldout_mae_reversal.json` |
| Sixth backbone (ResNet50) | `experiments/reviewer_round3/resnet50_late_fusion.py` | `resnet50_late_fusion.json` |
| Every quoted number, recomputed from raw predictions | `scripts/verify_skeleton_numbers.py` | `skeleton_numbers.json` |

`paper/results_frozen/README.md` documents the format, the sign conventions, and
which laterality pass these values come from.

`scripts/make_fig_where_gain.py` draws the earlier single-panel version of
Figure 4 and is kept for reference. Shared figure style — serif, 8/7 pt, 180 mm,
600 dpi — is in `scripts/fig_style.py`.

## Requirements

Python 3.10.18, PyTorch 1.12.1 + CUDA 11.3, timm 1.0.27, XGBoost 3.2.0,
NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2, matplotlib 3.9.4. Pinned versions
are in `requirements.txt`; `environment.yml` is the equivalent conda
specification; `ENVIRONMENT.md` records the environment the reported numbers were
produced in. `extract_cirrus.py` and the `ocr_*.py` scripts additionally need the
`tesseract` and `poppler-utils` system binaries.

```bash
conda env create -f environment.yml && conda activate hvf
# or
pip install -r requirements-dev.txt

make check     # config consistency + tests + the identifier scan
```

Paths are resolved from the environment, so no absolute path is baked into the
code:

```bash
export HVF_ROOT=/path/to/this/repo        # default: the repository directory
export HVF_DATA_ROOT=/path/to/private     # parent of cirrus_out/
```

`ENVIRONMENT.md` lists the rest.

**Inline comments are in Korean.** They record why a particular crop window,
coordinate convention, cohort filter or correction was chosen, and are kept in
the language they were written in rather than translated after the fact. The
parts a reader needs in order to use the repository are in English: this file,
`ENVIRONMENT.md`, `docs/coordinate_frame_convention.md`,
`paper/results_frozen/README.md`, `config/params.yaml`, and an English summary at
the top of every entry point's module docstring.

**The pins are deliberate and are not kept current.** They record the environment
the reported numbers were produced in, so dependency scanners will flag
advisories against them. Upgrading is expected to move the numbers: the reported
differences are of the order of 0.01 dB. Anyone reusing this code on new data
should upgrade and re-validate; anyone reproducing the reported values should
install exactly these versions.

## Design rules the code follows

1. No identifying data in the repository. Per-eye tables and predictions stay
   local; the two figures that embed a study eye take its identifiers as
   arguments.
2. Deterministic execution: seeds and library versions are pinned, and the fold
   assignment is patient-level and fixed.
3. The fusion weight is fitted inside the fold, never on the evaluation split.
   `scripts/verify_fusion_integrity.py` checks this.
4. No absolute paths: every root comes from `paths.py` and the environment.
5. Constants are recorded in one place. `config/params.yaml` holds the value of
   every reported constant, and `tests/test_config.py` asserts that each one
   still matches the literal in the source file it was taken from. The training
   code is not rewired to read the YAML — see the note at the top of the file for
   why — so this is drift detection, not externalised configuration.

## Verification

```bash
make check                                  # runs on a fresh clone
make verify                                 # needs the withheld predictions
```

`make verify` recomputes every quoted number from raw predictions
(`scripts/verify_skeleton_numbers.py`), checks the fold hygiene of the fusion
weight (`scripts/verify_fusion_integrity.py`), re-derives the held-out MAE
reversal, reimplements every metric independently
(`experiments/metric_sanity.py`), and re-runs the patient-clustered tests
(`scripts/patient_cluster_test.py`).

`scripts/verify_no_phi.py` scans for identifiers across file names, the working
tree, binary containers, `HEAD` and the full history, and separately checks the
prose for internal status notes. It proves its own patterns fire on a synthetic
probe before reporting anything, and exits non-zero if any layer is not clean or
could not be checked.

`scripts/check_numbers.py` cross-checks the manuscript source against these
artefacts. It needs the manuscript tree, which is not distributed here, so it
will not run from a clone.

## The two laterality passes

Several analyses under `experiments/` read predictions from two parallel trees:

```
experiments/laterality_qfix/step4_work/A/   # pass A, before the laterality fix
experiments/laterality_qfix/step4_work/B/   # pass B, after it — the paper's numbers
```

Each tree is a copy of this repository plus that pass's `runs/` directory.
Neither is distributed, because both hold the modelling tables and per-eye
predictions. To reproduce those analyses, copy this repository into both paths
and place the corresponding predictions under `<tree>/runs/`. Scripts that need
this assert on the tree root at import, so they fail loudly rather than silently
reading the wrong pass. `docs/coordinate_frame_convention.md` explains what the
fix changed and by how much.

## Citation

There is no `CITATION.cff` yet: the author list for the accompanying manuscript
is not final. The file will be added once it is, rather than shipping a list
that would have to be corrected in a public repository.

## License

MIT — see `LICENSE`. Release of the code is separate from any release of data or
trained weights.
