# Reproduction entry points.
#
#   make check     config consistency + tests + PHI scan  (runs on a fresh clone)
#   make tables    the six manuscript tables              (needs predictions)
#   make figures   the six manuscript figures             (needs predictions)
#   make verify    recompute every quoted number          (needs predictions)
#
# Only `make check` runs from a clone on its own. Everything below it reads
# per-eye predictions under runs/, which are withheld -- see README.

PY := python

.PHONY: all check config-check test phi tables figures verify env clean

all: tables figures

# -- runs on a fresh clone --------------------------------------------------
check: config-check test phi

config-check:
	$(PY) -c "from hvf_config import load_config, seed; c = load_config(); \
	print('config OK - seed', seed(), '- folds', c['constants']['split']['cv_folds'], \
	'- backbones', len(c['constants']['backbones']['ensemble5']))"

test:
	pytest -q

phi:
	$(PY) scripts/verify_no_phi.py

# -- need the withheld predictions under runs/ ------------------------------
tables:
	$(PY) scripts/build_trivial_baselines.py
	$(PY) scripts/build_final_model_comparison.py
	$(PY) scripts/build_backbone_matrix_table.py
	$(PY) scripts/aggregation_comparability.py
	$(PY) scripts/ensemble_decisive_test.py
	$(PY) scripts/build_severity_region.py

figures:
	$(PY) scripts/make_fig_heldout_quantiles.py
	$(PY) scripts/make_fig_where_gain_2panel.py
	$(PY) scripts/make_fig_bland_altman_single.py
	$(PY) experiments/forest_robustness/make_fig_forest.py
	@echo "Figures 1 and 6 embed one study eye and take its identifiers as arguments:"
	@echo "  $(PY) scripts/make_fig_pipeline.py <patient_id> <eye>"
	@echo "  $(PY) scripts/make_case_heatmap.py <patient_id> <eye>"

verify:
	$(PY) scripts/verify_skeleton_numbers.py
	$(PY) scripts/verify_fusion_integrity.py
	$(PY) scripts/verify_heldout_mae_reversal.py
	$(PY) experiments/metric_sanity.py
	$(PY) scripts/patient_cluster_test.py

env:
	pip install -r requirements-dev.txt

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
