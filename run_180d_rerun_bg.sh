#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hvf

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
[ -e "${CONDA_PREFIX}/lib/libstdc++.so.6" ] || ln -sf libstdc++.so.6.0.34 "${CONDA_PREFIX}/lib/libstdc++.so.6"
cd "$HVF_ROOT"

mkdir -p runs
LOG="runs/rerun_180d_all.log"
echo "=== $(date -Iseconds) 180d rerun start ===" | tee -a "$LOG"

echo "--- [XGB][180d][flip][full_26] ---" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from experiments_priority import run_xgb_cv, FEATURE_SETS, ROOT

csv_path = ROOT / "ml_final_180d_excl_empty_flip.csv"
feat_od, feat_os = FEATURE_SETS["full_26"]
res = run_xgb_cv(csv_path, feat_od, feat_os, val_folds=None, fast=False)
out = ROOT / "runs" / "rerun_180d_xgb_full26_flip.json"
out.write_text(json.dumps({"csv": str(csv_path.name), **res}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved: {out}")
print(f"CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}, TEST={res['test_mae']:.3f}")
PY

echo "--- [DL][180d][flip][image-only][5-fold] ---" | tee -a "$LOG"
python train.py \
  --csv ml_final_180d_excl_empty_flip.csv \
  --out_dir runs/rerun_180d_img_only_flip \
  --batch_size 8 \
  --num_workers 4 \
  2>&1 | tee -a "$LOG"

echo "--- [DL][180d][flip][multimodal][5-fold] ---" | tee -a "$LOG"
python train.py \
  --csv ml_final_180d_excl_empty_flip.csv \
  --use_tabular \
  --out_dir runs/rerun_180d_multimodal_flip \
  --batch_size 8 \
  --num_workers 4 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Iseconds) 180d rerun done ===" | tee -a "$LOG"
