#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hvf
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$HVF_ROOT"

CSV=ml_final_90d_excl_empty_flip.csv
LOG=runs/phase1_train.log

echo "=== $(date -Iseconds) Phase1 시작 ===" | tee -a "$LOG"

echo "--- image-only 5-fold ---" | tee -a "$LOG"
python train.py \
  --csv "$CSV" \
  --out_dir runs/img_only_90d_flip \
  --batch_size 8 \
  --num_workers 4 \
  2>&1 | tee -a "$LOG"

echo "--- multimodal 5-fold ---" | tee -a "$LOG"
python train.py \
  --csv "$CSV" \
  --use_tabular \
  --out_dir runs/multimodal_90d_flip \
  --batch_size 8 \
  --num_workers 4 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Iseconds) Phase1 완료 ===" | tee -a "$LOG"
