#!/usr/bin/env bash
# Phase C-ours: 180d B0 (Inception, thickness-only) 5-fold CV — sensitivity
# 90d Phase C와 동일 하이퍼 + patient-level cv_fold split
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

OUT="runs/phasec_180d_b0_imgonly_paper_geom_5fold"
LOG="${OUT}.log"

if [[ -f "${OUT}/results.json" ]]; then
  echo "skip: ${OUT}/results.json already exists"
  exit 0
fi

mkdir -p "$OUT"
echo "=== Phase C 180d B0 5-fold 시작 $(date) ===" | tee "$LOG"

"$PY" -u train.py \
  --csv ml_final_180d_excl_empty_flip.csv \
  --input_geometry paper \
  --optimizer rmsprop \
  --lr 1e-4 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --batch_size 64 \
  --epochs 300 \
  --patience 100 \
  --num_workers 4 \
  --seed 42 \
  --out_dir "$OUT" \
  2>&1 | tee -a "$LOG"

echo "=== Phase C 180d B0 5-fold 완료 $(date) ===" | tee -a "$LOG"
