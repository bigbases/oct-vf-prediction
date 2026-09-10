#!/usr/bin/env bash
# P1 go/no-go: fold 0만 — XGB OOF export + CNN 1-fold 학습(val dump) + 잔차 ρ probe
# 실행 전 승인 필요 (CNN 학습 = GPU). XGB만 먼저 돌리려면 STEP 1만 실행.
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

CSV="ml_final_90d_excl_empty_flip.csv"
OOF_DIR="runs/oof"
CNN_OUT="runs/fusion_probe_cnn_fold0"

echo "=== STEP 1: XGB val OOF (fold 0) ==="
"$PY" -u scripts/export_xgb_oof.py \
  --csv "$CSV" --tag 90d --folds 0 --device cuda \
  --out_dir "$OOF_DIR"

echo "=== STEP 2: CNN Phase C B0 fold 0 + val_preds dump ==="
mkdir -p "$CNN_OUT"
"$PY" -u train.py \
  --csv "$CSV" \
  --fold 0 \
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
  --dump_val_preds \
  --out_dir "$CNN_OUT" \
  2>&1 | tee "${CNN_OUT}.log"

echo "=== STEP 3: 잔차 상관 probe ==="
"$PY" -u scripts/probe_fusion_residual_corr.py \
  --xgb "${OOF_DIR}/xgb_90d_fold0_val.npz" \
  --cnn "${CNN_OUT}/val_preds_fold0.npz"

echo "=== probe 완료 $(date) ==="
echo "ρ < RMSE_XGB/RMSE_CNN 이면 5-fold GO 검토, 그렇지 않으면 P2 전환"
