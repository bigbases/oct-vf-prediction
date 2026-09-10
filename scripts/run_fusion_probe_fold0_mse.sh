#!/usr/bin/env bash
# fold 0: val_MSE early-stop (repro 동일) vs 기존 MAE-stop probe 비교용
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

CNN_OUT="runs/fusion_probe_cnn_fold0_mse"
OOF_XGB="runs/oof/xgb_90d_fold0_val.npz"

echo "=== CNN fold0 (early_stop_metric=mse) ==="
mkdir -p "$CNN_OUT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u train.py \
  --csv ml_final_90d_excl_empty_flip.csv \
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
  --early_stop_metric mse \
  --dump_val_preds \
  --out_dir "$CNN_OUT" \
  2>&1 | tee "${CNN_OUT}.log"

echo "=== 잔차 ρ (MAE-stop CNN과 비교) ==="
"$PY" -u scripts/probe_fusion_residual_corr.py \
  --xgb "$OOF_XGB" \
  --cnn "${CNN_OUT}/val_preds_fold0.npz" \
  --tag "MSE-stop"

echo "=== 완료. MAE-stop 기준: runs/fusion_probe_cnn_fold0/val_preds_fold0.npz ==="
