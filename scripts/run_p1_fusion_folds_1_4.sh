#!/usr/bin/env bash
# P1 fold 1–4 일괄: XGB OOF export + CNN 학습 (GPU 2장 병렬)
# fold 0: XGB runs/oof/xgb_90d_fold0_val.npz, CNN runs/fusion_probe_cnn_fold0/
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

echo "=== STEP A0: CNN fold0 test npz (기존 ckpt) ==="
"$PY" -u scripts/dump_cnn_preds_from_ckpt.py \
  --fold 0 \
  --ckpt_dir runs/fusion_probe_cnn_fold0 \
  --out_dir runs/fusion_probe_cnn_fold0

echo "=== STEP A: XGB val+test OOF folds 0–4 (fold0 test npz 보강 포함) ==="
"$PY" -u scripts/export_xgb_oof.py \
  --csv ml_final_90d_excl_empty_flip.csv \
  --tag 90d \
  --folds 0 1 2 3 4 \
  --device cuda \
  --out_dir runs/oof

echo "=== STEP B: CNN folds 1–4 (GPU0: 1,2 / GPU1: 3,4) ==="
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_p1_cnn_folds_1_4.sh 1 2 \
  > runs/p1_fusion_cnn_90d_gpu0.log 2>&1 &
echo "gpu0 pid=$!"
CUDA_VISIBLE_DEVICES=1 OUT_DIR=runs/p1_fusion_cnn_90d nohup bash scripts/run_p1_cnn_folds_1_4.sh 3 4 \
  > runs/p1_fusion_cnn_90d_gpu1.log 2>&1 &
echo "gpu1 pid=$!"
echo "진행: runs/p1_fusion_cnn_90d_gpu{0,1}.log"
echo "완료 후: python scripts/fit_p1_global_fusion.py"
