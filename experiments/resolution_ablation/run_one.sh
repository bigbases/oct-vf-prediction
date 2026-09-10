#!/bin/bash
# 공간 해상도 절제: fold 0 한 폴드, 정본 phasec_b0_inception_resnet_v2_5fold 와 동일 설정.
# 사용법: run_one.sh <blocks_n> <gpu_id>
set -euo pipefail
N=$1; GPU=$2
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/n${N}"
mkdir -p "$OUT"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_VISIBLE_DEVICES=$GPU "${PY:-python}" \
  experiments/resolution_ablation/train_res.py \
  --blocks_n "$N" \
  --csv ml_final_90d_excl_empty_flip.csv \
  --input_geometry paper \
  --backbone inception_resnet_v2 \
  --fold 0 \
  --epochs 300 --batch_size 64 --lr 1e-4 --weight_decay 1e-5 \
  --optimizer rmsprop --momentum 0.9 \
  --patience 100 --early_stop_metric mae \
  --seed 42 --num_workers 0 \
  --out_dir "$OUT" \
  --dump_val_preds \
  > "$OUT/train.log" 2>&1
