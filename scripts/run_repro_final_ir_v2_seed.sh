#!/usr/bin/env bash
# 최종 IR-v2 image-only 5-fold 파이프라인을 seed만 바꿔 재현 (multi-seed 신뢰도).
# run_backbone_5fold_three.sh 의 IR-v2 COMMON과 동일 + --seed + test preds 덤프.
# 사용법: SEED=43 GPU=0 bash scripts/run_repro_final_ir_v2_seed.sh
set -euo pipefail
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"
SEED="${SEED:?SEED required}"
GPU="${GPU:-0}"
OUT="runs/repro_final_ir_v2_5fold_s${SEED}"
LOG="${OUT}.log"

if [[ -f "${OUT}/results.json" ]]; then
  echo "skip: ${OUT}/results.json exists"; exit 0
fi
mkdir -p "$OUT"
echo "=== IR-v2 5-fold seed=${SEED} on GPU ${GPU} 시작 $(date) ===" | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train.py \
  --csv ml_final_90d_excl_empty_flip.csv \
  --input_geometry paper \
  --backbone inception_resnet_v2 \
  --optimizer rmsprop \
  --lr 1e-4 \
  --epochs 300 \
  --patience 100 \
  --batch_size 64 \
  --seed "$SEED" \
  --dump_val_preds \
  --dump_test_preds \
  --out_dir "$OUT" \
  2>&1 | tee -a "$LOG"

echo "=== seed=${SEED} 완료 $(date) ===" | tee -a "$LOG"
"$PY" -c "
import json, numpy as np
r=json.load(open('${OUT}/results.json'))
ep=[x['best_epoch'] for x in r['results']]
v=[x['best_val_rmse'] for x in r['results']]
t=[x['best_test_rmse'] for x in r['results']]
print('seed ${SEED}: best_epoch', ep)
print(f'  val_rmse mean={np.mean(v):.3f}±{np.std(v):.3f} | test_rmse mean={np.mean(t):.3f}±{np.std(t):.3f}')
" | tee -a "$LOG"
