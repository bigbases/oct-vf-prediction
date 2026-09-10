#!/usr/bin/env bash
# P1 CNN: fold 1–4 (Phase C B0, MAE early-stop, val+test npz dump)
# fold 0은 runs/fusion_probe_cnn_fold0/ 에 이미 완료.
# 사용법: bash scripts/run_p1_cnn_folds_1_4.sh <fold1> <fold2> ...
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

OUT="${OUT_DIR:-runs/p1_fusion_cnn_90d}"
FOLDS=("$@")
if [[ ${#FOLDS[@]} -eq 0 ]]; then
  echo "ERROR: fold 번호를 인자로 주세요 (예: 1 2 또는 3 4)" >&2
  exit 1
fi

mkdir -p "$OUT"
COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --input_geometry paper
  --optimizer rmsprop
  --lr 1e-4
  --momentum 0.0
  --weight_decay 0.0
  --batch_size 64
  --epochs 300
  --patience 100
  --num_workers 4
  --seed 42
  --early_stop_metric mae
  --dump_val_preds
  --dump_test_preds
)

echo "=== P1 CNN folds ${FOLDS[*]} 시작 $(date) | out=$OUT ==="
for F in "${FOLDS[@]}"; do
  if [[ -f "$OUT/val_preds_fold${F}.npz" && -f "$OUT/test_preds_fold${F}.npz" ]]; then
    echo "--- skip fold $F (npz 존재) ---"
    continue
  fi
  echo "=== fold $F ==="
  "$PY" -u train.py "${COMMON[@]}" \
    --fold "$F" \
    --out_dir "$OUT" \
    2>&1 | tee "${OUT}/fold${F}.log"
done
echo "=== P1 CNN folds ${FOLDS[*]} 완료 $(date) ==="
