#!/usr/bin/env bash
# 워크스트림 B: 90d Inception-v3 image-only
# 사용법: bash scripts/run_b_experiments_90d.sh B1|B3
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

EXP="${1:?B1 또는 B3 필요}"
SEEDS=(42 43 44 45 46)
COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --backbone inception_v3
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)

case "$EXP" in
  B1) EXTRA=(--freeze_backbone); TAG=b1_freeze ;;
  B3) EXTRA=(--input_mode thickness_deviation); TAG=b3_deviation ;;
  *) echo "ERROR: EXP는 B1 또는 B3" >&2; exit 1 ;;
esac

echo "=== B 실험 ${EXP} (${TAG}) 시작 $(date) ==="
for SEED in "${SEEDS[@]}"; do
  OUT="runs/bexp_90d_${TAG}_s${SEED}"
  if [[ -f "$OUT/results.json" ]]; then
    echo "--- skip: $OUT ---"
    continue
  fi
  echo "=== ${EXP} seed ${SEED} $(date) ==="
  "$PY" -u train_paper_track.py "${COMMON[@]}" "${EXTRA[@]}" \
    --seed "$SEED" --out_dir "$OUT" \
    2>&1 | tee "${OUT}.log"
done
echo "=== B 실험 ${EXP} 완료 $(date) ==="
