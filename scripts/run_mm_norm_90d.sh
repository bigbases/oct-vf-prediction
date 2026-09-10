#!/usr/bin/env bash
# Step 1 검증: 90d multimodal + tabular z-score 정규화 (repro 트랙)
# 사용법: bash scripts/run_mm_norm_90d.sh <backbone> <seed1> <seed2> ...
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

BK="$1"; shift
SEEDS=("$@")

COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --use_tabular
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)

echo "=== mm_norm 시작 $(date) | backbone=${BK} seeds=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
  OUT="runs/mm_norm_${BK}_s${SEED}"
  if [[ -f "$OUT/results.json" ]]; then
    echo "--- skip (이미 존재): $OUT ---"
    continue
  fi
  echo "=== ${BK} seed ${SEED} 시작 $(date) ==="
  "$PY" -u train_paper_track.py "${COMMON[@]}" \
    --backbone "$BK" --seed "$SEED" --out_dir "$OUT" \
    2>&1 | tee "${OUT}.log"
done
echo "=== mm_norm 완료 $(date) | backbone=${BK} seeds=${SEEDS[*]} ==="
