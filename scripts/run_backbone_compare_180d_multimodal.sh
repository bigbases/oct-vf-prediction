#!/usr/bin/env bash
# 180d multimodal 백본 비교 (repro + --use_tabular, tabular z-score는 train 쪽 자동)
# 사용법: bash scripts/run_backbone_compare_180d_multimodal.sh <backbone1> <backbone2> ...
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

BACKBONES=("$@")
if [[ ${#BACKBONES[@]} -eq 0 ]]; then
  echo "ERROR: 백본을 1개 이상 인자로 주세요" >&2
  exit 1
fi

SEEDS=(42 43 44 45 46)
COMMON=(
  --csv ml_final_180d_excl_empty_flip.csv
  --use_tabular
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)

echo "=== 180d multimodal 백본 비교 시작 $(date) | backbones: ${BACKBONES[*]} ==="
for BK in "${BACKBONES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="runs/bbcmp_180d_${BK}_multimodal_s${SEED}"
    if [[ -f "$OUT/results.json" ]]; then
      echo "--- skip (이미 존재): $OUT ---"
      continue
    fi
    echo "=== ${BK} seed ${SEED} 시작 $(date) ==="
    "$PY" -u train_paper_track.py "${COMMON[@]}" \
      --backbone "$BK" --seed "$SEED" --out_dir "$OUT" \
      2>&1 | tee "${OUT}.log"
  done
done
echo "=== 180d multimodal 백본 비교 완료 $(date) | backbones: ${BACKBONES[*]} ==="
