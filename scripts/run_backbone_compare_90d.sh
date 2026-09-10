#!/usr/bin/env bash
# 워크스트림 A: 90d image-only 백본 비교
# 사용법: bash run_backbone_compare_90d.sh <backbone1> <backbone2> ...
# repro 트랙 동일 조건 (epochs 300, batch 64, rmsprop, lr 1e-4, patience 100), seed 42-46
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
  --csv ml_final_90d_excl_empty_flip.csv
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)

echo "=== 백본 비교 시작 $(date) | backbones: ${BACKBONES[*]} ==="
for BK in "${BACKBONES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="runs/bbcmp_90d_${BK}_imgonly_s${SEED}"
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
echo "=== 백본 비교 완료 $(date) | backbones: ${BACKBONES[*]} ==="
