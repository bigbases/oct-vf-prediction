#!/usr/bin/env bash
# 180d repro: 9:1 random split × 5 seeds (42–46), image-only + multimodal each
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"

PY="${PY:-python}"
COMMON=(
  --csv ml_final_180d_excl_empty_flip.csv
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)
SEEDS=(42 43 44 45 46)
MASTER_LOG=runs/repro_180d_5seed_seq_flip_lr1e4_p100.log

log() { echo "=== $* ==="; }

log "Repro 180d 5-seed 시작 $(date)"

for SEED in "${SEEDS[@]}"; do
  IMG_DIR="runs/repro_180d_imgonly_flip_papertrack_lr1e4_p100_s${SEED}"
  MM_DIR="runs/repro_180d_multimodal_flip_papertrack_lr1e4_p100_s${SEED}"

  log "seed ${SEED} image-only 시작 $(date)"
  "$PY" -u train_paper_track.py "${COMMON[@]}" --seed "$SEED" \
    --out_dir "$IMG_DIR" \
    2>&1 | tee "runs/repro_180d_imgonly_flip_papertrack_lr1e4_p100_s${SEED}.log"

  log "seed ${SEED} multimodal 시작 $(date)"
  "$PY" -u train_paper_track.py "${COMMON[@]}" --use_tabular --seed "$SEED" \
    --out_dir "$MM_DIR" \
    2>&1 | tee "runs/repro_180d_multimodal_flip_papertrack_lr1e4_p100_s${SEED}.log"
done

log "5-seed 집계"
"$PY" -u scripts/summarize_repro_5seed.py \
  --seeds 42 43 44 45 46 \
  --prefix runs/repro_180d \
  --suffix flip_papertrack_lr1e4_p100 \
  2>&1 | tee -a "$MASTER_LOG"

log "Repro 180d 5-seed 전체 완료 $(date)"
