#!/usr/bin/env bash
# 90d repro: 9:1 × 3 seeds (42–44), image-only + multimodal (6 runs)
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"

PY="${PY:-python}"
COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --epochs 300
  --batch_size 64
  --optimizer rmsprop
  --lr 1e-4
  --num_workers 4
  --patience 100
)
SEEDS=(42 43 44)
MASTER_LOG=runs/repro_90d_3seed_seq_flip_lr1e4_p100.log

log() { echo "=== $* ==="; }

log "Repro 90d 3-seed 시작 $(date)"

for SEED in "${SEEDS[@]}"; do
  IMG_DIR="runs/repro_90d_imgonly_flip_papertrack_lr1e4_p100_s${SEED}"
  MM_DIR="runs/repro_90d_multimodal_flip_papertrack_lr1e4_p100_s${SEED}"

  if [[ -f "$IMG_DIR/results.json" ]]; then
    log "seed ${SEED} image-only: skip (results exist)"
  else
    log "seed ${SEED} image-only 시작 $(date)"
    "$PY" -u train_paper_track.py "${COMMON[@]}" --seed "$SEED" \
      --out_dir "$IMG_DIR" \
      2>&1 | tee "runs/repro_90d_imgonly_flip_papertrack_lr1e4_p100_s${SEED}.log"
  fi

  if [[ -f "$MM_DIR/results.json" ]]; then
    log "seed ${SEED} multimodal: skip (results exist)"
  else
    log "seed ${SEED} multimodal 시작 $(date)"
    "$PY" -u train_paper_track.py "${COMMON[@]}" --use_tabular --seed "$SEED" \
      --out_dir "$MM_DIR" \
      2>&1 | tee "runs/repro_90d_multimodal_flip_papertrack_lr1e4_p100_s${SEED}.log"
  fi
done

log "3-seed 집계"
"$PY" -u scripts/summarize_repro_5seed.py \
  --seeds 42 43 44 \
  --prefix runs/repro_90d \
  --suffix flip_papertrack_lr1e4_p100 \
  --out runs/repro_90d_3seed_summary.json \
  2>&1 | tee -a "$MASTER_LOG"

log "Repro 90d 3-seed 전체 완료 $(date)"
