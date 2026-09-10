#!/usr/bin/env bash
# Wait for 180d 5-seed (s46 multimodal), summarize + write interpretation, then 90d 3-seed
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"

PY="${PY:-python}"
S46_MM="runs/repro_180d_multimodal_flip_papertrack_lr1e4_p100_s46/results.json"
MASTER_LOG=runs/repro_180d_5seed_seq_flip_lr1e4_p100.log

echo "=== 180d s46 multimodal 완료 대기 $(date) ==="
while [[ ! -f "$S46_MM" ]]; do
  if ! pgrep -f "train_paper_track.py.*repro_180d_multimodal.*_s46" >/dev/null 2>&1; then
    if [[ ! -f "$S46_MM" ]]; then
      echo "ERROR: s46 multimodal 프로세스 없는데 results.json 없음 $(date)" >&2
      exit 1
    fi
  fi
  sleep 60
done
echo "=== 180d 5-seed 완료 확인 $(date) ==="

"$PY" -u scripts/summarize_repro_5seed.py \
  --seeds 42 43 44 45 46 \
  --prefix runs/repro_180d \
  --suffix flip_papertrack_lr1e4_p100 \
  --out runs/repro_180d_5seed_summary.json \
  2>&1 | tee -a "$MASTER_LOG"

"$PY" -u scripts/write_repro_180d_interpretation.py \
  2>&1 | tee runs/repro_180d_5seed_interpretation.txt

echo "=== 90d 3-seed 시작 $(date) ===" | tee -a runs/repro_90d_3seed_seq_flip_lr1e4_p100.log
bash scripts/run_repro_90d_3seed.sh 2>&1 | tee -a runs/repro_90d_3seed_seq_flip_lr1e4_p100.log
