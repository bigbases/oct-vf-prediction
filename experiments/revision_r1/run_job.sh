#!/usr/bin/env bash
# One R1 training job on one GPU, with the submitted CNN settings.
#   run_job.sh <gpu> <name> <launcher.py> [extra train.py args...]
# Output: runs/<name>/ (or probes/<name>/ when R1_PROBE=1), log: logs/<name>.log with a
# unix-time prefix on every line (the submitted logs carry no per-epoch time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${HVF_ROOT:-$(cd "$HERE/../.." && pwd)}"
PY="${PY:-python}"
GPU="$1"; NAME="$2"; LAUNCHER="$3"; shift 3
SUB=runs; [ "${R1_PROBE:-0}" = 1 ] && SUB=probes
OUT="$HERE/$SUB/$NAME"
LOG="$HERE/logs/$NAME.log"
COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --input_geometry paper
  --optimizer rmsprop
  --lr 1e-4
  --epochs 300
  --patience 100
  --batch_size 64
  --dump_val_preds
  --dump_test_preds
)
cd "$REPO"
echo "$(date '+%F %T') start $NAME gpu=$GPU" >> "$HERE/logs/queue.log"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$LAUNCHER" "${COMMON[@]}" "$@" --out_dir "$OUT" 2>&1 \
  | "$PY" -u -c "import sys,time
for l in sys.stdin: print(f'{time.time():.1f} {l}', end='', flush=True)" > "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "$(date '+%F %T') end $NAME rc=$rc" >> "$HERE/logs/queue.log"
exit $rc
