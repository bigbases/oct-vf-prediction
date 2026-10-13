#!/usr/bin/env bash
# Second probe of a backbone that did not match: same command, compared with the first
# probe (run-to-run determinism) and with the submitted run.
#   GPU=<n> PY=<python> run_probes_repeat.sh "<backbone> <submitted run dir>" ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${HVF_ROOT:-$(cd "$HERE/../.." && pwd)}"
PY="${PY:-python}"
export R1_PROBE=1
GPU="${GPU:-1}"
for spec in "$@"; do
  set -- $spec; bb=$1; ref=$2
  "$HERE/run_job.sh" "$GPU" "probe2_$bb" train.py --backbone "$bb" --fold 0 --epochs 3
  echo "$(date '+%F %T') $bb probe2-vs-probe1: $("$PY" "$HERE/probe_compare.py" "$HERE/probes/probe2_$bb/results.json" "$HERE/probes/probe_$bb/results.json" | head -1) | probe2-vs-submitted: $("$PY" "$HERE/probe_compare.py" "$HERE/probes/probe2_$bb/results.json" "$REPO/runs/$ref/results.json" | head -1)" >> "$HERE/logs/probes.log"
done
echo "$(date '+%F %T') repeat probes done" >> "$HERE/logs/probes.log"
