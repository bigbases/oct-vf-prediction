#!/usr/bin/env bash
# Guard probes: fold 0, 3 epochs, no manipulated variable. Each must match the submitted
# run's first three epochs exactly (probe_compare.py).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${HVF_ROOT:-$(cd "$HERE/../.." && pwd)}"
PY="${PY:-python}"
export R1_PROBE=1
P=(--fold 0 --epochs 3)
probe() {  # gpu name launcher ref_dir args...
  local g=$1 n=$2 l=$3 ref=$4; shift 4
  local t0=$(date +%s)
  "$HERE/run_job.sh" "$g" "$n" "$l" "$@" "${P[@]}"
  local t1=$(date +%s)
  echo "$(date '+%F %T') probe $n wall_s=$((t1-t0)) $("$PY" "$HERE/probe_compare.py" "$HERE/probes/$n/results.json" "$REPO/runs/$ref/results.json" | head -1)" >> "$HERE/logs/probes.log"
}
( probe 0 probe_inception_v3 train.py phasec_b0_clip_mse_5fold --backbone inception_v3
  probe 0 probe_vgg16 train.py phasec_b0_vgg16_5fold --backbone vgg16
  probe 0 probe_r14none_irv2 "$HERE/train_r14.py" phasec_b0_inception_resnet_v2_5fold --backbone inception_resnet_v2 --floor_rule none ) &
( probe 1 probe_xception train.py phasec_b0_xception_5fold --backbone xception
  probe 1 probe_densenet121 train.py phasec_b0_densenet121_5fold --backbone densenet121 ) &
wait
echo "$(date '+%F %T') probes done" >> "$HERE/logs/probes.log"
