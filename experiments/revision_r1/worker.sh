#!/usr/bin/env bash
# One GPU worker over a shared file queue. Jobs are queue/pending/<NN>_<name>.job, one
# line: <launcher> [train.py args...]. Lowest NN first. A job is claimed by an atomic mv
# into running/. The worker waits for <wait_pid> (the queue script already on this GPU)
# to exit, then keeps pulling; it polls an empty queue every 60 s until queue/STOP exists.
#   worker.sh <gpu> [wait_pid]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="$1"; WAIT="${2:-}"
Q="$HERE/queue"
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done; fi
while [ ! -e "$Q/STOP" ]; do
  job=""
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    if mv "$Q/pending/$f" "$Q/running/$f" 2>/dev/null; then job="$f"; break; fi
  done
  if [ -z "$job" ]; then sleep 60; continue; fi
  name="${job%.job}"; name="${name#*_}"
  read -r launcher args < "$Q/running/$job"
  case "$launcher" in /*) ;; train.py) ;; *) launcher="$HERE/$launcher" ;; esac
  # shellcheck disable=SC2086
  "$HERE/run_job.sh" "$GPU" "$name" "$launcher" $args
  mv "$Q/running/$job" "$Q/done/$job"
done
