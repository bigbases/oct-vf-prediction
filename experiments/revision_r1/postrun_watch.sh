#!/usr/bin/env bash
# Re-run the analysis a finished queue job feeds, then the numbers table, so results do not
# wait for a session. Runs beside worker.sh (which only trains). CPU only; never freezes or
# commits. Stops when queue/pending and queue/running are both empty.
#   setsid nohup ./postrun_watch.sh > logs/postrun_watch.nohup 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
Q="$HERE/queue"
LOG="$HERE/logs/postrun.log"
cd "$HERE" || exit 1
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
queue_empty() { [ -z "$(ls "$Q/pending")" ] && [ -z "$(ls "$Q/running")" ]; }

say "watch start pid=$$"
while read -r _d _t kind name rest; do
  [ "$kind" = end ] || continue
  case "$name" in r11_*) a=r11 ;; r13_*) a=r13 ;; r14_*) a=r14 ;; *) a= ;; esac
  if [ "${rest%% *}" != rc=0 ]; then say "skip $name ($rest)"
  elif [ -z "$a" ]; then say "no analysis for $name"
  else
    say "after $name: ${a}_analysis.py"
    "$PY" "${a}_analysis.py" > "logs/postrun_${a}.out" 2>&1; rc=$?
    grep -E '^guard|missing|arms available' "logs/postrun_${a}.out" >> "$LOG"
    n=$(grep -c '^guard' "logs/postrun_${a}.out"); bad=$(grep '^guard' "logs/postrun_${a}.out" | grep -vc ' ok$')
    say "${a}_analysis rc=$rc guards=$n not_ok=$bad"
    if [ "$rc" -eq 0 ] && [ "$a" = r13 ]; then   # added 2026-10-14: the rerun-frame check rides on r13
      "$PY" r13_rerun_frame.py > logs/postrun_r13_rerun.out 2>&1; rc=$?
      grep -E '^guard|reruns used' logs/postrun_r13_rerun.out | grep -v ' ok$' >> "$LOG"
      say "r13_rerun_frame rc=$rc"
    fi
    if [ "$rc" -eq 0 ]; then "$PY" numbers_table.py >> "$LOG" 2>&1; say "numbers_table rc=$?"; fi
  fi
  sleep 5   # worker.sh moves the job to done/ right after run_job.sh returns
  if queue_empty; then say "queue empty, watch end"; break; fi
done < <(tail -n 0 -F "$HERE/logs/queue.log" 2>/dev/null)
pkill -P $$ tail 2>/dev/null
exit 0
