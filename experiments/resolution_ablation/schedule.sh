#!/bin/bash
# 해상도 절제 파일럿 큐 실행기 — 남의 GPU 작업과 충돌하지 않게 양보하며 돌린다.
#
# 규칙:
#   1) GPU 하나당 이 실험 런은 최대 1개. (2개씩 쌓지 않는다)
#   2) 다른 사용자/다른 프로세스가 점유 중인 GPU 에는 올리지 않는다.
#   3) 여유 메모리가 MIN_FREE_MIB 미만이면 올리지 않는다.
#   4) 올릴 자리가 없으면 그냥 기다린다. 끼어들지 않는다.
#
# 사용법: schedule.sh 32 16 8 4 2
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE=experiments/resolution_ablation
MIN_FREE_MIB=13000        # 런 1개가 약 9.5 GB. 여유 포함.
POLL=60
ME=$(id -un)

# 이 스크립트가 띄운 런들의 pid: n -> pid
declare -A RUNPID
declare -A RUNGPU

gpu_is_free() {   # $1 = gpu index. 다른 프로세스 없고 메모리 여유 있으면 0
  local g=$1
  local free
  free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -lt "$MIN_FREE_MIB" ] && return 1
  # 이 GPU 위의 compute 프로세스 중 내가 이 실험으로 띄운 것이 아닌 게 있으면 점유중
  local uuid pid
  uuid=$(nvidia-smi --id="$g" --query-gpu=uuid --format=csv,noheader)
  while IFS=, read -r puuid ppid _; do
    puuid=$(echo "$puuid" | xargs); ppid=$(echo "$ppid" | xargs)
    [ "$puuid" = "$uuid" ] || continue
    return 1        # 어떤 프로세스든 있으면 양보 (내 다른 런 포함 = GPU당 1개 규칙)
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
  return 0
}

QUEUE=("$@")
echo "[$(date +%F_%T)] 큐: ${QUEUE[*]}  (GPU당 최대 1개, 점유 중이면 대기)"

while [ ${#QUEUE[@]} -gt 0 ] || [ ${#RUNPID[@]} -gt 0 ]; do
  # 끝난 런 수확
  for n in "${!RUNPID[@]}"; do
    if ! kill -0 "${RUNPID[$n]}" 2>/dev/null; then
      if [ -f "$HERE/runs/n$n/results.json" ]; then
        echo "[$(date +%F_%T)] n=$n 완료 (gpu ${RUNGPU[$n]})"
      else
        echo "[$(date +%F_%T)] !! n=$n 비정상 종료 (gpu ${RUNGPU[$n]}) — $HERE/runs/n$n/train.log 확인"
      fi
      unset "RUNPID[$n]" "RUNGPU[$n]"
    fi
  done

  # 자리 있으면 하나 올리기
  if [ ${#QUEUE[@]} -gt 0 ]; then
    for g in 0 1; do
      [ ${#QUEUE[@]} -eq 0 ] && break
      if gpu_is_free "$g"; then
        n=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
        echo "[$(date +%F_%T)] n=$n 시작 -> gpu $g"
        nohup ./$HERE/run_one.sh "$n" "$g" > "$HERE/runs/n$n.nohup" 2>&1 &
        RUNPID[$n]=$!; RUNGPU[$n]=$g
        sleep 45     # 메모리 잡을 때까지 기다렸다가 다음 GPU 판정
      fi
    done
  fi
  sleep $POLL
done
echo "[$(date +%F_%T)] 큐 전부 종료"
