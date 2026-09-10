#!/usr/bin/env bash
# 우선순위 실험 전체 (XGB → DL pilot), nohup용
set -euo pipefail
ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hvf
export HVF_ROOT="$ROOT"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
[ -e "${CONDA_PREFIX}/lib/libstdc++.so.6" ] || ln -sf libstdc++.so.6.0.34 "${CONDA_PREFIX}/lib/libstdc++.so.6"

LOG=runs/priority_all.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) priority_all start ==="
bash run_priority_xgb.sh
echo "=== $(date -Iseconds) XGB done, DL pilot start ==="
bash run_priority_pilot.sh
echo "=== $(date -Iseconds) priority_all done ==="
