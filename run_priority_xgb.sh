#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hvf
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# libstdc++.so.6 symlink (Ubuntu 18.04 GLIBCXX)
[ -e "${CONDA_PREFIX}/lib/libstdc++.so.6" ] || ln -sf libstdc++.so.6.0.34 "${CONDA_PREFIX}/lib/libstdc++.so.6"
cd "$HVF_ROOT"
python experiments_priority.py --resume --out runs/priority_xgb_results.json
