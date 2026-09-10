#!/usr/bin/env bash
# 우선순위 DL 1-fold 파일럿 (fold 0, 현재 전처리=ColorJitter만)
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hvf
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
[ -e "${CONDA_PREFIX}/lib/libstdc++.so.6" ] || ln -sf libstdc++.so.6.0.34 "${CONDA_PREFIX}/lib/libstdc++.so.6"
cd "$HVF_ROOT"

CSV=ml_final_90d_excl_empty_flip.csv
LOG=runs/priority_pilot.log
mkdir -p runs

echo "=== $(date -Iseconds) priority DL pilot (fold 0) ===" | tee "$LOG"

run_one() {
  local name=$1
  shift
  echo "--- $name ---" | tee -a "$LOG"
  python train.py --csv "$CSV" --fold 0 --num_workers 4 "$@" \
    2>&1 | tee -a "$LOG"
}

# P1: 전처리 수정 후 재학습 (기준 비교)
run_one "multimodal_f0" --use_tabular --out_dir runs/pilot_multimodal_f0
run_one "img_only_f0" --out_dir runs/pilot_img_only_f0

# P2: flip ablation (no_flip, multimodal)
run_one "multimodal_f0_no_flip" --use_tabular \
  --csv ml_final_90d_excl_empty.csv --out_dir runs/pilot_multimodal_f0_no_flip

# P3: freeze backbone
run_one "multimodal_f0_freeze" --use_tabular --freeze_backbone \
  --out_dir runs/pilot_multimodal_f0_freeze

echo "=== $(date -Iseconds) pilot done ===" | tee -a "$LOG"
