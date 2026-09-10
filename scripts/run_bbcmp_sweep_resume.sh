#!/usr/bin/env bash
# 백본 6종 × 5-seed sweep 재개 (90d multimodal → 180d imgonly → 180d multimodal)
# GPU 0/1에 백본 3개씩 병렬 (이전 bbcmp_90d imgonly 분할과 동일)
#
# 사용법:
#   bash scripts/run_bbcmp_sweep_resume.sh          # 전체 3단계 순차
#   bash scripts/run_bbcmp_sweep_resume.sh 90d_mm   # 90d multimodal만
#   bash scripts/run_bbcmp_sweep_resume.sh 180d_io  # 180d image-only만
#   bash scripts/run_bbcmp_sweep_resume.sh 180d_mm  # 180d multimodal만
set -euo pipefail

export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"

BK_A=(inception_v3 resnet18 resnet50)
BK_B=(densenet121 efficientnet_b0 vgg16)
ALL=("${BK_A[@]}" "${BK_B[@]}")

cleanup_partial() {
  local tag="$1"  # multimodal | imgonly
  local win="$2"  # 90d | 180d
  for bk in "${ALL[@]}"; do
    for seed in 42 43 44 45 46; do
      local d="runs/bbcmp_${win}_${bk}_${tag}_s${seed}"
      if [[ -d "$d" && ! -f "$d/results.json" ]]; then
        echo "[cleanup] 미완료 삭제: $d"
        rm -rf "$d" "${d}.log"
      fi
    done
  done
}

run_pair() {
  local script="$1"
  shift
  echo "=== GPU0: ${BK_A[*]} | GPU1: ${BK_B[*]} | $script ==="
  CUDA_VISIBLE_DEVICES=0 bash "$script" "${BK_A[@]}" \
    >"runs/$(basename "${script%.sh}")_gpu0.out" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 bash "$script" "${BK_B[@]}" \
    >"runs/$(basename "${script%.sh}")_gpu1.out" 2>&1 &
  local p1=$!
  wait "$p0" "$p1"
}

phase_90d_mm() {
  cleanup_partial multimodal 90d
  run_pair scripts/run_backbone_compare_90d_multimodal.sh
  "$PY" scripts/summarize_backbone_compare.py --window 90d --tag multimodal \
    --backbones "${ALL[@]}" --out runs/bbcmp_90d_multimodal_summary.json
}

phase_180d_io() {
  cleanup_partial imgonly 180d
  run_pair scripts/run_backbone_compare_180d.sh
  "$PY" scripts/summarize_backbone_compare.py --window 180d --tag imgonly \
    --backbones "${ALL[@]}" --out runs/bbcmp_180d_imgonly_summary.json
}

phase_180d_mm() {
  cleanup_partial multimodal 180d
  run_pair scripts/run_backbone_compare_180d_multimodal.sh
  "$PY" scripts/summarize_backbone_compare.py --window 180d --tag multimodal \
    --backbones "${ALL[@]}" --out runs/bbcmp_180d_multimodal_summary.json
}

PY="${PY:-python}"
MODE="${1:-all}"

case "$MODE" in
  all)
    phase_90d_mm
    phase_180d_io
    phase_180d_mm
    ;;
  90d_mm) phase_90d_mm ;;
  180d_io) phase_180d_io ;;
  180d_mm) phase_180d_mm ;;
  *)
    echo "ERROR: unknown mode $MODE (all|90d_mm|180d_io|180d_mm)" >&2
    exit 1
    ;;
esac

echo "=== bbcmp sweep done $(date) mode=$MODE ==="
