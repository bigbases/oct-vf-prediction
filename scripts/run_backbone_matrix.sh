#!/usr/bin/env bash
# 백본 매트릭스 순차 학습 (한 GPU에 여러 job 순차)
# 사용: run_backbone_matrix.sh <GPU_ID> <spec1> <spec2> ...
#   spec = mode:backbone   (mode=img|mm)
# 예:   run_backbone_matrix.sh 0 mm:inception_resnet_v2 img:densenet121
set -u
ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python}"
cd "$ROOT" || exit 1
export HVF_ROOT="$ROOT"

GPU="$1"; shift
export CUDA_VISIBLE_DEVICES="$GPU"

COMMON="--csv ml_final_90d_excl_empty_flip.csv --input_geometry paper \
--epochs 300 --batch_size 64 --lr 1e-4 --weight_decay 1e-5 \
--optimizer rmsprop --momentum 0.9 --patience 100 --early_stop_metric mae \
--num_workers 4 --seed 42 --dump_val_preds --dump_test_preds"

echo "[GPU $GPU] 시작 $(date '+%F %T') jobs: $*"

for spec in "$@"; do
  mode="${spec%%:*}"
  bb="${spec##*:}"
  if [ "$mode" = "mm" ]; then
    tab="--use_tabular"
    out="runs/phasec_mm_${bb}_5fold"
  else
    tab=""
    out="runs/phasec_b0_${bb}_5fold"
  fi
  mkdir -p "$out"

  if [ -f "$out/results.json" ] && [ -f "$out/DONE" ]; then
    echo "[GPU $GPU] SKIP (이미 완료): $out"
    continue
  fi

  echo "[GPU $GPU] >>> $spec → $out  $(date '+%F %T')"
  $PY -u train.py $COMMON $tab --backbone "$bb" --out_dir "$out" \
      > "$out/train.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$out/DONE"
    echo "[GPU $GPU] <<< DONE $spec (rc=0) $(date '+%F %T')"
  else
    echo "[GPU $GPU] <<< FAIL $spec (rc=$rc) $(date '+%F %T')"
  fi
done

echo "[GPU $GPU] 전체 종료 $(date '+%F %T')"
