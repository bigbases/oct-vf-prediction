#!/usr/bin/env bash
# Park(메인) + VGG16 + Inception-v3(재현) 5-fold
set -euo pipefail
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --input_geometry paper
  --optimizer rmsprop
  --lr 1e-4
  --epochs 300
  --patience 100
  --batch_size 64
  --dump_val_preds
)

run_one() {
  local BK="$1" OUT="$2"
  echo ""
  echo "========== 5-fold: $BK → $OUT $(date) =========="
  "$PY" -u train.py "${COMMON[@]}" --backbone "$BK" --out_dir "$OUT" \
    2>&1 | tee "${OUT}.log"
  "$PY" -c "
import json, numpy as np
r=json.load(open('${OUT}/results.json'))
print('--- ${BK} fold별 test_rmse ---')
for x in r['results']:
    print(f\"  fold {x['fold']}: val_rmse={x['best_val_rmse']:.3f} test_rmse={x['best_test_rmse']:.3f} (ep {x['best_epoch']})\")
t=[x['best_test_rmse'] for x in r['results']]
v=[x['best_val_rmse'] for x in r['results']]
print(f\"  test_rmse mean={np.mean(t):.3f} ± {np.std(t):.3f}\")
print(f\"  val_rmse  mean={np.mean(v):.3f} ± {np.std(v):.3f}\")
"
}

run_one inception_resnet_v2 runs/phasec_b0_inception_resnet_v2_5fold
run_one vgg16 runs/phasec_b0_vgg16_5fold

echo ""
echo "========== Inception-v3: 기존 phasec_b0_clip_mse_5fold 재사용 (재현) =========="
"$PY" -c "
import json, numpy as np
OUT='runs/phasec_b0_clip_mse_5fold'
r=json.load(open(f'{OUT}/results.json'))
print('--- inception_v3 fold별 test_rmse (기존 run) ---')
for x in r['results']:
    print(f\"  fold {x['fold']}: val_rmse={x['best_val_rmse']:.3f} test_rmse={x['best_test_rmse']:.3f} (ep {x['best_epoch']})\")
t=[x['best_test_rmse'] for x in r['results']]
v=[x['best_val_rmse'] for x in r['results']]
print(f\"  test_rmse mean={np.mean(t):.3f} ± {np.std(t):.3f}\")
print(f\"  val_rmse  mean={np.mean(v):.3f} ± {np.std(v):.3f}\")
"

echo ""
echo "=== 3-backbone 5-fold 완료 $(date) ==="
