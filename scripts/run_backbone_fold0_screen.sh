#!/usr/bin/env bash
# fold 0 백본 스크리닝: overfit PASS → fold0 학습 (버그 수정 train.py 설정)
set -euo pipefail
export HVF_ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$HVF_ROOT"
PY="${PY:-python}"

# 단일샘플 overfit 점검에 쓸 안(眼). 식별자를 스크립트에 박지 않는다.
OF_PID="${OVERFIT_PID:?OVERFIT_PID 를 지정하라 (점검용 한 안의 patient_id)}"
OF_EYE="${OVERFIT_EYE:-OD}"
OF_VFD="${OVERFIT_VFD:?OVERFIT_VFD 를 지정하라 (그 안의 vf_date, YYYYMMDD)}"

BACKBONES=(inception_resnet_v2 xception vgg16 densenet121)
REF_TEST=8.895
REF_VAL=10.575

COMMON=(
  --csv ml_final_90d_excl_empty_flip.csv
  --fold 0
  --input_geometry paper
  --optimizer rmsprop
  --lr 1e-4
  --epochs 300
  --patience 100
  --batch_size 64
)

echo "=== 백본 fold0 스크리닝 | 기준 Inception-v3 fold0 test=${REF_TEST} val=${REF_VAL} ==="
SUMMARY="$HVF_ROOT/runs/backbone_fold0_screen_summary.json"
echo '{"backbones":[]}' > "$SUMMARY"

for BK in "${BACKBONES[@]}"; do
  echo ""
  echo "########## $BK ##########"
  OF_DIR="runs/overfit_${BK}"
  echo "--- [1/2] 단일샘플 overfit ---"
  if ! "$PY" -u scripts/overfit_single_sample.py \
      --backbone "$BK" --epochs 200 --lr 3e-4 --log_every 25 \
      --patient_id "$OF_PID" --eye "$OF_EYE" --vf_date "$OF_VFD" \
      --out_dir "$OF_DIR" 2>&1 | tee "${OF_DIR}.log"; then
    echo "FAIL: $BK overfit 스크립트 오류 — 중단"
    exit 1
  fi
  PASS=$("$PY" -c "
import json
r=json.load(open('${OF_DIR}/results.json'))
fr=r['final_rmse']
tr=min(x['train_rmse'] for x in r['history'])
ok = fr < 1.5 or tr < 1.0
print('PASS' if ok else 'FAIL', fr, tr)
if not ok: raise SystemExit(1)
")
  echo "overfit: $PASS"

  OUT="runs/bbcmp_fold0_${BK}_clipmse"
  echo "--- [2/2] fold 0 학습 ---"
  "$PY" -u train.py "${COMMON[@]}" --backbone "$BK" --out_dir "$OUT" \
    2>&1 | tee "${OUT}.log"
  "$PY" -c "
import json
r=json.load(open('${OUT}/results.json'))['results'][0]
print(f\"fold0 { '${BK}' }: val_rmse={r['best_val_rmse']:.3f} test_rmse={r['best_test_rmse']:.3f}\")
"
done

echo ""
echo "=== 완료 ==="
