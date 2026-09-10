#!/usr/bin/env bash
# 우선순위 실험 진행 확인
ROOT="${HVF_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

echo "=== 프로세스 ==="
pgrep -af "experiments_priority|run_priority_all|train.py.*pilot" || echo "(없음)"

echo ""
echo "=== XGB 결과 (9개 목표) ==="
if [ -f runs/priority_xgb_results.json ]; then
  python3 -c "
import json
r=json.load(open('runs/priority_xgb_results.json'))
print(f'완료: {len(r)}/9')
for k,v in sorted(r.items()):
    print(f'  {k}: test={v[\"test_mae\"]:.3f} cv={v[\"cv_mean\"]:.3f}')
"
else
  echo "  runs/priority_xgb_results.json 아직 없음 (첫 설정 학습 중일 수 있음)"
fi

echo ""
echo "=== 로그 tail ==="
tail -8 runs/priority_xgb.log 2>/dev/null || echo "  (로그 없음)"
tail -5 runs/priority_pilot.log 2>/dev/null || echo "  (DL pilot 아직 시작 전)"
