"""
Quadrant + Clock-hour OCR 정확도 샘플 검증
- 처음 N 페어에 대해 parse_quadrants_cv / parse_clockhours_cv 실행
- 결과 표 + avg_rnfl과의 일관성(diff) 표시
- 사용자가 각 케이스의 정답을 quadrants/clockhours.png에서 시각 확인하여 비교
"""
import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ocr_oct_values import parse_quadrants_cv, parse_clockhours_cv

from paths import ROOT
BASE = str(ROOT)
ML   = os.path.join(BASE, 'ml_final_90d.csv')
OCT  = os.path.join(BASE, 'oct_values.csv')

# 페어 → rnfl_dir 매핑 (ml_dataset.csv 사용)
ml_full = list(csv.DictReader(open(os.path.join(BASE, 'ml_dataset.csv'), encoding='utf-8-sig')))
rnfl_map = {(r['patient_id'], r['eye'], r['vf_date']): r['rnfl_dir'] for r in ml_full}

# oct_values.csv에서 avg_rnfl 매핑
oct_idx = {}
for r in csv.DictReader(open(OCT, encoding='utf-8-sig')):
    oct_idx[(r['patient_id'], r['eye'])] = r

# 90d 첫 10페어
sample_rows = list(csv.DictReader(open(ML, encoding='utf-8-sig')))[:10]

N_SAMPLES = 10
print(f'\n=== 샘플 {N_SAMPLES}개 RNFL Quadrant + Clock-hour OCR 검증 ===\n')

results = []
done_paths = set()
for r in sample_rows[:N_SAMPLES]:
    pid, eye, vfd = r['patient_id'], r['eye'], r['vf_date']
    rnfl_dir = rnfl_map.get((pid, eye, vfd), '')
    if not rnfl_dir or rnfl_dir in done_paths:
        # 양안 통합 리포트라 같은 폴더가 중복으로 나옴 → OD 한 번만 처리
        if rnfl_dir in done_paths:
            print(f'[{pid} {eye}] (양안 리포트 중복, OD 결과 공유)')
            continue
        print(f'[{pid} {eye}] rnfl_dir 없음 → 스킵')
        continue
    done_paths.add(rnfl_dir)

    quad_path = os.path.join(rnfl_dir, 'quadrants.png')
    clk_path  = os.path.join(rnfl_dir, 'clockhours.png')

    if not (os.path.exists(quad_path) and os.path.exists(clk_path)):
        print(f'[{pid}] 이미지 없음 → 스킵')
        continue

    q = parse_quadrants_cv(quad_path)
    c = parse_clockhours_cv(clk_path)

    oct_od = oct_idx.get((pid, 'OD'), {})
    oct_os = oct_idx.get((pid, 'OS'), {})
    avg_od = float(oct_od.get('od_avg_rnfl') or 0) or None
    avg_os = float(oct_os.get('os_avg_rnfl') or 0) or None

    # Quadrant
    q_od = [q[f'od_{d}'] for d in 'STIN']
    q_os = [q[f'os_{d}'] for d in 'STIN']
    q_od_m = sum(v for v in q_od if v) / max(1, sum(1 for v in q_od if v))
    q_os_m = sum(v for v in q_os if v) / max(1, sum(1 for v in q_os if v))

    # Clock-hour
    c_od = [c[f'od_h{n:02d}'] for n in range(1, 13)]
    c_os = [c[f'os_h{n:02d}'] for n in range(1, 13)]
    c_od_m = sum(v for v in c_od if v) / max(1, sum(1 for v in c_od if v))
    c_os_m = sum(v for v in c_os if v) / max(1, sum(1 for v in c_os if v))

    print(f'\n──── {pid} ────')
    print(f'  Quadrant OD (S/T/I/N): {q_od}   mean={q_od_m:.1f}   '
          f'avg_rnfl={avg_od}   diff={abs(q_od_m-avg_od):.1f}' if avg_od else f'  Quadrant OD: {q_od}   mean={q_od_m:.1f}')
    print(f'  Quadrant OS (S/T/I/N): {q_os}   mean={q_os_m:.1f}   '
          f'avg_rnfl={avg_os}   diff={abs(q_os_m-avg_os):.1f}' if avg_os else f'  Quadrant OS: {q_os}   mean={q_os_m:.1f}')
    print(f'  Clock-hour OD (1~12):  {c_od}   mean={c_od_m:.1f}   '
          f'diff={abs(c_od_m-avg_od):.1f}' if avg_od else f'  Clock-hour OD: {c_od}   mean={c_od_m:.1f}')
    print(f'  Clock-hour OS (1~12):  {c_os}   mean={c_os_m:.1f}   '
          f'diff={abs(c_os_m-avg_os):.1f}' if avg_os else f'  Clock-hour OS: {c_os}   mean={c_os_m:.1f}')

    results.append({
        'pid': pid, 'q_od': q_od, 'q_os': q_os, 'c_od': c_od, 'c_os': c_os,
        'avg_od': avg_od, 'avg_os': avg_os,
        'q_od_diff': abs(q_od_m-avg_od) if avg_od else None,
        'q_os_diff': abs(q_os_m-avg_os) if avg_os else None,
        'c_od_diff': abs(c_od_m-avg_od) if avg_od else None,
        'c_os_diff': abs(c_os_m-avg_os) if avg_os else None,
    })

# 요약
print(f'\n=== 요약 (Quadrant/Clock-hour 평균 vs avg_rnfl diff) ===')
print(f'{"PID":12} {"Q_OD":>8} {"Q_OS":>8} {"C_OD":>8} {"C_OS":>8}   (단위: μm)')
for r in results:
    def fmt(x): return f'{x:.1f}' if x else '  N/A'
    print(f'{r["pid"]:12} {fmt(r["q_od_diff"]):>8} {fmt(r["q_os_diff"]):>8} '
          f'{fmt(r["c_od_diff"]):>8} {fmt(r["c_os_diff"]):>8}')

# 결측 카운트
print(f'\n=== 결측 카운트 ===')
print(f'{"PID":12} {"Q_OD결측":>10} {"Q_OS결측":>10} {"C_OD결측":>10} {"C_OS결측":>10}')
for r in results:
    print(f'{r["pid"]:12} {sum(1 for v in r["q_od"] if v is None):>10} '
          f'{sum(1 for v in r["q_os"] if v is None):>10} '
          f'{sum(1 for v in r["c_od"] if v is None):>10} '
          f'{sum(1 for v in r["c_os"] if v is None):>10}')
