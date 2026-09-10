"""
ml_final.csv 통합 빌드
ml_dataset + sfa_thresholds + oct_values → ml_final_90d.csv / ml_final_180d.csv
"""
import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from paths import ROOT
os.chdir(ROOT)

PT = [f'p{i+1:02d}' for i in range(54)]

OCT_COLS = [
    'avg_gcl_od', 'avg_gcl_os', 'min_gcl_od', 'min_gcl_os',
    'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'od_avg_rnfl', 'os_avg_rnfl',
    'od_vert_cd', 'os_vert_cd',
]

# ── 로드
ml  = list(csv.DictReader(open('ml_dataset.csv',    encoding='utf-8-sig')))
sfa = list(csv.DictReader(open('sfa_thresholds.csv', encoding='utf-8-sig')))
oct = list(csv.DictReader(open('oct_values.csv',     encoding='utf-8-sig')))

# ── 인덱스
sfa_idx = {(r['patient_id'], r['eye'], r['vf_date']): r for r in sfa}
oct_idx = {(r['patient_id'], r['eye'], r['oct_date']): r for r in oct}
ml_idx  = {(r['patient_id'], r['eye'], r['oct_date']): r for r in ml}

# ── 출력 컬럼 순서
FIELDS = (
    ['patient_id', 'eye', 'vf_date', 'oct_date', 'gap_days', 'split']
    + PT
    + ['n_detected', 'n_missing']
    + OCT_COLS
)

# ── 통합
merged = []
missing_sfa = []
missing_oct = []

for r in ml:
    pid  = r['patient_id']
    eye  = r['eye']
    vfd  = r['vf_date']
    octd = r['oct_date']

    sfa_row = sfa_idx.get((pid, eye, vfd))
    oct_row = oct_idx.get((pid, eye, octd))

    if sfa_row is None:
        missing_sfa.append((pid, eye, vfd))
        continue
    if oct_row is None:
        missing_oct.append((pid, eye, octd))
        continue

    row = {
        'patient_id': pid,
        'eye':        eye,
        'vf_date':    vfd,
        'oct_date':   octd,
        'gap_days':   r['gap_days'],
        'split':      r['split'],
    }
    for p in PT:
        row[p] = sfa_row.get(p, '')
    row['n_detected'] = sfa_row.get('n_detected', '')
    row['n_missing']  = sfa_row.get('n_missing', '')
    for c in OCT_COLS:
        row[c] = oct_row.get(c, '')

    merged.append(row)

if missing_sfa:
    print(f'[경고] sfa_thresholds 매칭 실패: {len(missing_sfa)}건')
    for x in missing_sfa: print(f'  {x}')
if missing_oct:
    print(f'[경고] oct_values 매칭 실패: {len(missing_oct)}건')
    for x in missing_oct: print(f'  {x}')

# ── gap 필터별 저장
def save(rows, path, label):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    splits = Counter(r['split'] for r in rows)
    print(f'\n{label}: {path}')
    print(f'  총 {len(rows)}행')
    print(f'  split: train={splits["train"]}, val={splits["val"]}, test={splits["test"]}')
    gaps = [int(r['gap_days']) for r in rows]
    print(f'  gap_days: 0일={sum(1 for g in gaps if g==0)}, '
          f'1-30일={sum(1 for g in gaps if 1<=g<=30)}, '
          f'31-90일={sum(1 for g in gaps if 31<=g<=90)}, '
          f'91-180일={sum(1 for g in gaps if 91<=g<=180)}')

rows_90  = [r for r in merged if int(r['gap_days']) <= 90]
rows_180 = [r for r in merged if int(r['gap_days']) <= 180]

# 빈 이미지(반대쪽 눈 미검사) 제외 버전
def is_empty_img(pid, eye, octd):
    ml_row = ml_idx.get((pid, eye, octd))
    if not ml_row:
        return False
    gca_dir = ml_row.get('gca_dir', '')
    img_fname = 'od_thickness_map.png' if eye == 'OD' else 'os_thickness_map.png'
    img_path = os.path.join(gca_dir, img_fname)
    return os.path.exists(img_path) and os.path.getsize(img_path) < 10_000

rows_90_excl  = [r for r in rows_90  if not is_empty_img(r['patient_id'], r['eye'], r['oct_date'])]
rows_180_excl = [r for r in rows_180 if not is_empty_img(r['patient_id'], r['eye'], r['oct_date'])]

save(rows_90,       'ml_final_90d.csv',           'ml_final_90d         (gap ≤ 90일,  전체)')
save(rows_90_excl,  'ml_final_90d_excl_empty.csv', 'ml_final_90d_excl    (gap ≤ 90일,  빈 이미지 제외)')
save(rows_180,      'ml_final_180d.csv',           'ml_final_180d        (gap ≤ 180일, 전체)')
save(rows_180_excl, 'ml_final_180d_excl_empty.csv','ml_final_180d_excl   (gap ≤ 180일, 빈 이미지 제외)')

print(f'\n컬럼 수: {len(FIELDS)} ({len(PT)} VF points + {len(OCT_COLS)} OCT + 6 meta)')
