"""
Phase B 적용: sfa_review_master.csv → sfa_thresholds.csv 업데이트 + 로그
"""
import csv, os, sys, shutil
from datetime import datetime
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from paths import ROOT
os.chdir(ROOT)

MASTER  = 'sfa_review_master.csv'
SFA     = 'sfa_thresholds.csv'
LOG     = 'sfa_review_log.csv'
BACKUP  = 'sfa_thresholds_before_phase_b.csv'

PT = [f'p{i+1:02d}' for i in range(54)]

# ── 백업
if not os.path.exists(BACKUP):
    shutil.copy(SFA, BACKUP)
    print(f'백업: {BACKUP}')

# ── 현재 sfa_thresholds.csv
rows = list(csv.DictReader(open(SFA, encoding='utf-8-sig')))
row_idx = {(r['patient_id'], r['eye'], r['vf_date']): i
           for i, r in enumerate(rows)}

# ── CSV 읽기
master_rows = list(csv.DictReader(open(MASTER, encoding='utf-8-sig')))

ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 통계
stat_status = Counter()
stat_changes_per_pair = Counter()
changes = []  # (pid, eye, date, point, old, new, suspect_category)
errors = []

# 의심 카테고리 매핑 (수정 사유 분류용)
def classify_change(old, new):
    try:
        old_i = int(float(old)) if old not in ('','-1',None) else (-1 if old=='-1' else None)
    except: old_i = None
    try:
        new_i = int(float(new)) if new not in ('','-1',None) else (-1 if new=='-1' else None)
    except: new_i = None

    if old_i is None and new_i is not None: return 'add_missing'
    if old_i is not None and new_i is None: return 'remove'
    if old_i == -1 and new_i != -1: return 'unfreeze_neg'
    if new_i == -1 and old_i != -1: return 'set_neg'

    if old_i is not None and new_i is not None:
        if old_i in (44, 11, 22, 33) and new_i < 10:
            return 'marker_adjacent'
        if abs(old_i - new_i) >= 10:
            return 'digit_error'
        if 5 <= abs(old_i - new_i) < 10:
            return 'medium_error'
        if 1 <= abs(old_i - new_i) < 5:
            return 'small_error'
    return 'other'

# CSV 행 순회
for mr in master_rows:
    pid = str(mr.get('patient_id') or '').strip()
    eye = str(mr.get('eye') or '').strip()
    date = str(mr.get('vf_date') or '').strip()
    status = str(mr.get('status') or 'pending').strip().lower()
    note = mr.get('reviewer_note') or ''

    if not pid: continue

    stat_status[status] += 1

    if status == 'pending':
        continue

    key = (pid, eye, date)
    if key not in row_idx:
        errors.append(f'행 없음: {key}')
        continue
    i = row_idx[key]
    cur_row = rows[i]

    n_changes_this_pair = 0
    for p in PT:
        new_v = mr.get(p, '')
        old_v = cur_row.get(p, '')

        # 정규화
        if new_v is None or (isinstance(new_v, str) and new_v.strip() == ''):
            new_str = ''
        else:
            try: new_str = str(int(float(str(new_v).strip())))
            except: new_str = str(new_v).strip()

        if new_str != old_v:
            cur_row[p] = new_str
            cat = classify_change(old_v, new_str)
            changes.append((pid, eye, date, p, old_v, new_str, cat, status, note))
            n_changes_this_pair += 1

    if n_changes_this_pair > 0:
        stat_changes_per_pair[n_changes_this_pair] += 1

if errors:
    print(f'[오류] {len(errors)}건')
    for e in errors[:10]: print(f'  {e}')

# ── 저장
fields = list(rows[0].keys())
with open(SFA, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── 마스터 로그 append
log_fields = ['tier','patient_id','eye','vf_date','point','current_value',
              'corrected_value','action','category','timestamp','reviewer_note']
log_existing = []
if os.path.exists(LOG):
    log_existing = list(csv.DictReader(open(LOG, encoding='utf-8-sig')))

new_log = []
for c in changes:
    new_log.append({
        'tier': 'B',
        'patient_id': c[0], 'eye': c[1], 'vf_date': c[2],
        'point': c[3], 'current_value': c[4], 'corrected_value': c[5],
        'action': 'error', 'category': c[6],
        'timestamp': ts, 'reviewer_note': c[8],
    })

with open(LOG, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=log_fields)
    w.writeheader()
    w.writerows(log_existing)
    w.writerows(new_log)

# ── 리포트
print(f'\n=== Phase B 적용 결과 ===')
print(f'\n[Status 분포]')
for s, c in stat_status.most_common():
    print(f'  {s:<10}: {c}')

print(f'\n[수정 통계]')
print(f'  총 수정 셀: {len(changes)}개')
print(f'  영향 받은 페어: {sum(stat_changes_per_pair.values())}개')

if stat_changes_per_pair:
    print(f'  페어당 수정 셀 수 분포:')
    for k in sorted(stat_changes_per_pair.keys()):
        print(f'    {k}개 셀 수정: {stat_changes_per_pair[k]} 페어')

# 패턴 분류
cat_counter = Counter(c[6] for c in changes)
print(f'\n[수정 패턴]')
for cat, cnt in cat_counter.most_common():
    print(f'  {cat:<20}: {cnt}')

# 자주 틀리는 위치
loc_counter = Counter((c[1], c[3]) for c in changes)
print(f'\n[자주 틀리는 위치 Top 10]')
for (eye, p), cnt in loc_counter.most_common(10):
    print(f'  {eye} {p}: {cnt}건')

# 자주 나오는 잘못된 값
val_counter = Counter(c[4] for c in changes if c[4] not in ('','-1'))
print(f'\n[자주 나오는 잘못된 OCR 원본 값 Top 10]')
for v, cnt in val_counter.most_common(10):
    print(f'  "{v}": {cnt}건')

print(f'\n저장:')
print(f'  sfa_thresholds.csv (수정 적용)')
print(f'  {LOG} ({len(log_existing) + len(new_log)}행)')
print(f'  백업: {BACKUP}')
print(f'  소스: {MASTER}')
