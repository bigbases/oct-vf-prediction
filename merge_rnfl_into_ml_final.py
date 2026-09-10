"""
기존 ml_final_*.csv에 RNFL quadrant + clock-hour 컬럼 merge.

- non-flip ml_final  → rnfl_detail.csv / rnfl_detail_180d.csv
- flip ml_final      → rnfl_detail_flip.csv / rnfl_detail_180d_flip.csv
  (clockhours.csv / clockhours_flip.csv와 동일 구조)

조인 키: (patient_id, eye, oct_date)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))

RNFL_COLS = (
    ['rnfl_q_s', 'rnfl_q_t', 'rnfl_q_i', 'rnfl_q_n']
    + [f'rnfl_h{i:02d}' for i in range(1, 13)]
)

MERGE_PAIRS = [
    ('ml_final_90d.csv', 'clockhours.csv'),
    ('ml_final_90d_excl_empty.csv', 'clockhours.csv'),
    ('ml_final_90d_flip.csv', 'clockhours_flip.csv'),
    ('ml_final_90d_excl_empty_flip.csv', 'clockhours_flip.csv'),
    ('ml_final_180d.csv', 'clockhours_180d.csv'),
    ('ml_final_180d_excl_empty.csv', 'clockhours_180d.csv'),
    ('ml_final_180d_flip.csv', 'clockhours_180d_flip.csv'),
    ('ml_final_180d_excl_empty_flip.csv', 'clockhours_180d_flip.csv'),
]


def load_rnfl_index(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    return {(r['patient_id'], r['eye'], r['oct_date']): r for r in rows}


def merge_file(ml_path: Path, rnfl_path: Path) -> None:
    ml_rows = list(csv.DictReader(open(ml_path, encoding='utf-8-sig')))
    if not ml_rows:
        print(f'  [skip] empty: {ml_path.name}')
        return

    rnfl_idx = load_rnfl_index(rnfl_path)
    base_fields = list(ml_rows[0].keys())
    # 이미 merge된 경우 RNFL 컬럼 제거 후 재삽입
    out_fields = [f for f in base_fields if f not in RNFL_COLS]
    insert_at = out_fields.index('cv_fold') if 'cv_fold' in out_fields else len(out_fields)
    out_fields = out_fields[:insert_at] + RNFL_COLS + out_fields[insert_at:]

    missing = []
    merged_rows = []
    for r in ml_rows:
        key = (r['patient_id'], r['eye'], r['oct_date'])
        rnfl = rnfl_idx.get(key)
        if rnfl is None:
            missing.append(key)
        out = {f: r.get(f, '') for f in out_fields if f not in RNFL_COLS}
        for c in RNFL_COLS:
            out[c] = rnfl.get(c, '') if rnfl else ''
        merged_rows.append(out)

    with open(ml_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(merged_rows)

    filled = sum(1 for r in merged_rows if any(r.get(c, '') not in ('', None) for c in RNFL_COLS))
    print(f'  {ml_path.name}: {len(merged_rows)}행, RNFL 채움 {filled}행, 미매칭 {len(missing)}')
    if missing:
        for k in missing[:5]:
            print(f'    - {k}')
        if len(missing) > 5:
            print(f'    ... 외 {len(missing) - 5}건')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', help='ml_final 파일명만 처리')
    args = parser.parse_args()

    pairs = MERGE_PAIRS
    if args.only:
        only = set(args.only)
        pairs = [(m, r) for m, r in pairs if m in only]

    print(f'ROOT: {ROOT}\n')
    for ml_name, rnfl_name in pairs:
        ml_path = ROOT / ml_name
        rnfl_path = ROOT / rnfl_name
        if not ml_path.exists():
            print(f'[skip] 없음: {ml_name}')
            continue
        if not rnfl_path.exists():
            print(f'[skip] RNFL 없음: {rnfl_name} (for {ml_name})')
            continue
        print(f'{ml_name} <- {rnfl_name}')
        merge_file(ml_path, rnfl_path)

    print('\n완료.')


if __name__ == '__main__':
    main()
