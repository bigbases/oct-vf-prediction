#!/usr/bin/env python3
"""OCT(GCA+RNFL) tabular 수치만 추출 — 환자별 정리.

입력: ml_final_{90,180}d_excl_empty_flip.csv
출력 (tag=90d|180d):
  runs/oct_tabular_{tag}.csv
  runs/oct_tabular_{tag}_target_eye.csv
  runs/oct_tabular_by_patient_{tag}/
  runs/oct_tabular_{tag}_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

META = ['patient_id', 'eye', 'vf_date', 'oct_date', 'gap_days', 'split', 'cv_fold']
GCA_OD = ['avg_gcl_od', 'min_gcl_od',
          'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n']
GCA_OS = ['avg_gcl_os', 'min_gcl_os',
          'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n']
RNFL_SUM = ['od_avg_rnfl', 'os_avg_rnfl', 'od_vert_cd', 'os_vert_cd']
RNFL_TAB = ['rnfl_q_s', 'rnfl_q_t', 'rnfl_q_i', 'rnfl_q_n'] + [f'rnfl_h{i:02d}' for i in range(1, 13)]
OCT_ALL = GCA_OD + GCA_OS + RNFL_SUM + RNFL_TAB

TARGET_OD = [
    'avg_gcl_od', 'min_gcl_od',
    'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n',
    'od_avg_rnfl', 'od_vert_cd',
] + RNFL_TAB
TARGET_OS = [
    'avg_gcl_os', 'min_gcl_os',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'os_avg_rnfl', 'os_vert_cd',
] + RNFL_TAB
TARGET_RENAME = [
    'avg_gcl', 'min_gcl',
    's_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n',
    'avg_rnfl', 'vert_cd',
] + [c.replace('rnfl_', '') for c in RNFL_TAB]


def load_rows(csv_path: Path) -> list[dict]:
    return list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def target_eye_row(r: dict) -> dict:
    src = TARGET_OD if r['eye'] == 'OD' else TARGET_OS
    out = {k: r.get(k, '') for k in META}
    for new, old in zip(TARGET_RENAME, src):
        out[new] = r.get(old, '')
    return out


def export(src: Path, tag: str) -> dict:
    rows = load_rows(src)
    rows.sort(key=lambda r: (str(r['patient_id']), r['eye'], r['vf_date']))

    ou_fields = META + OCT_ALL
    ou_rows = [{k: r.get(k, '') for k in ou_fields} for r in rows]
    out_ou = ROOT / f'runs/oct_tabular_{tag}.csv'
    write_csv(out_ou, ou_fields, ou_rows)

    te_fields = META + TARGET_RENAME
    te_rows = [target_eye_row(r) for r in rows]
    out_te = ROOT / f'runs/oct_tabular_{tag}_target_eye.csv'
    write_csv(out_te, te_fields, te_rows)

    by_pat_dir = ROOT / f'runs/oct_tabular_by_patient_{tag}'
    by_pat_dir.mkdir(parents=True, exist_ok=True)
    pat: dict[str, list] = defaultdict(list)
    for r in ou_rows:
        pat[str(r['patient_id'])].append(r)
    for pid, pr in sorted(pat.items()):
        write_csv(by_pat_dir / f'{pid}.csv', ou_fields, pr)

    meta = {
        'source': str(src.relative_to(ROOT)),
        'tag': tag,
        'n_visits': len(rows),
        'n_patients': len(pat),
        'outputs': {
            'all_visits_ou': str(out_ou.relative_to(ROOT)),
            'all_visits_target_eye': str(out_te.relative_to(ROOT)),
            'per_patient_dir': str(by_pat_dir.relative_to(ROOT)),
        },
        'oct_columns_ou': OCT_ALL,
        'oct_columns_target_eye': TARGET_RENAME,
        'note': 'VF p01-p54 제외. rnfl_q/h는 flip CSV 기준(OD 좌표계). target_eye는 eye 컬럼 기준 26 feature.',
    }
    meta_path = ROOT / f'runs/oct_tabular_{tag}_meta.json'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[{tag}] 방문 {len(rows)} / 환자 {len(pat)}', flush=True)
    print(f'  {out_ou}', flush=True)
    print(f'  {out_te}', flush=True)
    print(f'  {by_pat_dir}/ ({len(pat)} files)', flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--tag', default=None, help='출력 접두 (기본: csv 파일명에서 90d/180d 추론)')
    args = ap.parse_args()

    src = ROOT / args.csv
    if not src.is_file():
        raise SystemExit(f'없음: {src}')

    tag = args.tag
    if tag is None:
        name = src.name
        if '180d' in name:
            tag = '180d'
        elif '90d' in name:
            tag = '90d'
        else:
            tag = src.stem.replace('ml_final_', '').replace('_excl_empty_flip', '')

    export(src, tag)


if __name__ == '__main__':
    main()
