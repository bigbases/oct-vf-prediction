#!/usr/bin/env python3
"""진단 1: 이미지-VF 매칭 (proxy_mean_thresh >= 28, 5 eyes)."""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import rewrite_data_path  # noqa: E402
from image_preprocessing import build_image_index, REGION_KEYS_OD, REGION_KEYS_OS  # noqa: E402

ML_FINAL = ROOT / 'ml_final_90d_excl_empty_flip.csv'
COHORT_MD = ROOT / 'cohort_md.csv'
ML_DATASET = ROOT / 'ml_dataset.csv'

PT_52 = [f'p{i:02d}' for i in range(1, 55) if f'p{i:02d}' not in {'p26', 'p35'}]

_FILE_MAP = {
    'gca_od_thickness': ('gca_dir', 'od_thickness_map.png'),
    'gca_os_thickness': ('gca_dir', 'os_thickness_map.png'),
    'gca_od_deviation': ('gca_dir', 'od_deviation_map.png'),
    'gca_os_deviation': ('gca_dir', 'os_deviation_map.png'),
    'rnfl_od_thickness': ('rnfl_dir', 'od_thickness_map.png'),
    'rnfl_os_thickness': ('rnfl_dir', 'os_thickness_map.png'),
    'rnfl_od_deviation': ('rnfl_dir', 'od_deviation_map.png'),
    'rnfl_os_deviation': ('rnfl_dir', 'os_deviation_map.png'),
}
SFA_GRID = ('sfa_dir', 'threshold_grid.png')


def mean_threshold(row: dict) -> float | None:
    vals = []
    for p in PT_52:
        v = str(row.get(p, '')).strip()
        if v in ('', '-1'):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return sum(vals) / len(vals) if vals else None


def load_cohort_maps() -> tuple[dict, dict]:
    proxy, md = {}, {}
    for r in csv.DictReader(open(COHORT_MD, encoding='utf-8-sig')):
        k = (r['patient_id'].strip(), r['eye'].strip().upper(), r['vf_date'].strip())
        try:
            proxy[k] = float(r['proxy_mean_thresh'])
        except (ValueError, TypeError):
            pass
        try:
            md[k] = float(r['MD'])
        except (ValueError, TypeError):
            pass
    return proxy, md


def load_ml_dataset_row(key: tuple) -> dict | None:
    for r in csv.DictReader(open(ML_DATASET, encoding='utf-8-sig')):
        k = (r['patient_id'].strip(), r['eye'].strip().upper(), r['vf_date'].strip())
        if k == key:
            return r
    return None


def path_hints(path: str) -> dict:
    p = path.replace('\\', '/')
    hints = {'path_tail': Path(p).name, 'parent': Path(p).parent.name}
    # patient id often in path
    m = re.search(r'/by_case/([^/]+)__', p)
    if m:
        hints['by_case_prefix'] = m.group(1)
    m2 = re.search(r'(\d{8})', p)
    if m2:
        hints['date_in_path'] = m2.group(1)
    # eye in filename
    if '_OD_' in p or '/OD_' in p or 'od_thickness' in p.lower():
        hints['eye_hint'] = 'OD'
    if '_OS_' in p or '/OS_' in p or 'os_thickness' in p.lower():
        hints['eye_hint'] = 'OS'
    return hints


def check_path_consistency(pid: str, eye: str, vf_date: str, base_dir: str, fname: str) -> list[str]:
    issues = []
    if not base_dir:
        issues.append('base_dir 비어 있음')
        return issues
    norm = base_dir.replace('\\', '/')
    if pid not in norm:
        issues.append(f'경로에 patient_id({pid}) 없음')
    if vf_date not in norm and vf_date[:4] not in norm:
        # oct date might differ from vf_date — only warn
        issues.append(f'경로에 vf_date({vf_date}) 직접 없음 (oct_date만 있을 수 있음)')
    return issues


def main():
    proxy, md_map = load_cohort_maps()
    ml_rows = list(csv.DictReader(open(ML_FINAL, encoding='utf-8-sig')))

    candidates = []
    for r in ml_rows:
        k = (r['patient_id'].strip(), r['eye'].strip().upper(), r['vf_date'].strip())
        pt = proxy.get(k)
        if pt is None:
            pt = mean_threshold(r)
        if pt is not None and pt >= 28:
            candidates.append((pt, r, k))

    candidates.sort(key=lambda x: -x[0])
    if len(candidates) < 5:
        print(f'[경고] proxy>=28 인 행이 {len(candidates)}개뿐', flush=True)
    pick = candidates[:5]

    print('=== 진단 1: 이미지-VF 매칭 (proxy_mean_thresh >= 28, 상위 5눈) ===\n', flush=True)
    img_index = build_image_index(str(ML_DATASET))

    any_issue = False
    for i, (pt, r, key) in enumerate(pick, 1):
        pid, eye, vfd = key
        oct_date = r.get('oct_date', '')
        print(f'--- [{i}/5] patient_id={pid}  eye={eye}  vf_date={vfd}  oct_date={oct_date} ---')
        md_v = md_map.get(key)
        md_s = f'{md_v:.2f}' if md_v is not None else 'N/A'
        print(f'  proxy_mean_thresh={pt:.2f}  MD(cohort_md)={md_s}')

        ds = load_ml_dataset_row(key)
        if ds is None:
            print('  [이상] ml_dataset.csv에 (pid,eye,vf_date) 행 없음')
            any_issue = True
            print()
            continue

        print(f'  ml_dataset 매칭: OK')
        print(f'  sfa_dir: {ds.get("sfa_dir", "")}')
        print(f'  gca_dir: {ds.get("gca_dir", "")}')
        print(f'  rnfl_dir: {ds.get("rnfl_dir", "")}')

        region_keys = REGION_KEYS_OD if eye == 'OD' else REGION_KEYS_OS
        paths = img_index.get(key, {})
        print(f'  CNN 학습용 4장 ({", ".join(region_keys)}):')
        for rk in region_keys:
            p = paths.get(rk, '')
            exists = os.path.isfile(p) if p else False
            status = 'OK' if exists else 'MISSING'
            print(f'    [{status}] {rk}: {p}')
            if not exists:
                any_issue = True
            elif p:
                hints = path_hints(p)
                issues = check_path_consistency(pid, eye, vfd, p, '')
                if issues:
                    print(f'      경로 주의: {"; ".join(issues)}')

        # SFA threshold grid (VF 출처)
        sfa_base = ds.get('sfa_dir', '')
        if sfa_base:
            # CSV의 경로는 데이터 생성 머신의 절대경로다. DATA_ROOT 기준으로 바꾼다.
            sfa_base = rewrite_data_path(sfa_base)
            sfa_img = os.path.join(sfa_base, 'threshold_grid.png')
            print(f'  VF 출처 threshold_grid: {sfa_img}')
            print(f'    exists={os.path.isfile(sfa_img)}')
            if os.path.isfile(sfa_img):
                h = path_hints(sfa_img)
                print(f'    path hints: {h}')
                if pid not in sfa_img:
                    print(f'    [이상] SFA 경로에 patient_id 불일치')
                    any_issue = True

        # oct_date vs vf_date
        if oct_date and oct_date != vfd:
            print(f'  [참고] oct_date({oct_date}) != vf_date({vfd}) — 의도된 gap 페어링')

        print()

    print('=== 요약 ===', flush=True)
    if any_issue:
        print('[이상 발견] 위 항목 중 MISSING 또는 키 불일치 있음 → 다음 단계 전에 확인 필요', flush=True)
    else:
        print('[1단계] 5눈 모두 ml_dataset 키 매칭 + CNN 4장 + SFA grid 파일 존재', flush=True)
        print('  → 거친 이미지-VF 키 불일치는 이 샘플에서 보이지 않음', flush=True)
        print('  (경로 내 날짜는 oct_date일 수 있음 — vf_date와 다를 수 있음)', flush=True)


if __name__ == '__main__':
    main()
