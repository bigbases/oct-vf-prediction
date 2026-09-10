#!/usr/bin/env python3
"""cohort_md.csv 의 MD 열이 라벨과 정합하는지 검증한다. 읽기 전용.

생성 코드가 저장소에 없어 재현이 불가능하다. 그래서 정합성으로 근거를 댄다.
  1. MD ~ 라벨 평균감도 회귀 (52점 / 54점 두 정의)
  2. proxy_mean_thresh 의 교차 일치가 양자화에 의한 우연인지
env: hvf
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLIND = {25, 34}            # 0-based 인덱스. make_case_heatmap.py 와 동일


def rows(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


def label_vec(r):
    """p01..p54 를 float 배열로. 빈칸은 nan."""
    out = np.full(54, np.nan)
    for i in range(54):
        v = (r.get(f'p{i+1:02d}') or '').strip()
        if v not in ('', 'NA', 'nan'):
            out[i] = float(v)
    return out


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + intercept
    resid = y - pred
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(n=int(len(x)), r=r, r2=r * r, slope=float(slope),
                intercept=float(intercept), resid_sd=float(resid.std(ddof=2)),
                resid_top10_abs=[round(float(v), 3)
                                 for v in np.sort(np.abs(resid))[::-1][:10]])


def main():
    ml = rows(ROOT / 'ml_final_90d_excl_empty_flip.csv')
    md = rows(ROOT / 'cohort_md.csv')
    key = lambda r: (r['patient_id'], r['eye'], r['vf_date'])
    mdx = {key(r): r for r in md}

    out = {'note': 'cohort_md.csv 정합성 검증. 읽기 전용. 개별 식별자·값 미포함.',
           'n_ml_rows': len(ml), 'n_md_rows': len(md)}

    # --- 1. MD 대 평균 감도 -------------------------------------------------
    pairs = {'blind_excluded_52': [], 'blind_included_54': []}
    n_matched = n_md_present = 0
    for r in ml:
        m = mdx.get(key(r))
        if m is None:
            continue
        n_matched += 1
        s = (m.get('MD') or '').strip()
        if s in ('', 'NA', 'nan'):
            continue
        v = label_vec(r)
        if np.isnan(v).any():
            continue
        n_md_present += 1
        mdv = float(s)
        m52 = float(np.mean([v[i] for i in range(54) if i not in BLIND]))
        pairs['blind_excluded_52'].append((m52, mdv))
        pairs['blind_included_54'].append((float(v.mean()), mdv))
    out['n_key_matched'] = n_matched
    out['n_with_md_and_full_labels'] = n_md_present
    out['regression'] = {k: ols([a for a, _ in v], [b for _, b in v])
                         for k, v in pairs.items() if v}

    # --- 2. proxy_mean_thresh -------------------------------------------------
    means52, means54 = [], []
    for r in ml:
        v = label_vec(r)
        if np.isnan(v).any():
            continue
        means54.append(float(v.mean()))
        means52.append(float(np.mean([v[i] for i in range(54) if i not in BLIND])))

    def collisions(vals, tol=0.005):
        """서로 tol 이내인 **서로 다른 행 쌍**에 관여하는 행 수."""
        a = np.sort(np.asarray(vals))
        hit = 0
        arr = np.asarray(vals)
        for i, x in enumerate(arr):
            d = np.abs(arr - x)
            d[i] = np.inf
            if (d <= tol).any():
                hit += 1
        return int(hit), int(len(arr))

    c54 = collisions(means54)
    c52 = collisions(means52)
    out['proxy_collision'] = {
        'note': ('54개 정수 감도의 평균은 S/54 로 양자화된다. 0.005 이내 일치는 '
                 '합이 정확히 같다는 뜻이므로 배경 충돌이 흔할 수 있다.'),
        'n_rows_full_labels': c54[1],
        'rows_colliding_within_0.005_mean54': c54[0],
        'rows_colliding_within_0.005_mean52': c52[0],
    }

    # proxy_mean_thresh 가 어느 정의와 맞는가
    defs = {}
    cand = {}
    for r in ml:
        m = mdx.get(key(r))
        if m is None:
            continue
        s = (m.get('proxy_mean_thresh') or '').strip()
        if s in ('', 'NA', 'nan'):
            continue
        v = label_vec(r)
        if np.isnan(v).any():
            continue
        p = float(s)
        cand.setdefault('mean54_postclip', []).append(abs(p - v.mean()))
        cand.setdefault('mean52_postclip', []).append(
            abs(p - np.mean([v[i] for i in range(54) if i not in BLIND])))
    for k, v in cand.items():
        a = np.asarray(v)
        defs[k] = {'n': int(a.size), 'median_abs_diff': round(float(np.median(a)), 4),
                   'frac_within_0.005': round(float((a <= 0.005).mean()), 4),
                   'frac_within_0.05': round(float((a <= 0.05).mean()), 4)}
    out['proxy_definition_match'] = defs
    out['proxy_definition_caveat'] = (
        '정본 CSV 의 p01..p54 는 -1 을 0 으로 매핑한 뒤의 값이다. 매핑 전 raw '
        'threshold 나 total deviation 과의 대조는 원자료가 없어 불가능하다.')

    dst = ROOT / 'experiments' / 'metadata_audit' / 'md_consistency.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
