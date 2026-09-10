#!/usr/bin/env python3
"""
0단계: 코호트 VF 중증도·분포 감사 (GPU 0, 학습 없음).

핵심 설계:
  (A) 중증도·MD proxy = 이미 학습에 쓰는 52점 threshold에서 계산 (이미지 MD OCR 안 함)
  (B) 인쇄 MD/VFI/PSD가 CSV에 있으면 검증용 상관만 (층화·학습에는 (A)만)
  신뢰도(False NEG / Fixation Loss 등): 있으면 기록만, 필터링 안 함

<0> 인코딩 (확인됨, 2026-06):
  ocr_threshold.parse_value: 문자열에 '<' 포함 → -1
  sfa_thresholds: -1은 p26/p35(맹점)에만 존재, 비맹점 52열에는 -1 없음
  ml_final 52열: -1 없음, 0은 유효값(학습 mask=1) — <0·저감도는 0으로 저장된 케이스 다수
  → parse: ''/결측·-1만 무효; 0 dB는 평균에 포함 (image_preprocessing과 동일)

산출:
  runs/cohort_vf_audit.json
  runs/cohort_vf_audit.txt
  runs/cohort_vf_severity.csv  (눈별 라벨, 1단계 merge 키)

사용법:
  python scripts/audit_cohort_vf.py
  python scripts/audit_cohort_vf.py --csv ml_final_180d_excl_empty_flip.csv --tag 180d
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PT_ALL = [f'p{i:02d}' for i in range(1, 55)]
BLIND_SPOT = frozenset({'p26', 'p35'})
PT_52 = [p for p in PT_ALL if p not in BLIND_SPOT]

SEVERITY_ORDER = ['near_normal', 'mild', 'moderate', 'advanced', 'unknown']

# mean threshold (dB) — 순서 proxy, 정식 age-corrected MD 아님
def severity_from_mean_threshold(mean_thr: float) -> str:
    if np.isnan(mean_thr):
        return 'unknown'
    if mean_thr >= 28:
        return 'near_normal'
    if mean_thr >= 24:
        return 'mild'
    if mean_thr >= 18:
        return 'moderate'
    return 'advanced'


def parse_threshold(cell: Any, point: str) -> Tuple[float, bool]:
    """(값, 유효). image_preprocessing.py와 동일 원칙."""
    if cell is None:
        return np.nan, False
    s = str(cell).strip()
    if s == '' or s.lower() == 'nan':
        return np.nan, False
    if s == '-1':
        return np.nan, False
    try:
        return float(s), True
    except ValueError:
        return np.nan, False


def first_col(row: dict, names: tuple[str, ...]) -> Optional[str]:
    for n in names:
        if n in row and str(row[n]).strip() not in ('', 'nan'):
            return str(row[n]).strip()
    return None


def text_histogram(values: np.ndarray, bins: list[float], width: int = 40) -> list[str]:
    counts, edges = np.histogram(values, bins=bins)
    mx = int(counts.max()) if counts.size else 1
    lines = []
    for i, c in enumerate(counts):
        bar = '#' * int(round(c / mx * width)) if mx else ''
        lines.append(f'  [{edges[i]:5.1f},{edges[i+1]:5.1f})  {c:4d}  {bar}')
    return lines


def audit(csv_path: Path, tag: str) -> tuple[dict, np.ndarray, list[dict]]:
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig', newline='')))
    missing_cols = [p for p in PT_52 if p not in (rows[0].keys() if rows else [])]
    if missing_cols:
        raise SystemExit(
            f'[중단] threshold 컬럼 없음: {missing_cols[:5]}... '
            f'(p01–p54, 맹점 {sorted(BLIND_SPOT)} 제외 규칙 확인)'
        )

    per_eye_rows: list[dict] = []
    all_pts: list[float] = []
    enc_counter: Counter = Counter()

    for idx, row in enumerate(rows):
        vals, mask = [], []
        for p in PT_52:
            v, ok = parse_threshold(row.get(p), p)
            raw = str(row.get(p, '')).strip()
            if raw:
                enc_counter[raw] += 1
            if ok:
                vals.append(v)
                mask.append(True)
            else:
                vals.append(np.nan)
                mask.append(False)

        arr = np.array(vals, dtype=np.float64)
        m = np.array(mask, dtype=bool)
        n_valid = int(m.sum())
        if n_valid == 0:
            mean_thr = min_thr = median_thr = np.nan
        else:
            valid = arr[m]
            mean_thr = float(np.mean(valid))
            min_thr = float(np.min(valid))
            median_thr = float(np.median(valid))
            all_pts.extend(valid.tolist())

        sev = severity_from_mean_threshold(mean_thr)

        per_eye_rows.append({
            'row_idx': idx,
            'patient_id': row.get('patient_id', ''),
            'eye': row.get('eye', ''),
            'vf_date': row.get('vf_date', ''),
            'oct_date': row.get('oct_date', ''),
            'cv_fold': row.get('cv_fold', ''),
            'n_valid_points': n_valid,
            'mean_threshold_db': round(mean_thr, 2) if not np.isnan(mean_thr) else None,
            'median_threshold_db': round(median_thr, 2) if not np.isnan(median_thr) else None,
            'min_threshold_db': round(min_thr, 2) if not np.isnan(min_thr) else None,
            'severity_proxy': sev,
            # 신뢰도: CSV에 있으면 기록만 (현재 ml_final에는 없음 → None)
            'false_neg_pct': first_col(row, ('false_neg', 'false_negative', 'false_neg_pct')),
            'false_pos_pct': first_col(row, ('false_pos', 'false_positive', 'false_pos_pct')),
            'fixation_loss': first_col(row, ('fixation_loss', 'fixation_losses')),
            # (B) 인쇄 MD 검증용 — ml_final에 없으면 None
            'printed_md_db': first_col(row, ('md', 'MD', 'md24_2', 'MD24-2', 'md_24_2')),
            'printed_psd_db': first_col(row, ('psd', 'PSD', 'psd24_2', 'PSD24-2')),
            'printed_vfi_pct': first_col(row, ('vfi', 'VFI', 'vfi_pct')),
        })

    pts = np.array(all_pts, dtype=np.float64)
    means = np.array(
        [r['mean_threshold_db'] for r in per_eye_rows if r['mean_threshold_db'] is not None],
        dtype=np.float64,
    )

    sev_counts = Counter(r['severity_proxy'] for r in per_eye_rows)
    n_eyes = len(per_eye_rows)

    def numeric_token_count(target: float) -> int:
        total = 0
        for token, count in enc_counter.items():
            try:
                if float(token) == target:
                    total += count
            except ValueError:
                continue
        return total

    n_minus1 = numeric_token_count(-1.0)
    n_zero = numeric_token_count(0.0)

    summary: dict = {
        'tag': tag,
        'csv': str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path),
        'methodology': {
            'severity_source': '(A) computed from 52 threshold points in CSV (same as training labels)',
            'no_image_md_ocr': True,
            'printed_md_use': '(B) validation correlation only if column present',
            'reliability': 'record if present, never filter (preserve N=280)',
            'severity_is_proxy': (
                'mean threshold order proxy — NOT age-corrected MD; '
                'stage-1 stratification until EMR MD merged'
            ),
        },
        'encoding_audit': {
            'ocr_threshold_rule': "parse_value: text contains '<' → -1 (displayed as <0)",
            'ml_final_52pt': {
                'note': (
                    'Matches training: numeric 0 is valid; numeric -1/empty is '
                    'masked. Counts combine lexical variants such as 0 and 0.0.'
                ),
                'top_raw_tokens': dict(enc_counter.most_common(12)),
                'count_raw_minus1': n_minus1,
                'count_raw_zero': n_zero,
            },
            'blind_spots_excluded': sorted(BLIND_SPOT),
        },
        'n_rows': len(rows),
        'n_eyes': n_eyes,
        'n_valid_points': int(pts.size),
        'target': {
            'type': 'Humphrey 24-2 total threshold (dB)',
            'n_points': len(PT_52),
            'not_TD': True,
        },
        'point_level_threshold_db': {
            'median': float(np.median(pts)),
            'mean': float(np.mean(pts)),
            'std': float(np.std(pts)),
            'pct_ge_30': round(100 * float((pts >= 30).mean()), 1),
            'pct_lt_10': round(100 * float((pts < 10).mean()), 1),
        },
        'per_eye_mean_threshold_db': {
            'median': float(np.median(means)),
            'mean': float(np.mean(means)),
        },
        'severity_proxy_counts': dict(sev_counts),
        'severity_proxy_pct': {
            k: round(100 * sev_counts[k] / n_eyes, 1) for k in sev_counts
        },
        'stage1_note': (
            'Proxy severity supports ordering for internal stratified RMSE; '
            'EMR MD still needed for paper-aligned normal vs glaucoma bins. '
            'Do NOT claim causal cohort effect from near_normal cut alone (circularity).'
        ),
        'paper_reference_rmse_db': {
            'overall': 4.79,
            'normal_cohort': 3.27,
            'glaucoma_cohort': 5.27,
        },
    }

    # (B) printed MD vs mean threshold
    pairs = [
        (r['mean_threshold_db'], r['printed_md_db'])
        for r in per_eye_rows
        if r['mean_threshold_db'] is not None and r['printed_md_db'] is not None
    ]
    if pairs:
        xs, ys = [], []
        for m, pmd in pairs:
            try:
                ys.append(float(str(pmd).replace('−', '-').split()[0]))
                xs.append(float(m))
            except ValueError:
                pass
        if len(xs) >= 5:
            r = float(np.corrcoef(xs, ys)[0, 1])
            summary['validation_mean_thr_vs_printed_md'] = {
                'n': len(xs),
                'pearson_r': round(r, 3),
                'note': 'High |r| → mean_thr tracks printed MD; still use (A) for stratification',
            }
    else:
        summary['validation_mean_thr_vs_printed_md'] = {
            'n': 0,
            'note': 'printed MD not in CSV — optional future merge or header OCR',
        }

    rel_present = sum(
        1 for r in per_eye_rows
        if any(r[k] for k in ('false_neg_pct', 'false_pos_pct', 'fixation_loss'))
    )
    summary['reliability_columns'] = {
        'n_eyes_with_any': rel_present,
        'in_ml_final': rel_present > 0,
        'action': 'extend OCR/header parse later; do not filter on reliability',
    }

    return summary, means, per_eye_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--tag', default='90d')
    ap.add_argument('--out-json', default='runs/cohort_vf_audit.json')
    ap.add_argument('--out-txt', default='runs/cohort_vf_audit.txt')
    ap.add_argument('--out-csv', default='runs/cohort_vf_severity.csv')
    args = ap.parse_args()

    csv_path = ROOT / args.csv
    summary, means, per_eye_rows = audit(csv_path, args.tag)

    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)

    out_json = ROOT / args.out_json
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    out_csv = ROOT / args.out_csv
    if per_eye_rows:
        fields = list(per_eye_rows[0].keys())
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_eye_rows)

    lines = [
        f'=== Cohort VF audit ({args.tag}) ===',
        f'CSV: {summary["csv"]}',
        f'Eyes: {summary["n_eyes"]}  Valid points: {summary["n_valid_points"]}',
        '',
        '--- Encoding (52pt, excl blind) ---',
        f'  OCR rule: {summary["encoding_audit"]["ocr_threshold_rule"]}',
        f'  ml_final: -1 count={summary["encoding_audit"]["ml_final_52pt"]["count_raw_minus1"]}, '
        f'0 count={summary["encoding_audit"]["ml_final_52pt"]["count_raw_zero"]}',
        '',
        '--- Point-level threshold ---',
        f'  median={summary["point_level_threshold_db"]["median"]:.1f}  '
        f'mean={summary["point_level_threshold_db"]["mean"]:.2f}  '
        f'<10 dB: {summary["point_level_threshold_db"]["pct_lt_10"]}%  '
        f'>=30 dB: {summary["point_level_threshold_db"]["pct_ge_30"]}%',
        '',
        '--- Per-eye mean threshold histogram ---',
        *text_histogram(means, bins=[0, 5, 10, 15, 18, 21, 24, 27, 30, 33, 40]),
        '',
        '--- Severity proxy (NOT EMR diagnosis) ---',
    ]
    for k in SEVERITY_ORDER:
        if k in summary['severity_proxy_counts']:
            c = summary['severity_proxy_counts'][k]
            p = summary['severity_proxy_pct'][k]
            lines.append(f'  {k:14s}  {c:4d}  ({p:5.1f}%)')

    v = summary.get('validation_mean_thr_vs_printed_md', {})
    lines += [
        '',
        f'--- (B) mean_thr vs printed MD: {v}',
        f'--- Reliability cols filled: {summary["reliability_columns"]}',
        '',
        f'JSON: {out_json}',
        f'Severity CSV: {out_csv}',
    ]
    out_txt = ROOT / args.out_txt
    out_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
