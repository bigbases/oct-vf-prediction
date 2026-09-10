#!/usr/bin/env python3
"""
MD 층별 stratified RMSE — 5-fold val OOF (≈240행) + cohort_md.csv.

층 (MD dB, 논문 잣대):
  normal:   MD > -3
  early:    -6 < MD <= -3
  moderate: -12 < MD <= -6
  advanced: MD <= -12

사용법:
  python scripts/stratified_rmse.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz, masked_overall_rmse_mae  # noqa: E402

COHORT_MD = ROOT / 'cohort_md.csv'
FUSION_W_PATH = ROOT / 'runs/p1_global_fusion.json'
OUT_JSON = ROOT / 'runs/stratified_rmse_clip_mse_5fold.json'

XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
_CLIP_MSE_DIR = ROOT / 'runs/phasec_b0_clip_mse_5fold'
CNN_VAL = {k: _CLIP_MSE_DIR / f'val_preds_fold{k}.npz' for k in range(5)}

STRATA = [
    ('normal', lambda md: md > -3, 'paper ref RMSE ~3.27'),
    ('early', lambda md: -6 < md <= -3, ''),
    ('moderate', lambda md: -12 < md <= -6, ''),
    ('advanced', lambda md: md <= -12, 'paper ref RMSE ~5.27 (glaucoma)'),
]


def norm_key(pid, eye, vf_date) -> Tuple[str, str, str]:
    return (
        str(pid).strip(),
        str(eye).strip().upper(),
        str(vf_date).strip().replace('-', '').replace('/', ''),
    )


def load_cohort_md(path: Path) -> Dict[Tuple[str, str, str], dict]:
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    out = {}
    dup = 0
    for r in rows:
        k = norm_key(r['patient_id'], r['eye'], r['vf_date'])
        if k in out:
            dup += 1
        try:
            md = float(r['MD'])
        except (ValueError, KeyError, TypeError):
            md = float('nan')
        out[k] = {
            'MD': md,
            'proxy_mean_thresh': r.get('proxy_mean_thresh', ''),
            'low_reliability': r.get('low_reliability', ''),
        }
    return out, dup, len(rows)


def stack_val_oof(paths: dict[int, Path], label: str) -> dict:
    keys_all, preds, labels, masks = [], [], [], []
    missing = []
    for k in sorted(paths):
        p = paths[k]
        if not p.is_file():
            missing.append((k, str(p)))
            continue
        d = load_oof_npz(p)
        keys_all.extend(d['keys'])
        preds.append(d['pred'])
        labels.append(d['labels'])
        masks.append(d['mask'])
    if missing:
        raise FileNotFoundError(f'{label} val npz 누락: {missing}')
    return {
        'keys': keys_all,
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'meta': {'stacked': label, 'n_folds': len(preds)},
    }


def md_stratum(md: float) -> Optional[str]:
    if not np.isfinite(md):
        return None
    for name, fn, _ in STRATA:
        if fn(md):
            return name
    return None


def metrics_for_indices(
    pred: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    idx: List[int],
) -> Optional[dict]:
    if not idx:
        return None
    p = pred[idx]
    y = labels[idx]
    m = mask[idx]
    if m.sum() == 0:
        return None
    rmse, mae = masked_overall_rmse_mae(p, y, m)
    return {
        'n_eyes': len(idx),
        'n_points': int(m.sum()),
        'rmse': rmse,
        'mae': mae,
    }


def main():
    print('=== Step 1: npz 구조 확인 ===', flush=True)
    for p in [XGB_VAL[0], CNN_VAL[0]]:
        z = np.load(p, allow_pickle=True)
        print(f'  {p.name}: files={list(z.files)}, pred={z["pred"].shape}', flush=True)

    print('\n=== Step 2: OOF stack + MD merge ===', flush=True)
    xgb = stack_val_oof(XGB_VAL, 'XGB')
    cnn = stack_val_oof(CNN_VAL, 'CNN')
    print(f'  XGB stacked rows: {len(xgb["keys"])}', flush=True)
    print(f'  CNN stacked rows: {len(cnn["keys"])}', flush=True)

    ax, bx, common = align_oof(xgb, cnn)
    print(f'  XGB∩CNN aligned rows: {len(common)}', flush=True)
    if len(common) < 200:
        print('[경고] aligned < 200 — 키 불일치 가능', flush=True)

    md_map, n_dup, n_md_rows = load_cohort_md(COHORT_MD)
    print(f'  cohort_md rows: {n_md_rows} (dup keys: {n_dup})', flush=True)

    labels = ax['labels']
    mask = ax['mask'] & bx['mask']
    pred_x = ax['pred']
    pred_c = bx['pred']

    w = 0.84
    if FUSION_W_PATH.is_file():
        w = float(json.load(open(FUSION_W_PATH, encoding='utf-8'))['w_xgb_global'])
    pred_f = w * pred_x + (1.0 - w) * pred_c

    missing_md = []
    by_stratum: Dict[str, List[int]] = {s[0]: [] for s in STRATA}
    merged_indices = []
    for i, k in enumerate(common):
        nk = norm_key(*k)
        if nk not in md_map or not np.isfinite(md_map[nk]['MD']):
            missing_md.append(k)
            continue
        merged_indices.append(i)
        st = md_stratum(md_map[nk]['MD'])
        if st:
            by_stratum[st].append(i)

    print(f'  OOF rows with MD: {len(merged_indices)} / {len(common)}', flush=True)
    if len(merged_indices) < 200:
        print('[STOP] merge 후 행 수가 예상(~240)보다 많이 적음', flush=True)
        if missing_md[:3]:
            print('  missing keys sample:', missing_md[:3], flush=True)
        sys.exit(1)

    print('\n=== Step 3: MD 층별 분포 (OOF with MD) ===', flush=True)
    n_tot = len(merged_indices)
    for name, _, note in STRATA:
        ni = len(by_stratum[name])
        print(f'  {name:10s}  {ni:3d} eyes ({100*ni/n_tot:5.1f}%)  {note}', flush=True)

    print(f'\n=== Step 4: 층별 RMSE/MAE (5-fold val OOF, w_fusion={w:.2f}) ===', flush=True)
    print(f'{"stratum":10s} {"n":>4s}  {"XGB RMSE":>9s} {"CNN RMSE":>9s} {"Fus RMSE":>9s}  {"CNN MAE":>8s}', flush=True)
    print('-' * 58, flush=True)

    results = {'n_oof_aligned': len(common), 'n_with_md': n_tot, 'w_fusion': w, 'strata': {}}

    for name, _, note in STRATA:
        idx = by_stratum[name]
        mx = metrics_for_indices(pred_x, labels, mask, idx)
        mc = metrics_for_indices(pred_c, labels, mask, idx)
        mf = metrics_for_indices(pred_f, labels, mask, idx)
        if mx is None:
            print(f'{name:10s}    0   —', flush=True)
            continue
        print(
            f'{name:10s} {mx["n_eyes"]:4d}  '
            f'{mx["rmse"]:9.3f} {mc["rmse"]:9.3f} {mf["rmse"]:9.3f}  '
            f'{mc["mae"]:8.3f}',
            flush=True,
        )
        results['strata'][name] = {
            'n_eyes': mx['n_eyes'],
            'n_points': mx['n_points'],
            'xgb': mx,
            'cnn': mc,
            'fusion': mf,
            'note': note,
        }

    # overall on merged
    ov_x = metrics_for_indices(pred_x, labels, mask, merged_indices)
    ov_c = metrics_for_indices(pred_c, labels, mask, merged_indices)
    results['overall_merged'] = {'xgb': ov_x, 'cnn': ov_c}
    print(f'\n전체(OOF+MD, n={n_tot}): XGB RMSE={ov_x["rmse"]:.3f}  CNN RMSE={ov_c["rmse"]:.3f}', flush=True)

    cnn_norm = results['strata'].get('normal', {}).get('cnn')
    if cnn_norm:
        rm = cnn_norm['rmse']
        print(f'\n*** 판정: CNN normal층 RMSE = {rm:.3f} dB ***', flush=True)
        if rm <= 4.0:
            interp = '≈ 논문 정상 3.27 근처 → 격차는 입력/규모·중증 층 쪽'
        elif rm <= 5.5:
            interp = '3.27보다 높으나 5~6대 → 부분 재현, 전체 격차는 moderate/advanced·입력'
        else:
            interp = '5~6 이상 → 쉬운 케이스도 못 맞춤 → 전처리/구현 의심'
        print(f'    해석: {interp}', flush=True)
        results['verdict_normal_cnn'] = {'rmse': rm, 'interpretation': interp}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
