#!/usr/bin/env python3
"""
수정 CNN (phasec_b0_clip_mse_5fold) + XGB val OOF late fusion 가중 스윕 + MD 층별 RMSE.

사용법:
  python scripts/fit_clip_mse_global_fusion.py
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
from oof_common import (  # noqa: E402
    align_oof,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
    residual_pearson_rho,
)

COHORT_MD = ROOT / 'cohort_md.csv'
CNN_VAL = {k: ROOT / f'runs/phasec_b0_clip_mse_5fold/val_preds_fold{k}.npz' for k in range(5)}
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
OUT_JSON = ROOT / 'runs/clip_mse_global_fusion.json'
OLD_W = 0.84  # p1_global_fusion.json (buggy CNN)

STRATA = [
    ('normal', lambda md: md > -3),
    ('early', lambda md: -6 < md <= -3),
    ('moderate', lambda md: -12 < md <= -6),
    ('advanced', lambda md: md <= -12),
]


def norm_key(pid, eye, vf_date) -> Tuple[str, str, str]:
    return (
        str(pid).strip(),
        str(eye).strip().upper(),
        str(vf_date).strip().replace('-', '').replace('/', ''),
    )


def load_cohort_md(path: Path) -> Dict[Tuple[str, str, str], float]:
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        k = norm_key(r['patient_id'], r['eye'], r['vf_date'])
        try:
            out[k] = float(r['MD'])
        except (ValueError, KeyError, TypeError):
            out[k] = float('nan')
    return out


def stack_oof(paths: dict[int, Path], label: str) -> dict:
    keys_all, preds, labels, masks = [], [], [], []
    for k in sorted(paths):
        d = load_oof_npz(paths[k])
        keys_all.extend(d['keys'])
        preds.append(d['pred'])
        labels.append(d['labels'])
        masks.append(d['mask'])
    return {
        'keys': keys_all,
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'meta': {'stacked': label},
    }


def fuse(px, pc, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def metrics_by_stratum(
    pred_x, pred_c, pred_f, labels, mask, keys, md_map
) -> Dict[str, dict]:
    by: Dict[str, List[int]] = {s[0]: [] for s in STRATA}
    for i, k in enumerate(keys):
        nk = norm_key(*k)
        md = md_map.get(nk, float('nan'))
        if not np.isfinite(md):
            continue
        for name, fn in STRATA:
            if fn(md):
                by[name].append(i)
                break

    out = {}
    for name, idx in by.items():
        if not idx:
            continue
        ii = np.array(idx)
        m = mask[ii]
        y = labels[ii]
        mx = {'rmse': masked_overall_rmse_mae(pred_x[ii], y, m)[0],
              'mae': masked_overall_rmse_mae(pred_x[ii], y, m)[1],
              'n_eyes': len(idx)}
        mc = {'rmse': masked_overall_rmse_mae(pred_c[ii], y, m)[0],
              'mae': masked_overall_rmse_mae(pred_c[ii], y, m)[1],
              'n_eyes': len(idx)}
        mf = {'rmse': masked_overall_rmse_mae(pred_f[ii], y, m)[0],
              'mae': masked_overall_rmse_mae(pred_f[ii], y, m)[1],
              'n_eyes': len(idx)}
        out[name] = {'n_eyes': len(idx), 'xgb': mx, 'cnn': mc, 'fusion': mf}
    return out


def main():
    print('=== clip_mse CNN + XGB global fusion (5-fold val OOF) ===', flush=True)
    xgb = stack_oof(XGB_VAL, 'XGB')
    cnn = stack_oof(CNN_VAL, 'CNN')
    ax, bx, common = align_oof(xgb, cnn)
    mask = ax['mask'] & bx['mask']
    labels = ax['labels']
    px, pc = ax['pred'], bx['pred']

    rmse_x, mae_x = masked_overall_rmse_mae(px, labels, mask)
    rmse_c, mae_c = masked_overall_rmse_mae(pc, labels, mask)
    rho, n_pts = residual_pearson_rho(px, pc, labels, mask)
    fus_rmse, w_opt = optimal_fusion_rmse(px, pc, labels, mask)
    _, mae_f = masked_overall_rmse_mae(fuse(px, pc, w_opt), labels, mask)

    rmse_old, _ = masked_overall_rmse_mae(fuse(px, pc, OLD_W), labels, mask)

    print(f'  OOF aligned rows: {len(common)}  VF points: {n_pts}', flush=True)
    print(f'  XGB-alone   RMSE={rmse_x:.3f}  MAE={mae_x:.3f}', flush=True)
    print(f'  CNN-alone   RMSE={rmse_c:.3f}  MAE={mae_c:.3f}  (이전 buggy CNN OOF ~12 normal)', flush=True)
    print(f'  ρ={rho:.4f}  σ_XGB/σ_CNN={rmse_x/rmse_c:.4f}', flush=True)
    print(f'  **최적 w_xgb={w_opt:.3f}**  (w_cnn={1-w_opt:.3f})  fusion RMSE={fus_rmse:.3f}  MAE={mae_f:.3f}', flush=True)
    print(f'  이전 P1 w_xgb={OLD_W:.2f} (CNN 0.16) → fusion RMSE={rmse_old:.3f}', flush=True)
    print(f'  vs XGB: fusion {"↓" if fus_rmse < rmse_x else "↑"} {abs(fus_rmse-rmse_x):.3f}', flush=True)
    print(f'  vs CNN: fusion {"↓" if fus_rmse < rmse_c else "↑"} {abs(fus_rmse-rmse_c):.3f}', flush=True)

    md_map = load_cohort_md(COHORT_MD)
    pf_opt = fuse(px, pc, w_opt)
    pf_old = fuse(px, pc, OLD_W)

    print('\n=== MD 층별 RMSE (OOF+MD) ===', flush=True)
    print(f'{"stratum":10s} {"n":>4s}  {"XGB":>8s} {"CNN":>8s} {"Fus*":>8s} {"Fus0.84":>8s}', flush=True)
    print('-' * 52, flush=True)
    strata_opt = metrics_by_stratum(px, pc, pf_opt, labels, mask, common, md_map)
    strata_old = metrics_by_stratum(px, pc, pf_old, labels, mask, common, md_map)
    for name, _ in STRATA:
        if name not in strata_opt:
            continue
        so = strata_opt[name]
        ro = strata_old[name]['fusion']['rmse']
        print(
            f'{name:10s} {so["n_eyes"]:4d}  '
            f'{so["xgb"]["rmse"]:8.3f} {so["cnn"]["rmse"]:8.3f} '
            f'{so["fusion"]["rmse"]:8.3f} {ro:8.3f}',
            flush=True,
        )

    # sweep curve sample
    print('\n=== w_xgb 스윕 (일부) ===', flush=True)
    for w in [0.0, 0.16, 0.5, 0.84, w_opt, 1.0]:
        r, _ = masked_overall_rmse_mae(fuse(px, pc, w), labels, mask)
        mark = ' <-- opt' if abs(w - w_opt) < 0.01 else (' (old P1)' if abs(w - OLD_W) < 0.01 else '')
        print(f'  w_xgb={w:.2f}  w_cnn={1-w:.2f}  RMSE={r:.3f}{mark}', flush=True)

    result = {
        'cnn_source': 'runs/phasec_b0_clip_mse_5fold',
        'xgb_source': 'runs/oof/xgb_90d_fold{k}_val.npz',
        'n_oof_rows_aligned': len(common),
        'n_vf_points': int(n_pts),
        'w_xgb_optimal': float(w_opt),
        'w_cnn_optimal': float(1.0 - w_opt),
        'w_xgb_previous_p1': OLD_W,
        'oof': {
            'xgb_rmse': rmse_x, 'xgb_mae': mae_x,
            'cnn_rmse': rmse_c, 'cnn_mae': mae_c,
            'fusion_rmse_opt': fus_rmse, 'fusion_mae_opt': mae_f,
            'fusion_rmse_w084': rmse_old,
            'rho': rho,
        },
        'strata_optimal_w': strata_opt,
        'strata_fusion_w084': {k: v['fusion'] for k, v in strata_old.items()},
        'comparison': {
            'old_p1_cnn_normal_rmse_buggy': 12.06,
            'new_cnn_normal_rmse': strata_opt.get('normal', {}).get('cnn', {}).get('rmse'),
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
