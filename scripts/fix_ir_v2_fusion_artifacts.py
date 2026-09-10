#!/usr/bin/env python3
"""IR-v2 late fusion 가중·OOF fusion 예측 고정 저장 + 층별 RMSE JSON."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import (  # noqa: E402
    align_oof,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
    residual_pearson_rho,
    save_oof_npz,
)

W_XGB_FIXED = 0.47
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
FUSION_OUT = ROOT / 'runs/ir_v2_fusion_oof'
FUSION_JSON = ROOT / 'runs/ir_v2_global_fusion.json'
STRATA_JSON = ROOT / 'runs/stratified_rmse_ir_v2.json'
COHORT_MD = ROOT / 'cohort_md.csv'

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


def fuse(px, pc, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def rmse_idx(pred, labels, mask, idx: List[int]) -> Tuple[float, float]:
    ii = np.array(idx)
    return masked_overall_rmse_mae(pred[ii], labels[ii], mask[ii])


def main():
    md_map = load_cohort_md(COHORT_MD)
    FUSION_OUT.mkdir(parents=True, exist_ok=True)

    # --- per-fold fusion npz + stack ---
    stack_keys, stack_pred, stack_labels, stack_mask = [], [], [], []
    fold_metrics = []

    for k in range(5):
        ax = load_oof_npz(XGB_VAL[k])
        bx = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(ax, bx)
        m = ax['mask'] & bx['mask']
        labels = ax['labels']
        pf = fuse(ax['pred'], bx['pred'], W_XGB_FIXED)
        meta = {
            'model': 'late_fusion',
            'w_xgb': W_XGB_FIXED,
            'w_cnn': 1.0 - W_XGB_FIXED,
            'cnn_backbone': 'inception_resnet_v2',
            'cnn_dir': str(CNN_DIR),
            'xgb_npz': str(XGB_VAL[k]),
            'cv_fold': k,
            'split': 'val',
        }
        out_path = FUSION_OUT / f'fusion_preds_fold{k}.npz'
        save_oof_npz(out_path, common, pf, labels, m, meta)
        fr, fm = masked_overall_rmse_mae(pf, labels, m)
        fold_metrics.append({'fold': k, 'n': len(common), 'fusion_rmse': fr, 'fusion_mae': fm})

        stack_keys.extend(common)
        stack_pred.append(pf)
        stack_labels.append(labels)
        stack_mask.append(m)

    stack_pred = np.concatenate(stack_pred, axis=0)
    stack_labels = np.concatenate(stack_labels, axis=0)
    stack_mask = np.concatenate(stack_mask, axis=0)
    save_oof_npz(
        FUSION_OUT / 'fusion_preds_oof_stack.npz',
        stack_keys,
        stack_pred,
        stack_labels,
        stack_mask,
        {'model': 'late_fusion', 'w_xgb': W_XGB_FIXED, 'stacked_folds': 5},
    )

    # --- global OOF metrics (recompute w for audit) ---
    keys_all, px_list, pc_list, lab_list, m_list = [], [], [], [], []
    for k in range(5):
        ax = load_oof_npz(XGB_VAL[k])
        bx = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(ax, bx)
        m = ax['mask'] & bx['mask']
        keys_all.extend(common)
        px_list.append(ax['pred'])
        pc_list.append(bx['pred'])
        lab_list.append(ax['labels'])
        m_list.append(m)
    px = np.concatenate(px_list, axis=0)
    pc = np.concatenate(pc_list, axis=0)
    labels = np.concatenate(lab_list, axis=0)
    mask = np.concatenate(m_list, axis=0)
    common = keys_all

    rx, mx = masked_overall_rmse_mae(px, labels, mask)
    rc, mc = masked_overall_rmse_mae(pc, labels, mask)
    rf_fix, mf_fix = masked_overall_rmse_mae(fuse(px, pc, W_XGB_FIXED), labels, mask)
    rf_opt, w_opt = optimal_fusion_rmse(px, pc, labels, mask)
    _, mf_opt = masked_overall_rmse_mae(fuse(px, pc, w_opt), labels, mask)
    rho, n_pts = residual_pearson_rho(px, pc, labels, mask)

    # strata at fixed w
    by: Dict[str, List[int]] = {s[0]: [] for s in STRATA}
    for i, key in enumerate(common):
        md = md_map.get(norm_key(*key), float('nan'))
        if not np.isfinite(md):
            continue
        for name, fn in STRATA:
            if fn(md):
                by[name].append(i)
                break

    pf = fuse(px, pc, W_XGB_FIXED)
    strata = {}
    for name, idx in by.items():
        if not idx:
            continue
        xr, _ = rmse_idx(px, labels, mask, idx)
        cr, _ = rmse_idx(pc, labels, mask, idx)
        fr, fm = rmse_idx(pf, labels, mask, idx)
        strata[name] = {
            'n_eyes': len(idx),
            'xgb_rmse': xr,
            'cnn_rmse': cr,
            'fusion_rmse': fr,
            'fusion_mae': fm,
        }

    fusion_doc = {
        'status': 'frozen_for_vae_phase',
        'cnn_backbone': 'inception_resnet_v2',
        'cnn_run_dir': str(CNN_DIR.relative_to(ROOT)),
        'xgb_oof': 'runs/oof/xgb_90d_fold{k}_val.npz',
        'w_xgb_global': W_XGB_FIXED,
        'w_cnn_global': round(1.0 - W_XGB_FIXED, 2),
        'w_xgb_optimal_recomputed': w_opt,
        'note': 'w=0.47 고정 (OOF RMSE 최적 0.47). VAE·멀티모달 전 기준.',
        'n_oof_rows': len(common),
        'n_vf_points': int(mask.sum()),
        'oof': {
            'xgb_rmse': rx, 'xgb_mae': mx,
            'cnn_rmse': rc, 'cnn_mae': mc,
            'fusion_rmse_fixed_w': rf_fix, 'fusion_mae_fixed_w': mf_fix,
            'fusion_rmse_optimal_w': rf_opt, 'fusion_mae_optimal_w': mf_opt,
            'rho': rho,
        },
        'per_fold_fusion_rmse': fold_metrics,
        'fusion_oof_dir': str(FUSION_OUT.relative_to(ROOT)),
        'artifacts': {
            'per_fold': [f'fusion_preds_fold{k}.npz' for k in range(5)],
            'stack': 'fusion_preds_oof_stack.npz',
        },
    }
    FUSION_JSON.write_text(json.dumps(fusion_doc, ensure_ascii=False, indent=2), encoding='utf-8')

    strata_doc = {
        'backbone': 'inception_resnet_v2',
        'w_fusion': W_XGB_FIXED,
        'n_with_md': sum(len(v) for v in by.values()),
        'strata': strata,
        'overall_oof': {
            'cnn_rmse': rc,
            'xgb_rmse': rx,
            'fusion_rmse': rf_fix,
        },
        'improvement_timeline': {
            'buggy_cnn_normal_rmse': 12.060776710510254,
            'buggy_source': 'runs/stratified_rmse_oof.json (output ReLU head, fusion_probe CNN)',
            'fixed_inception_v3_normal_cnn': 4.923570156097412,
            'fixed_inception_v3_source': 'runs/phasec_b0_clip_mse_5fold + stratified_rmse_clip_mse_5fold.json',
            'ir_v2_normal_cnn': strata.get('normal', {}).get('cnn_rmse'),
            'ir_v2_advanced_cnn': strata.get('advanced', {}).get('cnn_rmse'),
        },
    }
    STRATA_JSON.write_text(json.dumps(strata_doc, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'저장: {FUSION_JSON}', flush=True)
    print(f'저장: {STRATA_JSON}', flush=True)
    print(f'OOF fusion npz: {FUSION_OUT}/fusion_preds_fold{{0..4}}.npz + stack', flush=True)
    print(f'고정 w={W_XGB_FIXED} fusion OOF RMSE={rf_fix:.3f} (optimal w={w_opt:.3f} → {rf_opt:.3f})', flush=True)


if __name__ == '__main__':
    main()
