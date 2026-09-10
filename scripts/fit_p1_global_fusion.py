#!/usr/bin/env python3
"""
P1 마무리: 5-fold val OOF 전역 융합 가중 1회 + holdout test 평가.

- val OOF: fold 0–4 npz를 (patient_id, eye, vf_date)로 concat (~240행)
- 가중: w·XGB + (1-w)·CNN, w는 **전역 OOF에서 RMSE 1회** 최적화 (fold별 재튜닝 금지)
- test: fold별 test_preds 평균 후 동일 w 적용, 교집합 행에서만 metric

사용법:
  python scripts/fit_p1_global_fusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import (  # noqa: E402
    align_oof,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
    residual_pearson_rho,
    row_key,
)

# fold 0 CNN은 probe 디렉터리, 1–4는 p1_fusion_cnn_90d
CNN_VAL = {
    0: ROOT / 'runs/fusion_probe_cnn_fold0/val_preds_fold0.npz',
    1: ROOT / 'runs/p1_fusion_cnn_90d/val_preds_fold1.npz',
    2: ROOT / 'runs/p1_fusion_cnn_90d/val_preds_fold2.npz',
    3: ROOT / 'runs/p1_fusion_cnn_90d/val_preds_fold3.npz',
    4: ROOT / 'runs/p1_fusion_cnn_90d/val_preds_fold4.npz',
}
CNN_TEST = {k: p.parent / f'test_preds_fold{k}.npz' for k, p in CNN_VAL.items()}
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}


def stack_oof(paths: dict[int, Path], label: str) -> dict:
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
        raise FileNotFoundError(f'{label} 누락: {missing}')
    return {
        'keys': keys_all,
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'meta': {'stacked': label},
    }


def fuse_pred(px, pc, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def main():
    print('=== P1 global fusion (5-fold val OOF stack) ===', flush=True)
    xgb = stack_oof(XGB_VAL, 'XGB val')
    cnn = stack_oof(CNN_VAL, 'CNN val')
    ax, bx, common_keys = align_oof(xgb, cnn)
    mask = ax['mask'] & bx['mask']
    labels = ax['labels']

    rmse_x, mae_x = masked_overall_rmse_mae(ax['pred'], labels, mask)
    rmse_c, mae_c = masked_overall_rmse_mae(bx['pred'], labels, mask)
    rho, n_pts = residual_pearson_rho(ax['pred'], bx['pred'], labels, mask)
    fus_rmse, w_xgb = optimal_fusion_rmse(ax['pred'], bx['pred'], labels, mask)
    _, mae_f = masked_overall_rmse_mae(
        fuse_pred(ax['pred'], bx['pred'], w_xgb), labels, mask
    )

    print(f'  OOF rows (aligned): {len(common_keys)}  VF points: {n_pts}', flush=True)
    print(f'  XGB-alone  RMSE={rmse_x:.3f}  MAE={mae_x:.3f}', flush=True)
    print(f'  CNN-alone  RMSE={rmse_c:.3f}  MAE={mae_c:.3f}', flush=True)
    print(f'  ρ={rho:.4f}  threshold σ_XGB/σ_CNN={rmse_x/rmse_c:.4f}', flush=True)
    print(f'  **global w_xgb={w_xgb:.3f}**  fusion OOF RMSE={fus_rmse:.3f}  MAE={mae_f:.3f}', flush=True)

    def stack_test(paths: dict[int, Path], label: str) -> dict:
        preds = []
        keys_ref = None
        base = None
        for k in sorted(paths):
            p = paths[k]
            if not p.is_file():
                raise FileNotFoundError(f'{label} test npz 없음: {p}')
            d = load_oof_npz(p)
            if keys_ref is None:
                keys_ref = d['keys']
                base = d
            elif d['keys'] != keys_ref:
                raise ValueError(f'{label} fold {k} test keys 불일치')
            preds.append(d['pred'])
        return {
            'keys': keys_ref,
            'pred': np.mean(np.stack(preds, axis=0), axis=0),
            'labels': base['labels'],
            'mask': base['mask'],
            'meta': base.get('meta', {}),
        }

    cnn_te = stack_test(CNN_TEST, 'CNN')
    xgb_te = stack_test(XGB_TEST, 'XGB')
    tx, bx, _ = align_oof(xgb_te, cnn_te)
    tmask = tx['mask'] & bx['mask']
    tlabels = tx['labels']
    trmse_x, tmae_x = masked_overall_rmse_mae(tx['pred'], tlabels, tmask)
    trmse_c, tmae_c = masked_overall_rmse_mae(bx['pred'], tlabels, tmask)
    tfus = fuse_pred(tx['pred'], bx['pred'], w_xgb)
    trmse_f, tmae_f = masked_overall_rmse_mae(tfus, tlabels, tmask)

    print('\n=== Holdout test (aligned, CNN=test fold-mean, XGB=test fold-mean) ===', flush=True)
    print(f'  test rows aligned: {len(tx["keys"])}', flush=True)
    print(f'  XGB-alone  RMSE={trmse_x:.3f}  MAE={tmae_x:.3f}', flush=True)
    print(f'  CNN-alone  RMSE={trmse_c:.3f}  MAE={tmae_c:.3f}', flush=True)
    print(f'  fusion (w={w_xgb:.3f}) RMSE={trmse_f:.3f}  MAE={tmae_f:.3f}', flush=True)

    out = ROOT / 'runs/p1_global_fusion.json'
    result = {
        'n_oof_rows_aligned': len(common_keys),
        'n_vf_points': int(n_pts),
        'w_xgb_global': float(w_xgb),
        'oof': {
            'xgb_rmse': rmse_x, 'xgb_mae': mae_x,
            'cnn_rmse': rmse_c, 'cnn_mae': mae_c,
            'fusion_rmse': fus_rmse, 'fusion_mae': mae_f,
            'rho': rho,
        },
        'test': {
            'n_rows_aligned': len(tx['keys']),
            'xgb_rmse': trmse_x, 'xgb_mae': tmae_x,
            'cnn_rmse': trmse_c, 'cnn_mae': tmae_c,
            'fusion_rmse': trmse_f, 'fusion_mae': tmae_f,
        },
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {out}', flush=True)


if __name__ == '__main__':
    main()
