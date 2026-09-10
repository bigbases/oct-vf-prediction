#!/usr/bin/env python3
"""
P1 go/no-go: XGB vs CNN val-fold 잔차 상관 ρ (fold 0 probe 등).

동일 (patient_id, eye, vf_date) 교집합에서만 metric·ρ 계산.
ρ < RMSE_XGB/RMSE_CNN 이면 단순 2-model 가정에서 CNN 비중 > 0 여지.

사용법:
  python scripts/probe_fusion_residual_corr.py \\
      --xgb runs/oof/xgb_90d_fold0_val.npz \\
      --cnn runs/fusion_probe_cnn_fold0/val_preds_fold0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import (  # noqa: E402
    align_oof,
    fusion_ceiling_note,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
    residual_pearson_rho,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xgb', required=True, help='export_xgb_oof.py 출력 npz')
    ap.add_argument('--cnn', required=True, help='train.py --dump_val_preds 출력 npz')
    ap.add_argument('--tag', default='', help='출력 라벨 (예: MAE-stop / MSE-stop)')
    args = ap.parse_args()

    xa = load_oof_npz(ROOT / args.xgb if not Path(args.xgb).is_absolute() else Path(args.xgb))
    xb = load_oof_npz(ROOT / args.cnn if not Path(args.cnn).is_absolute() else Path(args.cnn))
    ax, bx, keys = align_oof(xa, xb)

    # labels는 동일해야 함 (교집합)
    if not np.allclose(ax['labels'], bx['labels'], equal_nan=True):
        diff = np.abs(ax['labels'] - bx['labels'])
        m = ax['mask'] & bx['mask']
        if (diff[m] > 1e-3).any():
            print('[경고] 교집합에서 labels 불일치 — CSV/마스크 확인', flush=True)

    mask = ax['mask'] & bx['mask']
    labels = ax['labels']

    rmse_x, mae_x = masked_overall_rmse_mae(ax['pred'], labels, mask)
    rmse_c, mae_c = masked_overall_rmse_mae(bx['pred'], labels, mask)
    rho, n_pts = residual_pearson_rho(ax['pred'], bx['pred'], labels, mask)
    thresh = rmse_x / rmse_c if rmse_c > 0 else float('nan')

    fus_rmse, fus_w = optimal_fusion_rmse(ax['pred'], bx['pred'], labels, mask)
    tag = f' [{args.tag}]' if args.tag else ''
    print(f'\n=== Fusion probe (aligned rows only){tag} ===', flush=True)
    print(f'  XGB npz val n={len(xa["keys"])}  CNN npz val n={len(xb["keys"])}  aligned n={len(keys)}', flush=True)
    print(f'  masked VF points (pooled): {n_pts}', flush=True)
    print(f'  XGB-alone  RMSE={rmse_x:.3f}  MAE={mae_x:.3f}', flush=True)
    print(f'  CNN-alone  RMSE={rmse_c:.3f}  MAE={mae_c:.3f}', flush=True)
    print(f'  residual ρ(e_xgb, e_cnn) = {rho:.4f}', flush=True)
    print(f'  ρ threshold (σ_XGB/σ_CNN)   = {thresh:.4f}', flush=True)
    print(f'  optimal fusion RMSE (val OOF, w·XGB+(1-w)·CNN) = {fus_rmse:.3f}  (w_xgb={fus_w:.2f})', flush=True)
    print(f'  → {fusion_ceiling_note(rmse_x, rmse_c, rho)}', flush=True)

    if np.isfinite(rho) and np.isfinite(thresh):
        if rho < 0.5 * thresh:
            print('  [판정] ρ가 threshold보다 훨씬 낮음 → 5-fold fusion GO 강하게 권장', flush=True)
        elif rho < thresh:
            print('  [판정] ρ < threshold → 5-fold fusion GO (이득 크기는 ρ에 비례)', flush=True)
        else:
            print('  [판정] ρ ≥ threshold → 5-fold fusion STOP, P2(구조 개선) 권장', flush=True)


if __name__ == '__main__':
    main()
