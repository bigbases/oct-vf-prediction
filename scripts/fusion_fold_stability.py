#!/usr/bin/env python3
"""fold별 fusion 가중·RMSE 안정성 (val OOF + holdout test, global w=0.60)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from image_preprocessing import build_image_index, build_dataloaders  # noqa: E402
from oof_common import (  # noqa: E402
    align_oof,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
)
from train import VFModel, predict_loader  # noqa: E402

GLOBAL_W = 0.60
CNN_DIR = ROOT / 'runs/phasec_b0_clip_mse_5fold'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
OUT_JSON = ROOT / 'runs/fusion_fold_stability.json'


def fuse(px, pc, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def fold_metrics(px, pc, labels, mask, w_global: float):
    m = mask.astype(bool)
    rx, _ = masked_overall_rmse_mae(px, labels, m)
    rc, _ = masked_overall_rmse_mae(pc, labels, m)
    rg, _ = masked_overall_rmse_mae(fuse(px, pc, w_global), labels, m)
    ropt, w_opt = optimal_fusion_rmse(px, pc, labels, m)
    return {
        'w_opt': w_opt,
        'rmse_xgb': rx,
        'rmse_cnn': rc,
        'rmse_fusion_global': rg,
        'rmse_fusion_opt': ropt,
        'fusion_beats_xgb_global': rg < rx,
    }


def cnn_test_preds(fold: int, device: torch.device, csv_path: str, image_index) -> dict:
    args = SimpleNamespace(
        use_tabular=False,
        use_deviation=False,
        n_output=52,
        batch_size=64,
        num_workers=0,
        image_h=161,
        image_w=161,
        model_h=161,
        model_w=322,
    )
    _, _, test_loader = build_dataloaders(
        csv_path, image_index, fold=fold,
        batch_size=args.batch_size, num_workers=0,
        image_size=(args.image_h, args.image_w),
    )
    model = VFModel(
        n_output=args.n_output,
        use_tabular=False,
        use_deviation=False,
        model_input_size=(args.model_h, args.model_w),
    ).to(device)
    ckpt = CNN_DIR / f'best_fold{fold}.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    bundle = predict_loader(model, test_loader, device)
    keys = list(zip(bundle['patient_id'], bundle['eye'], bundle['vf_date']))
    return {
        'keys': keys,
        'pred': bundle['pred'],
        'labels': bundle['labels'],
        'mask': bundle['mask'],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = str(ROOT / 'ml_final_90d_excl_empty_flip.csv')
    image_index = build_image_index()

    print(f'=== fold별 fusion 안정성 (global w_xgb={GLOBAL_W}) ===\n', flush=True)

    val_rows = []
    test_rows = []

    for k in range(5):
        ax = load_oof_npz(XGB_VAL[k])
        bx = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, _ = align_oof(ax, bx)
        mask = ax['mask'] & bx['mask']
        vm = fold_metrics(ax['pred'], bx['pred'], ax['labels'], mask, GLOBAL_W)
        vm['fold'] = k
        vm['split'] = 'val'
        val_rows.append(vm)

        tx = load_oof_npz(XGB_TEST[k])
        cx = cnn_test_preds(k, device, csv_path, image_index)
        cx = {
            'keys': cx['keys'],
            'pred': cx['pred'],
            'labels': cx['labels'],
            'mask': cx['mask'],
            'meta': {},
        }
        tx, cx, _ = align_oof(tx, cx)
        tmask = tx['mask'] & cx['mask']
        tm = fold_metrics(tx['pred'], cx['pred'], tx['labels'], tmask, GLOBAL_W)
        tm['fold'] = k
        tm['split'] = 'test'
        test_rows.append(tm)

    def summarize(rows, split: str):
        print(f'--- {split} ---', flush=True)
        print(f'{"fold":>4s} {"w_opt":>6s} {"XGB":>8s} {"CNN":>8s} {"Fus0.60":>8s} {"Fus_opt":>8s} {"F<XGB":>5s}', flush=True)
        for r in rows:
            print(
                f'{r["fold"]:4d} {r["w_opt"]:6.3f} {r["rmse_xgb"]:8.3f} {r["rmse_cnn"]:8.3f} '
                f'{r["rmse_fusion_global"]:8.3f} {r["rmse_fusion_opt"]:8.3f} '
                f'{"Y" if r["fusion_beats_xgb_global"] else "N":>5s}',
                flush=True,
            )
        for key in ('w_opt', 'rmse_xgb', 'rmse_cnn', 'rmse_fusion_global', 'rmse_fusion_opt'):
            v = [r[key] for r in rows]
            print(f'  {key:20s} mean={np.mean(v):.3f}  std={np.std(v):.3f}  range=[{min(v):.3f},{max(v):.3f}]', flush=True)
        n_win = sum(1 for r in rows if r['fusion_beats_xgb_global'])
        print(f'  fusion(0.60) < XGB: {n_win}/5 folds\n', flush=True)

    summarize(val_rows, 'val (fold별 val OOF)')
    summarize(test_rows, 'test (동일 holdout, fold별 best CNN ckpt)')

    OUT_JSON.write_text(
        json.dumps({'global_w_xgb': GLOBAL_W, 'val': val_rows, 'test': test_rows}, indent=2),
        encoding='utf-8',
    )
    print(f'저장: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
