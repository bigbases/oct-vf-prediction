#!/usr/bin/env python3
"""IR-v2 late fusion 5-fold holdout test (w 고정, 재학습 없음).

fold k: XGB test npz + CNN test preds(ckpt dump) → 키 정렬 → w·XGB + (1-w)·CNN.

출력: runs/fusion_5fold_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from image_preprocessing import build_image_index, build_dataloaders  # noqa: E402
from oof_common import align_oof, load_oof_npz, masked_overall_rmse_mae  # noqa: E402
from train import VFModel, predict_loader, set_seed  # noqa: E402

W_XGB = 0.47
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
OUT_JSON = ROOT / 'runs/fusion_5fold_test.json'
CNN_TEST_NPZ_DIR = CNN_DIR  # test_preds_fold{k}.npz alongside ckpt


def fuse(px: np.ndarray, pc: np.ndarray, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def dump_cnn_test_npz(fold: int, device: torch.device) -> Path:
    ckpt = CNN_DIR / f'best_fold{fold}.pt'
    out_path = CNN_TEST_NPZ_DIR / f'test_preds_fold{fold}.npz'
    if out_path.is_file():
        return out_path

    set_seed(42)
    csv_path = str(ROOT / 'ml_final_90d_excl_empty_flip.csv')
    image_index = build_image_index()
    _, _, test_loader = build_dataloaders(
        csv_path, image_index, fold=fold, batch_size=64, num_workers=4,
        image_size=(161, 161),
    )
    model = VFModel(
        model_input_size=(161, 322),
        backbone='inception_resnet_v2',
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    bundle = predict_loader(model, test_loader, device)

    meta = {
        'split': 'test',
        'fold': fold,
        'ckpt': str(ckpt),
        'backbone': 'inception_resnet_v2',
    }
    np.savez_compressed(
        out_path,
        pred=bundle['pred'].astype(np.float32),
        labels=bundle['labels'].astype(np.float32),
        mask=bundle['mask'].astype(bool),
        patient_id=bundle['patient_id'],
        eye=bundle['eye'],
        vf_date=bundle['vf_date'],
        meta_json=np.array([json.dumps(meta)], dtype=object),
    )
    print(f'  dumped CNN test: {out_path} n={len(bundle["pred"])}', flush=True)
    return out_path


def npz_to_oof_dict(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    keys = list(zip(z['patient_id'], z['eye'], z['vf_date']))
    return {
        'keys': keys,
        'pred': z['pred'],
        'labels': z['labels'],
        'mask': z['mask'],
        'meta': json.loads(str(z['meta_json'][0])) if 'meta_json' in z else {},
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== IR-v2 late fusion 5-fold test (w_xgb={W_XGB}) device={device} ===', flush=True)

    per_fold = []
    fold_rmses, fold_maes = [], []
    solo_xgb_rmses, solo_cnn_rmses = [], []

    for k in range(5):
        xgb_path = XGB_TEST[k]
        if not xgb_path.is_file():
            raise FileNotFoundError(xgb_path)

        cnn_path = dump_cnn_test_npz(k, device)
        xgb = load_oof_npz(xgb_path)
        cnn = npz_to_oof_dict(cnn_path)

        print(f'\nfold {k}: XGB n={len(xgb["keys"])} CNN n={len(cnn["keys"])}', flush=True)
        ax, bx, common = align_oof(xgb, cnn)
        m = ax['mask'] & bx['mask']
        labels = ax['labels']
        n_eyes = len(common)

        rx, mx = masked_overall_rmse_mae(ax['pred'], labels, m)
        rc, mc = masked_overall_rmse_mae(bx['pred'], labels, m)
        pf = fuse(ax['pred'], bx['pred'], W_XGB)
        rf, mf = masked_overall_rmse_mae(pf, labels, m)

        per_fold.append({
            'fold': k,
            'n_eyes_aligned': n_eyes,
            'n_xgb_raw': len(xgb['keys']),
            'n_cnn_raw': len(cnn['keys']),
            'xgb': {'rmse': rx, 'mae': mx},
            'cnn': {'rmse': rc, 'mae': mc},
            'fusion_w047': {'rmse': rf, 'mae': mf, 'w_xgb': W_XGB, 'w_cnn': 1.0 - W_XGB},
        })
        fold_rmses.append(rf)
        fold_maes.append(mf)
        solo_xgb_rmses.append(rx)
        solo_cnn_rmses.append(rc)
        print(
            f'  aligned n={n_eyes}  XGB={rx:.3f}  CNN={rc:.3f}  fusion={rf:.3f}',
            flush=True,
        )

    result = {
        'protocol': '5-fold holdout test, per-fold train excludes val fold k; w=0.47 fixed (OOF global)',
        'cnn_backbone': 'inception_resnet_v2',
        'cnn_run_dir': str(CNN_DIR.relative_to(ROOT)),
        'w_xgb': W_XGB,
        'reference_oof_fusion_rmse': 8.064,
        'reference_fold0_test_fusion_rmse': 8.439,
        'per_fold': per_fold,
        'summary': {
            'fusion_rmse_mean': float(np.mean(fold_rmses)),
            'fusion_rmse_std': float(np.std(fold_rmses)),
            'fusion_mae_mean': float(np.mean(fold_maes)),
            'fusion_mae_std': float(np.std(fold_maes)),
            'xgb_rmse_mean': float(np.mean(solo_xgb_rmses)),
            'xgb_rmse_std': float(np.std(solo_xgb_rmses)),
            'cnn_rmse_mean': float(np.mean(solo_cnn_rmses)),
            'cnn_rmse_std': float(np.std(solo_cnn_rmses)),
        },
        'note': (
            'Each fold uses model trained without fold k val; test set is same 37 holdout eyes. '
            'Mean±std is across 5 fold-specific models on the same test set (not a single model).'
        ),
    }

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    s = result['summary']
    print('\n=== Summary (5-fold test, aligned) ===', flush=True)
    print(f'  fusion RMSE: {s["fusion_rmse_mean"]:.3f} ± {s["fusion_rmse_std"]:.3f}', flush=True)
    print(f'  fusion MAE:  {s["fusion_mae_mean"]:.3f} ± {s["fusion_mae_std"]:.3f}', flush=True)
    print(f'  XGB   RMSE: {s["xgb_rmse_mean"]:.3f} ± {s["xgb_rmse_std"]:.3f}', flush=True)
    print(f'  CNN   RMSE: {s["cnn_rmse_mean"]:.3f} ± {s["cnn_rmse_std"]:.3f}', flush=True)
    print(f'저장: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
