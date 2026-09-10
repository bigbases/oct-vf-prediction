#!/usr/bin/env python3
"""백본 매트릭스 집계 — image-only vs multimodal(concat) vs late-fusion (재학습 X).

각 (backbone, mode)에 대해:
  - OOF CNN RMSE : val_preds_fold{k}.npz 5-fold concat, masked RMSE
  - Late fusion  : XGB val OOF와 정렬 → w grid 튜닝(OOF) → test per-fold 적용
  - Test CNN     : test_preds_fold{k}.npz per-fold RMSE mean±std (누락 시 ckpt에서 dump)
mode:
  image  = image-only CNN (late fusion = w*XGB + (1-w)*CNN)
  mm     = concat multimodal CNN (그 자체가 이미 tabular 사용; fusion은 참고로 XGB와도 계산)

출력: runs/backbone_matrix_table.json, runs/backbone_matrix_table.md
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
from oof_common import align_oof, load_oof_npz  # noqa: E402
from train import VFModel, predict_loader, set_seed  # noqa: E402

XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
CSV = 'ml_final_90d_excl_empty_flip.csv'

# (backbone, mode) -> run_dir
RUN_DIRS = {
    ('inception_v3', 'image'): 'runs/phasec_b0_clip_mse_5fold',
    ('inception_resnet_v2', 'image'): 'runs/phasec_b0_inception_resnet_v2_5fold',
    ('vgg16', 'image'): 'runs/phasec_b0_vgg16_5fold',
    ('xception', 'image'): 'runs/phasec_b0_xception_5fold',
    ('densenet121', 'image'): 'runs/phasec_b0_densenet121_5fold',
    ('inception_v3', 'mm'): 'runs/phasec_mm_inception_v3_5fold',
    ('inception_resnet_v2', 'mm'): 'runs/phasec_mm_inception_resnet_v2_5fold',
    ('vgg16', 'mm'): 'runs/phasec_mm_vgg16_5fold',
    ('xception', 'mm'): 'runs/phasec_mm_xception_5fold',
    ('densenet121', 'mm'): 'runs/phasec_mm_densenet121_5fold',
}
BACKBONES = ['inception_v3', 'inception_resnet_v2', 'vgg16', 'xception', 'densenet121']


def rmse_mae(pred, labels, mask):
    m = mask.astype(bool)
    d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def opt_w(px, pc, labels, mask, grid=101):
    m = mask.astype(bool)
    ya, yb, y = px[m], pc[m], labels[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, grid):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (float(r), float(w))
    return best[1], best[0]


def ensure_test_preds(run_dir: Path, backbone: str, use_tabular: bool, device):
    """test_preds_fold{k}.npz 없으면 best_fold{k}.pt로 dump."""
    missing = [k for k in range(5) if not (run_dir / f'test_preds_fold{k}.npz').is_file()]
    if not missing:
        return
    image_index = build_image_index()
    for k in missing:
        ckpt = run_dir / f'best_fold{k}.pt'
        if not ckpt.is_file():
            print(f'  [warn] {ckpt} 없음 — skip', flush=True)
            continue
        set_seed(42)
        _, _, test_loader = build_dataloaders(
            str(ROOT / CSV), image_index, fold=k, batch_size=64,
            num_workers=4, image_size=(161, 161),
        )
        model = VFModel(
            model_input_size=(161, 322), backbone=backbone, use_tabular=use_tabular,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        b = predict_loader(model, test_loader, device)
        meta = {'split': 'test', 'fold': k, 'backbone': backbone, 'use_tabular': use_tabular}
        np.savez_compressed(
            run_dir / f'test_preds_fold{k}.npz',
            pred=b['pred'].astype(np.float32), labels=b['labels'].astype(np.float32),
            mask=b['mask'].astype(bool), patient_id=b['patient_id'], eye=b['eye'],
            vf_date=b['vf_date'], meta_json=np.array([json.dumps(meta)], dtype=object),
        )
        print(f'  dumped test preds fold{k}: {run_dir.name}', flush=True)


def npz_dict(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        'keys': list(zip(z['patient_id'], z['eye'], z['vf_date'])),
        'pred': z['pred'], 'labels': z['labels'], 'mask': z['mask'], 'meta': {},
    }


def eval_run(backbone: str, mode: str, device) -> dict | None:
    run_dir = ROOT / RUN_DIRS[(backbone, mode)]
    if not (run_dir / 'results.json').is_file():
        return None
    args = json.loads((run_dir / 'results.json').read_text())['args']
    use_tab = bool(args.get('use_tabular', False))

    # test preds 보장
    ensure_test_preds(run_dir, backbone, use_tab, device)

    # ---- OOF (val) ----
    xv, cv, lv, mv = [], [], [], []
    for k in range(5):
        vp = run_dir / f'val_preds_fold{k}.npz'
        if not vp.is_file():
            return {'backbone': backbone, 'mode': mode, 'status': f'missing {vp.name}'}
        xz = load_oof_npz(XGB_VAL[k])
        cz = npz_dict(vp)
        ax, bx, _ = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        xv.append(ax['pred']); cv.append(bx['pred']); lv.append(ax['labels']); mv.append(m)
    XV, CV, LV, MV = (np.concatenate(a) for a in (xv, cv, lv, mv))
    cnn_oof_rmse, cnn_oof_mae = rmse_mae(CV, LV, MV)
    w, fus_oof_rmse = opt_w(XV, CV, LV, MV)
    fus_oof_mae = rmse_mae(w * XV + (1 - w) * CV, LV, MV)[1]

    # ---- TEST per-fold ----
    cnn_test, fus_test = [], []
    for k in range(5):
        tp = run_dir / f'test_preds_fold{k}.npz'
        if not tp.is_file():
            continue
        xz = load_oof_npz(XGB_TEST[k])
        cz = npz_dict(tp)
        ax, bx, _ = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        cnn_test.append(rmse_mae(bx['pred'], ax['labels'], m)[0])
        fus_test.append(rmse_mae(w * ax['pred'] + (1 - w) * bx['pred'], ax['labels'], m)[0])

    return {
        'backbone': backbone, 'mode': mode, 'run_dir': str(run_dir.relative_to(ROOT)),
        'use_tabular': use_tab,
        'cnn_oof_rmse': round(cnn_oof_rmse, 3), 'cnn_oof_mae': round(cnn_oof_mae, 3),
        'w_xgb': round(w, 2),
        'fusion_oof_rmse': round(fus_oof_rmse, 3), 'fusion_oof_mae': round(fus_oof_mae, 3),
        'cnn_test_rmse_mean': round(float(np.mean(cnn_test)), 3) if cnn_test else None,
        'cnn_test_rmse_std': round(float(np.std(cnn_test)), 3) if cnn_test else None,
        'fusion_test_rmse_mean': round(float(np.mean(fus_test)), 3) if fus_test else None,
        'fusion_test_rmse_std': round(float(np.std(fus_test)), 3) if fus_test else None,
        'cnn_test_per_fold': [round(x, 3) for x in cnn_test],
        'fusion_test_per_fold': [round(x, 3) for x in fus_test],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    rows = []
    for bb in BACKBONES:
        for mode in ('image', 'mm'):
            print(f'\n=== {bb} / {mode} ===', flush=True)
            r = eval_run(bb, mode, device)
            if r is None:
                print('  (미완료 — results.json 없음)', flush=True)
                continue
            rows.append(r)
            print(f'  {r.get("status","")} CNN_oof={r.get("cnn_oof_rmse")} '
                  f'fus_oof={r.get("fusion_oof_rmse")}(w={r.get("w_xgb")}) '
                  f'CNN_test={r.get("cnn_test_rmse_mean")} fus_test={r.get("fusion_test_rmse_mean")}',
                  flush=True)

    out = {'protocol': 'Phase C 5-fold, 90d, seed42', 'csv': CSV, 'rows': rows}
    (ROOT / 'runs/backbone_matrix_table.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# 백본 매트릭스: image-only vs multimodal(concat) vs late-fusion (90d, 5-fold)', '',
          '| Backbone | Mode | CNN OOF RMSE | Fusion OOF RMSE (w) | CNN Test RMSE | Fusion Test RMSE |',
          '|----------|------|--------------|---------------------|---------------|------------------|']
    for r in rows:
        if 'cnn_oof_rmse' not in r:
            continue
        ct = f"{r['cnn_test_rmse_mean']}±{r['cnn_test_rmse_std']}" if r['cnn_test_rmse_mean'] else '—'
        ft = f"{r['fusion_test_rmse_mean']}±{r['fusion_test_rmse_std']}" if r['fusion_test_rmse_mean'] else '—'
        md.append(f"| {r['backbone']} | {r['mode']} | {r['cnn_oof_rmse']} | "
                  f"{r['fusion_oof_rmse']} (w={r['w_xgb']}) | {ct} | {ft} |")
    (ROOT / 'runs/backbone_matrix_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n저장: runs/backbone_matrix_table.json / .md', flush=True)


if __name__ == '__main__':
    main()
