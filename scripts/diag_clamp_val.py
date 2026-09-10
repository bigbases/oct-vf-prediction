#!/usr/bin/env python3
"""clamp + val plateau 진단: train/eval 동일성, 예측값, epoch별 val 변화."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from image_preprocessing import PT, build_image_index, build_dataloaders  # noqa: E402
from train import VFModel, set_seed, masked_mse, masked_mae, train_one_epoch  # noqa: E402


@torch.no_grad()
def val_predictions(model, val_loader, device):
    model.eval()
    batch = next(iter(val_loader))
    images = batch['images'].to(device)
    labels = batch['labels'].to(device)
    mask = batch['mask'].to(device)
    pred = model(images)
    return pred, labels, mask, batch


def pred_fingerprint(pred: torch.Tensor, mask: torch.Tensor) -> dict:
    p = pred[0].cpu().numpy()
    m = mask[0].cpu().numpy()
    pv = p[m]
    return {
        'min': float(pv.min()),
        'max': float(pv.max()),
        'mean': float(pv.mean()),
        'std': float(pv.std()),
        'n_at_-5': int((np.abs(pv + 5) < 1e-4).sum()),
        'n_at_45': int((np.abs(pv - 45) < 1e-4).sum()),
        'hash': float(np.sum(pv * 1e3).round() % 1e12),
    }


def check_mode_independence(model, val_loader, device):
    print('=== 1) clamp: train vs eval 모드 (동일 forward) ===', flush=True)
    batch = next(iter(val_loader))
    images = batch['images'].to(device)
    model.train()
    p_train = model(images).detach()
    model.eval()
    p_eval = model(images)
    diff = (p_train - p_eval).abs().max().item()
    print(f'  max|pred_train - pred_eval| = {diff:.6e}  (0이면 모드 무관 동일)', flush=True)
    print(f'  train pred min/max: {p_train.min().item():.4f} / {p_train.max().item():.4f}', flush=True)
    print(f'  eval  pred min/max: {p_eval.min().item():.4f} / {p_eval.max().item():.4f}', flush=True)

    # pre-clamp vs post-clamp (head raw)
    model.eval()
    with torch.no_grad():
        gca_thick = images[:, 0]
        rnfl_thick = images[:, 2]
        x = torch.cat([gca_thick, rnfl_thick], dim=3)
        x = torch.nn.functional.interpolate(
            x, size=model.model_input_size, mode='bilinear', align_corners=False,
        )
        feat = model.image_branch(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        raw = model.head(feat)
        clamped = torch.clamp(raw, -5.0, 45.0)
    n_clip_low = (raw < -5).sum().item()
    n_clip_high = (raw > 45).sum().item()
    print(f'  head raw min/max: {raw.min().item():.4f} / {raw.max().item():.4f}', flush=True)
    print(f'  clamp hit: {n_clip_low} pts <-5, {n_clip_high} pts >45 (total {raw.numel()})', flush=True)
    print(f'  clamped min/max: {clamped.min().item():.4f} / {clamped.max().item():.4f}', flush=True)


def print_sample_points(pred, labels, mask, title: str):
    p = pred[0].cpu().numpy()
    y = labels[0].cpu().numpy()
    m = mask[0].cpu().numpy()
    print(f'\n=== {title} (첫 val 샘플, 유효점만) ===', flush=True)
    fp = pred_fingerprint(pred, mask)
    print(f'  pred min/max/mean/std: {fp["min"]:.2f} {fp["max"]:.2f} {fp["mean"]:.2f} {fp["std"]:.4f}', flush=True)
    print(f'  at -5: {fp["n_at_-5"]}  at 45: {fp["n_at_45"]}  / {m.sum()} valid', flush=True)
    print(f'  {"pt":>4s}  {"label":>8s}  {"pred":>8s}  {"raw?":>8s}', flush=True)
    # raw for first sample
    for i, pt in enumerate(PT):
        if not m[i]:
            continue
        print(f'  {pt:>4s}  {y[i]:8.2f}  {p[i]:8.2f}', flush=True)


def epoch_val_mse(model, val_loader, device):
    model.eval()
    mse_sum, n = 0.0, 0
    for batch in val_loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        pred = model(images)
        nv = mask.sum().item()
        mse_sum += masked_mse(pred, labels, mask).item() * nv
        n += nv
    return mse_sum / max(n, 1)


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = str(ROOT / 'ml_final_90d_excl_empty_flip.csv')
    train_loader, val_loader, _ = build_dataloaders(
        csv_path, build_image_index(), fold=0, batch_size=64, num_workers=0, image_size=(161, 161),
    )

    model = VFModel(n_output=52, use_tabular=False, use_deviation=False, model_input_size=(161, 322)).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5, momentum=0.9)

    check_mode_independence(model, val_loader, device)

    print('\n=== 2–3) epoch별 val_mse / 예측 fingerprint (smoke와 동일 설정) ===', flush=True)
    history = []
    for epoch in range(1, 11):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device)
        vmse = epoch_val_mse(model, val_loader, device)
        pred, labels, mask, _ = val_predictions(model, val_loader, device)
        fp = pred_fingerprint(pred, mask)
        vrmse = float(np.sqrt(vmse))
        vmae = masked_mae(pred, labels, mask).item()
        history.append({'ep': epoch, 'val_mse': vmse, 'fp': fp})
        print(
            f'  ep{epoch:2d} train_mae={tr_mae:.3f} val_mse={vmse:.7f} val_rmse={vrmse:.4f} val_mae={vmae:.3f} '
            f'pred=[{fp["min"]:.2f},{fp["max"]:.2f}] mean={fp["mean"]:.2f} @-5={fp["n_at_-5"]} @45={fp["n_at_45"]}',
            flush=True,
        )
        if epoch in (1, 2, 9, 10):
            print_sample_points(pred, labels, mask, f'ep{epoch} 첫 val 샘플')

    print('\n=== val_mse epoch간 변화 ===', flush=True)
    for i in range(1, len(history)):
        d = history[i]['val_mse'] - history[i - 1]['val_mse']
        same = abs(d) < 1e-6
        print(
            f'  ep{history[i-1]["ep"]}→{history[i]["ep"]}: Δval_mse={d:+.7f}  '
            f'pred_hash {history[i-1]["fp"]["hash"]:.0f} → {history[i]["fp"]["hash"]:.0f}  '
            f'identical_mse={same}',
            flush=True,
        )


if __name__ == '__main__':
    main()
