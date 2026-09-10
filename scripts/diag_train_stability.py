#!/usr/bin/env python3
"""학습 발산 진단: 입력/라벨, step별 loss·grad norm, epoch별 val 예측 범위."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from image_preprocessing import build_image_index, build_dataloaders  # noqa: E402
from train import VFModel, set_seed, masked_mse, masked_mae  # noqa: E402


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return float(np.sqrt(total))


def scan_loader(name: str, loader, device, max_batches: int = 3):
    print(f'\n=== [{name}] 배치 스캔 (최대 {max_batches}배치) ===', flush=True)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x = batch['images']
        y = batch['labels']
        m = batch['mask']
        print(
            f'  batch {bi}: images shape={tuple(x.shape)} '
            f'min={x.min().item():.4f} max={x.max().item():.4f} '
            f'nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()}',
            flush=True,
        )
        yv = y[m]
        if yv.numel():
            print(
                f'           labels(valid) min={yv.min().item():.2f} max={yv.max().item():.2f} '
                f'nan={torch.isnan(y).any().item()}',
                flush=True,
            )


@torch.no_grad()
def pred_stats(model, loader, device):
    mins, maxs, n_nan, n_inf = [], [], 0, 0
    for batch in loader:
        pred = model(batch['images'].to(device))
        mins.append(pred.min().item())
        maxs.append(pred.max().item())
        n_nan += int(torch.isnan(pred).sum().item())
        n_inf += int(torch.isinf(pred).sum().item())
    return min(mins), max(maxs), n_nan, n_inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--max_steps_per_epoch', type=int, default=0,
                    help='0=전체 step, 양수면 epoch당 step 수 제한')
    args = ap.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = str(ROOT / args.csv)
    image_index = build_image_index()
    train_loader, val_loader, _ = build_dataloaders(
        csv_path, image_index, fold=args.fold,
        batch_size=args.batch_size, num_workers=0,
        image_size=(161, 161),
    )

    scan_loader('train', train_loader, device)
    scan_loader('val', val_loader, device)

    model = VFModel(
        n_output=52, use_tabular=False, use_deviation=False,
        model_input_size=(161, 322),
    ).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9,
    )

    print('\n=== clip 위치: loss.backward() → clip_grad_norm_(max_norm=1.0) → optimizer.step() ===', flush=True)
    print(f'device={device}  lr={args.lr}  train_batches={len(train_loader)}  val_batches={len(val_loader)}', flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        step = 0
        for batch in train_loader:
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = masked_mse(pred, labels, mask)
            if not torch.isfinite(loss):
                print(f'  ep{epoch} step{step}: non-finite loss — skip', flush=True)
                continue
            loss.backward()
            gn_before = grad_norm(model)
            gn_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gn_after = grad_norm(model)
            optimizer.step()
            step += 1

            with torch.no_grad():
                pmn, pmx = pred.min().item(), pred.max().item()
                pn = int(torch.isnan(pred).sum().item())
                pi = int(torch.isinf(pred).sum().item())

            if step <= 5 or step % max(1, len(train_loader) // 3) == 0:
                print(
                    f'  ep{epoch:2d} step{step:3d} | loss={loss.item():.4f} '
                    f'grad_norm before={gn_before:.3f} returned_clip={float(gn_clip):.3f} after={gn_after:.3f} '
                    f'pred=[{pmn:.2f},{pmx:.2f}] nan={pn} inf={pi}',
                    flush=True,
                )

        model.eval()
        val_mse, val_mae = 0.0, 0.0
        n = 0
        for batch in val_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            pred = model(images)
            nv = mask.sum().item()
            val_mse += masked_mse(pred, labels, mask).item() * nv
            val_mae += masked_mae(pred, labels, mask).item() * nv
            n += nv
        val_mse /= max(n, 1)
        val_mae /= max(n, 1)
        pmin, pmax, pnan, pinf = pred_stats(model, val_loader, device)
        print(
            f'  ep{epoch:2d} EPOCH END | val_mse={val_mse:.4f} val_rmse={np.sqrt(val_mse):.4f} val_mae={val_mae:.3f} '
            f'val_pred=[{pmin:.2f},{pmax:.2f}] nan={pnan} inf={pinf}',
            flush=True,
        )


if __name__ == '__main__':
    main()
