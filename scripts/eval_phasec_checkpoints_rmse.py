"""
Phase C (train.py) 체크포인트에서 val/test RMSE·MAE 후처리.

train.py는 MAE만 저장하므로, best_fold{N}.pt 로드 후 masked MSE→RMSE 계산.

사용법:
  HVF_ROOT="$(pwd)" python scripts/eval_phasec_checkpoints_rmse.py \\
    --out_dir runs/phasec_90d_b0_imgonly_paper_geom_5fold

  # results.json 없이 수동 지정
  python scripts/eval_phasec_checkpoints_rmse.py --out_dir runs/... \\
    --csv ml_final_90d_excl_empty_flip.csv --input_geometry paper
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT))

from image_preprocessing import build_image_index, build_dataloaders  # noqa: E402
from train import VFModel, masked_mae, masked_mse  # noqa: E402


def _resolve_geometry(args) -> SimpleNamespace:
    if args.input_geometry == 'paper':
        args.image_h, args.image_w = 161, 161
        args.model_h, args.model_w = 161, 322
    else:
        args.image_h, args.image_w = 224, 224
        args.model_h, args.model_w = 299, 299
    return args


def _load_train_args(out_dir: Path, cli) -> SimpleNamespace:
    results_path = out_dir / 'results.json'
    if results_path.is_file():
        merged = dict(json.load(open(results_path, encoding='utf-8')).get('args', {}))
    else:
        merged = {}
    if cli.csv:
        merged['csv'] = cli.csv
    if cli.input_geometry:
        merged['input_geometry'] = cli.input_geometry
    if not merged.get('csv'):
        raise SystemExit(f'{results_path}에 csv 없음 — --csv 지정 필요')
    merged.setdefault('input_geometry', 'paper')
    merged.setdefault('batch_size', 64)
    merged.setdefault('num_workers', 4)
    merged.setdefault('n_output', 52)
    return _resolve_geometry(SimpleNamespace(**merged))


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        pred = model(images, tabular)
        n_valid = mask.sum().item()
        mae_sum += masked_mae(pred, labels, mask).item() * n_valid
        mse_sum += masked_mse(pred, labels, mask).item() * n_valid
        n += n_valid
    if n == 0:
        return float('inf'), float('inf')
    mse = mse_sum / n
    mae = mae_sum / n
    return mse, mae


def eval_fold(
    fold: int,
    out_dir: Path,
    csv_path: str,
    image_index: dict,
    args: SimpleNamespace,
    device: torch.device,
) -> dict | None:
    ckpt = out_dir / f'best_fold{fold}.pt'
    if not ckpt.is_file():
        return None

    _, val_loader, test_loader = build_dataloaders(
        csv_path, image_index, fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_h, args.image_w),
    )

    model = VFModel(
        n_output=args.n_output,
        use_tabular=getattr(args, 'use_tabular', False),
        freeze_backbone=getattr(args, 'freeze_backbone', False),
        use_deviation=getattr(args, 'use_deviation', False),
        model_input_size=(args.model_h, args.model_w),
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    val_mse, val_mae = eval_loader(model, val_loader, device)
    test_mse, test_mae = eval_loader(model, test_loader, device)

    return {
        'fold': fold,
        'checkpoint': str(ckpt),
        'val_mse': val_mse,
        'val_rmse': float(math.sqrt(val_mse)),
        'val_mae': val_mae,
        'test_mse': test_mse,
        'test_rmse': float(math.sqrt(test_mse)),
        'test_mae': test_mae,
        'n_val': len(val_loader.dataset),
        'n_test': len(test_loader.dataset),
    }


def _summary(rows: list[dict], key_prefix: str) -> dict:
    rmse = np.array([r[f'{key_prefix}_rmse'] for r in rows])
    mae = np.array([r[f'{key_prefix}_mae'] for r in rows])
    return {
        f'{key_prefix}_rmse_mean': float(rmse.mean()),
        f'{key_prefix}_rmse_std': float(rmse.std(ddof=0)),
        f'{key_prefix}_mae_mean': float(mae.mean()),
        f'{key_prefix}_mae_std': float(mae.std(ddof=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='Phase C run dir (best_fold*.pt)')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--input_geometry', choices=['legacy', 'paper'], default=None)
    parser.add_argument('--use_tabular', action='store_true')
    parser.add_argument('--use_deviation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_output', type=int, default=52)
    parser.add_argument('--folds', nargs='*', type=int, default=None,
                        help='특정 fold만 (기본: out_dir 내 best_fold*.pt 전부)')
    args_cli = parser.parse_args()

    out_dir = ROOT / args_cli.out_dir if not Path(args_cli.out_dir).is_absolute() else Path(args_cli.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f'out_dir 없음: {out_dir}')

    train_args = _load_train_args(out_dir, args_cli)
    if args_cli.batch_size is not None:
        train_args.batch_size = args_cli.batch_size
    if args_cli.num_workers is not None:
        train_args.num_workers = args_cli.num_workers

    if args_cli.folds:
        folds = args_cli.folds
    else:
        folds = sorted(
            int(p.stem.replace('best_fold', ''))
            for p in out_dir.glob('best_fold*.pt')
        )

    if not folds:
        raise SystemExit(f'체크포인트 없음: {out_dir}/best_fold*.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = str(ROOT / train_args.csv)
    image_index = build_image_index()

    print(f'=== Phase C RMSE 후처리 ===', flush=True)
    print(f'out_dir: {out_dir}', flush=True)
    print(f'csv: {csv_path}', flush=True)
    print(f'folds: {folds}', flush=True)
    print(f'device: {device}', flush=True)

    per_fold = []
    for fold in folds:
        row = eval_fold(fold, out_dir, csv_path, image_index, train_args, device)
        if row is None:
            print(f'  [skip] fold {fold}: checkpoint 없음', flush=True)
            continue
        per_fold.append(row)
        print(
            f'  fold {row["fold"]}: val_rmse {row["val_rmse"]:.4f} val_mae {row["val_mae"]:.3f}'
            f' | test_rmse {row["test_rmse"]:.4f} test_mae {row["test_mae"]:.3f}',
            flush=True,
        )

    if not per_fold:
        raise SystemExit('평가 가능한 fold 없음')

    summary = {
        'n_folds': len(per_fold),
        **_summary(per_fold, 'val'),
        **_summary(per_fold, 'test'),
    }

    print(f'\n=== 집계 ({len(per_fold)} fold) ===', flush=True)
    print(
        f'  Val  RMSE: {summary["val_rmse_mean"]:.4f} ± {summary["val_rmse_std"]:.4f}'
        f'  | MAE: {summary["val_mae_mean"]:.3f} ± {summary["val_mae_std"]:.3f}',
        flush=True,
    )
    print(
        f'  Test RMSE: {summary["test_rmse_mean"]:.4f} ± {summary["test_rmse_std"]:.4f}'
        f'  | MAE: {summary["test_mae_mean"]:.3f} ± {summary["test_mae_std"]:.3f}',
        flush=True,
    )

    out_json = out_dir / 'results_rmse.json'
    payload = {
        'out_dir': str(out_dir),
        'csv': csv_path,
        'train_args': vars(train_args),
        'per_fold': per_fold,
        'summary': summary,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n저장: {out_json}', flush=True)


if __name__ == '__main__':
    main()
