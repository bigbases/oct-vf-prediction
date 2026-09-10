#!/usr/bin/env python3
"""학습 없이 best_fold{k}.pt 로 val/test 예측 npz 저장 (fold 0 test 보강용)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from image_preprocessing import build_image_index, build_dataloaders  # noqa: E402
from train import VFModel, predict_loader, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--input_geometry', default='paper', choices=['legacy', 'paper'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir or args.ckpt_dir)
    ckpt = Path(args.ckpt_dir) / f'best_fold{args.fold}.pt'
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    if args.input_geometry == 'paper':
        image_size = (161, 161)
        model_size = (161, 322)
    else:
        image_size = (224, 224)
        model_size = (299, 299)

    csv_path = str(ROOT / args.csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_index = build_image_index()
    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path, image_index, fold=args.fold, batch_size=64, num_workers=4,
        image_size=image_size,
    )
    model = VFModel(model_input_size=model_size).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        bundle = predict_loader(model, loader, device)
        path = out_dir / f'{split_name}_preds_fold{args.fold}.npz'
        meta = {'split': split_name, 'fold': args.fold, 'ckpt': str(ckpt)}
        np.savez_compressed(
            path,
            pred=bundle['pred'].astype(np.float32),
            labels=bundle['labels'].astype(np.float32),
            mask=bundle['mask'].astype(bool),
            patient_id=bundle['patient_id'],
            eye=bundle['eye'],
            vf_date=bundle['vf_date'],
            meta_json=np.array([json.dumps(meta)], dtype=object),
        )
        print(f'저장: {path} (n={len(bundle["pred"])})')


if __name__ == '__main__':
    main()
