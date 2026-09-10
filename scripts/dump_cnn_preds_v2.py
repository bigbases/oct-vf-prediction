#!/usr/bin/env python3
"""학습 없이 best_fold{k}.pt 로 val/test 예측 npz 저장 (osflip 실험용 확장판).

기존 scripts/dump_cnn_preds_from_ckpt.py 의 두 결함을 고친 신규 파일이다
(기존 파일은 무손상 보존):
  (a) build_dataloaders() 에 flip_os_images 를 전달하지 않아, flip 으로 학습한
      체크포인트에 안 flip 된 입력으로 예측을 뽑던 문제 → --flip_os_images 추가
  (b) VFModel() 을 backbone 인자 없이 호출해 기본값 inception_v3 가 쓰이던 문제
      (IR-v2 체크포인트와 불일치) → --backbone 추가

num_workers 는 기준 런(results.json: num_workers=0)과 맞춰 기본 0.
"""
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
    ap.add_argument('--backbone', default='inception_resnet_v2',
                    help='체크포인트의 백본. VFModel 기본값(inception_v3)과 다르면 반드시 지정')
    ap.add_argument('--flip_os_images', action='store_true',
                    help='OS 이미지 좌우반전. 체크포인트 학습 시 설정과 반드시 일치시킬 것')
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--splits', default='val,test',
                    help='쉼표구분: val,test 중 저장할 것')
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir or args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        csv_path, image_index, fold=args.fold, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=image_size,
        flip_os_images=args.flip_os_images,
    )
    model = VFModel(model_input_size=model_size, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    want = {s.strip() for s in args.splits.split(',') if s.strip()}
    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        if split_name not in want:
            continue
        bundle = predict_loader(model, loader, device)
        path = out_dir / f'{split_name}_preds_fold{args.fold}.npz'
        meta = {
            'split': split_name, 'fold': args.fold, 'ckpt': str(ckpt),
            'backbone': args.backbone, 'flip_os_images': bool(args.flip_os_images),
            'input_geometry': args.input_geometry, 'seed': args.seed,
        }
        np.savez_compressed(
            path,
            pred=bundle['pred'].astype(np.float32),
            labels=bundle['labels'].astype(np.float32),
            mask=bundle['mask'].astype(bool),
            patient_id=bundle['patient_id'],
            eye=bundle['eye'],
            vf_date=bundle['vf_date'],
            meta_json=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object),
        )
        print(f'저장: {path} (n={len(bundle["pred"])}) flip={args.flip_os_images} '
              f'backbone={args.backbone}')


if __name__ == '__main__':
    main()
