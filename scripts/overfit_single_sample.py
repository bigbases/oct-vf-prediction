#!/usr/bin/env python3
"""
단일 샘플 overfit 테스트 — 파이프라인(이미지→모델→라벨) 연결 검증.

사용법:
  python scripts/overfit_single_sample.py --patient_id <id> --eye OD --vf_date <YYYYMMDD>
  python scripts/overfit_single_sample.py --patient_id <id> --vf_date <YYYYMMDD> --epochs 500
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from image_preprocessing import build_image_index, VFDataset, get_transforms, PT  # noqa: E402
from train import (  # noqa: E402
    VFModel,
    set_seed,
    masked_mse,
    masked_mae,
    train_one_epoch,
    evaluate_metrics,
)

# 대상 안(眼)은 인자로 받는다. 환자 식별자를 코드에 남기지 않기 위한 것이다.
DEFAULT_EYE = 'OD'


def find_row(csv_path: Path, pid: str, eye: str, vf_date: str) -> dict:
    eye = eye.upper()
    for r in csv.DictReader(open(csv_path, encoding='utf-8-sig')):
        if (r['patient_id'].strip() == pid and r['eye'].strip().upper() == eye
                and r['vf_date'].strip() == vf_date):
            return r
    raise SystemExit(f'행 없음: {pid} {eye} {vf_date} in {csv_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--patient_id', required=True)
    ap.add_argument('--eye', default=DEFAULT_EYE)
    ap.add_argument('--vf_date', required=True)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='runs/overfit_single_sample')
    ap.add_argument('--backbone', default='inception_v3',
                    choices=['inception_v3', 'inception_resnet_v2', 'xception', 'vgg16', 'densenet121'])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = ROOT / args.csv
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row = find_row(csv_path, args.patient_id, args.eye, args.vf_date)
    image_index = build_image_index(str(ROOT / 'ml_dataset.csv'))
    image_size = (161, 161)
    model_size = (161, 322)

    # 증강 없음(val transform) — 1장 암기 테스트
    tfm = get_transforms('val', image_size)
    ds = VFDataset([row], image_index, tfm, 'train')
    if len(ds) != 1:
        raise SystemExit(f'VFDataset len={len(ds)} (이미지 4장 누락?). skip_log={ds.skip_log}')

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = VFModel(
        n_output=52,
        use_tabular=False,
        use_deviation=False,
        model_input_size=model_size,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.0)

    print('=== 단일 샘플 overfit ===', flush=True)
    print(f'  sample: {args.patient_id} {args.eye} vf_date={args.vf_date}', flush=True)
    print(f'  device={device}  epochs={args.epochs}  lr={args.lr}', flush=True)
    print(f'  backbone={args.backbone}  pipeline: VFModel, 4-map, thickness→161x322 (Phase C B0)', flush=True)
    print(f'  transform: val (no ColorJitter)', flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = train_one_epoch(model, loader, optimizer, device)
        tr_rmse = float(np.sqrt(tr_loss))
        history.append({'epoch': epoch, 'train_mse': tr_loss, 'train_rmse': tr_rmse, 'train_mae': tr_mae})
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'  ep {epoch:4d}  train_mse={tr_loss:.4f}  train_rmse={tr_rmse:.4f}  train_mae={tr_mae:.4f}', flush=True)

    # 최종 평가 + 예측
    model.eval()
    batch = next(iter(loader))
    images = batch['images'].to(device)
    labels = batch['labels'].to(device)
    mask = batch['mask'].to(device)
    with torch.no_grad():
        pred = model(images)
    fin_mse = masked_mse(pred, labels, mask).item()
    fin_mae = masked_mae(pred, labels, mask).item()
    fin_rmse = float(np.sqrt(fin_mse))

    pred_np = pred[0].cpu().numpy()
    lab_np = labels[0].cpu().numpy()
    msk_np = mask[0].cpu().numpy()

    print('\n=== 최종 (동일 1샘플) ===', flush=True)
    print(f'  train_rmse={fin_rmse:.4f}  train_mae={fin_mae:.4f}', flush=True)
    if fin_rmse < 1.5:
        verdict = 'PASS — 파이프라인 암기 가능, 일반화/규모/입력형식 이슈 쪽'
    elif fin_rmse < 5.0:
        verdict = 'PARTIAL — 어느 정도 학습되나 완전 암기 실패 (lr/epoch/augment 확인)'
    else:
        verdict = 'FAIL — 라벨·입력·모델 연결 버그 의심'
    print(f'  판정: {verdict}', flush=True)

    print('\n=== 예측 vs 라벨 (52점, |err| 큰 순 12개) ===', flush=True)
    rows_pt = []
    for i, p in enumerate(PT):
        if not msk_np[i]:
            continue
        rows_pt.append((abs(pred_np[i] - lab_np[i]), p, lab_np[i], pred_np[i]))
    rows_pt.sort(reverse=True)
    print(f'  {"pt":>4s}  {"label":>8s}  {"pred":>8s}  {"|err|":>8s}', flush=True)
    for err, p, y, p_hat in rows_pt[:12]:
        print(f'  {p:>4s}  {y:8.2f}  {p_hat:8.2f}  {err:8.2f}', flush=True)
    print(f'  ... (유효점 {len(rows_pt)}개 중 상위 12개 오차)', flush=True)

    # 전체 요약 통계
    err = np.abs(pred_np[msk_np] - lab_np[msk_np])
    print(f'\n  유효점 MAE={err.mean():.3f}  RMSE={np.sqrt((err**2).mean()):.3f}  max|err|={err.max():.3f}', flush=True)

    result = {
        'backbone': args.backbone,
        'sample': {'patient_id': args.patient_id, 'eye': args.eye, 'vf_date': args.vf_date},
        'epochs': args.epochs,
        'lr': args.lr,
        'final_rmse': fin_rmse,
        'final_mae': fin_mae,
        'verdict': verdict,
        'history_tail': history[-20:],
        'history': history,
    }
    (out_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {out_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
