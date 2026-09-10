"""Prior-work-matched training track: a single random holdout split instead of
five-fold cross-validation, used only to compare image-only against multimodal
under the earlier paper's protocol.

train_paper_track.py
논문 일치 트랙(근사): 단일 random holdout split로 image-only vs multimodal 비교

- 입력 geometry: 161x322 고정
- optimizer: RMSprop 기본값
- split: train set에서 val_ratio만큼 random holdout (논문 validation_split 유사)
- 선택 기준: val_loss(MSE)
"""
import os
import sys
import csv
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from image_preprocessing import (
    build_image_index, VFDataset, get_transforms, compute_tabular_stats,
    N_TABULAR, N_VF,
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone(name: str):
    """ImageNet-pretrained backbone → (feature_extractor, out_dim).
    분류 head(fc/classifier)는 Identity로 치환해 feature만 반환한다.
    입력 크기는 호출부에서 161x322로 통일(모든 백본 adaptive pooling 지원)."""
    if name == 'inception_v3':
        m = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
        m.AuxLogits = None
    elif name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
    elif name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_feat = m.classifier.in_features
        m.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif name == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[6].in_features
        m.classifier[6] = nn.Identity()
    else:
        raise ValueError(f'unknown backbone: {name}')
    return m, in_feat


class VFModel(nn.Module):
    def __init__(
        self,
        n_output: int = N_VF,
        use_tabular: bool = False,
        n_tabular: int = N_TABULAR,
        backbone: str = 'inception_v3',
        input_mode: str = 'thickness_only',
    ):
        super().__init__()
        self.use_tabular = use_tabular
        self.backbone_name = backbone
        self.input_mode = input_mode
        if input_mode not in ('thickness_only', 'thickness_deviation'):
            raise ValueError(f'unknown input_mode: {input_mode}')

        self.image_branch, in_feat = build_backbone(backbone)

        if use_tabular:
            self.tabular_branch = nn.Sequential(
                nn.Linear(n_tabular, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 64), nn.ReLU(inplace=True),
            )
            fc_in = in_feat + 64
        else:
            self.tabular_branch = None
            fc_in = in_feat

        self.head = nn.Sequential(
            nn.Linear(fc_in, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_output), nn.ReLU(inplace=True),
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """OCT map tensor → backbone 입력 [B,3,H,W]."""
        if self.input_mode == 'thickness_only':
            x = torch.cat([images[:, 0], images[:, 2]], dim=3)  # GCA·RNFL thickness
            return F.interpolate(x, size=(161, 322), mode='bilinear', align_corners=False)
        # thickness + deviation 4장 (공간 해상도 유지: map당 ~161px)
        x = torch.cat([images[:, i] for i in range(4)], dim=3)
        return F.interpolate(x, size=(161, 644), mode='bilinear', align_corners=False)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        x = self._prepare_images(images)

        feat = self.image_branch(x)
        if isinstance(feat, tuple):
            feat = feat[0]

        if self.use_tabular:
            tfeat = self.tabular_branch(tabular)
            feat = torch.cat([feat, tfeat], dim=1)
        return self.head(feat)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    diff2 = (pred - target) ** 2 * m
    return diff2.sum() / m.sum().clamp(min=1.0)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    diff = torch.abs(pred - target) * m
    return diff.sum() / m.sum().clamp(min=1.0)


def split_rows(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    # holdout test는 따로 리포팅만 하고, 논문식 validation split은 train 부분에서 수행
    train_pool = [r for r in rows if r.get('cv_fold') != 'test']
    test_rows = [r for r in rows if r.get('cv_fold') == 'test']
    if len(train_pool) < 2:
        raise ValueError("학습 가능한 행 수가 너무 적습니다.")

    rng = random.Random(seed)
    idx = list(range(len(train_pool)))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(train_pool) * val_ratio)))
    n_val = min(n_val, len(train_pool) - 1)
    val_idx = set(idx[:n_val])
    train_rows = [train_pool[i] for i in range(len(train_pool)) if i not in val_idx]
    val_rows = [train_pool[i] for i in range(len(train_pool)) if i in val_idx]
    return train_rows, val_rows, test_rows


def make_loaders(args, image_index):
    rows = list(csv.DictReader(open(str(ROOT / args.csv), encoding='utf-8-sig')))
    train_rows, val_rows, test_rows = split_rows(rows, args.val_ratio, args.seed)
    tab_fill, tab_std = compute_tabular_stats(train_rows)

    train_ds = VFDataset(train_rows, image_index, get_transforms('train', (161, 161)), 'train', tab_fill, tab_std)
    val_ds = VFDataset(val_rows, image_index, get_transforms('val', (161, 161)), 'val', tab_fill, tab_std)
    test_ds = VFDataset(test_rows, image_index, get_transforms('test', (161, 161)), 'test', tab_fill, tab_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


def eval_loader(model, loader, device) -> Tuple[float, float]:
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
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
    return mse_sum / n, mae_sum / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ml_final_180d_excl_empty_flip.csv')
    parser.add_argument('--use_tabular', action='store_true')
    parser.add_argument('--backbone', default='inception_v3',
                        choices=['inception_v3', 'resnet18', 'resnet50',
                                 'densenet121', 'efficientnet_b0', 'vgg16'])
    parser.add_argument('--input_mode', default='thickness_only',
                        choices=['thickness_only', 'thickness_deviation'],
                        help='thickness_only=GCA+RNFL thickness (기본), '
                             'thickness_deviation=4 maps incl. deviation')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='backbone 가중치 고정, regression head만 학습 (B1)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='rmsprop')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='runs/repro_default')
    parser.add_argument('--start_epoch', type=int, default=1, help='1-indexed first epoch to train (resume)')
    parser.add_argument('--resume_weights', default=None, help='path to .pt state_dict')
    parser.add_argument('--resume_state', default=None, help='checkpoint.json from prior run')
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== 설정 ===', flush=True)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    print(f'device: {"cuda" if torch.cuda.is_available() else "cpu"}', flush=True)

    image_index = build_image_index()
    train_loader, val_loader, test_loader = make_loaders(args, image_index)
    print(f'data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VFModel(
        use_tabular=args.use_tabular,
        backbone=args.backbone,
        input_mode=args.input_mode,
    ).to(device)
    if args.freeze_backbone:
        for p in model.image_branch.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.image_branch.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'freeze_backbone: {n_frozen} params frozen, {n_train} trainable', flush=True)
    if args.resume_weights:
        wpath = Path(args.resume_weights)
        if not wpath.is_file():
            raise FileNotFoundError(f'resume_weights not found: {wpath}')
        model.load_state_dict(torch.load(wpath, map_location=device))
        print(f'resumed weights: {wpath}', flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
        )
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = -1
    best_val_mse = float('inf')
    best_val_mae = None
    best_val_rmse = None
    best_test_mae = None
    best_test_rmse = None
    patience_left = args.patience
    history = []

    if args.resume_state:
        spath = Path(args.resume_state)
        if not spath.is_file():
            raise FileNotFoundError(f'resume_state not found: {spath}')
        with open(spath, encoding='utf-8') as f:
            ckpt = json.load(f)
        history = list(ckpt.get('history', []))
        best_epoch = ckpt.get('best_epoch', -1)
        best_val_mse = ckpt.get('best_val_mse', float('inf'))
        best_val_mae = ckpt.get('best_val_mae')
        best_val_rmse = ckpt.get('best_val_rmse')
        best_test_rmse = ckpt.get('best_holdout_test_rmse')
        best_test_mae = ckpt.get('best_holdout_test_mae')
        patience_left = ckpt.get('patience_left', args.patience)
        print(
            f'resumed state: last_ep={ckpt.get("last_completed_epoch")} best_ep={best_epoch} '
            f'patience_left={patience_left}',
            flush=True,
        )

    if args.start_epoch > 1:
        print(f'training from epoch {args.start_epoch} to {args.epochs}', flush=True)

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_mae_sum, n = 0.0, 0.0, 0
        for batch in train_loader:
            images = batch['images'].to(device)
            tabular = batch['tabular'].to(device) if model.use_tabular else None
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            pred = model(images, tabular)
            loss = masked_mse(pred, labels, mask)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            n_valid = mask.sum().item()
            tr_loss_sum += loss.item() * n_valid
            tr_mae_sum += masked_mae(pred.detach(), labels, mask).item() * n_valid
            n += n_valid

        train_mse = tr_loss_sum / max(n, 1)
        train_mae = tr_mae_sum / max(n, 1)
        train_rmse = float(np.sqrt(train_mse))
        val_mse, val_mae = eval_loader(model, val_loader, device)
        val_rmse = float(np.sqrt(val_mse))
        msg = (
            f'ep {epoch:4d}/{args.epochs} | train_mse {train_mse:.4f} | train_rmse {train_rmse:.4f} | train_mae {train_mae:.3f}'
            f' | val_mse {val_mse:.4f} | val_rmse {val_rmse:.4f} | val_mae {val_mae:.3f}'
        )

        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_left = args.patience
            best_test_mse, best_test_mae = eval_loader(model, test_loader, device) if len(test_loader.dataset) > 0 else (None, None)
            best_test_rmse = float(np.sqrt(best_test_mse)) if best_test_mse is not None else None
            torch.save(model.state_dict(), out_dir / 'best.pt')
            if best_test_mae is not None:
                msg += f'  ★ holdout_test_rmse {best_test_rmse:.4f} | holdout_test_mae {best_test_mae:.3f}'
        else:
            patience_left -= 1

        print(msg, flush=True)
        history.append({
            'epoch': epoch,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
        })

        if patience_left <= 0:
            print(f'early stop @ ep {epoch} (best ep {best_epoch})', flush=True)
            break

    result = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse if best_epoch >= 0 else None,
        'best_val_rmse': best_val_rmse,
        'best_val_mae': best_val_mae,
        'best_holdout_test_rmse': best_test_rmse,
        'best_holdout_test_mae': best_test_mae,
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'n_holdout_test': len(test_loader.dataset),
        'history': history,
        'resumed_from_epoch': args.start_epoch if args.start_epoch > 1 else None,
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n=== 최종 ===', flush=True)
    if best_epoch >= 0:
        print(
            f'best ep {best_epoch} | val_mse {best_val_mse:.4f} | val_rmse {best_val_rmse:.4f} | val_mae {best_val_mae:.3f}',
            flush=True,
        )
        if best_test_mae is not None:
            print(f'holdout_test_rmse {best_test_rmse:.4f} | holdout_test_mae {best_test_mae:.3f}', flush=True)
    print(f'결과 저장: {out_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
