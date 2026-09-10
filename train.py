"""
train.py
멀티모달 VF 예측 모델 학습 (선행연구 climyth/VFbySD-OCT 재현 + tabular 확장)

구조 (논문 동일):
  GCA 두께맵 + RNFL 두께맵 가로 concat → InceptionV3(ImageNet pretrained)
  → GAP → Dense(1024 → 512 → 256 → 52) → ReLU  (p26/p35 제외)
  + (옵션) tabular → MLP(64 → 64) → GAP feature와 concat

Loss: masked MSE (p26/p35 제외, 결측/−1 마스킹)
Metric: masked MAE (dB)

사용법:
  # image-only baseline (전체 5-fold)
  python train.py --csv ml_final_90d_excl_empty.csv

  # image + tabular (전체 5-fold)
  python train.py --csv ml_final_90d_excl_empty.csv --use_tabular

  # 1 fold만 빠르게 (스모크 테스트)
  python train.py --fold 0 --epochs 2 --batch_size 4
"""
import os, sys, csv, json, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from image_preprocessing import (
    build_image_index, VFDataset, get_transforms, build_dataloaders,
    OCT_FEATURES_OD, OCT_FEATURES_OS, N_TABULAR, N_VF, PT,
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))


# ─────────────────────────────────────────────
# 1. 재현성
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# 2. 모델
# ─────────────────────────────────────────────
BACKBONE_CHOICES = (
    'inception_v3',
    'inception_resnet_v2',
    'xception',
    'vgg16',
    'densenet121',
)


def build_image_backbone(name: str) -> Tuple[nn.Module, int]:
    """ImageNet pretrained backbone → (module, feature_dim). 분류 head는 Identity."""
    if name == 'inception_v3':
        m = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True,
        )
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
        m.AuxLogits = None
    elif name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_feat = m.classifier.in_features
        m.classifier = nn.Identity()
    elif name in ('inception_resnet_v2', 'xception', 'vgg16'):
        import timm
        timm_name = 'legacy_xception' if name == 'xception' else name
        m = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool='avg')
        # timm vgg16: num_features=512 but forward → 4096 (penultimate FC path)
        in_feat = 4096 if name == 'vgg16' else m.num_features
    else:
        raise ValueError(f'unknown backbone: {name}')
    return m, in_feat


class VFModel(nn.Module):
    """
    선행연구 재현 모델 (climyth/VFbySD-OCT)
      입력: GCA 두께맵 + RNFL 두께맵 가로 concat → 299x299로 리사이즈
      백본: InceptionV3 (ImageNet pretrained)
      head: Dense 1024 → 512 → 256 → 52 (ReLU, p26/p35 제외)
      (옵션) tabular 10 → MLP → concat
    """
    def __init__(
        self,
        n_output: int = N_VF,
        use_tabular: bool = False,
        n_tabular: int = N_TABULAR,
        freeze_backbone: bool = False,
        use_deviation: bool = False,
        model_input_size: Tuple[int, int] = (299, 299),
        backbone: str = 'inception_v3',
    ):
        super().__init__()
        self.n_output = n_output
        self.use_tabular = use_tabular
        self.use_deviation = use_deviation
        self.model_input_size = model_input_size
        self.backbone_name = backbone

        self.image_branch, in_feat = build_image_backbone(backbone)

        if freeze_backbone:
            for p in self.image_branch.parameters():
                p.requires_grad = False

        # Tabular MLP (옵션)
        if use_tabular:
            self.tabular_branch = nn.Sequential(
                nn.Linear(n_tabular, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            fc_in = in_feat + 64
        else:
            self.tabular_branch = None
            fc_in = in_feat

        # 논문 head: 1024 → 512 → 256 → 52, 마지막 층은 선형 회귀 출력
        self.head = nn.Sequential(
            nn.Linear(fc_in, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512),   nn.ReLU(inplace=True),
            nn.Linear(512, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, n_output),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        """
        images: [B, 4, 3, 224, 224] — (GCA_thick, GCA_dev, RNFL_thick, RNFL_dev)
                                       논문 재현은 thickness 2장만 사용
        tabular: [B, 10] (use_tabular=True 시)
        반환: [B, n_output]
        """
        gca_thick  = images[:, 0]   # [B, 3, 224, 224]
        gca_dev    = images[:, 1]   # [B, 3, 224, 224]
        rnfl_thick = images[:, 2]   # [B, 3, 224, 224]
        rnfl_dev   = images[:, 3]   # [B, 3, 224, 224]

        if self.use_deviation:
            # 4-panel (thickness + deviation) horizontal concat
            x = torch.cat([gca_thick, gca_dev, rnfl_thick, rnfl_dev], dim=3)  # [B, 3, 224, 896]
        else:
            # paper baseline: thickness 2장만 사용
            x = torch.cat([gca_thick, rnfl_thick], dim=3)  # [B, 3, 224, 448]
        x = F.interpolate(x, size=self.model_input_size,
                          mode='bilinear', align_corners=False)

        # InceptionV3 forward (training 모드에서 InceptionOutputs 튜플 방지)
        # aux_logits=True여도 AuxLogits=None이면 자동 단일 텐서 반환
        img_feat = self.image_branch(x)
        if isinstance(img_feat, tuple):
            img_feat = img_feat[0]

        if self.use_tabular:
            tab_feat = self.tabular_branch(tabular)
            feat = torch.cat([img_feat, tab_feat], dim=1)
        else:
            feat = img_feat

        return self.head(feat)


# ─────────────────────────────────────────────
# 3. Loss / Metric
# ─────────────────────────────────────────────
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """blind spot/결측 마스킹된 MSE"""
    m = mask.float()
    diff2 = (pred - target) ** 2 * m
    return diff2.sum() / m.sum().clamp(min=1.0)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """blind spot/결측 마스킹된 MAE (dB)"""
    m = mask.float()
    diff = torch.abs(pred - target) * m
    return diff.sum() / m.sum().clamp(min=1.0)


# ─────────────────────────────────────────────
# 4. Train / Validate
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    loss_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        images  = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels  = batch['labels'].to(device)
        mask    = batch['mask'].to(device)

        optimizer.zero_grad()
        pred = model(images, tabular)
        loss = masked_mse(pred, labels, mask)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n_valid = mask.sum().item()
        loss_sum += loss.item() * n_valid
        mae_sum  += masked_mae(pred.detach(), labels, mask).item() * n_valid
        n += n_valid
    return loss_sum / max(n, 1), mae_sum / max(n, 1)


@torch.no_grad()
def evaluate_metrics(model, loader, device) -> Tuple[float, float]:
    """masked val MSE, MAE (repro early-stop은 MSE, Phase C 기본은 MAE)."""
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        images  = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels  = batch['labels'].to(device)
        mask    = batch['mask'].to(device)
        pred = model(images, tabular)
        n_valid = mask.sum().item()
        mse_sum += masked_mse(pred, labels, mask).item() * n_valid
        mae_sum += masked_mae(pred, labels, mask).item() * n_valid
        n += n_valid
    denom = max(n, 1)
    return mse_sum / denom, mae_sum / denom


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    _, mae = evaluate_metrics(model, loader, device)
    return mae


@torch.no_grad()
def predict_loader(model, loader, device) -> Dict:
    """배치별 예측 + 행 키 수집 (late-fusion OOF dump용)."""
    model.eval()
    preds, labels, masks = [], [], []
    pids, eyes, dates = [], [], []
    for batch in loader:
        images  = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels_b = batch['labels'].to(device)
        mask_b   = batch['mask'].to(device)
        pred = model(images, tabular)
        preds.append(pred.cpu().numpy())
        labels.append(labels_b.cpu().numpy())
        masks.append(mask_b.cpu().numpy())
        pids.extend(batch['patient_id'])
        eyes.extend(batch['eye'])
        dates.extend(batch['vf_date'])
    return {
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'patient_id': np.array(pids, dtype=object),
        'eye': np.array(eyes, dtype=object),
        'vf_date': np.array(dates, dtype=object),
    }


# ─────────────────────────────────────────────
# 5. 1 fold 학습
# ─────────────────────────────────────────────
def train_fold(csv_path: str, image_index: Dict, fold: int, args, out_dir: Path) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _rng_state():
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])

    def _advance_legacy_test_loader_rng():
        """과거 실행의 test-loader base-seed 소비만 재현하되 held-out 데이터는 읽지 않는다.

        기존 코드는 validation이 개선될 때마다 test loader를 평가해 DataLoader의
        base-seed를 하나 소비했다. 그 값을 checkpoint 선택에는 쓰지 않았지만,
        소비된 RNG가 다음 epoch/fold의 shuffle에 영향을 줬다. PyTorch 1.12의
        DataLoader iterator와 같은 base-seed draw만 수행해 canonical RNG 궤적을
        보존한다.
        """
        torch.empty((), dtype=torch.int64).random_(generator=test_loader.generator)

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path, image_index, fold=fold,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=(args.image_h, args.image_w),
        flip_os_images=getattr(args, 'flip_os_images', False),
        target_floor=getattr(args, 'target_floor', None),
    )

    model = VFModel(
        n_output=args.n_output,
        use_tabular=args.use_tabular,
        freeze_backbone=args.freeze_backbone,
        use_deviation=args.use_deviation,
        model_input_size=(args.model_h, args.model_w),
        backbone=getattr(args, 'backbone', 'inception_v3'),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
        )
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    stop_on_mse = getattr(args, 'early_stop_metric', 'mae') == 'mse'
    best_val_mae = float('inf')
    best_val_mse = float('inf')
    best_test_mae = None
    best_test_mse = None
    best_epoch = -1
    patience_left = args.patience

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = evaluate_metrics(model, val_loader, device)
        val_rmse = float(np.sqrt(val_mse))
        msg = (
            f'  [fold {fold}] ep {epoch:3d}/{args.epochs} | train_loss {tr_loss:.4f} | train_mae {tr_mae:.3f}'
            f' | val_mse {val_mse:.4f} | val_rmse {val_rmse:.4f} | val_mae {val_mae:.3f}'
        )

        if not (np.isfinite(tr_loss) and np.isfinite(val_mae) and np.isfinite(val_mse)):
            print(f'  [fold {fold}] ep {epoch}: NaN 감지 — 학습 중단 (tabular 결측/impute 확인)', flush=True)
            break

        if stop_on_mse:
            improved = val_mse < best_val_mse - 1e-6
        else:
            improved = val_mae < best_val_mae - 1e-4

        if improved:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), out_dir / f'best_fold{fold}.pt')
            _advance_legacy_test_loader_rng()
            msg += '  ★ checkpoint'
        else:
            patience_left -= 1

        print(msg, flush=True)
        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_mae': tr_mae,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
        })

        if patience_left <= 0:
            print(f'  [fold {fold}] early stop @ ep {epoch} (best ep {best_epoch})', flush=True)
            break

    def _dump_preds(loader, split_name: str, n_split: int) -> Path:
        bundle = predict_loader(model, loader, device)
        path = out_dir / f'{split_name}_preds_fold{fold}.npz'
        meta = {
            'model': 'cnn_phasec',
            'csv': Path(csv_path).name,
            'cv_fold': str(fold),
            'split': split_name,
            'best_epoch': best_epoch,
            'use_tabular': args.use_tabular,
            'input_geometry': args.input_geometry,
            'early_stop_metric': args.early_stop_metric,
            'n_train': len(train_loader.dataset),
            'n_val': len(val_loader.dataset),
            'n_test': len(test_loader.dataset),
            'n_split': n_split,
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
        print(f'  [fold {fold}] {split_name} 예측 저장: {path} (n={len(bundle["pred"])})', flush=True)
        return path

    val_pred_path = None
    test_pred_path = None
    if best_epoch < 0:
        best_val_mae = None
        best_val_mse = None
        print(f'  [fold {fold}] 유효한 체크포인트 없음', flush=True)
    else:
        ckpt = out_dir / f'best_fold{fold}.pt'
        model.load_state_dict(torch.load(ckpt, map_location=device))
        # Held-out set은 학습·epoch 선택 중 열람하지 않고 최종 checkpoint에서 1회 평가한다.
        state_before_held_out = _rng_state()
        best_test_mse, best_test_mae = evaluate_metrics(model, test_loader, device)
        _restore_rng_state(state_before_held_out)
        print(
            f'  [fold {fold}] final held-out evaluation: '
            f'rmse {np.sqrt(best_test_mse):.3f} | mae {best_test_mae:.3f}',
            flush=True,
        )
        if getattr(args, 'dump_val_preds', False):
            val_pred_path = _dump_preds(val_loader, 'val', len(val_loader.dataset))
        if getattr(args, 'dump_test_preds', False):
            state_before_dump = _rng_state()
            test_pred_path = _dump_preds(test_loader, 'test', len(test_loader.dataset))
            _restore_rng_state(state_before_dump)

    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'early_stop_metric': args.early_stop_metric,
        'best_val_mse': best_val_mse if best_epoch >= 0 else None,
        'best_val_rmse': float(np.sqrt(best_val_mse)) if best_epoch >= 0 else None,
        'best_val_mae': best_val_mae if best_epoch >= 0 else None,
        'best_test_mse': best_test_mse,
        'best_test_rmse': float(np.sqrt(best_test_mse)) if best_test_mse is not None else None,
        'best_test_mae': best_test_mae,
        'history': history,
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'n_test': len(test_loader.dataset),
        'val_pred_path': str(val_pred_path) if val_pred_path else None,
        'test_pred_path': str(test_pred_path) if test_pred_path else None,
    }


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    parser.add_argument('--use_tabular', action='store_true',
                        help='tabular 브랜치 추가 (OCT+RNFL quadrant/clock-hour)')
    parser.add_argument('--use_deviation', action='store_true',
                        help='deviation map도 이미지 입력에 포함 (4장 concat)')
    parser.add_argument('--input_geometry', choices=['legacy', 'paper'], default='legacy',
                        help='legacy: 224->concat->299, paper: 161x161->concat->161x322')
    parser.add_argument('--backbone', default='inception_v3', choices=BACKBONE_CHOICES,
                        help='CNN 백본 (Phase C image-only)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='백본 동결 (head만 학습)')
    parser.add_argument('--n_output', type=int, default=N_VF,
                        help='VF 출력 차원 (기본 52: p26/p35 제외)')
    parser.add_argument('--fold', type=int, default=None,
                        help='특정 fold만 (생략 시 5-fold 전체)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='RMSprop momentum (optimizer=rmsprop에서 사용)')
    parser.add_argument('--patience', type=int, default=15,
                        help='early stopping patience')
    parser.add_argument('--early_stop_metric', choices=['mae', 'mse'], default='mae',
                        help='early stop 기준: mae(Phase C 기본) | mse(repro/train_paper_track 동일)')
    parser.add_argument('--flip_os_images', action='store_true',
                        help='OS 이미지를 좌우반전해 OD 프레임으로 정규화 (라벨/tabular는 '
                             '*_flip.csv에서 이미 정규화됨). 기본 False = 기존 결과 재현')
    parser.add_argument('--target_floor', type=float, default=None,
                        help='라벨(타깃)만 이 값 이상으로 클램프. Hasan 2025 normalised TS '
                             '재현용(=14). 입력은 손대지 않음. 기본 None = 기존 결과 재현. '
                             '주의: 이 축은 §5.3 민감도 전용이며 헤드라인이 아님')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='runs/default',
                        help='체크포인트 / 결과 저장 경로')
    parser.add_argument('--allow_existing_out_dir', action='store_true',
                        help='기존 checkpoint/prediction/results 덮어쓰기 허용')
    parser.add_argument('--dump_val_preds', action='store_true',
                        help='best epoch 가중치로 val fold 예측 npz 저장 (fusion OOF)')
    parser.add_argument('--dump_test_preds', action='store_true',
                        help='best epoch 가중치로 holdout test 예측 npz 저장 (fold별, 이후 평균)')
    args = parser.parse_args()

    if args.input_geometry == 'paper':
        # 논문 정렬 모드: 각 맵 161x161으로 맞춘 뒤 thickness 2장 concat -> 161x322
        args.image_h, args.image_w = 161, 161
        args.model_h, args.model_w = 161, 322
    else:
        # 기존 모드: 각 맵 224x224, concat 후 299x299
        args.image_h, args.image_w = 224, 224
        args.model_h, args.model_w = 299, 299

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_artifacts = (
        list(out_dir.glob('best_fold*.pt'))
        + list(out_dir.glob('val_preds_fold*.npz'))
        + list(out_dir.glob('test_preds_fold*.npz'))
        + ([out_dir / 'results.json'] if (out_dir / 'results.json').exists() else [])
    )
    if existing_artifacts and not args.allow_existing_out_dir:
        raise SystemExit(
            f'{out_dir}에 기존 학습 산출물 {len(existing_artifacts)}개가 있어 중단. '
            '새 --out_dir를 사용하거나 의도한 경우 --allow_existing_out_dir를 지정하라.'
        )

    print('=== 설정 ===', flush=True)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    print(f'device: {"cuda" if torch.cuda.is_available() else "cpu"}', flush=True)

    csv_path = str(ROOT / args.csv)
    image_index = build_image_index()

    folds = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]
    results = []
    for fold in folds:
        print(f'\n=== Fold {fold} 시작 ===', flush=True)
        result = train_fold(csv_path, image_index, fold, args, out_dir)
        results.append(result)
        if result['best_epoch'] >= 0:
            print(f'  fold {fold} 완료: best ep {result["best_epoch"]}, '
                  f'val_rmse {result["best_val_rmse"]:.3f}, val_mae {result["best_val_mae"]:.3f}, '
                  f'test_rmse {result["best_test_rmse"]:.3f}, test_mae {result["best_test_mae"]:.3f}',
                  flush=True)
        else:
            print(f'  fold {fold} 실패: 유효 epoch 없음', flush=True)

    # CV 집계
    ok = [r for r in results if r.get('best_epoch', -1) >= 0]
    if len(ok) >= 1:
        val_maes  = np.array([r['best_val_mae']  for r in ok])
        test_maes = np.array([r['best_test_mae'] for r in ok])
        print(f'\n=== 5-fold CV 결과 ===', flush=True)
        print(f'  Val MAE  : {val_maes.mean():.3f} ± {val_maes.std():.3f}', flush=True)
        print(f'  Test MAE : {test_maes.mean():.3f} ± {test_maes.std():.3f}', flush=True)

    # 결과 저장
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'results': results}, f, ensure_ascii=False, indent=2)
    print(f'\n결과 저장: {out_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
