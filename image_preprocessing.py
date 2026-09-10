"""Build the model input tensors from the OCT thickness maps.

This is where the cropping, resizing and side-by-side concatenation of Section
3.2 happen: _crop_gca_thickness, _crop_rnfl_thickness and the Resize transforms
below.

Image Preprocessing Pipeline
OCT 이미지(GCA + RNFL) → 모델 입력 텐서

입력 구성 (eye에 따라 4장, clockhours는 tabular로만 사용):
  OD: gca_od_thickness, gca_od_deviation, rnfl_od_thickness, rnfl_od_deviation
  OS: gca_os_thickness, gca_os_deviation, rnfl_os_thickness, rnfl_os_deviation

사용법:
  from image_preprocessing import build_image_index, VFDataset, get_transforms
"""
import os, csv, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from paths import ROOT, DATA_ROOT, rewrite_data_path  # noqa: F401

# ── 이미지 종류 정의
IMG_SIZE = 224

REGION_KEYS_OD = [
    'gca_od_thickness', 'gca_od_deviation',
    'rnfl_od_thickness', 'rnfl_od_deviation',
]
REGION_KEYS_OS = [
    'gca_os_thickness', 'gca_os_deviation',
    'rnfl_os_thickness', 'rnfl_os_deviation',
]
N_IMGS = 4  # 입력 이미지 수 (clockhours 제외)

PT_ALL = [f'p{i+1:02d}' for i in range(54)]
BLIND_POINTS = frozenset({'p26', 'p35'})  # Humphrey 생리적 맹점 (선행연구 동일)
PT = [p for p in PT_ALL if p not in BLIND_POINTS]  # 52-point 타깃
N_VF = len(PT)

# target eye 기준 tabular (GCA/RNFL OCT 수치 + RNFL quadrant/clock-hour)
RNFL_TABULAR = (
    ['rnfl_q_s', 'rnfl_q_t', 'rnfl_q_i', 'rnfl_q_n']
    + [f'rnfl_h{i:02d}' for i in range(1, 13)]
)
OCT_FEATURES_OD = [
    'avg_gcl_od', 'min_gcl_od',
    'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n',
    'od_avg_rnfl', 'od_vert_cd',
] + RNFL_TABULAR
OCT_FEATURES_OS = [
    'avg_gcl_os', 'min_gcl_os',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'os_avg_rnfl', 'os_vert_cd',
] + RNFL_TABULAR
N_TABULAR = len(OCT_FEATURES_OD)


def tabular_vector(row: Dict, eye: str) -> List[float]:
    """target eye tabular 벡터 (결측은 nan)."""
    feats = OCT_FEATURES_OD if eye == 'OD' else OCT_FEATURES_OS
    out = []
    for f in feats:
        v = row.get(f, '')
        out.append(float(v) if v not in ('', None) else float('nan'))
    return out


def compute_tabular_fill(rows: List[Dict]) -> np.ndarray:
    """train 행 기준 컬럼별 평균 impute (XGB와 동일 원칙)."""
    if not rows:
        return np.zeros(N_TABULAR, dtype=np.float32)
    arr = np.array(
        [tabular_vector(r, r['eye']) for r in rows],
        dtype=np.float32,
    )
    col_means = np.nanmean(arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    return col_means.astype(np.float32)


def compute_tabular_stats(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """train 행 기준 컬럼별 (평균, 표준편차).

    CNN multimodal의 tabular z-score 정규화용. raw OCT 값은 스케일 편차가 커
    (vert_cd ~0.6 vs rnfl_h06 ~105) backbone feature와 동일 스케일로 맞추지 않으면
    tabular branch가 사실상 학습되지 않는다. 결측 대치 평균과 동일한 평균을 반환해
    (impute 후 표준화) 일관성을 유지한다. std==0/nan 컬럼은 1.0으로 둔다.
    """
    if not rows:
        return (np.zeros(N_TABULAR, dtype=np.float32),
                np.ones(N_TABULAR, dtype=np.float32))
    arr = np.array(
        [tabular_vector(r, r['eye']) for r in rows],
        dtype=np.float32,
    )
    mean = np.nanmean(arr, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.nanstd(arr, axis=0)
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ── 텍스트/캡션 제거 crop (deviation맵·고정크기 이미지용)
# GCA/RNFL thickness맵은 아래 자동검출 함수 사용.
TEXT_CROP = {
    'gca_od_deviation':  ((323, 372), (0,  43, 323, 367)),
    'rnfl_od_deviation': ((310, 360), (0,  24, 310, 334)),
    'gca_os_deviation':  ((324, 372), (0,  43, 324, 367)),
    'rnfl_os_deviation': ((310, 360), (0,  24, 310, 334)),
}


def _crop_gca_thickness(img: Image.Image) -> Image.Image:
    """GCA thickness맵: 파란 경계선 밀도 기반으로 컬러 히트맵 영역 자동검출."""
    arr = np.array(img.convert('RGB')).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue = (b > r + 40) & (b > g + 40) & (b > 80)
    row_d = blue.sum(axis=1)
    col_d = blue.sum(axis=0)
    rows = np.where(row_d > 30)[0]
    cols = np.where(col_d > 30)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img  # 검출 실패 시 원본 반환
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return img.crop(box)


def _crop_rnfl_thickness(img: Image.Image, top_margin: int = 16) -> Image.Image:
    """RNFL thickness맵: 정사각형 역산으로 좌측 컬러바 제거."""
    W, H = img.size
    sq = H - top_margin
    arr = np.array(img.convert('RGB')).astype(float)
    max_c = np.maximum(np.maximum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
    min_c = np.minimum(np.minimum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
    sat = (max_c - min_c) / (max_c + 1e-5)
    colored = (sat > 0.2) & (max_c > 30)
    cols = np.where(colored.any(axis=0))[0]
    if len(cols) == 0:
        return img  # 검출 실패 시 원본 반환
    right = int(cols[-1]) + 1
    left = max(0, right - sq)
    return img.crop((left, top_margin, right, H))


# ─────────────────────────────────────────────
# 1. 이미지 경로 인덱스 빌드
# ─────────────────────────────────────────────
def build_image_index(ml_dataset_path: str = None) -> Dict[Tuple, Dict]:
    """
    ml_dataset.csv 기반으로 (patient_id, eye, vf_date) → 이미지경로 dict 생성
    반환: {(pid, eye, vf_date): {region_key: filepath, ...}}
    """
    if ml_dataset_path is None:
        ml_dataset_path = str(ROOT / 'ml_dataset.csv')

    ml = list(csv.DictReader(open(ml_dataset_path, encoding='utf-8-sig')))
    index = {}

    _FILE_MAP = {
        'gca_od_thickness': ('gca_dir', 'od_thickness_map.png'),
        'gca_os_thickness': ('gca_dir', 'os_thickness_map.png'),
        'gca_od_deviation': ('gca_dir', 'od_deviation_map.png'),
        'gca_os_deviation': ('gca_dir', 'os_deviation_map.png'),
        'rnfl_od_thickness': ('rnfl_dir', 'od_thickness_map.png'),
        'rnfl_os_thickness': ('rnfl_dir', 'os_thickness_map.png'),
        'rnfl_od_deviation': ('rnfl_dir', 'od_deviation_map.png'),
        'rnfl_os_deviation': ('rnfl_dir', 'os_deviation_map.png'),
        'rnfl_clockhours':   ('rnfl_dir', 'clockhours.png'),
    }

    for row in ml:
        key = (row['patient_id'], row['eye'], row['vf_date'])
        paths = {}
        for rk, (base_col, fname) in _FILE_MAP.items():
            # CSV의 경로는 데이터 생성 머신의 절대경로다. DATA_ROOT 기준으로 바꾼다.
            base = rewrite_data_path(row.get(base_col, ''))
            p = os.path.join(base, fname) if base else ''
            if p and os.path.exists(p):
                paths[rk] = p
        index[key] = paths

    return index


# ─────────────────────────────────────────────
# 2. Transform 정의
# ─────────────────────────────────────────────
def get_transforms(
    mode: str = 'train',
    image_size: Union[int, Tuple[int, int]] = IMG_SIZE,
) -> transforms.Compose:
    """
    mode: 'train' | 'val' | 'test'
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # 주의: 4장 이미지가 같은 해부학 위치를 가리키므로 공간을 어긋나게 만드는
    # augmentation (RandomCrop, RandomHorizontalFlip 등)은 금지. 강도 약한 ColorJitter만 허용.
    if isinstance(image_size, int):
        resize_size = (image_size, image_size)
    else:
        resize_size = image_size

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class VFDataset(Dataset):
    """
    OCT 이미지 + 정형 OCT 값 → VF 54점 예측 Dataset

    Args:
        rows: ml_final_*.csv 행 리스트
        image_index: build_image_index() 결과
        transform: torchvision transform
        mode: 'train' | 'val' | 'test'  (augmentation 제어용)
    """
    def __init__(
        self,
        rows: List[Dict],
        image_index: Dict,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'val',
        tabular_fill: Optional[np.ndarray] = None,
        tabular_std: Optional[np.ndarray] = None,
        flip_os_images: bool = False,
        target_floor: Optional[float] = None,
    ):
        self.rows = rows
        self.image_index = image_index
        self.transform = transform or get_transforms(mode)
        self.tabular_fill = tabular_fill
        self.tabular_std = tabular_std
        # OS 이미지를 좌우반전해 OD 프레임으로 정규화할지 (기본 False = 기존 동작).
        # 라벨/tabular는 *_flip.csv에서 이미 OD 프레임으로 정규화돼 있으나 이미지는
        # 촬영 프레임 그대로였다. 이 옵션은 그 불일치를 제거한다.
        self.flip_os_images = flip_os_images
        # 라벨(타깃)만 floor로 클램프. None = 기존 동작(기존 전 결과 재현성 보존).
        # Hasan 2025는 14 dB 미만 타깃을 14로 올린 좌표계에서 학습했다.
        self.target_floor = target_floor
        self._build_valid_index()

    def _build_valid_index(self):
        """이미지가 완전히 있는 행만 유효 인덱스로"""
        self.valid_idx = []
        self.skip_log = []
        for i, r in enumerate(self.rows):
            key = (r['patient_id'], r['eye'], r['vf_date'])
            paths = self.image_index.get(key, {})
            region_keys = REGION_KEYS_OD if r['eye'] == 'OD' else REGION_KEYS_OS
            missing = [rk for rk in region_keys if rk not in paths]
            if missing:
                self.skip_log.append({
                    'patient_id': r['patient_id'], 'eye': r['eye'],
                    'vf_date': r['vf_date'], 'missing_regions': missing,
                })
            else:
                self.valid_idx.append(i)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[self.valid_idx[idx]]
        eye = row['eye']
        key = (row['patient_id'], eye, row['vf_date'])
        paths = self.image_index[key]
        region_keys = REGION_KEYS_OD if eye == 'OD' else REGION_KEYS_OS

        # 이미지 로드: 4장을 각각 [3, H, W]로 변환한 뒤 [4, 3, H, W]로 적층.
        imgs = []
        for rk in region_keys:
            img = Image.open(paths[rk]).convert('RGB')
            if 'thickness' in rk and 'gca' in rk:
                img = _crop_gca_thickness(img)
            elif 'thickness' in rk and 'rnfl' in rk:
                img = _crop_rnfl_thickness(img)
            else:
                spec = TEXT_CROP.get(rk)
                if spec is not None and img.size == spec[0]:
                    img = img.crop(spec[1])
            # ★ 반드시 crop 이후 / transform 이전.
            # _crop_gca_thickness·_crop_rnfl_thickness·TEXT_CROP 좌표는 촬영 방향
            # 원본 기준으로 캘리브레이션돼 있어, 먼저 뒤집으면 crop이 깨진다.
            if self.flip_os_images and eye == 'OS':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.transform:
                img = self.transform(img)   # [3, H, W]
            imgs.append(img)
        images = torch.stack(imgs, dim=0)   # [4, 3, H, W]

        # 정형 tabular (결측 → train 평균 impute, 이후 train 통계로 z-score 정규화)
        tab = np.array(tabular_vector(row, eye), dtype=np.float32)
        if self.tabular_fill is not None:
            nan_mask = np.isnan(tab)
            tab[nan_mask] = self.tabular_fill[nan_mask]
        if self.tabular_std is not None:
            # tabular_fill(=train 평균) 기준 표준화: impute된 결측은 0이 됨
            tab = (tab - self.tabular_fill) / self.tabular_std
        tabular = torch.from_numpy(tab)

        # 라벨 + 마스크
        labels = []
        mask = []
        for p in PT:
            v = row.get(p, '')
            if v == '' or v is None:
                labels.append(0.0); mask.append(0)
            elif v == '-1':
                labels.append(0.0); mask.append(0)  # blind spot 제외
            else:
                try:
                    fv = float(v)
                    # target_floor: Hasan 2025 'normalised TS' 재현용.
                    # None(기본) = 기존 동작. 타깃만 올리고 입력은 손대지 않는다.
                    if self.target_floor is not None and fv < self.target_floor:
                        fv = self.target_floor
                    labels.append(fv); mask.append(1)
                except:
                    labels.append(0.0); mask.append(0)

        return {
            'images':  images,                                  # [5, 3, 224, 224]
            'tabular': tabular,                                 # [N_TABULAR]
            'labels':  torch.tensor(labels, dtype=torch.float32),  # [52]
            'mask':    torch.tensor(mask,   dtype=torch.bool),     # [52]
            'patient_id': row['patient_id'],
            'eye': eye,
            'vf_date': row['vf_date'],
        }


# ─────────────────────────────────────────────
# 4. DataLoader 빌드 헬퍼
# ─────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    image_index: Dict,
    fold: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: Union[int, Tuple[int, int]] = IMG_SIZE,
    flip_os_images: bool = False,
    target_floor: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    cv_fold 컬럼 기준으로 train/val/test DataLoader 반환
    fold: 0-4 중 어느 fold를 val로 쓸지
    flip_os_images: OS 이미지를 좌우반전(OD 정규화). 기본 False = 기존 동작 보존.
    target_floor: 라벨(타깃)만 이 값으로 클램프. None = 기존 동작 보존.
                  Hasan 2025 normalised TS 재현용(=14). 입력은 손대지 않는다.
    """
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))

    train_rows = [r for r in rows if r['cv_fold'] not in (str(fold), 'test')]
    val_rows   = [r for r in rows if r['cv_fold'] == str(fold)]
    test_rows  = [r for r in rows if r['cv_fold'] == 'test']

    tabular_fill, tabular_std = compute_tabular_stats(train_rows)

    train_ds = VFDataset(
        train_rows, image_index, get_transforms('train', image_size), 'train', tabular_fill, tabular_std,
        flip_os_images=flip_os_images, target_floor=target_floor,
    )
    val_ds   = VFDataset(
        val_rows, image_index, get_transforms('val', image_size), 'val', tabular_fill, tabular_std,
        flip_os_images=flip_os_images, target_floor=target_floor,
    )
    test_ds  = VFDataset(
        test_rows, image_index, get_transforms('test', image_size), 'test', tabular_fill, tabular_std,
        flip_os_images=flip_os_images, target_floor=target_floor,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 5. 단독 실행: 정합성 리포트
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=== 이미지 경로 인덱스 빌드 ===')
    idx = build_image_index()
    print(f'인덱스 키 수: {len(idx)}')

    for tag, path in [
        ('90d',  str(ROOT / 'ml_final_90d.csv')),
        ('180d', str(ROOT / 'ml_final_180d.csv')),
    ]:
        print(f'\n=== {tag} 정합성 검증 ===')
        rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
        ds = VFDataset(rows, idx, get_transforms('val'), 'val')

        print(f'  전체 행: {len(rows)}')
        print(f'  이미지 완전 OK: {len(ds.valid_idx)} ({len(ds.valid_idx)/len(rows)*100:.1f}%)')
        print(f'  이미지 누락 행: {len(ds.skip_log)}')

        if ds.skip_log:
            for s in ds.skip_log[:5]:
                print(f'    {s["patient_id"]} {s["eye"]} {s["vf_date"]}: 누락={s["missing_regions"]}')

        # 샘플 1개 로드 테스트
        if len(ds) > 0:
            sample = ds[0]
            print(f'\n  샘플 로드 테스트:')
            print(f'    images:  {sample["images"].shape}   (4장 × 3ch × {IMG_SIZE}×{IMG_SIZE})')
            print(f'    tabular: {sample["tabular"].shape}  (OCT 요약지표 {N_TABULAR}개, target eye)')
            print(f'    labels:  {sample["labels"].shape}  valid={sample["mask"].sum().item()}개')
            print(f'    patient: {sample["patient_id"]} {sample["eye"]}')

        # fold=0 기준 DataLoader 테스트
        tl, vl, tel = build_dataloaders(path, idx, fold=0, batch_size=4)
        tb = next(iter(tl))
        print(f'\n  DataLoader 배치 테스트 (fold=0, bs=4):')
        print(f'    train: {len(tl.dataset)}개 / val: {len(vl.dataset)}개 / test: {len(tel.dataset)}개')
        print(f'    배치 images: {tb["images"].shape}')
        print(f'    배치 tabular: {tb["tabular"].shape}')
        print(f'    배치 labels: {tb["labels"].shape}')
        print(f'    배치 mask 유효점 합: {tb["mask"].sum().item()}개')
