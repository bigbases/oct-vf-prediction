"""Late-fusion OOF 예측 공통: 행 키 정렬, masked metric, 잔차 상관."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROW_KEY_FIELDS = ('patient_id', 'eye', 'vf_date')


def row_key(row: Dict) -> Tuple[str, str, str]:
    return tuple(row[f] for f in ROW_KEY_FIELDS)


def keys_to_arrays(keys: Sequence[Tuple[str, str, str]]):
    return {
        'patient_id': np.array([k[0] for k in keys], dtype=object),
        'eye': np.array([k[1] for k in keys], dtype=object),
        'vf_date': np.array([k[2] for k in keys], dtype=object),
    }


def save_oof_npz(
    path: Path,
    keys: Sequence[Tuple[str, str, str]],
    pred: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    meta: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = keys_to_arrays(keys)
    np.savez_compressed(
        path,
        pred=pred.astype(np.float32),
        labels=labels.astype(np.float32),
        mask=mask.astype(bool),
        meta_json=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object),
        **kw,
    )


def load_oof_npz(path: Path) -> Dict:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta_json'][0]))
    keys = list(zip(z['patient_id'], z['eye'], z['vf_date']))
    return {
        'keys': keys,
        'pred': z['pred'],
        'labels': z['labels'],
        'mask': z['mask'],
        'meta': meta,
    }


def align_oof(a: Dict, b: Dict) -> Tuple[Dict, Dict, List[Tuple[str, str, str]]]:
    """(patient_id, eye, vf_date) 교집합만 유지, 동일 순서."""
    ia = {k: i for i, k in enumerate(a['keys'])}
    ib = {k: i for i, k in enumerate(b['keys'])}
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError('교집합 행이 0입니다.')
    ai = [ia[k] for k in common]
    bi = [ib[k] for k in common]
    pick = lambda d, idx: {
        'keys': common,
        'pred': d['pred'][idx],
        'labels': d['labels'][idx],
        'mask': d['mask'][idx],
        'meta': d['meta'],
    }
    return pick(a, ai), pick(b, bi), common


def masked_overall_rmse_mae(pred, labels, mask) -> Tuple[float, float]:
    m = mask.astype(bool)
    diff = pred - labels
    diff[~m] = np.nan
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    return rmse, mae


def residual_pearson_rho(pred_a, pred_b, labels, mask) -> Tuple[float, int]:
    """잔차 e_a=y-pred_a, e_b=y-pred_b 의 Pearson ρ (masked pooled)."""
    m = mask.astype(bool)
    ea = (labels - pred_a)[m]
    eb = (labels - pred_b)[m]
    n = int(ea.size)
    if n < 3:
        return float('nan'), n
    ea = ea.astype(np.float64)
    eb = eb.astype(np.float64)
    ea -= ea.mean()
    eb -= eb.mean()
    denom = np.sqrt((ea ** 2).sum() * (eb ** 2).sum())
    if denom < 1e-12:
        return float('nan'), n
    return float((ea * eb).sum() / denom), n


def optimal_fusion_rmse(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_grid: int = 101,
) -> Tuple[float, float]:
    """val OOF에서 w·pred_a + (1-w)·pred_b RMSE 최소 (probe 게이트용)."""
    m = mask.astype(bool)
    ya = pred_a[m].astype(np.float64)
    yb = pred_b[m].astype(np.float64)
    y = labels[m].astype(np.float64)
    if ya.size < 3:
        return float('nan'), float('nan')
    best_w, best_rmse = 0.0, float('inf')
    for w in np.linspace(0.0, 1.0, n_grid):
        fused = w * ya + (1.0 - w) * yb
        rmse = float(np.sqrt(np.mean((fused - y) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, float(w)
    return best_rmse, best_w


def fusion_ceiling_note(rmse_xgb: float, rmse_cnn: float, rho: float) -> str:
    """ρ < σ_XGB/σ_CNN 이면 CNN 비중 > 0 가능 (단순 2-model 가정)."""
    thresh = rmse_xgb / rmse_cnn if rmse_cnn > 0 else float('nan')
    if not np.isfinite(rho) or not np.isfinite(thresh):
        return 'ρ 또는 RMSE 비교 불가'
    if rho >= thresh:
        return f'ρ={rho:.3f} ≥ σ_XGB/σ_CNN≈{thresh:.3f} → 융합 ≈ XGB-alone (CNN 비중→0)'
    return f'ρ={rho:.3f} < σ_XGB/σ_CNN≈{thresh:.3f} → 융합 여지 있음 (5-fold GO 검토)'
