#!/usr/bin/env python3
"""백본별 image-only OOF / late fusion / MD 층별 RMSE 비교."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import (  # noqa: E402
    align_oof,
    load_oof_npz,
    masked_overall_rmse_mae,
    optimal_fusion_rmse,
)

COHORT_MD = ROOT / 'cohort_md.csv'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
BACKBONES = {
    'inception_v3': ROOT / 'runs/phasec_b0_clip_mse_5fold',
    'inception_resnet_v2': ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold',
    'vgg16': ROOT / 'runs/phasec_b0_vgg16_5fold',
}
OUT_JSON = ROOT / 'runs/backbone_fusion_strata_compare.json'

STRATA = [
    ('normal', lambda md: md > -3),
    ('early', lambda md: -6 < md <= -3),
    ('moderate', lambda md: -12 < md <= -6),
    ('advanced', lambda md: md <= -12),
]


def norm_key(pid, eye, vf_date) -> Tuple[str, str, str]:
    return (
        str(pid).strip(),
        str(eye).strip().upper(),
        str(vf_date).strip().replace('-', '').replace('/', ''),
    )


def load_cohort_md(path: Path) -> Dict[Tuple[str, str, str], float]:
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        k = norm_key(r['patient_id'], r['eye'], r['vf_date'])
        try:
            out[k] = float(r['MD'])
        except (ValueError, KeyError, TypeError):
            out[k] = float('nan')
    return out


def stack_cnn(cnn_dir: Path) -> dict:
    keys_all, preds, labels, masks = [], [], [], []
    for k in range(5):
        d = load_oof_npz(cnn_dir / f'val_preds_fold{k}.npz')
        keys_all.extend(d['keys'])
        preds.append(d['pred'])
        labels.append(d['labels'])
        masks.append(d['mask'])
    return {
        'keys': keys_all,
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'meta': {'stacked': str(cnn_dir)},
    }


def fuse(px, pc, w: float) -> np.ndarray:
    return w * px + (1.0 - w) * pc


def stratum_indices(keys, md_map) -> Dict[str, List[int]]:
    by: Dict[str, List[int]] = {s[0]: [] for s in STRATA}
    for i, k in enumerate(keys):
        md = md_map.get(norm_key(*k), float('nan'))
        if not np.isfinite(md):
            continue
        for name, fn in STRATA:
            if fn(md):
                by[name].append(i)
                break
    return by


def rmse_at_indices(pred, labels, mask, idx: List[int]) -> float:
    if not idx:
        return float('nan')
    ii = np.array(idx)
    r, _ = masked_overall_rmse_mae(pred[ii], labels[ii], mask[ii])
    return r


def analyze_backbone(name: str, cnn_dir: Path, xgb: dict, md_map: dict) -> dict:
    cnn = stack_cnn(cnn_dir)
    ax, bx, common = align_oof(xgb, cnn)
    mask = ax['mask'] & bx['mask']
    labels = ax['labels']
    px, pc = ax['pred'], bx['pred']

    cnn_rmse, _ = masked_overall_rmse_mae(pc, labels, mask)
    xgb_rmse, _ = masked_overall_rmse_mae(px, labels, mask)
    fus_rmse, w_opt = optimal_fusion_rmse(px, pc, labels, mask)
    pf = fuse(px, pc, w_opt)

    by = stratum_indices(common, md_map)
    strata = {}
    for sname, idx in by.items():
        strata[sname] = {
            'n_eyes': len(idx),
            'cnn': rmse_at_indices(pc, labels, mask, idx),
            'xgb': rmse_at_indices(px, labels, mask, idx),
            'fusion': rmse_at_indices(pf, labels, mask, idx),
        }

    return {
        'cnn_dir': str(cnn_dir.relative_to(ROOT)),
        'oof_cnn_rmse': cnn_rmse,
        'oof_xgb_rmse': xgb_rmse,
        'w_xgb_opt': w_opt,
        'w_cnn_opt': 1.0 - w_opt,
        'oof_fusion_rmse': fus_rmse,
        'strata': strata,
    }


def main():
    xgb_paths = {k: XGB_VAL[k] for k in range(5)}
    keys_all, preds, labels, masks = [], [], [], []
    for k in sorted(xgb_paths):
        d = load_oof_npz(xgb_paths[k])
        keys_all.extend(d['keys'])
        preds.append(d['pred'])
        labels.append(d['labels'])
        masks.append(d['mask'])
    xgb = {
        'keys': keys_all,
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'mask': np.concatenate(masks, axis=0),
        'meta': {'stacked': 'xgb'},
    }
    md_map = load_cohort_md(COHORT_MD)

    results = {}
    for name, cdir in BACKBONES.items():
        results[name] = analyze_backbone(name, cdir, xgb, md_map)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=== 백본별 OOF (val, MD merge n≈235) ===', flush=True)
    print(f'{"backbone":22s} {"CNN":>7s} {"fus":>7s} {"w_xgb":>6s} {"normal":>7s} {"adv CNN":>8s} {"adv fus":>8s}', flush=True)
    print('-' * 72, flush=True)
    for name, r in results.items():
        st = r['strata']
        print(
            f'{name:22s} {r["oof_cnn_rmse"]:7.3f} {r["oof_fusion_rmse"]:7.3f} '
            f'{r["w_xgb_opt"]:6.3f} {st["normal"]["cnn"]:7.3f} '
            f'{st["advanced"]["cnn"]:8.3f} {st["advanced"]["fusion"]:8.3f}',
            flush=True,
        )
    print(f'\n저장: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
