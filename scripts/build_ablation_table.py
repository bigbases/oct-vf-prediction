#!/usr/bin/env python3
"""IR-v2 late fusion ablation 표 — 기존 산출물만 집계 (재학습 없음).

출력:
  runs/ablation_ir_v2_baseline.json
  runs/ablation_ir_v2_baseline.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _rnd(x: float, d: int = 2) -> float:
    return round(float(x), d)


def _mean_std(vals: list[float]) -> dict:
    a = np.array(vals, dtype=np.float64)
    return {'mean': _rnd(a.mean()), 'std': _rnd(a.std())}


def main():
    ig = json.loads((ROOT / 'runs/ir_v2_global_fusion.json').read_text())
    ft = json.loads((ROOT / 'runs/fusion_5fold_test.json').read_text())
    bx = json.loads((ROOT / 'baseline_xgb_results.json').read_text())['90d_excl_empty_flip']
    bb = json.loads((ROOT / 'runs/backbone_final_compare.json').read_text())
    ir = bb['backbones']['inception_resnet_v2']

    w = ig['w_xgb_global']
    oof = ig['oof']

    # test: fusion_5fold_test uses aligned n=37 (consistent across rows)
    pf = ft['per_fold']
    test_xgb_rmse = _mean_std([f['xgb']['rmse'] for f in pf])
    test_xgb_mae = _mean_std([f['xgb']['mae'] for f in pf])
    test_cnn_rmse = _mean_std([f['cnn']['rmse'] for f in pf])
    test_cnn_mae = _mean_std([f['cnn']['mae'] for f in pf])
    test_fus_rmse = _mean_std([f['fusion_w047']['rmse'] for f in pf])
    test_fus_mae = _mean_std([f['fusion_w047']['mae'] for f in pf])

    # CNN test from phasec (same protocol, may differ slightly from aligned dump)
    cnn_test_phasec = {
        'rmse_mean': ir['test_rmse_mean'],
        'rmse_std': ir['test_rmse_std'],
        'per_fold': ir['test_rmse_per_fold'],
        'note': 'Unaligned holdout; fusion_5fold_test CNN uses key-aligned preds',
    }

    rows = [
        {
            'id': 'xgb_tabular',
            'label': 'XGB tabular',
            'modality': 'tabular',
            'oof_rmse': _rnd(oof['xgb_rmse']),
            'oof_mae': _rnd(oof['xgb_mae']),
            'test_rmse': f"{test_xgb_rmse['mean']} ± {test_xgb_rmse['std']}",
            'test_mae': f"{test_xgb_mae['mean']} ± {test_xgb_mae['std']}",
            'test_rmse_mean': test_xgb_rmse['mean'],
            'test_rmse_std': test_xgb_rmse['std'],
            'test_mae_mean': test_xgb_mae['mean'],
            'test_mae_std': test_xgb_mae['std'],
            'source_oof': 'runs/ir_v2_global_fusion.json',
            'source_test': 'runs/fusion_5fold_test.json (aligned n=37)',
        },
        {
            'id': 'cnn_ir_v2',
            'label': 'CNN IR-v2 (image-only)',
            'modality': 'image',
            'oof_rmse': _rnd(oof['cnn_rmse']),
            'oof_mae': _rnd(oof['cnn_mae']),
            'test_rmse': f"{test_cnn_rmse['mean']} ± {test_cnn_rmse['std']}",
            'test_mae': f"{test_cnn_mae['mean']} ± {test_cnn_mae['std']}",
            'test_rmse_mean': test_cnn_rmse['mean'],
            'test_rmse_std': test_cnn_rmse['std'],
            'test_mae_mean': test_cnn_mae['mean'],
            'test_mae_std': test_cnn_mae['std'],
            'source_oof': 'runs/ir_v2_global_fusion.json',
            'source_test': 'runs/fusion_5fold_test.json (aligned n=37)',
        },
        {
            'id': 'late_fusion',
            'label': f'Late fusion (w_xgb={w})',
            'modality': 'tabular+image',
            'oof_rmse': _rnd(oof['fusion_rmse_fixed_w']),
            'oof_mae': _rnd(oof['fusion_mae_fixed_w']),
            'test_rmse': f"{test_fus_rmse['mean']} ± {test_fus_rmse['std']}",
            'test_mae': f"{test_fus_mae['mean']} ± {test_fus_mae['std']}",
            'test_rmse_mean': test_fus_rmse['mean'],
            'test_rmse_std': test_fus_rmse['std'],
            'test_mae_mean': test_fus_mae['mean'],
            'test_mae_std': test_fus_mae['std'],
            'w_xgb': w,
            'w_cnn': ig['w_cnn_global'],
            'source_oof': 'runs/ir_v2_global_fusion.json',
            'source_test': 'runs/fusion_5fold_test.json',
        },
    ]

    for r in rows:
        if r['id'] == 'late_fusion':
            r['rho_oof_xgb_cnn'] = _rnd(oof.get('rho', float('nan')), 4)

    # deltas vs fusion (test rmse mean)
    fus_t = test_fus_rmse['mean']
    for r in rows:
        if r['id'] != 'late_fusion':
            r['delta_test_rmse_vs_fusion'] = _rnd(r['test_rmse_mean'] - fus_t)

    # optional negative controls (fold0 only — different comparison scope)
    mm_path = ROOT / 'runs/phasec_mm_v3_fold0_concat_zscore/results.json'
    v3_io_path = ROOT / 'runs/phasec_b0_clip_mse_5fold/results.json'
    negative = []
    if mm_path.is_file():
        mm = json.loads(mm_path.read_text())['results'][0]
        negative.append({
            'id': 'concat_mm_v3_fold0',
            'label': 'Concat multimodal v3 (fold0, post-ReLU fix)',
            'val_rmse': _rnd(mm['best_val_rmse']),
            'test_rmse': _rnd(mm['best_test_rmse']),
            'test_mae': _rnd(mm['best_test_mae']),
            'source': str(mm_path.relative_to(ROOT)),
            'note': 'End-to-end concat; fold0 only — not 5-fold aggregate',
        })
    if v3_io_path.is_file():
        v3 = json.loads(v3_io_path.read_text())['results'][0]
        negative.append({
            'id': 'cnn_v3_io_fold0',
            'label': 'CNN v3 image-only (fold0)',
            'val_rmse': _rnd(v3['best_val_rmse']),
            'test_rmse': _rnd(v3['best_test_rmse']),
            'test_mae': _rnd(v3['best_test_mae']),
            'source': 'runs/phasec_b0_clip_mse_5fold/results.json fold0',
            'note': 'Reference for concat comparison on same fold',
        })

    out = {
        'title': 'Ablation: XGB / CNN IR-v2 / Late fusion (90d primary)',
        'protocol': {
            'csv': 'ml_final_90d_excl_empty_flip.csv',
            'cv': '5-fold patient-level',
            'cnn': 'Inception-ResNet-v2 image-only, masked MSE, linear head, grad clip 1.0',
            'xgb': '52 independent XGBRegressor, 26 tabular features, no scaling',
            'fusion': f'late fusion w_xgb={w} tuned once on 5-fold val OOF (n=240)',
            'oof_split': '5-fold val OOF stack',
            'test_split': 'holdout test n=37, per-fold models, key-aligned XGB∩CNN',
        },
        'main_table': rows,
        'per_fold_test': [
            {
                'fold': f['fold'],
                'n_aligned': f['n_eyes_aligned'],
                'xgb_rmse': _rnd(f['xgb']['rmse']),
                'cnn_rmse': _rnd(f['cnn']['rmse']),
                'fusion_rmse': _rnd(f['fusion_w047']['rmse']),
            }
            for f in pf
        ],
        'oof_meta': {
            'n_oof_rows': ig['n_oof_rows'],
            'residual_rho_xgb_cnn': _rnd(oof.get('rho', float('nan')), 4),
            'rho_source': 'runs/ir_v2_global_fusion.json',
        },
        'cnn_test_phasec_unaligned': cnn_test_phasec,
        'xgb_test_baseline_xgb': {
            'test_rmse_mean': _rnd(bx['test_rmse_mean']),
            'test_rmse_std': _rnd(bx['test_rmse_std']),
            'note': '5-fold CV protocol; fold_test_rmses mean — may differ from aligned fusion_5fold_test XGB',
        },
        'negative_controls_fold0': negative,
        'sources': [
            'runs/ir_v2_global_fusion.json',
            'runs/fusion_5fold_test.json',
            'runs/backbone_final_compare.json',
            'baseline_xgb_results.json',
        ],
    }

    json_path = ROOT / 'runs/ablation_ir_v2_baseline.json'
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    md_lines = [
        '# Ablation: IR-v2 Late Fusion Baseline (90d)',
        '',
        '**Protocol:** Phase C · 5-fold CV · `ml_final_90d_excl_empty_flip.csv`',
        '',
        '| Model | OOF RMSE | OOF MAE | Test RMSE (5-fold, aligned) | Test MAE | Δ test RMSE vs fusion |',
        '|-------|----------|---------|----------------------------|----------|----------------------|',
    ]
    for r in rows:
        delta = r.get('delta_test_rmse_vs_fusion', '—')
        if r['id'] == 'late_fusion':
            delta = '—'
        md_lines.append(
            f"| {r['label']} | {r['oof_rmse']} | {r['oof_mae']} | {r['test_rmse']} | {r['test_mae']} | {delta} |"
        )
    md_lines += [
        '',
        f'- Fusion weight: **w_xgb = {w}** (OOF grid once, n=240 eyes)',
        f'- Test: holdout **n=37**, 5 fold-specific models, same w',
        '',
        '### Per-fold test RMSE (aligned)',
        '',
        '| fold | XGB | CNN | Fusion |',
        '|------|-----|-----|--------|',
    ]
    for pf_row in out['per_fold_test']:
        md_lines.append(
            f"| {pf_row['fold']} | {pf_row['xgb_rmse']} | {pf_row['cnn_rmse']} | {pf_row['fusion_rmse']} |"
        )
    if negative:
        md_lines += [
            '',
            '### Negative controls (fold0 only — not in main 5-fold aggregate)',
            '',
            '| Model | val RMSE | test RMSE | test MAE |',
            '|-------|----------|-----------|----------|',
        ]
        for n in negative:
            md_lines.append(
                f"| {n['label']} | {n.get('val_rmse', '—')} | {n['test_rmse']} | {n['test_mae']} |"
            )

    md_path = ROOT / 'runs/ablation_ir_v2_baseline.md'
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    print(json.dumps(out['main_table'], ensure_ascii=False, indent=2), flush=True)
    print(f'\n저장: {json_path}', flush=True)
    print(f'저장: {md_path}', flush=True)


if __name__ == '__main__':
    main()
