"""Summary branch: 52 independent XGBoost regressors on the 26 OCT summary
parameters, five-fold patient-level cross-validation, blind-spot points
excluded, left eyes mapped to right-eye coordinates.

Tabular XGBoost Baseline: target-eye OCT + RNFL quadrant/clock-hour → VF 52점
- 5-fold patient-level CV (cv_fold)
- p26/p35 제외 (선행연구 동일)
- flip CSV 기준 (OS→OD 좌표계)

사용법:
  HVF_ROOT="$(pwd)" python -u baseline_xgb.py --device cuda
  python -u baseline_xgb.py --only 90d_excl_empty_flip --device cpu
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))

PT_ALL = [f'p{i+1:02d}' for i in range(54)]
PT = [p for p in PT_ALL if p not in ('p26', 'p35')]
N_PT = len(PT)

RNFL_TABULAR = (
    ['rnfl_q_s', 'rnfl_q_t', 'rnfl_q_i', 'rnfl_q_n']
    + [f'rnfl_h{i:02d}' for i in range(1, 13)]
)
FEATURES_OD = [
    'avg_gcl_od', 'min_gcl_od',
    'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n',
    'od_avg_rnfl', 'od_vert_cd',
] + RNFL_TABULAR
FEATURES_OS = [
    'avg_gcl_os', 'min_gcl_os',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'os_avg_rnfl', 'os_vert_cd',
] + RNFL_TABULAR


def _feat_list(eye: str):
    return FEATURES_OD if eye == 'OD' else FEATURES_OS


def _xgb_kwargs(device: str) -> dict:
    kw = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    if device == 'cuda':
        kw['device'] = 'cuda'
    return kw


def resolve_device(requested: str) -> str:
    """CPU-only XGBoost build를 CUDA로 잘못 기록하지 않도록 실제 장치를 확정한다."""
    if requested == 'cuda' and not bool(xgb.build_info().get('USE_CUDA', False)):
        print('[warning] XGBoost가 CPU-only build라 --device cpu로 대체', flush=True)
        return 'cpu'
    return requested


def load_dataset(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    X, Y, masks, folds = [], [], [], []
    for r in rows:
        feats = []
        for f in _feat_list(r['eye']):
            v = r.get(f, '')
            feats.append(float(v) if v not in ('', None) else np.nan)

        y_row, mask_row = [], []
        for p in PT:
            v = r.get(p, '')
            if v == '' or v is None:
                y_row.append(np.nan)
                mask_row.append(False)
            elif v == '-1':
                y_row.append(-1.0)
                mask_row.append(False)
            else:
                try:
                    y_row.append(float(v))
                    mask_row.append(True)
                except ValueError:
                    y_row.append(np.nan)
                    mask_row.append(False)

        X.append(feats)
        Y.append(y_row)
        masks.append(mask_row)
        folds.append(r.get('cv_fold', '0'))

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(masks, dtype=bool),
        folds,
    )


def masked_mae(pred, true, mask):
    diff = np.abs(pred - true)
    diff[~mask] = np.nan
    per_point = np.nanmean(diff, axis=0)
    overall = np.nanmean(diff)
    return per_point, overall


def masked_rmse(pred, true, mask):
    diff2 = (pred - true) ** 2
    diff2[~mask] = np.nan
    per_point = np.sqrt(np.nanmean(diff2, axis=0))
    overall = float(np.sqrt(np.nanmean(diff2)))
    return per_point, overall


def _fit_points(X_tr, Y_tr, M_tr, X_pred, device, label):
    preds = np.full((len(X_pred), N_PT), np.nan)
    xgb_kw = _xgb_kwargs(device)
    for pi in range(N_PT):
        valid_tr = M_tr[:, pi]
        if valid_tr.sum() < 10:
            continue
        if pi % 13 == 0:
            print(f'    {label} point {pi + 1}/{N_PT} ({PT[pi]})', flush=True)
        model = XGBRegressor(**xgb_kw)
        model.fit(X_tr[valid_tr], Y_tr[valid_tr, pi])
        preds[:, pi] = model.predict(X_pred)
    return preds


def run_cv(path, tag, device: str):
    print(f'\n{"=" * 60}', flush=True)
    print(f'[{tag}] 5-fold CV  device={device}  features={len(FEATURES_OD)}  targets={N_PT}', flush=True)
    print(f'{"=" * 60}', flush=True)

    X, Y, masks, folds = load_dataset(path)
    fold_ids = sorted(set(folds) - {'test'})
    fold_maes, fold_rmses = [], []
    fold_test_maes, fold_test_rmses = [], []

    te_idx = [i for i, f in enumerate(folds) if f == 'test']
    X_te, Y_te, M_te = X[te_idx], Y[te_idx], masks[te_idx]

    for val_fold in fold_ids:
        print(f'  CV fold {val_fold} ...', flush=True)
        tr_idx = [i for i, f in enumerate(folds) if f != val_fold and f != 'test']
        va_idx = [i for i, f in enumerate(folds) if f == val_fold]

        X_tr, Y_tr, M_tr = X[tr_idx], Y[tr_idx], masks[tr_idx]
        X_va, Y_va, M_va = X[va_idx], Y[va_idx], masks[va_idx]

        col_means = np.nanmean(X_tr, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        X_tr_imp = np.where(np.isnan(X_tr), col_means, X_tr)
        X_va_imp = np.where(np.isnan(X_va), col_means, X_va)
        X_te_imp = np.where(np.isnan(X_te), col_means, X_te)

        preds_va = _fit_points(X_tr_imp, Y_tr, M_tr, X_va_imp, device, f'fold {val_fold} val')

        _, fold_mae = masked_mae(preds_va, Y_va, M_va)
        _, fold_rmse = masked_rmse(preds_va, Y_va, M_va)
        fold_maes.append(fold_mae)
        fold_rmses.append(fold_rmse)
        print(f'  fold {val_fold} val: MAE = {fold_mae:.3f}  RMSE = {fold_rmse:.3f} dB  (n={len(va_idx)})', flush=True)

        preds_te = _fit_points(X_tr_imp, Y_tr, M_tr, X_te_imp, device, f'fold {val_fold} test')
        _, fold_test_mae = masked_mae(preds_te, Y_te, M_te)
        _, fold_test_rmse = masked_rmse(preds_te, Y_te, M_te)
        fold_test_maes.append(fold_test_mae)
        fold_test_rmses.append(fold_test_rmse)
        print(
            f'  fold {val_fold} test: MAE = {fold_test_mae:.3f}  RMSE = {fold_test_rmse:.3f} dB  (n={len(te_idx)})',
            flush=True,
        )

    print(f'\n  CV val  MAE : {np.mean(fold_maes):.3f} ± {np.std(fold_maes):.3f} dB', flush=True)
    print(f'  CV val  RMSE: {np.mean(fold_rmses):.3f} ± {np.std(fold_rmses):.3f} dB', flush=True)
    print(f'  CV test MAE : {np.mean(fold_test_maes):.3f} ± {np.std(fold_test_maes):.3f} dB', flush=True)
    print(f'  CV test RMSE: {np.mean(fold_test_rmses):.3f} ± {np.std(fold_test_rmses):.3f} dB', flush=True)

    tv_idx = [i for i, f in enumerate(folds) if f != 'test']

    X_tv, Y_tv, M_tv = X[tv_idx], Y[tv_idx], masks[tv_idx]
    X_te, Y_te, M_te = X[te_idx], Y[te_idx], masks[te_idx]

    col_means = np.nanmean(X_tv, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X_tv_imp = np.where(np.isnan(X_tv), col_means, X_tv)
    X_te_imp = np.where(np.isnan(X_te), col_means, X_te)

    print('  full holdout fit (train=all non-test) ...', flush=True)
    preds_te = _fit_points(X_tv_imp, Y_tv, M_tv, X_te_imp, device, 'holdout test')

    pp_mae, overall_mae = masked_mae(preds_te, Y_te, M_te)
    pp_rmse, overall_rmse = masked_rmse(preds_te, Y_te, M_te)
    print(f'\n  Holdout test MAE : {overall_mae:.3f} dB  (n={len(te_idx)})', flush=True)
    print(f'  Holdout test RMSE: {overall_rmse:.3f} dB', flush=True)

    return {
        'csv': str(path),
        'device': device,
        'n_rows': int(len(X)),
        'n_features': len(FEATURES_OD),
        'n_targets': N_PT,
        'cv_mean': float(np.mean(fold_maes)),
        'cv_std': float(np.std(fold_maes)),
        'cv_rmse_mean': float(np.mean(fold_rmses)),
        'cv_rmse_std': float(np.std(fold_rmses)),
        'test_mae_mean': float(np.mean(fold_test_maes)),
        'test_mae_std': float(np.std(fold_test_maes)),
        'test_rmse_mean': float(np.mean(fold_test_rmses)),
        'test_rmse_std': float(np.std(fold_test_rmses)),
        'test_mae': float(overall_mae),
        'test_rmse': float(overall_rmse),
        'per_point_mae': {PT[i]: float(pp_mae[i]) for i in range(N_PT) if not np.isnan(pp_mae[i])},
        'per_point_rmse': {PT[i]: float(pp_rmse[i]) for i in range(N_PT) if not np.isnan(pp_rmse[i])},
        'fold_maes': [float(m) for m in fold_maes],
        'fold_rmses': [float(m) for m in fold_rmses],
        'fold_test_maes': [float(m) for m in fold_test_maes],
        'fold_test_rmses': [float(m) for m in fold_test_rmses],
    }


DEFAULT_DATASETS = [
    ('90d_excl_empty_flip', 'ml_final_90d_excl_empty_flip.csv'),
    ('180d_excl_empty_flip', 'ml_final_180d_excl_empty_flip.csv'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='baseline_xgb_results.json')
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    datasets = DEFAULT_DATASETS
    if args.only:
        only = set(args.only)
        datasets = [(t, f) for t, f in datasets if t in only]

    print(f'device={args.device}', flush=True)

    results = {}
    for tag, fname in datasets:
        path = ROOT / fname
        if not path.exists():
            print(f'[skip] {tag}: 없음 — {path}', flush=True)
            continue
        results[tag] = run_cv(path, tag, args.device)

    print(f'\n{"=" * 60}', flush=True)
    print('요약', flush=True)
    print(f'{"=" * 60}', flush=True)
    for tag, res in results.items():
        print(
            f'{tag:<22} test MAE {res["test_mae_mean"]:.3f}±{res["test_mae_std"]:.3f}  '
            f'RMSE {res["test_rmse_mean"]:.3f}±{res["test_rmse_std"]:.3f}  '
            f'(holdout {res["test_mae"]:.3f}/{res["test_rmse"]:.3f})',
            flush=True,
        )

    out_path = ROOT / args.out
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n저장: {out_path}', flush=True)


if __name__ == '__main__':
    main()
