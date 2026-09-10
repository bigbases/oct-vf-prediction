"""
우선순위 실험 일괄 실행 (XGB: 빠름 / DL은 run_priority_pilot.sh)

사용:
  HVF_ROOT="$(pwd)" bash run_priority_xgb.sh
  HVF_ROOT=... python experiments_priority.py
"""
from __future__ import annotations

import os

# Ubuntu 18.04: scipy/xgboost need conda libstdc++
_conda_lib = os.environ.get('CONDA_PREFIX')
if _conda_lib:
    _lib = os.path.join(_conda_lib, 'lib')
    os.environ['LD_LIBRARY_PATH'] = _lib + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

# baseline_xgb와 동일한 VF·RNFL 정의
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_xgb import (  # noqa: E402
    PT,
    N_PT,
    RNFL_TABULAR,
    FEATURES_OD,
    FEATURES_OS,
    masked_mae,
    ROOT,
)

GCL_FEATURES_OD = [
    'avg_gcl_od', 'min_gcl_od',
    'od_s_sup', 'od_s_sup_t', 'od_s_inf_t', 'od_s_inf', 'od_s_inf_n', 'od_s_sup_n',
    'od_avg_rnfl', 'od_vert_cd',
]
GCL_FEATURES_OS = [
    'avg_gcl_os', 'min_gcl_os',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'os_avg_rnfl', 'os_vert_cd',
]

FEATURE_SETS = {
    'full_26': (FEATURES_OD, FEATURES_OS),
    'gcl_only_10': (GCL_FEATURES_OD, GCL_FEATURES_OS),
    'rnfl_only_16': (RNFL_TABULAR, RNFL_TABULAR),  # eye-agnostic names
}

XGB_DATASETS = [
    ('90d_flip', 'ml_final_90d_excl_empty_flip.csv'),
    ('90d_no_flip', 'ml_final_90d_excl_empty.csv'),
    ('180d_flip', 'ml_final_180d_excl_empty_flip.csv'),
]


def load_X_with_features(path: Path, feat_od: list[str], feat_os: list[str]):
    import csv

    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    X, Y, masks, folds = [], [], [], []
    for r in rows:
        flist = feat_od if r['eye'] == 'OD' else feat_os
        feats = []
        for f in flist:
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
                y_row.append(float(v))
                mask_row.append(True)
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


def _xgb_params(fast: bool) -> dict:
    if fast:
        return dict(n_estimators=100, learning_rate=0.08, max_depth=4)
    return dict(n_estimators=300, learning_rate=0.05, max_depth=4)


def run_xgb_cv(
    path: Path,
    feat_od: list[str],
    feat_os: list[str],
    val_folds: list[str] | None = None,
    fast: bool = False,
):
    X, Y, masks, folds = load_X_with_features(path, feat_od, feat_os)
    fold_ids = sorted(set(folds) - {'test'})
    if val_folds is not None:
        fold_ids = [f for f in fold_ids if f in val_folds]

    xgb_kw = _xgb_params(fast)
    fold_maes = []
    for val_fold in fold_ids:
        tr_idx = [i for i, f in enumerate(folds) if f != val_fold and f != 'test']
        va_idx = [i for i, f in enumerate(folds) if f == val_fold]
        X_tr, Y_tr, M_tr = X[tr_idx], Y[tr_idx], masks[tr_idx]
        X_va, Y_va, M_va = X[va_idx], Y[va_idx], masks[va_idx]
        col_means = np.nanmean(X_tr, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        X_tr_imp = np.where(np.isnan(X_tr), col_means, X_tr)
        X_va_imp = np.where(np.isnan(X_va), col_means, X_va)
        preds_va = np.full((len(va_idx), N_PT), np.nan)
        for pi in range(N_PT):
            valid_tr = M_tr[:, pi]
            if valid_tr.sum() < 10:
                continue
            model = XGBRegressor(
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbosity=0,
                **xgb_kw,
            )
            model.fit(X_tr_imp[valid_tr], Y_tr[valid_tr, pi])
            preds_va[:, pi] = model.predict(X_va_imp)
        _, fold_mae = masked_mae(preds_va, Y_va, M_va)
        fold_maes.append(float(fold_mae))
        print(f'    fold {val_fold}: MAE {fold_mae:.3f}', flush=True)

    te_idx = [i for i, f in enumerate(folds) if f == 'test']
    tv_idx = [i for i, f in enumerate(folds) if f != 'test']
    X_tv, Y_tv, M_tv = X[tv_idx], Y[tv_idx], masks[tv_idx]
    X_te, Y_te, M_te = X[te_idx], Y[te_idx], masks[te_idx]
    col_means = np.nanmean(X_tv, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X_tv_imp = np.where(np.isnan(X_tv), col_means, X_tv)
    X_te_imp = np.where(np.isnan(X_te), col_means, X_te)
    preds_te = np.full((len(te_idx), N_PT), np.nan)
    for pi in range(N_PT):
        valid_tv = M_tv[:, pi]
        if valid_tv.sum() < 10:
            continue
        model = XGBRegressor(
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbosity=0,
            **xgb_kw,
        )
        model.fit(X_tv_imp[valid_tv], Y_tv[valid_tv, pi])
        preds_te[:, pi] = model.predict(X_te_imp)
    _, test_mae = masked_mae(preds_te, Y_te, M_te)
    print(f'    test: MAE {test_mae:.3f}', flush=True)

    return {
        'n_rows': len(X),
        'n_features': len(feat_od),
        'cv_mean': float(np.mean(fold_maes)),
        'cv_std': float(np.std(fold_maes)),
        'fold_maes': fold_maes,
        'test_mae': float(test_mae),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/priority_xgb_results.json')
    parser.add_argument('--fold0-only', action='store_true',
                        help='CV는 fold 0만 (파일럿용)')
    parser.add_argument('--resume', action='store_true',
                        help='기존 결과 JSON에서 완료된 항목 건너뜀')
    parser.add_argument('--fast', action='store_true',
                        help='n_estimators=100 (우선순위 스크리닝용, ~3x 빠름)')
    args = parser.parse_args()

    val_folds = ['0'] if args.fold0_only else None
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if args.resume and out.exists():
        all_results = json.loads(out.read_text(encoding='utf-8'))
        print(f'[resume] {len(all_results)}개 완료 항목 로드', flush=True)

    for ds_tag, fname in XGB_DATASETS:
        path = ROOT / fname
        if not path.exists():
            print(f'[skip] {ds_tag}: {path}')
            continue
        for feat_tag, (fod, fos) in FEATURE_SETS.items():
            key = f'{ds_tag}__{feat_tag}'
            if key in all_results:
                print(f'\n=== XGB {key} [skip] ===', flush=True)
                continue
            print(f'\n=== XGB {key} ===', flush=True)
            res = run_xgb_cv(path, fod, fos, val_folds=val_folds, fast=args.fast)
            all_results[key] = {'csv': fname, **res}
            print(f'  CV {res["cv_mean"]:.3f}±{res["cv_std"]:.3f}  Test {res["test_mae"]:.3f}', flush=True)
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n저장: {out}', flush=True)

    print('\n=== 요약 (Test MAE) ===')
    for k, v in sorted(all_results.items(), key=lambda x: x[1]['test_mae']):
        print(f'  {k:<35} test={v["test_mae"]:.3f}  cv={v["cv_mean"]:.3f}±{v["cv_std"]:.3f}')


if __name__ == '__main__':
    main()
