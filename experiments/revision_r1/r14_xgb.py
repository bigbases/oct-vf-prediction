#!/usr/bin/env python3
"""R1-4, XGBoost branch: refit with floor cells dropped from each location's training samples.

Plan: docs/revision_r1/prereg_R1-3_R1-4.md §3. The fitting code is the submitted one
(export_xgb_oof.py: row_to_xy, impute, _xgb_kwargs on CPU), imported from the pass-B tree.

  arm 0  the submitted rule. Guard: its predictions must equal the submitted pass-B
         prediction files (every row, every location) before arm 1 is fitted.
  arm 1  for location j, training rows whose cell j is a floor cell are left out.

Outputs (patient keys inside, never copied into the release tree):
  runs/r14_xgb/arm{0,1}/xgb_90d_fold{k}_{val,test}.npz
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))
sys.path.insert(0, str(HERE))

import export_xgb_oof as X                 # noqa: E402
from oof_common import row_key, save_oof_npz, load_oof_npz  # noqa: E402
from xgboost import XGBRegressor           # noqa: E402
from floor_cells import floor_cells, PT52  # noqa: E402

assert list(X.PT) == PT52
CSV = PASS_B / 'ml_final_90d_excl_empty_flip.csv'
OUT = HERE / 'runs' / 'r14_xgb'


def fit_predict(rows_tr, rows_out, floor, exclude):
    tr = [X.row_to_xy(r) for r in rows_tr]
    va = [X.row_to_xy(r) for r in rows_out]
    X_tr = np.stack([t[0] for t in tr]); Y_tr = np.stack([t[1] for t in tr]); M_tr = np.stack([t[2] for t in tr])
    X_va = np.stack([v[0] for v in va]); Y_va = np.stack([v[1] for v in va]); M_va = np.stack([v[2] for v in va])
    F_tr = np.stack([floor[row_key(r)] for r in rows_tr])
    cm = X.impute_train_stats(X_tr)
    X_tr_i, X_va_i = X.impute(X_tr, cm), X.impute(X_va, cm)
    preds = np.full((len(rows_out), X.N_PT), np.nan, dtype=np.float32)
    kw = X._xgb_kwargs('cpu')
    n_dropped = 0
    for pi in range(X.N_PT):
        valid = M_tr[:, pi].copy()
        if exclude:
            n_dropped += int((valid & F_tr[:, pi]).sum())
            valid &= ~F_tr[:, pi]
        if valid.sum() < 10:
            continue
        model = XGBRegressor(**kw)
        model.fit(X_tr_i[valid], Y_tr[valid, pi])
        preds[:, pi] = model.predict(X_va_i)
    return preds, Y_va, M_va, n_dropped


def main():
    rows = X.load_rows(CSV)
    floor, info = floor_cells(current=CSV)
    print('floor cells:', info['floor_cells'], f"({info['floor_pct']}%)")
    for arm, exclude in ((0, False), (1, True)):
        for k in map(str, range(5)):
            rows_tr = [r for r in rows if r['cv_fold'] not in (k, 'test')]
            for split, rows_out in (('val', [r for r in rows if r['cv_fold'] == k]),
                                    ('test', [r for r in rows if r['cv_fold'] == 'test'])):
                pred, lab, msk, nd = fit_predict(rows_tr, rows_out, floor, exclude)
                path = OUT / f'arm{arm}' / f'xgb_90d_fold{k}_{split}.npz'
                save_oof_npz(path, [row_key(r) for r in rows_out], pred, lab, msk,
                             meta={'model': 'xgb_tabular', 'r14_arm': arm, 'floor_rule':
                                   'exclude' if exclude else 'submitted', 'val_fold': k, 'split': split,
                                   'n_train': len(rows_tr), 'train_cells_dropped': nd, 'device': 'cpu'})
                if arm == 0:
                    ref = load_oof_npz(PASS_B / 'runs/oof' / f'xgb_90d_fold{k}_{split}.npz')
                    mine = load_oof_npz(path)
                    assert ref['keys'] == mine['keys'], 'row order differs from the submitted file'
                    diff = float(np.nanmax(np.abs(ref['pred'] - mine['pred'])))
                    print(f'guard arm0 fold {k} {split}: max |pred - submitted| = {diff:.2e}')
                    if diff > 1e-5:
                        raise SystemExit('arm 0 does not reproduce the submitted XGBoost predictions')
                else:
                    print(f'arm1 fold {k} {split}: training cells dropped {nd}')
    print('done')


if __name__ == '__main__':
    main()
