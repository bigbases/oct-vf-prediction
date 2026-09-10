#!/usr/bin/env python3
"""
XGB tabular OOF(val fold) 예측 export — late-fusion P1 선결.

fold k: train = cv_fold ∉ {k, test}, predict on val fold k (OOF for those rows).

사용법:
  HVF_ROOT="$(pwd)" python scripts/export_xgb_oof.py \\
      --csv ml_final_90d_excl_empty_flip.csv --tag 90d --folds 0 --device cuda

출력: runs/oof/xgb_{tag}_fold{k}_val.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import ROW_KEY_FIELDS, row_key, save_oof_npz  # noqa: E402

# baseline_xgb.py 와 동일
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

# ---- 반대편 눈 (--fellow both). 사전등록: docs/fellow_eye_ablation_prereg.md ----
# study 눈 피처 앞 10개와 1:1 대응한다. 나머지(RNFL clock/quadrant)는 반대편 값이 CSV 에 없다.
N_FELLOW = 10
# study OD → fellow 는 OS 눈. 원본 컬럼 이름이 해부학 기준이므로 그대로 대응한다.
FELLOW_FOR_OD = [
    'avg_gcl_os', 'min_gcl_os',
    'os_s_sup', 'os_s_sup_t', 'os_s_inf_t', 'os_s_inf', 'os_s_inf_n', 'os_s_sup_n',
    'os_avg_rnfl', 'os_vert_cd',
]
# study OS → fellow 는 OD 눈. **t/n 을 맞바꿔 대응시킨다.**
# flip CSV 는 OS 행의 os_* 섹터를 이미 스왑했으므로(structure-function 정규화), 같은 해부학
# 위치끼리 비교하려면 fellow(od_*) 도 스왑해야 한다. 실측 근거 = docs/fellow_eye_frame_check.md
FELLOW_FOR_OS = [
    'avg_gcl_od', 'min_gcl_od',
    'od_s_sup', 'od_s_sup_n', 'od_s_inf_n', 'od_s_inf', 'od_s_inf_t', 'od_s_sup_t',
    'od_avg_rnfl', 'od_vert_cd',
]


def _feat_list(eye: str):
    return FEATURES_OD if eye == 'OD' else FEATURES_OS


def _fellow_list(eye: str):
    return FELLOW_FOR_OD if eye == 'OD' else FELLOW_FOR_OS


def feature_names(eye: str, fellow: str):
    """실제로 쓰는 피처 이름 순서 (기록·검증용)."""
    names = list(_feat_list(eye))
    if fellow == 'both':
        fl = _fellow_list(eye)
        names += [f'fellow__{c}' for c in fl]
        names += [f'asym__{s}_minus_{f}' for s, f in zip(_feat_list(eye)[:N_FELLOW], fl)]
    return names


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
        print('  [warning] XGBoost가 CPU-only build라 --device cpu로 대체', flush=True)
        return 'cpu'
    return requested


def load_rows(path: Path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def row_to_xy(row, target_floor=None, fellow='none'):
    """target_floor: 라벨만 이 값 이상으로 클램프 (None=기존 동작).

    Hasan 2025 normalised TS 재현용. feature는 손대지 않는다.

    fellow='both': 반대편 눈 원값 10개 + 비대칭 차이 10개를 뒤에 붙인다 (26 → 46).
    기본값 'none' 은 기존 동작을 그대로 재현한다.
    """
    def num(col):
        v = row.get(col, '')
        return float(v) if v not in ('', None) else np.nan

    study = [num(f) for f in _feat_list(row['eye'])]
    feats = list(study)
    if fellow == 'both':
        fell = [num(c) for c in _fellow_list(row['eye'])]
        feats += fell
        feats += [s - f for s, f in zip(study[:N_FELLOW], fell)]
    y, m = [], []
    for p in PT:
        v = row.get(p, '')
        if v == '' or v is None or v == '-1':
            y.append(0.0)
            m.append(False)
        else:
            fv = float(v)
            if target_floor is not None and fv < target_floor:
                fv = target_floor
            y.append(fv)
            m.append(True)
    return np.array(feats, np.float32), np.array(y, np.float32), np.array(m, bool)


def impute_train_stats(X_tr: np.ndarray):
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    return col_means


def impute(X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), col_means, X)


def fit_predict_fold(
    rows_tr, rows_va, device: str, target_floor=None, fellow='none',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = [row_to_xy(r, target_floor, fellow) for r in rows_tr]
    va = [row_to_xy(r, target_floor, fellow) for r in rows_va]
    X_tr = np.stack([t[0] for t in tr])
    Y_tr = np.stack([t[1] for t in tr])
    M_tr = np.stack([t[2] for t in tr])
    X_va = np.stack([v[0] for v in va])
    Y_va = np.stack([v[1] for v in va])
    M_va = np.stack([v[2] for v in va])

    col_means = impute_train_stats(X_tr)
    X_tr_i = impute(X_tr, col_means)
    X_va_i = impute(X_va, col_means)

    preds = np.full((len(rows_va), N_PT), np.nan, dtype=np.float32)
    xgb_kw = _xgb_kwargs(device)
    for pi in range(N_PT):
        valid = M_tr[:, pi]
        if valid.sum() < 10:
            continue
        model = XGBRegressor(**xgb_kw)
        model.fit(X_tr_i[valid], Y_tr[valid, pi])
        preds[:, pi] = model.predict(X_va_i)

    return preds, Y_va, M_va


def export_fold(rows, val_fold: str, tag: str, csv_name: str, device: str, out_root: Path,
                target_floor=None, fellow='none', cohort_filter='all_csv_rows'):
    rows_tr = [r for r in rows if r['cv_fold'] not in (val_fold, 'test')]
    rows_va = [r for r in rows if r['cv_fold'] == val_fold]
    keys = [row_key(r) for r in rows_va]

    rows_te = [r for r in rows if r['cv_fold'] == 'test']
    print(f'  fold {val_fold}: train={len(rows_tr)} val={len(rows_va)} test={len(rows_te)}', flush=True)
    pred_va, labels_va, mask_va = fit_predict_fold(rows_tr, rows_va, device, target_floor, fellow)
    out_val = out_root / f'xgb_{tag}_fold{val_fold}_val.npz'
    save_oof_npz(
        out_val,
        keys,
        pred_va,
        labels_va,
        mask_va,
        meta={
            'model': 'xgb_tabular',
            'csv': csv_name,
            'val_fold': val_fold,
            'split': 'val',
            'n_train': len(rows_tr),
            'n_val': len(rows_va),
            'device': device,
            'cohort_filter': cohort_filter,
            'target_floor': target_floor,
            'fellow': fellow,
            'n_features': len(feature_names(rows_va[0]['eye'], fellow)) if rows_va else None,
        },
    )
    print(f'    val → {out_val}', flush=True)

    keys_te = [row_key(r) for r in rows_te]
    pred_te, labels_te, mask_te = fit_predict_fold(rows_tr, rows_te, device, target_floor, fellow)
    out_te = out_root / f'xgb_{tag}_fold{val_fold}_test.npz'
    save_oof_npz(
        out_te,
        keys_te,
        pred_te,
        labels_te,
        mask_te,
        meta={
            'model': 'xgb_tabular',
            'csv': csv_name,
            'val_fold': val_fold,
            'split': 'test',
            'n_train': len(rows_tr),
            'n_test': len(rows_te),
            'device': device,
            'cohort_filter': cohort_filter,
            'target_floor': target_floor,
            'fellow': fellow,
            'n_features': len(feature_names(rows_te[0]['eye'], fellow)) if rows_te else None,
        },
    )
    print(f'    test → {out_te}', flush=True)
    return out_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv')
    ap.add_argument('--tag', default='90d', help='출력 파일 접두 태그')
    ap.add_argument('--folds', nargs='+', default=['0', '1', '2', '3', '4'])
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--target_floor', type=float, default=None,
                    help='라벨만 이 값 이상으로 클램프 (Hasan normalised TS 재현=14). '
                         '기본 None = 기존 결과 재현. 반드시 --tag를 바꿔 별도 파일로 저장할 것')
    ap.add_argument('--fellow', choices=['none', 'both'], default='none',
                    help="'both' = 반대편 눈 원값 10 + 비대칭 차이 10 추가 (26 → 46). "
                         '사전등록 docs/fellow_eye_ablation_prereg.md. '
                         '기본 none = 기존 결과 재현. 반드시 --tag를 바꿔 별도 파일로 저장할 것')
    ap.add_argument('--out_dir', default='runs/oof')
    ap.add_argument(
        '--match-cnn-dir',
        default=None,
        help='이 CNN 디렉터리의 val/test NPZ key 합집합으로 모든 XGB split을 제한',
    )
    ap.add_argument('--overwrite', action='store_true',
                    help='기존 NPZ 덮어쓰기 허용. 기본은 기존 산출물이 있으면 중단')
    args = ap.parse_args()
    args.device = resolve_device(args.device)

    if (args.fellow != 'none' or args.target_floor is not None) and args.tag == '90d':
        raise SystemExit(
            '--fellow/--target_floor 를 쓸 때는 --tag 를 바꿔야 한다 '
            '(정본 xgb_90d_* 덮어쓰기 방지)'
        )

    csv_path = ROOT / args.csv
    rows = load_rows(csv_path)
    cohort_filter = 'all_csv_rows'
    if args.match_cnn_dir is not None:
        cnn_dir = Path(args.match_cnn_dir)
        if not cnn_dir.is_absolute():
            cnn_dir = ROOT / cnn_dir
        allowed = set()
        for fold in range(5):
            for split in ('val', 'test'):
                path = cnn_dir / f'{split}_preds_fold{fold}.npz'
                z = np.load(path, allow_pickle=True)
                allowed.update(
                    (str(pid), str(eye), str(date))
                    for pid, eye, date in zip(
                        z['patient_id'], z['eye'], z['vf_date']
                    )
                )
        before = len(rows)
        rows = [
            row for row in rows
            if (row['patient_id'], row['eye'], row['vf_date']) in allowed
        ]
        cohort_filter = f'matched_to_cnn:{cnn_dir.name}'
        print(
            f'CNN key manifest 적용: {before} → {len(rows)} rows '
            f'(allowed keys={len(allowed)})',
            flush=True,
        )
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    planned = [
        out_root / f'xgb_{args.tag}_fold{fold}_{split}.npz'
        for fold in args.folds
        for split in ('val', 'test')
    ]
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        names = ', '.join(path.name for path in existing[:3])
        raise SystemExit(
            f'기존 산출물 {len(existing)}개가 있어 중단: {names}. '
            '--overwrite 또는 새 --out_dir/--tag를 사용하라.'
        )

    print(f'XGB OOF export | {csv_path.name} | folds={args.folds} | device={args.device} '
          f'| fellow={args.fellow}', flush=True)
    if args.fellow == 'both':
        for eye in ('OD', 'OS'):
            print(f'  study {eye} 피처 {len(feature_names(eye, args.fellow))}개', flush=True)
    for vf in args.folds:
        export_fold(rows, str(vf), args.tag, args.csv, args.device, out_root,
                    args.target_floor, args.fellow, cohort_filter)
    print('완료', flush=True)


if __name__ == '__main__':
    main()
