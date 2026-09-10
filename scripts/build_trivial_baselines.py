#!/usr/bin/env python3
"""Trivial baseline 비교표 (재학습 없음) — "ML이 왜 필요한가" 근거.

성능 사다리:
  1) grand-mean      : 모든 point를 train 평균 dB 하나로 예측 (가장 단순)
  2) per-point mean  : point별 train 평균 dB (climatology; 공간 구조만 반영)
  3) linear (tabular): point별 Ridge 선형회귀 (OCT 26 feature) — XGB의 비선형 가치 대조
  4) XGB (tabular)   : 저장된 예측
  5) image-only CNN  : 저장된 예측 (대표 백본 IR-v2)
  6) late fusion     : w·XGB + (1-w)·CNN, w는 OOF에서 튜닝

baseline은 fold별로 train(=다른 4 fold의 val 합집합) 통계로만 적합 → 누수 없음.
출력: runs/trivial_baselines.json / .md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402
sys.path.insert(0, str(ROOT))
from image_preprocessing import OCT_FEATURES_OD, OCT_FEATURES_OS  # noqa: E402

CSV = ROOT / 'ml_final_90d_excl_empty_flip.csv'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'  # 대표 백본
RIDGE_ALPHA = 1.0


def nkey(pid, eye, vf):
    return (str(pid).strip(),
            str(eye).strip().upper(),
            str(vf).strip().replace('-', '').replace('/', ''))


def load_feat_map():
    """key -> (26,) raw tabular vector (결측 nan). target-eye 기준."""
    out = {}
    for r in csv.DictReader(open(CSV, encoding='utf-8-sig')):
        eye = str(r['eye']).strip().upper()
        feats = OCT_FEATURES_OD if eye == 'OD' else OCT_FEATURES_OS
        v = []
        for f in feats:
            x = r.get(f, '')
            v.append(float(x) if x not in ('', None) else np.nan)
        out[nkey(r['patient_id'], r['eye'], r['vf_date'])] = np.array(v, np.float64)
    return out


FEAT = load_feat_map()


def feats_for(keys):
    return np.array([FEAT[nkey(*k)] for k in keys], np.float64)


def pooled(pred, lab, m):
    m = m.astype(bool)
    d = (pred - lab)[m]
    return round(float(np.sqrt(np.mean(d ** 2))), 2), round(float(np.mean(np.abs(d))), 2)


def load_fold(xgb_paths, split):
    """fold별 정렬된 dict list: keys/labels/mask/xgb/cnn."""
    folds = []
    for k in range(5):
        xz = load_oof_npz(xgb_paths[k])
        fn = f'{"val" if split == "oof" else "test"}_preds_fold{k}.npz'
        cz = load_oof_npz(CNN_DIR / fn)
        ax, bx, common = align_oof(xz, cz)
        folds.append({
            'keys': common,
            'labels': ax['labels'], 'mask': ax['mask'] & bx['mask'],
            'xgb': ax['pred'], 'cnn': bx['pred'],
        })
    return folds


def per_point_mean(labels, mask):
    """(52,) masked point 평균. 값 없는 point는 전체 평균."""
    m = mask.astype(bool)
    out = np.zeros(labels.shape[1])
    g = labels[m].mean() if m.any() else 0.0
    for j in range(labels.shape[1]):
        col = m[:, j]
        out[j] = labels[col, j].mean() if col.any() else g
    return out, float(g)


def fit_pointwise_ridge(X, Y, M, alpha=RIDGE_ALPHA):
    """point별 Ridge (X: n×26 impute+standardize, Y n×52, M n×52 mask).
    반환: 예측 함수용 (mu, sd, W[52×27] with bias)."""
    mu = np.nanmean(X, 0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    Xi = np.where(np.isnan(X), mu, X)
    sd = Xi.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = (Xi - mu) / sd
    Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # bias
    p = Xb.shape[1]
    W = np.zeros((Y.shape[1], p))
    reg = alpha * np.eye(p)
    reg[-1, -1] = 0.0  # bias 미규제
    for j in range(Y.shape[1]):
        m = M[:, j].astype(bool)
        if m.sum() < 5:
            W[j, -1] = Y[m, j].mean() if m.any() else 0.0
            continue
        A = Xb[m]
        yj = Y[m, j]
        W[j] = np.linalg.solve(A.T @ A + reg, A.T @ yj)
    return mu, sd, W


def predict_ridge(params, X):
    mu, sd, W = params
    Xi = np.where(np.isnan(X), mu, X)
    Xs = (Xi - mu) / sd
    Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    return Xb @ W.T


def opt_w(x, c, lab, m):
    m = m.astype(bool)
    ya, yb, y = x[m], c[m], lab[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (r, w)
    return round(float(best[1]), 2)


def build_oof():
    folds = load_fold(XGB_VAL, 'oof')
    preds = {m: [] for m in ('grand', 'ppmean', 'linear', 'xgb', 'cnn')}
    labs, masks = [], []
    for k in range(5):
        others = [folds[j] for j in range(5) if j != k]
        tr_lab = np.concatenate([o['labels'] for o in others])
        tr_msk = np.concatenate([o['mask'] for o in others])
        tr_keys = [key for o in others for key in o['keys']]
        cur = folds[k]
        n = cur['labels'].shape[0]
        ppm, g = per_point_mean(tr_lab, tr_msk)
        # linear
        params = fit_pointwise_ridge(feats_for(tr_keys), tr_lab, tr_msk)
        lin = predict_ridge(params, feats_for(cur['keys']))
        preds['grand'].append(np.full((n, 52), g))
        preds['ppmean'].append(np.tile(ppm, (n, 1)))
        preds['linear'].append(lin)
        preds['xgb'].append(cur['xgb'])
        preds['cnn'].append(cur['cnn'])
        labs.append(cur['labels'])
        masks.append(cur['mask'])
    P = {m: np.concatenate(v) for m, v in preds.items()}
    L = np.concatenate(labs)
    M = np.concatenate(masks)
    return P, L, M


def build_test():
    valf = load_fold(XGB_VAL, 'oof')
    tr_lab = np.concatenate([o['labels'] for o in valf])
    tr_msk = np.concatenate([o['mask'] for o in valf])
    tr_keys = [key for o in valf for key in o['keys']]
    ppm, g = per_point_mean(tr_lab, tr_msk)
    params = fit_pointwise_ridge(feats_for(tr_keys), tr_lab, tr_msk)

    testf = load_fold(XGB_TEST, 'test')
    L = testf[0]['labels']
    M = testf[0]['mask']
    for f in testf[1:]:
        M = M & f['mask']
    keys = testf[0]['keys']
    n = L.shape[0]
    xgb = np.mean(np.stack([f['xgb'] for f in testf]), 0)
    cnn = np.mean(np.stack([f['cnn'] for f in testf]), 0)
    P = {
        'grand': np.full((n, 52), g),
        'ppmean': np.tile(ppm, (n, 1)),
        'linear': predict_ridge(params, feats_for(keys)),
        'xgb': xgb, 'cnn': cnn,
    }
    return P, L, M


def main():
    Po, Lo, Mo = build_oof()
    w = opt_w(Po['xgb'], Po['cnn'], Lo, Mo)
    Po['fusion'] = w * Po['xgb'] + (1 - w) * Po['cnn']

    Pt, Lt, Mt = build_test()
    Pt['fusion'] = w * Pt['xgb'] + (1 - w) * Pt['cnn']

    order = [('grand', 'grand-mean (상수)'), ('ppmean', 'per-point mean (climatology)'),
             ('linear', 'linear (tabular Ridge)'), ('xgb', 'XGB (tabular)'),
             ('cnn', 'image-only CNN (IR-v2)'), ('fusion', 'late fusion')]
    res = {'protocol': '90d, 5-fold, seed42, IR-v2 대표, pooled RMSE/MAE(dB), 52-point masked',
           'w_xgb': w, 'n_oof_eyes': int(Mo.shape[0]), 'n_test_eyes': int(Mt.shape[0]),
           'rows': []}
    print(f'{"model":30s} {"OOF R/M":>14s}   {"TEST R/M":>14s}', flush=True)
    print('-' * 64, flush=True)
    for key, label in order:
        ro, mo = pooled(Po[key], Lo, Mo)
        rt, mt = pooled(Pt[key], Lt, Mt)
        res['rows'].append({'model': key, 'label': label,
                            'oof': {'rmse': ro, 'mae': mo}, 'test': {'rmse': rt, 'mae': mt}})
        print(f'{label:30s} {ro:6.2f}/{mo:<6.2f}   {rt:6.2f}/{mt:<6.2f}', flush=True)

    (ROOT / 'runs/trivial_baselines.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# Trivial baseline 비교 (90d · 5-fold · seed42 · IR-v2 대표)', '',
          f'pooled RMSE / MAE (dB), 낮을수록 좋음. late fusion w_xgb={w}. '
          f'OOF n={res["n_oof_eyes"]} eyes, test n={res["n_test_eyes"]} eyes.',
          'baseline은 fold별 train(다른 4 fold) 통계로만 적합 → 누수 없음.', '',
          '| 모델 | OOF RMSE/MAE | Test RMSE/MAE |',
          '|------|--------------|---------------|']
    for r in res['rows']:
        bold = '**' if r['model'] == 'fusion' else ''
        md.append(f"| {bold}{r['label']}{bold} | {bold}{r['oof']['rmse']}/{r['oof']['mae']}{bold} "
                  f"| {bold}{r['test']['rmse']}/{r['test']['mae']}{bold} |")
    md += ['', '## 해석',
           '- **grand-mean / per-point mean**: 학습 없이 "평균 시야만 예측"하는 하한선.',
           '- **linear(tabular) → XGB**: 비선형(gradient boosting)의 추가 이득.',
           '- **XGB → late fusion**: 영상(두께맵) 추가 이득 = 멀티모달 상보성(핵심 공헌).',
           '- 사다리 전 구간에서 단조 개선이면 "각 구성요소가 실제로 기여함"을 정량 입증.']
    (ROOT / 'runs/trivial_baselines.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f'\n저장: runs/trivial_baselines.json / .md', flush=True)


if __name__ == '__main__':
    main()
