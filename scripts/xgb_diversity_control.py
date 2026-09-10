#!/usr/bin/env python3
"""XGB+XGB 다양성 대조 실험 (Discussion 다양성 가설 방어의 대칭 짝).

가설 분리:
  - 융합 이득이 "모델 다양성(비상관 오차)"만의 산물이라면, seed만 다른
    두 번째 XGB와의 융합에서도 비슷한 이득이 나와야 한다.
  - 이득이 "영상 표현의 추가 정보"라면 XGB+XGB 융합은 이득이 거의 없어야 한다.

설계:
  - XGB-B = 기존과 동일 kwargs에서 random_state만 42→43 (subsample 0.8이라
    seed가 실제로 다른 트리를 만든다).
  - fold별 train/val/test 분리는 export_xgb_oof.py와 동일 (cv_fold 컬럼).
  - fusion w는 XGB+CNN과 동일하게 OOF pooled RMSE 최소화로 1회 결정.
  - 비교: XGB-A 단독 / XGB-B 단독 / XGB-A+B 융합 / (참조) XGB-A+IR-v2 융합.
  - Wilcoxon per-eye RMSE, 잔차 상관도 함께 기록.

실행: HVF_ROOT 불필요, repo 루트 기준 상대경로.
  python scripts/xgb_diversity_control.py
출력: runs/xgb_diversity_control.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz, row_key  # noqa: E402
from export_xgb_oof import (  # noqa: E402
    load_rows, fit_predict_fold, _xgb_kwargs,
)
import export_xgb_oof as xo  # noqa: E402

CSV = 'ml_final_90d_excl_empty_flip.csv'
SEED_B = 43
XGB_A = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGT_A = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
IR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'


def make_seed_b_kwargs(device: str) -> dict:
    kw = _xgb_kwargs(device)
    kw['random_state'] = SEED_B
    return kw


def train_seed_b(rows):
    """fold별 XGB-B 예측 (val, test). export_xgb_oof.fit_predict_fold 재사용
    위해 _xgb_kwargs를 임시로 바꿔치기한다."""
    orig = xo._xgb_kwargs
    xo._xgb_kwargs = make_seed_b_kwargs
    out = {}
    try:
        for k in range(5):
            fk = str(k)
            rows_tr = [r for r in rows if r['cv_fold'] not in (fk, 'test')]
            rows_va = [r for r in rows if r['cv_fold'] == fk]
            rows_te = [r for r in rows if r['cv_fold'] == 'test']
            pv, lv, mv = fit_predict_fold(rows_tr, rows_va, 'cpu')
            pt, lt, mt = fit_predict_fold(rows_tr, rows_te, 'cpu')
            out[k] = dict(
                val=dict(keys=[row_key(r) for r in rows_va], pred=pv, lab=lv, mask=mv),
                test=dict(keys=[row_key(r) for r in rows_te], pred=pt, lab=lt, mask=mt),
            )
            print(f'  fold {k} done', flush=True)
    finally:
        xo._xgb_kwargs = orig
    return out


def align(a_keys, a_pred, b):
    idx = {kk: i for i, kk in enumerate(b['keys'])}
    ii = [idx[kk] for kk in a_keys]
    return b['pred'][ii], b['lab'][ii] if 'lab' in b else None, b['mask'][ii]


def gather(split, seed_b, ir_fname):
    """fold별로 A/B/IR 예측을 공통 키 순서로 모은다.

    val: fold별 OOF를 concat (240안).
    test: 정본 관례대로 5개 fold 모델의 예측을 눈별 평균 (37안).
    """
    per_fold = []
    for k in range(5):
        za = load_oof_npz((XGB_A if split == 'val' else XGT_A)[k])
        zc = load_oof_npz(IR / f'{split}_preds_fold{k}.npz')
        sb = seed_b[k][split]
        keys = [kk for kk in za['keys'] if kk in set(sb['keys']) and kk in set(zc['keys'])]
        def pick(z):
            idx = {kk: i for i, kk in enumerate(z['keys'])}
            ii = [idx[kk] for kk in keys]
            return z['pred'][ii], z['mask'][ii]
        pa, ma = pick(za)
        pb, mb = pick({'keys': sb['keys'], 'pred': sb['pred'], 'mask': sb['mask']})
        pc, mc = pick(zc)
        idx = {kk: i for i, kk in enumerate(za['keys'])}
        lab = za['labels'][[idx[kk] for kk in keys]]
        m = (ma & mb & mc).astype(bool)
        per_fold.append((keys, pa, pb, pc, lab, m))

    if split == 'val':
        cat = lambda i: np.concatenate([f[i] for f in per_fold])
        keys = [kk for f in per_fold for kk in f[0]]
        return cat(1), cat(2), cat(3), cat(4), cat(5), keys

    # test: 눈별로 5 fold 예측 평균
    keys = per_fold[0][0]
    assert all(f[0] == keys for f in per_fold), 'test key order mismatch'
    pa = np.mean([f[1] for f in per_fold], axis=0)
    pb = np.mean([f[2] for f in per_fold], axis=0)
    pc = np.mean([f[3] for f in per_fold], axis=0)
    lab = per_fold[0][4]
    m = per_fold[0][5]
    for f in per_fold[1:]:
        m = m & f[5]
    return pa, pb, pc, lab, m, keys


def pooled_rmse(p, l, m):
    d = (p - l)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def per_eye_rmse(p, l, m):
    out = []
    for i in range(p.shape[0]):
        mi = m[i]
        out.append(np.sqrt(np.mean((p[i, mi] - l[i, mi]) ** 2)))
    return np.array(out)


def best_w(pa, pb, l, m):
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean(((w * pa + (1 - w) * pb - l)[m]) ** 2))
        if r < best[0]:
            best = (r, round(float(w), 2))
    return best[1]


def resid_corr(pa, pb, l, m):
    ra, rb = (pa - l)[m], (pb - l)[m]
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(name, p_fus, p_base, l, m):
    ea, eb = per_eye_rmse(p_fus, l, m), per_eye_rmse(p_base, l, m)
    stat = wilcoxon(ea, eb)
    return {
        'fusion_rmse_mae': pooled_rmse(p_fus, l, m),
        'base_rmse_mae': pooled_rmse(p_base, l, m),
        'delta_pooled': round(pooled_rmse(p_fus, l, m)[0] - pooled_rmse(p_base, l, m)[0], 4),
        'win': f'{int((ea < eb).sum())}/{len(ea)}',
        'wilcoxon_p': float(stat.pvalue),
    }


def main():
    rows = load_rows(ROOT / CSV)
    print('training XGB-B (seed 43) ...', flush=True)
    seed_b = train_seed_b(rows)

    res = {'design': 'XGB-A(seed42, 기존 OOF) + XGB-B(seed43 재학습) late fusion. '
                     'w는 OOF pooled RMSE 최소화. 참조로 XGB-A+IR-v2.',
           'seed_b': SEED_B}
    for split in ('val', 'test'):
        pa, pb, pc, l, m, _ = gather(split, seed_b, IR)
        blk = {'n_eyes': int(pa.shape[0])}
        blk['xgb_a'] = pooled_rmse(pa, l, m)
        blk['xgb_b'] = pooled_rmse(pb, l, m)
        blk['resid_corr_xgbA_xgbB'] = resid_corr(pa, pb, l, m)
        blk['resid_corr_xgbA_cnn'] = resid_corr(pa, pc, l, m)
        if split == 'val':
            w_bb = best_w(pa, pb, l, m)
            w_bc = best_w(pa, pc, l, m)
            res['w_xgbA_xgbB'] = w_bb
            res['w_xgbA_cnn'] = w_bc
        w_bb, w_bc = res['w_xgbA_xgbB'], res['w_xgbA_cnn']
        fus_bb = w_bb * pa + (1 - w_bb) * pb
        fus_bc = w_bc * pa + (1 - w_bc) * pc
        blk['fusion_xgbA_xgbB_vs_xgbA'] = compare('bb', fus_bb, pa, l, m)
        blk['fusion_xgbA_cnn_vs_xgbA'] = compare('bc', fus_bc, pa, l, m)
        res['OOF' if split == 'val' else 'TEST'] = blk
        print(f'[{split}] done', flush=True)

    out = ROOT / 'runs/xgb_diversity_control.json'
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print('→', out)


if __name__ == '__main__':
    main()
