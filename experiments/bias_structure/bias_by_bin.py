#!/usr/bin/env python3
"""image branch 계통 편향의 감도 구간별 분포와 중심화 반사실.

배경: XGB 잔차 평균은 +0.027 dB 인데 CNN 은 +0.98~2.03 dB 로 시야를 과대예측한다
(experiments/weight_theory/). 원고 §4.4 는 fusion 이 image branch 대비 저감도
구간에서 이득을 낸다고 보고한다. 그 이득이 정보 때문인지 편향 상쇄 때문인지
가리는 것이 목적이다.

구간·MIN_PTS·w·마스크·기울기 정의는 전부 scripts/fusion_sensitivity_trend.py 와
scripts/fusion_eval_common.py 에서 import 한다. 새로 정의하지 않는다.
w 는 재적합하지 않는다 — 원래 값(OOF 는 fold별 nested, TEST 는 전체 OOF 적합)을
그대로 쓴다.

읽기 전용. 재학습 없음. 정본 runs/ 와 원고를 건드리지 않는다.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREES = {'A': REPO / 'experiments/laterality_qfix/step4_work/A',
         'B': REPO / 'experiments/laterality_qfix/step4_work/B'}
SEED = 42
N_BOOT = 10000


def load_modules(tree: Path):
    """섀도 트리의 scripts/ 를 import 한다. ROOT 가 __file__ 로 정해지므로
    경로를 갈아끼우면 그 트리의 산출물을 읽는다."""
    for m in ('fusion_eval_common', 'fusion_sensitivity_trend', 'oof_common',
              'fig_style'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tree / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        fst = importlib.import_module('fusion_sensitivity_trend')
        assert fec.ROOT == tree, f'ROOT 불일치: {fec.ROOT} != {tree}'
        return fec, fst
    finally:
        sys.path.pop(0)


def oof_raw_with_w(fec, oof, name):
    """fec.oof_raw 와 동일하되 fold별 w 와 fold 경계를 함께 돌려준다.
    중심화 반사실에서 같은 w 를 다시 써야 하기 때문이다."""
    xgb, cnn, lab, mask, pid, ws, bounds = [], [], [], [], [], [], []
    n = 0
    for k in range(5):
        tr = [oof[i] for i in range(5) if i != k]
        w = fec.fit_w(tr, name)
        f = oof[k]
        xgb.append(f['xgb']); cnn.append(f['cnns'][name])
        lab.append(f['lab']); mask.append(f['mask'])
        pid += [str(kk[0]) for kk in f['keys']]
        ws.append(w)
        bounds.append((n, n + f['xgb'].shape[0])); n += f['xgb'].shape[0]
    return (np.concatenate(xgb), np.concatenate(cnn), np.concatenate(lab),
            np.concatenate(mask), np.array(pid, dtype=object), ws, bounds)


def fuse(xgb, cnn, ws, bounds):
    """fold별 w 를 그대로 적용한다."""
    out = np.empty_like(xgb)
    for w, (a, b) in zip(ws, bounds):
        out[a:b] = w * xgb[a:b] + (1 - w) * cnn[a:b]
    return out


def bin_residual_means(fst, pred, lab, mask):
    """구간별 잔차 평균 — 마스크된 지점을 통째로 풀링한다."""
    m = mask.astype(bool)
    bi = fst.bin_index(lab[m])
    r = (pred - lab)[m]
    return np.array([float(r[bi == b].mean()) if (bi == b).any() else np.nan
                     for b in range(4)]), np.array([int((bi == b).sum()) for b in range(4)])


def center_by_bin(fst, cnn, lab, mask, bmeans):
    """구간별 잔차 평균을 뺀다. 구간은 참 감도로 정하므로 예측을 조건으로 걸지 않는다."""
    out = cnn.copy()
    bi_full = fst.bin_index(lab)
    for b in range(4):
        if np.isfinite(bmeans[b]):
            out[bi_full == b] -= bmeans[b]
    return out


def boot_ci(v, pid, rng):
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        means[b] = np.mean(v[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trend_block(fst, fus, cnn, lab, mask, pid):
    """§4.4 와 같은 계산: 안·구간 RMSE 차 -> 구간별 평균, 안별 기울기."""
    d = fst.eye_bin_rmse(fus, lab, mask) - fst.eye_bin_rmse(cnn, lab, mask)
    sl = fst.slopes(d)
    ok = np.isfinite(sl)
    sl_ok, pid_ok = sl[ok], pid[ok]
    rng = np.random.default_rng(SEED)
    lo, hi = boot_ci(sl_ok, pid_ok, rng)
    per_bin = []
    for b in range(4):
        col = d[:, b]; v = col[np.isfinite(col)]
        per_bin.append({'bin': fst.BIN_LABELS[b], 'n_eyes': int(v.size),
                        'mean_delta': float(v.mean()) if v.size else float('nan')})
    return {'mean_slope': float(sl_ok.mean()),
            'median_slope': float(np.median(sl_ok)),
            'slope_ci_lo': lo, 'slope_ci_hi': hi,
            'p_wilcoxon': float(wilcoxon(sl_ok).pvalue) if sl_ok.size >= 6 else float('nan'),
            'frac_positive': float(np.mean(sl_ok > 0)),
            'n_eyes_with_slope': int(sl_ok.size),
            'per_bin': per_bin}


def run_tree(pk, tree):
    fec, fst = load_modules(tree)
    oof, tst, test_lab, test_mask = fec.load_all(ensemble=True)
    names = fec.NAMES + [fec.ENSEMBLE]
    out = {}
    for nm in names:
        xo, co, lo, mo, pid_o, ws, bnd = oof_raw_with_w(fec, oof, nm)
        w_test = fec.fit_w([oof[k] for k in range(5)], nm)
        xt = np.mean([tst[k]['xgb'] for k in range(5)], 0)
        ct = np.mean([tst[k]['cnns'][nm] for k in range(5)], 0)
        pid_t = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)

        entry = {'w_oof_nested': [round(w, 4) for w in ws], 'w_test': float(w_test)}
        for split, (xp, cp, lb, mk, pd_, mkfuse) in (
            ('OOF', (xo, co, lo, mo, pid_o, lambda x, c: fuse(x, c, ws, bnd))),
            ('TEST', (xt, ct, test_lab, test_mask, pid_t,
                      lambda x, c: w_test * x + (1 - w_test) * c)),
        ):
            cnn_bm, n_bin = bin_residual_means(fst, cp, lb, mk)
            xgb_bm, _ = bin_residual_means(fst, xp, lb, mk)
            m = mk.astype(bool)
            g_cnn = float(((cp - lb)[m]).mean())
            g_xgb = float(((xp - lb)[m]).mean())

            cp_bc = center_by_bin(fst, cp, lb, mk, cnn_bm)
            cp_gc = cp - g_cnn

            blk = {
                'n_points_by_bin': n_bin.tolist(),
                'cnn_residual_mean_by_bin': cnn_bm.tolist(),
                'xgb_residual_mean_by_bin': xgb_bm.tolist(),
                'cnn_residual_mean_global': g_cnn,
                'xgb_residual_mean_global': g_xgb,
                'bias_range_across_bins': float(np.nanmax(cnn_bm) - np.nanmin(cnn_bm)),
                'variants': {
                    'original':      trend_block(fst, mkfuse(xp, cp), cp, lb, mk, pd_),
                    'bin_centered':  trend_block(fst, mkfuse(xp, cp_bc), cp_bc, lb, mk, pd_),
                    'global_centered': trend_block(fst, mkfuse(xp, cp_gc), cp_gc, lb, mk, pd_),
                },
            }
            entry[split] = blk
        out[nm] = entry
        o = entry['OOF']
        print(f'  [{pk}] {nm:<13} 편향/구간 {np.round(o["cnn_residual_mean_by_bin"],2)} '
              f'slope orig {o["variants"]["original"]["mean_slope"]:+.4f} '
              f'-> bin {o["variants"]["bin_centered"]["mean_slope"]:+.4f} '
              f'-> glob {o["variants"]["global_centered"]["mean_slope"]:+.4f}', flush=True)
    return out


def main():
    res = {'seed': SEED, 'n_boot': N_BOOT,
           'note': ('구간·MIN_PTS·w·마스크·기울기는 fusion_sensitivity_trend / '
                    'fusion_eval_common 에서 import. w 재적합 없음. '
                    'delta = fusion RMSE - image branch RMSE (같은 변형 안에서 비교).'),
           'passes': {}}
    for pk, tree in TREES.items():
        print(f'=== 패스 {pk} ===', flush=True)
        res['passes'][pk] = run_tree(pk, tree)
    p = REPO / 'experiments/bias_structure/bias_by_bin.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {p}')


if __name__ == '__main__':
    main()
