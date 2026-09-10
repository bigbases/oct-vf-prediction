#!/usr/bin/env python3
"""중심화 반사실 재계산 — 두 가지 결함을 고친다.

(1) 중심화 후 w 재적합
    image branch 를 중심화하면 최적 가중치가 달라진다. 중심화된 image branch 에
    대해 학습 폴드 OOF RMSE 를 최소화하는 w 를 다시 구한다 (fec.fit_w 그대로 사용,
    101-그리드). 원래 w 를 그대로 쓴 이전 결과는 공정한 대조가 아니었다.

(2) 중심화 상수를 학습 폴드에서 추정
    이전에는 평가 대상(OOF 전체 / test 전체)의 잔차 평균을 썼다. 여기서는
    fold k 에 대해 나머지 4 fold 의 잔차 평균을 쓰고, test 에 대해서는 OOF 5 fold
    전체의 잔차 평균을 쓴다. 평가 대상은 상수 추정에 들어가지 않는다.

두 결함을 따로 껐다 켤 수 있게 변형을 7종으로 둔다. 이전 결과(eval+fixedw)를
그대로 재현하는 변형을 포함해 대조가 가능하다.

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

# (라벨, 중심화 종류, 상수 출처, w 재적합)
VARIANTS = [
    ('original',                  'none',   'eval',  False),
    ('global_center_eval_fixedw', 'global', 'eval',  False),   # 이전 결과
    ('global_center_train_fixedw', 'global', 'train', False),   # (2)만
    ('global_center_train_refitw', 'global', 'train', True),    # (1)+(2)
    ('bin_center_eval_fixedw',    'bin',    'eval',  False),   # 이전 결과
    ('bin_center_train_fixedw',   'bin',    'train', False),   # (2)만
    ('bin_center_train_refitw',   'bin',    'train', True),     # (1)+(2)
]


def load_modules(tree: Path):
    for m in ('fusion_eval_common', 'fusion_sensitivity_trend', 'oof_common', 'fig_style'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tree / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        fst = importlib.import_module('fusion_sensitivity_trend')
        assert fec.ROOT == tree, f'ROOT 불일치: {fec.ROOT} != {tree}'
        return fec, fst
    finally:
        sys.path.pop(0)


def const_from_folds(fst, folds, name):
    """주어진 fold 묶음의 마스크된 지점에서 전역 평균과 구간별 평균을 낸다.
    비어 있는 구간은 전역 평균으로 대체한다."""
    r = np.concatenate([(f['cnns'][name] - f['lab'])[f['mask'].astype(bool)] for f in folds])
    bi = np.concatenate([fst.bin_index(f['lab'][f['mask'].astype(bool)]) for f in folds])
    g = float(r.mean())
    bm = np.array([float(r[bi == b].mean()) if (bi == b).any() else g for b in range(4)])
    return g, bm


def const_from_arrays(fst, cnn, lab, mask, name=None):
    m = mask.astype(bool)
    r = (cnn - lab)[m]
    bi = fst.bin_index(lab[m])
    g = float(r.mean())
    bm = np.array([float(r[bi == b].mean()) if (bi == b).any() else g for b in range(4)])
    return g, bm


def apply_center(fst, cnn, lab, const):
    if np.ndim(const) == 0:
        return cnn - float(const)
    out = cnn.copy()
    bi = fst.bin_index(lab)
    for b in range(4):
        out[bi == b] -= const[b]
    return out


def centered_folds(fst, folds, name, const):
    out = []
    for f in folds:
        g = dict(f)
        g['cnns'] = dict(f['cnns'])
        g['cnns'][name] = apply_center(fst, f['cnns'][name], f['lab'], const)
        out.append(g)
    return out


def boot_ci(v, pid, rng):
    """환자 클러스터 부트스트랩. 이전 스크립트의 루프와 같은 난수 스트림·같은 통계량."""
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    gsum = np.array([v[g].sum() for g in groups])
    gcnt = np.array([g.size for g in groups], dtype=float)
    idx = rng.integers(0, ng, size=(N_BOOT, ng))
    means = gsum[idx].sum(1) / gcnt[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trend_block(fst, fus, cnn, lab, mask, pid):
    d = fst.eye_bin_rmse(fus, lab, mask) - fst.eye_bin_rmse(cnn, lab, mask)
    sl = fst.slopes(d)
    ok = np.isfinite(sl)
    sl_ok, pid_ok = sl[ok], pid[ok]
    lo, hi = boot_ci(sl_ok, pid_ok, np.random.default_rng(SEED))
    per_bin = []
    for b in range(4):
        col = d[:, b]
        v = col[np.isfinite(col)]
        per_bin.append({'bin': fst.BIN_LABELS[b], 'n_eyes': int(v.size),
                        'mean_delta': float(v.mean()) if v.size else float('nan')})
    return {'mean_slope': float(sl_ok.mean()),
            'median_slope': float(np.median(sl_ok)),
            'slope_ci_lo': lo, 'slope_ci_hi': hi,
            'p_wilcoxon': float(wilcoxon(sl_ok).pvalue) if sl_ok.size >= 6 else float('nan'),
            'frac_positive': float(np.mean(sl_ok > 0)),
            'n_eyes_with_slope': int(sl_ok.size),
            'per_bin': per_bin}


def oof_variant(fec, fst, oof, name, kind, source, refit):
    """fold 별로 상수·w 를 정하고 held-out fold 에 적용한다."""
    g_all, bm_all = const_from_folds(fst, [oof[k] for k in range(5)], name)
    cnn_p, fus_p, lab_p, msk_p, pid_p, ws, consts = [], [], [], [], [], [], []
    for k in range(5):
        tr = [oof[i] for i in range(5) if i != k]
        f = oof[k]
        if kind == 'none':
            const = 0.0
        elif source == 'eval':
            const = g_all if kind == 'global' else bm_all
        else:
            g_tr, bm_tr = const_from_folds(fst, tr, name)
            const = g_tr if kind == 'global' else bm_tr
        cnn_k = apply_center(fst, f['cnns'][name], f['lab'], const)
        w = fec.fit_w(centered_folds(fst, tr, name, const) if refit else tr, name)
        fus_p.append(w * f['xgb'] + (1 - w) * cnn_k)
        cnn_p.append(cnn_k)
        lab_p.append(f['lab']); msk_p.append(f['mask'])
        pid_p += [str(kk[0]) for kk in f['keys']]
        ws.append(round(float(w), 4))
        consts.append(float(const) if np.ndim(const) == 0 else [round(float(x), 4) for x in const])
    blk = trend_block(fst, np.concatenate(fus_p), np.concatenate(cnn_p),
                      np.concatenate(lab_p), np.concatenate(msk_p),
                      np.array(pid_p, dtype=object))
    blk['w_by_fold'] = ws
    blk['center_const_by_fold'] = consts
    return blk


def test_variant(fec, fst, oof, tst, test_lab, test_mask, name, kind, source, refit):
    tx = np.mean([tst[k]['xgb'] for k in range(5)], 0)
    tc = np.mean([tst[k]['cnns'][name] for k in range(5)], 0)
    pid = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)
    folds = [oof[k] for k in range(5)]
    if kind == 'none':
        const = 0.0
    elif source == 'eval':
        g, bm = const_from_arrays(fst, tc, test_lab, test_mask)
        const = g if kind == 'global' else bm
    else:
        g, bm = const_from_folds(fst, folds, name)
        const = g if kind == 'global' else bm
    tc_c = apply_center(fst, tc, test_lab, const)
    w = fec.fit_w(centered_folds(fst, folds, name, const) if refit else folds, name)
    blk = trend_block(fst, w * tx + (1 - w) * tc_c, tc_c, test_lab, test_mask, pid)
    blk['w'] = round(float(w), 4)
    blk['center_const'] = (float(const) if np.ndim(const) == 0
                           else [round(float(x), 4) for x in const])
    return blk


def run_tree(pk, tree):
    fec, fst = load_modules(tree)
    oof, tst, test_lab, test_mask = fec.load_all(ensemble=True)
    out = {}
    for nm in fec.NAMES + [fec.ENSEMBLE]:
        entry = {'OOF': {'variants': {}}, 'TEST': {'variants': {}}}
        for lab, kind, source, refit in VARIANTS:
            entry['OOF']['variants'][lab] = oof_variant(fec, fst, oof, nm, kind, source, refit)
            entry['TEST']['variants'][lab] = test_variant(
                fec, fst, oof, tst, test_lab, test_mask, nm, kind, source, refit)
        out[nm] = entry
        o = entry['OOF']['variants']
        print(f'  [{pk}] {nm:<13} '
              f'orig {o["original"]["mean_slope"]:+.4f} | '
              f'glob eval/fix {o["global_center_eval_fixedw"]["mean_slope"]:+.4f} '
              f'-> train/fix {o["global_center_train_fixedw"]["mean_slope"]:+.4f} '
              f'-> train/refit {o["global_center_train_refitw"]["mean_slope"]:+.4f} | '
              f'bin train/refit {o["bin_center_train_refitw"]["mean_slope"]:+.4f}', flush=True)
    return out


def main():
    res = {'seed': SEED, 'n_boot': N_BOOT,
           'variants': {lab: {'kind': k, 'const_source': s, 'refit_w': r}
                        for lab, k, s, r in VARIANTS},
           'note': ('중심화 후 w 재적합(fec.fit_w, 101-그리드, 학습 폴드 RMSE 최소화)과 '
                    '학습 폴드 기반 상수 추정을 적용한 재계산. 재학습 없음. '
                    'delta = fusion RMSE - (중심화된) image branch RMSE.'),
           'passes': {}}
    for pk, tree in TREES.items():
        print(f'=== 패스 {pk} ===', flush=True)
        res['passes'][pk] = run_tree(pk, tree)
    p = REPO / 'experiments/bias_structure/bias_by_bin_refit.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {p}')


if __name__ == '__main__':
    main()
