#!/usr/bin/env python3
"""ResNet50 을 5백본 앙상블에 late fusion 으로 더한다 (등가중 6백본이 아니라).

sixth_backbone.py 는 6번째 CNN 을 **등가중 평균**으로 더해 8.2232 (악화) 를 냈다.
그 비교는 XGB 추가(스칼라 w 적합, 7.963)와 결합 방식이 달라 불리하게 기울어 있다.
여기서는 결합 방식을 맞춘다:

    y = w * y_ResNet50 + (1 - w) * y_5ens

w 는 식 (2)/XGB 추가와 **완전히 같은 규약**으로 적합한다 — 폴드 k 의 w 는 나머지
4 폴드의 유효 셀 pooled RMSE 를 최소화하는 스칼라 하나 (grid linspace(0,1,101)).
FEC.fit_w 를 두 임의 소스용으로 일반화한 것 외에 절차는 동일하다.

패스 B, out-of-fold, 저장된 예측만 읽는다. 재학습 없음.
basis A = XGB ∩ 기존 5백본 (가드 8.115 / 7.963 확인)
basis B = XGB ∩ 6백본       (보고는 여기서 — sixth_backbone.py 와 같은 분모)

부호 규약: delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.
원고와 정본 runs/ 를 건드리지 않는다.
출력: experiments/reviewer_round3/resnet50_late_fusion.{json,md}
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'
R50 = REPO / 'experiments/reviewer_round3/runs_r3/phasec_b0_resnet50_5fold'
OUT_JSON = REPO / 'experiments/reviewer_round3/resnet50_late_fusion.json'
N_BOOT, SEED = 5000, 42
GUARD_TOL = 5e-3
GUARDS = {'oof_ensemble5_pooled_rmse_basisA': 8.115,
          'oof_ensemble5_fusion_pooled_rmse_basisA': 7.963,
          'oof_summary_pooled_rmse_basisA': 8.6603,
          'oof_ensemble6_equal_pooled_rmse_basisB': 8.2232}


def load_fec():
    for m in ('fusion_eval_common', 'oof_common'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(TREE / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        assert fec.ROOT == TREE, f'ROOT 불일치: {fec.ROOT} != {TREE}'
        return fec
    finally:
        sys.path.pop(0)


FEC = load_fec()
sys.path.insert(0, str(TREE / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

NAMES5 = list(FEC.NAMES)
NEW = 'ResNet50'
NAMES6 = NAMES5 + [NEW]
E5, E6 = 'ENS5', 'ENS6'


def load_fold6(k: int, split: str) -> dict:
    """FEC.load_fold 와 같은 절차 + ResNet50 한 항목. sixth_backbone.py 와 동일."""
    xz = load_oof_npz(FEC.xgb_path(k, split))
    fn = f'{split}_preds_fold{k}.npz'
    czs = {nm: load_oof_npz(TREE / d / fn) for nm, d in FEC.BB.items()}
    czs[NEW] = load_oof_npz(R50 / fn)
    keys = set(xz['keys'])
    for cz in czs.values():
        keys &= set(cz['keys'])
    keys = sorted(keys)

    def pick(z):
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        ii = [idx[kk] for kk in keys]
        return z['pred'][ii], z['labels'][ii], z['mask'][ii]

    xp, lab, xm = pick(xz)
    cnns, cmask = {}, None
    for nm, cz in czs.items():
        cp, _, cm = pick(cz)
        cnns[nm] = cp
        cmask = cm if cmask is None else (cmask & cm)
    return dict(keys=keys, xgb=xp, cnns=cnns, lab=lab, mask=(xm & cmask).astype(bool))


def add_ens(d, names, tag):
    d['cnns'][tag] = np.mean(np.stack([d['cnns'][n] for n in names]), 0)
    return d


A = {k: add_ens(FEC.load_fold(k, 'val'), NAMES5, E5) for k in range(5)}
B = {k: add_ens(add_ens(load_fold6(k, 'val'), NAMES5, E5), NAMES6, E6) for k in range(5)}


# ------------------------------------------------------------------ 통계 도구
def cat(folds, f):
    return np.concatenate([f(folds[k]) for k in range(5)])


def pooled(pred, lab, mask):
    return float(np.sqrt(np.mean((pred - lab)[mask] ** 2)))


def pooled_mae(pred, lab, mask):
    return float(np.mean(np.abs(pred - lab)[mask]))


def boot_ci(v, pid, rng):
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    idx = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def src(f, tag):
    """'XGB' 는 f['xgb'], 그 밖은 f['cnns'][tag]."""
    return f['xgb'] if tag == 'XGB' else f['cnns'][tag]


def fit_w_pair(folds, tag_a: str, tag_b: str, obj: str = 'rmse') -> float:
    """FEC.fit_w 를 두 임의 소스용으로 일반화. y = w*a + (1-w)*b.

    FEC.fit_w(folds, cnn) 는 (tag_a='XGB', tag_b=cnn) 과 같은 함수다 — grid,
    유효 셀만, 목적함수, 동점 처리(첫 최소 채택)까지 동일하게 맞췄다.
    """
    a = np.concatenate([src(f, tag_a)[f['mask']] for f in folds])
    b = np.concatenate([src(f, tag_b)[f['mask']] for f in folds])
    lab = np.concatenate([f['lab'][f['mask']] for f in folds])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = w * a + (1 - w) * b - lab
        v = np.sqrt(np.mean(r ** 2)) if obj == 'rmse' else np.mean(np.abs(r))
        if v < best[0]:
            best = (v, float(w))
    return best[1]


def nested_fusion_pair(folds, tag_a: str, tag_b: str):
    """폴드 k 의 w 는 나머지 4폴드에서 적합. FEC.oof_arrays 와 같은 규약."""
    out, ws = [], []
    for k in range(5):
        tr = [folds[i] for i in range(5) if i != k]
        w = fit_w_pair(tr, tag_a, tag_b)
        ws.append(w)
        f = folds[k]
        out.append(w * src(f, tag_a) + (1 - w) * src(f, tag_b))
    return np.concatenate(out), ws


def block(label, pred, lab, mask):
    ei = FEC.eye_metric(pred, lab, mask, 'rmse')
    em = FEC.eye_metric(pred, lab, mask, 'mae')
    return dict(label=label, pooled_rmse=pooled(pred, lab, mask),
                pooled_mae=pooled_mae(pred, lab, mask),
                eye_rmse_mean=float(ei.mean()), eye_rmse_sd_pop=float(ei.std(ddof=0)),
                eye_mae_mean=float(em.mean()), eye_mae_sd_pop=float(em.std(ddof=0)),
                n_eyes=int(pred.shape[0]), n_cells=int(mask.sum()))


def compare(a_lab, b_lab, a, b, lab, mask, pid, kind='rmse'):
    ea = FEC.eye_metric(a, lab, mask, kind)
    eb = FEC.eye_metric(b, lab, mask, kind)
    ok = np.isfinite(ea) & np.isfinite(eb)
    d = ea[ok] - eb[ok]
    lo, hi = boot_ci(d, pid[ok], np.random.default_rng(SEED))
    # 두 예측이 셀 단위로 완전히 동일하면 모든 차이가 정확히 0 이라 Wilcoxon 이
    # 정의되지 않는다 (nan). 검정 실패가 아니라 퇴화이므로 그렇게 적는다.
    degenerate = bool(np.all(d == 0))
    return dict(a=a_lab, b=b_lab, metric=kind, n_eyes=int(ok.sum()),
                delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
                median_delta=float(np.median(d)),
                p_wilcoxon=None if degenerate else float(wilcoxon(ea[ok], eb[ok]).pvalue),
                degenerate_identical=degenerate,
                n_a_better=int((d < 0).sum()))


def w_unconstrained(folds, tag_a: str, tag_b: str) -> float:
    """제약 없는 최소제곱 최적 w (음수 허용). 진단용 — 보고 값은 아니다.

    r(w) = w*(a-b) + (b-lab) 이므로 w* = -<a-b, b-lab> / ||a-b||^2.
    격자가 [0,1] 이라 w=0 이 잡혔을 때, 그게 내부 최적인지 경계에 눌린 것인지 가른다.
    """
    a = np.concatenate([src(f, tag_a)[f['mask']] for f in folds])
    b = np.concatenate([src(f, tag_b)[f['mask']] for f in folds])
    lab = np.concatenate([f['lab'][f['mask']] for f in folds])
    diff = a - b
    return float(-np.dot(diff, b - lab) / np.dot(diff, diff))


def fixed_w_oof(folds, tag_a: str, tag_b: str, w: float):
    """모든 폴드에 같은 고정 w 를 적용한 OOF 예측. 진단용 곡선."""
    return np.concatenate([w * src(folds[k], tag_a) + (1 - w) * src(folds[k], tag_b)
                           for k in range(5)])


# --------------------------------------------------------------------- basis A
LA = cat(A, lambda f: f['lab'])
MA = cat(A, lambda f: f['mask']).astype(bool)
E5A = cat(A, lambda f: f['cnns'][E5])
XA = cat(A, lambda f: f['xgb'])
FUSA, _ = nested_fusion_pair(A, 'XGB', E5)

# basis B 는 등가중 6백본 가드에 필요하므로 먼저 만든다
LB = cat(B, lambda f: f['lab'])
MB = cat(B, lambda f: f['mask']).astype(bool)
E6B_pre = cat(B, lambda f: f['cnns'][E6])

got_map = {'oof_ensemble5_pooled_rmse_basisA': pooled(E5A, LA, MA),
           'oof_ensemble5_fusion_pooled_rmse_basisA': pooled(FUSA, LA, MA),
           'oof_summary_pooled_rmse_basisA': pooled(XA, LA, MA),
           'oof_ensemble6_equal_pooled_rmse_basisB': pooled(E6B_pre, LB, MB)}
guards = {n: dict(got=round(got_map[n], 4), want=w, tol=GUARD_TOL,
                  ok=bool(abs(got_map[n] - w) <= GUARD_TOL))
          for n, w in GUARDS.items()}

if not all(v['ok'] for v in guards.values()):
    OUT_JSON.write_text(json.dumps(
        {'status': 'GUARD_FAILED', 'guards': guards,
         'note': '가드 불일치. 결과를 내지 않는다.'}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('GUARD FAILED')
    print(json.dumps(guards, ensure_ascii=False, indent=1))
    sys.exit(1)
print('가드 통과.', {k: v['got'] for k, v in guards.items()})

# --------------------------------------------------------------------- basis B
PIDB = np.array([str(kk[0]) for k in range(5) for kk in B[k]['keys']], dtype=object)
E5B = cat(B, lambda f: f['cnns'][E5])
E6B = E6B_pre
R50B = cat(B, lambda f: f['cnns'][NEW])
XB = cat(B, lambda f: f['xgb'])

# 본 요청: w 적합 ResNet50 late fusion
FUS_R50, W_R50 = nested_fusion_pair(B, NEW, E5)
# 나란히 놓을 대조: XGB 추가 (같은 basis, 같은 규약)
FUS_XGB, W_XGB = nested_fusion_pair(B, 'XGB', E5)

keys_a = [kk for k in range(5) for kk in A[k]['keys']]
keys_b = [kk for k in range(5) for kk in B[k]['keys']]
basis_shift = dict(
    basis_a_eyes=len(keys_a), basis_b_eyes=len(keys_b),
    basis_a_cells=int(MA.sum()), basis_b_cells=int(MB.sum()),
    eyes_dropped=len(set(map(tuple, keys_a)) - set(map(tuple, keys_b))),
    note='basis B 는 ResNet50 의 키/마스크까지 교집합에 넣은 것이다.')

rows = [block('5백본 앙상블', E5B, LB, MB),
        block('6백본 등가중 (+ResNet50)', E6B, LB, MB),
        block('5백본 + ResNet50 late fusion (w 적합)', FUS_R50, LB, MB),
        block('5백본 + XGB fusion (w 적합)', FUS_XGB, LB, MB),
        block(f'{NEW} 단독', R50B, LB, MB),
        block('summary XGB 단독', XB, LB, MB)]

cmps = []
for kind in ('rmse', 'mae'):
    cmps.append(compare('R50 late fusion', '5백본 앙상블', FUS_R50, E5B, LB, MB, PIDB, kind))
    cmps.append(compare('R50 late fusion', '6백본 등가중', FUS_R50, E6B, LB, MB, PIDB, kind))
    cmps.append(compare('R50 late fusion', 'XGB fusion', FUS_R50, FUS_XGB, LB, MB, PIDB, kind))
    cmps.append(compare('XGB fusion', '5백본 앙상블', FUS_XGB, E5B, LB, MB, PIDB, kind))

# ------------------------------------------------- 진단: w=0 이 경계인가 내부인가
w_unc_r50 = [w_unconstrained([B[i] for i in range(5) if i != k], NEW, E5) for k in range(5)]
w_unc_xgb = [w_unconstrained([B[i] for i in range(5) if i != k], 'XGB', E5) for k in range(5)]
w_curve = [dict(w=round(float(w), 3),
                pooled_rmse=pooled(fixed_w_oof(B, NEW, E5, w), LB, MB),
                pooled_mae=pooled_mae(fixed_w_oof(B, NEW, E5, w), LB, MB))
           for w in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)]
diagnostics = dict(
    w_unconstrained_resnet50_per_fold=w_unc_r50,
    w_unconstrained_resnet50_mean=float(np.mean(w_unc_r50)),
    w_unconstrained_xgb_per_fold=w_unc_xgb,
    w_unconstrained_xgb_mean=float(np.mean(w_unc_xgb)),
    oof_curve_fixed_w_resnet50=w_curve,
    note='제약 없는 최소제곱 최적 w 가 음수면, 적합된 w=0 은 격자 경계에 눌린 것이 아니라 '
         '"ResNet50 을 조금이라도 섞으면 나빠진다"는 뜻이다. 곡선은 모든 폴드에 같은 고정 w 를 '
         '적용한 OOF 값으로, 적합 절차가 아니라 진단이다.')

g5 = pooled(E5B, LB, MB)
gains = dict(
    basis='B', ens5_pooled_rmse=g5,
    gain_add_6th_equal=pooled(E6B, LB, MB) - g5,
    gain_add_6th_late_fusion=pooled(FUS_R50, LB, MB) - g5,
    gain_add_xgb=pooled(FUS_XGB, LB, MB) - g5,
    note='음수 = 개선. 세 이득 전부 같은 basis B, 같은 출발점(5백본 앙상블)에서 잰다.')

payload = dict(
    note='ResNet50 을 5백본 앙상블에 late fusion(스칼라 w) 으로 더한 값. 패스 B, '
         'out-of-fold, 저장된 예측만 사용(재학습 없음). '
         'delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.',
    equation='y = w * y_ResNet50 + (1 - w) * y_5ens',
    w_convention='폴드 k 의 w 는 나머지 4폴드의 유효 셀 pooled RMSE 를 최소화 '
                 '(grid linspace(0,1,101)). 식 (2)/XGB 추가와 동일 규약.',
    tree=str(TREE), resnet50_run=str(R50), seed=SEED, n_boot=N_BOOT,
    backbones5=NAMES5, backbones6=NAMES6,
    guards=guards, basis_shift=basis_shift,
    nested_w_resnet50=W_R50, nested_w_resnet50_mean=float(np.mean(W_R50)),
    nested_w_xgb=W_XGB, nested_w_xgb_mean=float(np.mean(W_XGB)),
    rows_basisB=rows, comparisons_basisB=cmps, gains=gains,
    diagnostics=diagnostics,
    no_retrain=dict(
        resnet50_preds=str(R50),
        note='ResNet50 5-fold 예측은 2026-08-31 sixth_backbone 작업에서 이미 저장된 것을 '
             '읽기만 했다. 이번 작업에서 학습한 모델은 없다.'))
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
print('->', OUT_JSON)
for r in rows:
    print(f"  {r['label']:<38} pooled RMSE {r['pooled_rmse']:.4f}  MAE {r['pooled_mae']:.4f}  "
          f"per-eye {r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f}")
print()
print(f"  w(ResNet50) per-fold = {W_R50}  mean {np.mean(W_R50):.4f}")
print(f"  w(XGB)      per-fold = {W_XGB}  mean {np.mean(W_XGB):.4f}")
print()
for c in cmps:
    pv = '퇴화(전부 0)' if c['degenerate_identical'] else f"{c['p_wilcoxon']:.3g}"
    print(f"  {c['metric']:>4} {c['a']:<18} vs {c['b']:<16} d={c['delta']:+.4f} "
          f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={pv} "
          f"{c['n_a_better']}/{c['n_eyes']}")
print()
print(f"  제약없는 최적 w(R50) per-fold = {[round(v,4) for v in w_unc_r50]}  mean {np.mean(w_unc_r50):+.4f}")
print(f"  제약없는 최적 w(XGB) per-fold = {[round(v,4) for v in w_unc_xgb]}  mean {np.mean(w_unc_xgb):+.4f}")
print("  고정 w 곡선(R50):", "  ".join(f"w={c['w']}:{c['pooled_rmse']:.4f}" for c in w_curve))
print()
print(f"  5백본 {g5:.4f} | +6th 등가중 {gains['gain_add_6th_equal']:+.4f} | "
      f"+6th late fusion {gains['gain_add_6th_late_fusion']:+.4f} | "
      f"+XGB {gains['gain_add_xgb']:+.4f}")
