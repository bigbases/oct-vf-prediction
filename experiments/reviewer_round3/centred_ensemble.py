#!/usr/bin/env python3
"""(1) 중심화한 5백본 이미지 앙상블 '단독' 의 전체 오차 — 패스 B, out-of-fold.

배경: 중심화 반사실(experiments/bias_structure/)은 구간 기울기만 본다. 중심화한
앙상블 단독의 pooled RMSE / MAE 는 어디에도 없다. 그 값이 앙상블 fusion(7.963)과
같아지면 "요약 파라미터가 더하는 것은 보정" 이라는 결론이 닫히고, 여전히 fusion 이
낫다면 그 결론은 과장이다.

중심화 규약은 bias_by_bin_refit.py 의 `global_center_train_fixedw` 와 같다:
fold k 의 상수는 나머지 4 fold 의 마스크된 지점 잔차 평균이다. 평가 대상 fold 는
상수 추정에 들어가지 않는다. 구간중심화(oracle)는 참고로만 함께 낸다.

w 는 재적합하지 않는다 — fusion 행은 원고 tab:ceiling 규약(fold별 nested w)이다.
중심화 후 w 를 다시 맞춘 행은 보조로만 붙인다.

읽기 전용. 재학습 없음. 정본 runs/ 와 원고를 건드리지 않는다.
출력: experiments/reviewer_round3/centred_ensemble.{json,md}
env: hvf
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'      # 패스 B
OUT_JSON = REPO / 'experiments/reviewer_round3/centred_ensemble.json'
OUT_MD = REPO / 'experiments/reviewer_round3/centred_ensemble.md'
SEED, N_BOOT = 42, 5000

GUARD_TOL = 5e-3
GUARDS = {
    'oof_ensemble_pooled_rmse': 8.115,      # 중심화 전 앙상블 단독 (원고 tab:ceiling)
    'oof_ensemble_fusion_pooled_rmse': 7.963,   # 앙상블 fusion, nested w
    'oof_summary_pooled_rmse': 8.6603,      # 단일 XGB (Table 1)
}


def load_modules():
    for m in ('fusion_eval_common', 'fusion_sensitivity_trend', 'oof_common', 'fig_style'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(TREE / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        fst = importlib.import_module('fusion_sensitivity_trend')
        assert fec.ROOT == TREE, f'ROOT 불일치: {fec.ROOT} != {TREE}'
        return fec, fst
    finally:
        sys.path.pop(0)


FEC, FST = load_modules()
OOF, TST, TEST_LAB, TEST_MASK = FEC.load_all(ensemble=True)
ENS = FEC.ENSEMBLE


# ------------------------------------------------------------------ 통계
def pooled(pred, lab, mask):
    d = (pred - lab)[mask]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def eye_metric(pred, lab, mask, kind):
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.any():
            d = pred[i][m] - lab[i][m]
            out[i] = np.sqrt(np.mean(d ** 2)) if kind == 'rmse' else np.mean(np.abs(d))
    return out


def boot_ci(v, pid, rng):
    """환자 군집 부트스트랩 (direct_test.py 와 같은 통계량·같은 난수 소비)."""
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    ng = len(groups)
    idx = rng.integers(0, ng, (N_BOOT, ng))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def block(pred, lab, mask, tag):
    pr, pm = pooled(pred, lab, mask)
    er = eye_metric(pred, lab, mask, 'rmse')
    em = eye_metric(pred, lab, mask, 'mae')
    ok = np.isfinite(er)
    return {
        'label': tag,
        'pooled_rmse': pr, 'pooled_mae': pm,
        'eye_rmse_mean': float(er[ok].mean()),
        'eye_rmse_sd_pop': float(er[ok].std(ddof=0)),
        'eye_mae_mean': float(em[ok].mean()),
        'eye_mae_sd_pop': float(em[ok].std(ddof=0)),
        'n_eyes': int(ok.sum()),
    }


def compare(a_pred, b_pred, lab, mask, pid, a_tag, b_tag, kind='rmse'):
    """delta = M(a) - M(b) per eye. 음수 = a 우세."""
    ea, eb = eye_metric(a_pred, lab, mask, kind), eye_metric(b_pred, lab, mask, kind)
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb, p_ = ea[ok], eb[ok], pid[ok]
    d = ea - eb
    lo, hi = boot_ci(d, p_, np.random.default_rng(SEED))
    return {
        'a': a_tag, 'b': b_tag, 'metric': kind,
        'convention': f'delta = per-eye {kind.upper()}({a_tag}) - {kind.upper()}({b_tag}); 음수 = {a_tag} 우세',
        'n_eyes': int(ok.sum()), 'n_patients': int(np.unique(p_).size),
        'delta': float(d.mean()), 'ci_lo': lo, 'ci_hi': hi,
        'median_delta': float(np.median(d)),
        'p_wilcoxon': float(wilcoxon(ea, eb).pvalue),
        'n_a_better': int((d < 0).sum()),
    }


# ------------------------------------------------------ 중심화 상수 (학습 폴드)
def const_global(folds):
    r = np.concatenate([(f['cnns'][ENS] - f['lab'])[f['mask'].astype(bool)] for f in folds])
    return float(r.mean())


def const_bin(folds):
    r = np.concatenate([(f['cnns'][ENS] - f['lab'])[f['mask'].astype(bool)] for f in folds])
    bi = np.concatenate([FST.bin_index(f['lab'][f['mask'].astype(bool)]) for f in folds])
    g = float(r.mean())
    return np.array([float(r[bi == b].mean()) if (bi == b).any() else g for b in range(4)])


def apply_center(cnn, lab, const):
    if np.ndim(const) == 0:
        return cnn - float(const)
    out = cnn.copy()
    bi = FST.bin_index(lab)
    for b in range(4):
        out[bi == b] -= const[b]
    return out


def centered_folds(folds, const_of):
    """fit_w 를 중심화된 image branch 위에서 다시 돌리기 위한 사본."""
    out = []
    for f in folds:
        g = dict(f); g['cnns'] = dict(f['cnns'])
        g['cnns'][ENS] = apply_center(f['cnns'][ENS], f['lab'], const_of)
        out.append(g)
    return out


# ------------------------------------------------------------ OOF 조립
LAB = np.concatenate([OOF[k]['lab'] for k in range(5)])
MASK = np.concatenate([OOF[k]['mask'] for k in range(5)]).astype(bool)
XGB = np.concatenate([OOF[k]['xgb'] for k in range(5)])
CNN = np.concatenate([OOF[k]['cnns'][ENS] for k in range(5)])
PID = np.array([str(k[0]) for j in range(5) for k in OOF[j]['keys']], dtype=object)

fus, cnn_gc, cnn_bc, fus_gc_refit = [], [], [], []
w_nested, c_global, c_bin, w_gc_refit = [], [], [], []
for k in range(5):
    tr = [OOF[i] for i in range(5) if i != k]
    f = OOF[k]
    w = FEC.fit_w(tr, ENS)
    fus.append(w * f['xgb'] + (1 - w) * f['cnns'][ENS])
    w_nested.append(round(float(w), 4))

    cg = const_global(tr)
    cb = const_bin(tr)
    c_global.append(cg)
    c_bin.append([round(float(x), 4) for x in cb])
    ck = apply_center(f['cnns'][ENS], f['lab'], cg)
    cnn_gc.append(ck)
    cnn_bc.append(apply_center(f['cnns'][ENS], f['lab'], cb))

    wr = FEC.fit_w(centered_folds(tr, cg), ENS)
    w_gc_refit.append(round(float(wr), 4))
    fus_gc_refit.append(wr * f['xgb'] + (1 - wr) * ck)

FUS = np.concatenate(fus)
CNN_GC = np.concatenate(cnn_gc)
CNN_BC = np.concatenate(cnn_bc)
FUS_GC = np.concatenate(fus_gc_refit)

# ------------------------------------------------------------------ 가드
got = {
    'oof_ensemble_pooled_rmse': pooled(CNN, LAB, MASK)[0],
    'oof_ensemble_fusion_pooled_rmse': pooled(FUS, LAB, MASK)[0],
    'oof_summary_pooled_rmse': pooled(XGB, LAB, MASK)[0],
}
guards, all_ok = {}, True
for name, want in GUARDS.items():
    ok = abs(got[name] - want) <= GUARD_TOL
    all_ok &= ok
    guards[name] = {'got': round(got[name], 4), 'want': want, 'tol': GUARD_TOL, 'ok': bool(ok)}

if not all_ok:
    OUT_JSON.write_text(json.dumps(
        {'status': 'GUARD_FAILED', 'tree': str(TREE), 'guards': guards,
         'note': '가드 불일치. 결과를 내지 않는다.'}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('GUARD FAILED'); print(json.dumps(guards, ensure_ascii=False, indent=1))
    sys.exit(1)

# ------------------------------------------------------------------ 산출
rows = [
    block(XGB, LAB, MASK, 'summary (XGBoost 단독)'),
    block(CNN, LAB, MASK, '이미지 앙상블 (중심화 전)'),
    block(CNN_GC, LAB, MASK, '이미지 앙상블 (전역중심화, 학습폴드 상수)'),
    block(FUS, LAB, MASK, '앙상블 fusion (nested w, 중심화 없음)'),
    block(CNN_BC, LAB, MASK, '[참고/oracle] 이미지 앙상블 (구간중심화, 학습폴드 상수)'),
    block(FUS_GC, LAB, MASK, '[보조] 전역중심화 앙상블 + fusion (w 재적합)'),
]

comparisons = [
    compare(CNN_GC, FUS, LAB, MASK, PID, '전역중심화 앙상블', '앙상블 fusion', 'rmse'),
    compare(CNN_GC, FUS, LAB, MASK, PID, '전역중심화 앙상블', '앙상블 fusion', 'mae'),
    compare(CNN_GC, CNN, LAB, MASK, PID, '전역중심화 앙상블', '중심화 전 앙상블', 'rmse'),
    compare(FUS_GC, CNN_GC, LAB, MASK, PID, '전역중심화 후 fusion', '전역중심화 앙상블', 'rmse'),
]

const = {
    'definition': 'fold k 의 상수 = 나머지 4 fold 의 마스크된 지점에서의 (앙상블 예측 - 실측) 평균',
    'global_by_fold': [round(c, 4) for c in c_global],
    'global_mean': float(np.mean(c_global)),
    'global_sd_pop': float(np.std(c_global, ddof=0)),
    'global_min': float(np.min(c_global)), 'global_max': float(np.max(c_global)),
    'global_range': float(np.max(c_global) - np.min(c_global)),
    'global_all_folds_reference': round(const_global([OOF[k] for k in range(5)]), 4),
    'bin_by_fold': c_bin,
    'bin_labels': list(FST.BIN_LABELS),
}

payload = {
    'item': '(1) 중심화한 이미지 앙상블 단독의 성능',
    'tree': str(TREE), 'pass': 'B', 'split': 'out-of-fold',
    'seed': SEED, 'n_boot': N_BOOT,
    'ci': '환자 군집 부트스트랩 95% percentile',
    'note': ('전역중심화 = 앙상블 예측에서 학습 폴드 잔차 평균 스칼라 하나를 뺀다. '
             'w 재적합 없음(보조 행 제외). 재학습 없음, 저장된 예측만 사용. '
             'per-eye SD 는 모집단 분모(ddof=0).'),
    'guards': guards,
    'centering_constant': const,
    'w_nested_by_fold': w_nested,
    'w_refit_after_global_centering_by_fold': w_gc_refit,
    'rows': rows,
    'comparisons': comparisons,
}
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

print('guards OK')
for r in rows:
    print(f"  {r['label']:<46} pooled RMSE {r['pooled_rmse']:.4f}  MAE {r['pooled_mae']:.4f}  "
          f"per-eye RMSE {r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f}")
print()
for c in comparisons:
    print(f"  {c['metric']} {c['a']} vs {c['b']}: d={c['delta']:+.4f} "
          f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p_wilcoxon']:.3g} "
          f"{c['n_a_better']}/{c['n_eyes']}")
print(f"\n상수 (fold별): {const['global_by_fold']}  평균 {const['global_mean']:.4f} "
      f"SD {const['global_sd_pop']:.4f} 범위 {const['global_range']:.4f}")
print('->', OUT_JSON)
