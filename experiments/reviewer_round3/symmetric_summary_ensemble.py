#!/usr/bin/env python3
"""(2) 요약 분지의 대칭적 강화 — 패스 B, out-of-fold.

배경: head-to-head 는 5백본 앙상블 대 **단일** XGBoost 다. 이미지 쪽만 앙상블이므로
-0.852 dB 중 일부가 표현의 차이가 아니라 앙상블 효과일 수 있다. 요약 분지도 같은
방식으로 앙상블해서 그 몫을 떼어낸다.

두 가지 요약 앙상블을 만든다.
  (a) seed 앙상블  : 같은 XGBoost 를 random_state 42/43/44 로 적합해 평균.
                     하이퍼파라미터는 ablate_vert_cd.XGB_KW 그대로, seed 만 바꾼다.
  (b) 다중 모델 앙상블: XGBoost + RandomForest + ExtraTrees + Ridge + SVR 을 평균.
                     XGB 외 넷은 scikit-learn 기본값 그대로, 조정하지 않는다.

입력(26 파라미터)·폴드·52 지점별 구조·결측 대치는 전부 패스 B 트리의
scripts/ablate_vert_cd.py 에서 import 한다. 새로 정의하지 않는다.
평가 기준선(키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합)은 fusion_eval_common 을
그대로 쓴다 — direct_test.py 와 같은 분모여야 -0.852 와 나란히 읽힌다.

XGB 만 재적합한다. CNN 은 저장된 예측을 읽는다. 정본 runs/ 와 원고 미변경.
출력: experiments/reviewer_round3/symmetric_summary_ensemble.{json,md}
      (예측 npz 는 experiments/reviewer_round3/summary_preds/ 에만 쓴다)
env: hvf
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'
OUT_JSON = REPO / 'experiments/reviewer_round3/symmetric_summary_ensemble.json'
PRED_DIR = REPO / 'experiments/reviewer_round3/summary_preds'
SEED, N_BOOT = 42, 5000

GUARD_TOL = 5e-3
GUARDS = {
    'single_xgb_pooled_rmse': 8.6603,
    'single_xgb_pooled_mae': 6.3975,
    'image_ensemble_pooled_rmse': 8.115,
}
# 재현 확인용 (가드 아님): direct_test.md 의 5백본 앙상블 대 단일 XGB 안별 차이
REF_DELTA_RMSE = -0.852


def load_modules():
    for m in ('fusion_eval_common', 'oof_common', 'ablate_vert_cd'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(TREE / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        abl = importlib.import_module('ablate_vert_cd')
        assert fec.ROOT == TREE and abl.ROOT == TREE, 'ROOT 불일치'
        return fec, abl
    finally:
        sys.path.pop(0)


FEC, ABL = load_modules()
OOF, TST, TEST_LAB, TEST_MASK = FEC.load_all(ensemble=True)
ENS = FEC.ENSEMBLE
DROP = False        # 26 파라미터 기준선


# ------------------------------------------------------- 요약 분지 재적합
def make_model(kind: str, seed: int):
    """XGB 는 ablate_vert_cd.XGB_KW 에서 seed 만 바꾼다. 나머지는 sklearn 기본값."""
    if kind == 'xgb':
        kw = dict(ABL.XGB_KW); kw['random_state'] = seed
        return XGBRegressor(**kw)
    if kind == 'rf':
        return RandomForestRegressor(random_state=seed)
    if kind == 'et':
        return ExtraTreesRegressor(random_state=seed)
    if kind == 'ridge':
        return Ridge()
    if kind == 'svr':
        return SVR()
    raise ValueError(kind)


def design(rows):
    X = np.stack([ABL.row_to_xy(r, DROP)[0] for r in rows])
    Y = np.stack([ABL.row_to_xy(r, DROP)[1] for r in rows])
    M = np.stack([ABL.row_to_xy(r, DROP)[2] for r in rows])
    return X, Y, M


def fit_predict(rows_tr, rows_va, kind, seed):
    """ablate_vert_cd.fit_predict 와 같은 절차 — 모델만 갈아끼운다.
    결측 대치는 학습 폴드 열 평균, 지점별로 유효 표본 10 미만이면 건너뛴다."""
    Xtr, Ytr, Mtr = design(rows_tr)
    Xva, _, _ = design(rows_va)
    cm = np.nanmean(Xtr, axis=0); cm = np.where(np.isnan(cm), 0.0, cm)
    Xtr = np.where(np.isnan(Xtr), cm, Xtr)
    Xva = np.where(np.isnan(Xva), cm, Xva)
    pred = np.full((len(rows_va), ABL.N_PT), np.nan, np.float32)
    for pi in range(ABL.N_PT):
        valid = Mtr[:, pi]
        if valid.sum() < 10:
            continue
        mdl = make_model(kind, seed)
        mdl.fit(Xtr[valid], Ytr[valid, pi])
        pred[:, pi] = mdl.predict(Xva)
    return pred


def oof_predict(rows, kind, seed):
    """fold별 val 예측을 dict[key] -> 52벡터 로 돌려준다."""
    out = {}
    for k in range(5):
        vf = str(k)
        rows_tr = [r for r in rows if r['cv_fold'] not in (vf, 'test')]
        rows_va = [r for r in rows if r['cv_fold'] == vf]
        pv = fit_predict(rows_tr, rows_va, kind, seed)
        for r, p in zip(rows_va, pv):
            out[ABL.row_key(r)] = p
    return out


# --------------------------------------------- 공통 기준선 위로 재배열
KEYS = [k for j in range(5) for k in OOF[j]['keys']]
PID = np.array([str(k[0]) for k in KEYS], dtype=object)
LAB = np.concatenate([OOF[k]['lab'] for k in range(5)])
MASK = np.concatenate([OOF[k]['mask'] for k in range(5)]).astype(bool)
CNN_ENS = np.concatenate([OOF[k]['cnns'][ENS] for k in range(5)])
XGB_STORED = np.concatenate([OOF[k]['xgb'] for k in range(5)])


def align(pred_map):
    """공통 기준선의 키 순서로 재배열한다. 키는 (patient_id, eye, vf_date) 문자열 튜플."""
    rows = []
    for k in KEYS:
        kk = tuple(str(x) for x in k)
        if kk not in pred_map:
            raise KeyError(f'재적합 예측에 없는 키: {kk}')
        rows.append(pred_map[kk])
    return np.stack(rows)


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
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    ng = len(groups)
    idx = rng.integers(0, ng, (N_BOOT, ng))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def block(pred, tag):
    pr, pm = pooled(pred, LAB, MASK)
    er = eye_metric(pred, LAB, MASK, 'rmse')
    em = eye_metric(pred, LAB, MASK, 'mae')
    ok = np.isfinite(er)
    return {'label': tag, 'pooled_rmse': pr, 'pooled_mae': pm,
            'eye_rmse_mean': float(er[ok].mean()),
            'eye_rmse_sd_pop': float(er[ok].std(ddof=0)),
            'eye_mae_mean': float(em[ok].mean()),
            'eye_mae_sd_pop': float(em[ok].std(ddof=0)),
            'n_eyes': int(ok.sum())}


def compare(img, smy, smy_tag, kind):
    """direct_test.py 와 같은 부호 규약: delta = M(image) - M(summary), 음수 = image 우세."""
    ei, es = eye_metric(img, LAB, MASK, kind), eye_metric(smy, LAB, MASK, kind)
    ok = np.isfinite(ei) & np.isfinite(es)
    ei, es, p_ = ei[ok], es[ok], PID[ok]
    d = ei - es
    lo, hi = boot_ci(d, p_, np.random.default_rng(SEED))
    return {'image': '5백본 이미지 앙상블', 'summary': smy_tag, 'metric': kind,
            'convention': 'delta = per-eye M(image) - M(summary); 음수 = image 우세',
            'n_eyes': int(ok.sum()), 'n_patients': int(np.unique(p_).size),
            'delta': float(d.mean()), 'ci_lo': lo, 'ci_hi': hi,
            'median_delta': float(np.median(d)),
            'p_wilcoxon': float(wilcoxon(ei, es).pvalue),
            'n_image_better': int((d < 0).sum())}


# ------------------------------------------------------------------ 실행
def main() -> int:
    rows = list(csv.DictReader(open(ABL.CSV, encoding='utf-8-sig')))
    print(f'CSV {ABL.CSV.name}: {len(rows)} 행, feature {len(ABL.feats("OD", DROP))} 개', flush=True)

    members = [('xgb', 42), ('xgb', 43), ('xgb', 44),
               ('rf', 42), ('et', 42), ('ridge', 42), ('svr', 42)]
    preds, timing = {}, {}
    for kind, seed in members:
        t0 = time.time()
        preds[(kind, seed)] = align(oof_predict(rows, kind, seed))
        timing[f'{kind}_s{seed}'] = round(time.time() - t0, 1)
        pr, pm = pooled(preds[(kind, seed)], LAB, MASK)
        print(f'  {kind}_s{seed:<3} 적합 {timing[f"{kind}_s{seed}"]:>6.1f}s  '
              f'pooled RMSE {pr:.4f}  MAE {pm:.4f}', flush=True)

    X42 = preds[('xgb', 42)]

    # ------------------------------------------------------------ 가드
    pr42, pm42 = pooled(X42, LAB, MASK)
    ens_r, _ = pooled(CNN_ENS, LAB, MASK)
    got = {'single_xgb_pooled_rmse': pr42, 'single_xgb_pooled_mae': pm42,
           'image_ensemble_pooled_rmse': ens_r}
    guards, all_ok = {}, True
    for name, want in GUARDS.items():
        ok = abs(got[name] - want) <= GUARD_TOL
        all_ok &= ok
        guards[name] = {'got': round(got[name], 4), 'want': want, 'tol': GUARD_TOL, 'ok': bool(ok)}
    # 저장된 정본 seed42 예측과 셀 단위로 얼마나 벌어지는지 (보고용, 가드 아님)
    guards['refit_vs_stored_seed42_max_abs_cell_diff'] = float(
        np.abs(X42 - XGB_STORED)[MASK].max())
    guards['refit_vs_stored_seed42_stored_pooled_rmse'] = round(
        pooled(XGB_STORED, LAB, MASK)[0], 4)

    if not all_ok:
        OUT_JSON.write_text(json.dumps(
            {'status': 'GUARD_FAILED', 'guards': guards,
             'note': '가드 불일치. 결과를 내지 않는다.'}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        print('GUARD FAILED'); print(json.dumps(guards, ensure_ascii=False, indent=1))
        return 1
    print('\n가드 통과.', flush=True)

    # -------------------------------------------------------- 두 앙상블
    SEED_ENS = np.mean([preds[('xgb', s)] for s in (42, 43, 44)], 0)
    MULTI = np.mean([preds[('xgb', 42)], preds[('rf', 42)], preds[('et', 42)],
                     preds[('ridge', 42)], preds[('svr', 42)]], 0)

    rows_out = [block(X42, 'summary: 단일 XGBoost (seed 42)')]
    for kind, seed in members[1:]:
        rows_out.append(block(preds[(kind, seed)], f'summary 구성원: {kind} (seed {seed})'))
    rows_out += [
        block(SEED_ENS, 'summary: XGB seed 앙상블 (42/43/44)'),
        block(MULTI, 'summary: 다중 모델 앙상블 (XGB+RF+ET+Ridge+SVR)'),
        block(CNN_ENS, 'image: 5백본 이미지 앙상블'),
    ]

    comparisons = []
    for kind in ('rmse', 'mae'):
        comparisons.append(compare(CNN_ENS, X42, '단일 XGBoost (seed 42)', kind))
        comparisons.append(compare(CNN_ENS, SEED_ENS, 'XGB seed 앙상블 (42/43/44)', kind))
        comparisons.append(compare(CNN_ENS, MULTI, '다중 모델 앙상블 (5종)', kind))

    payload = {
        'item': '(2) 요약 분지의 대칭적 강화',
        'tree': str(TREE), 'pass': 'B', 'split': 'out-of-fold',
        'seed': SEED, 'n_boot': N_BOOT,
        'ci': '환자 군집 부트스트랩 95% percentile',
        'note': ('요약 분지만 재적합. 입력 26 파라미터·폴드·52 지점별 구조·결측 대치는 '
                 'ablate_vert_cd.py 에서 import. XGB 는 XGB_KW 에서 seed 만 변경, '
                 'RF/ET/Ridge/SVR 은 scikit-learn 기본값 그대로 (스케일링 없음). '
                 'per-eye SD 는 모집단 분모(ddof=0).'),
        'reference_delta_rmse_single_xgb': REF_DELTA_RMSE,
        'guards': guards, 'fit_seconds': timing,
        'members': [f'{k}_s{s}' for k, s in members],
        'sklearn_defaults_caveat': ('Ridge / SVR 은 표준화 없이 원 스케일 26 입력에 적합했다. '
                                    'SVR 기본값(RBF, C=1, epsilon=0.1)은 이 스케일에서 사실상 '
                                    '상수 예측에 가깝다 — 다중 모델 앙상블 행을 읽을 때 감안할 것. '
                                    '지시대로 조정하지 않았다.'),
        'rows': rows_out, 'comparisons': comparisons,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    print()
    for r in rows_out:
        print(f"  {r['label']:<48} pooled RMSE {r['pooled_rmse']:.4f}  MAE {r['pooled_mae']:.4f}  "
              f"per-eye {r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f}")
    print()
    for c in comparisons:
        print(f"  {c['metric']} image vs {c['summary']:<34} d={c['delta']:+.4f} "
              f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p_wilcoxon']:.3g} "
              f"{c['n_image_better']}/{c['n_eyes']}")

    # 예측 저장 (정본 밖)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRED_DIR / 'summary_oof_preds.npz',
        keys=np.array([['%s' % x for x in k] for k in KEYS], dtype=object),
        lab=LAB, mask=MASK,
        **{f'{k}_s{s}': preds[(k, s)] for k, s in members},
        seed_ensemble=SEED_ENS, multi_model_ensemble=MULTI)
    print(f'\n-> {OUT_JSON}\n-> {PRED_DIR}/summary_oof_preds.npz')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
