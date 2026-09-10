#!/usr/bin/env python3
"""채점 코드 sanity check — 답을 이미 아는 입력을 실제 metric 함수에 넣는다.

원고 수치는 저장소 곳곳에 흩어진 30여 개의 metric 함수가 만든다. 각각은 그럴듯해
보이지만 서로 같은 값을 준다는 보장이 없고, 어느 하나가 조용히 틀려도 결과는 여전히
"그럴듯한 dB"로 나온다. 그래서 값을 눈으로 보는 대신 **답이 정해진 입력**을 넣는다.

  T1  예측 = 실측  →  오차가 정확히 0인가
  T2  모든 예측에 상수 c  →  MAE 가 정확히 c 만큼 느는가
  T3  마스크된 지점에 극단값  →  결과가 안 변하는가
  T4  per-eye 평균/표준편차와 pooled 가 RMSE_pooled = sqrt(mu^2 + sigma^2) 를
      만족하는가 (원고 03_methods.tex:202 의 항등식)
  T5  두 모델을 바꿔 넣으면 Wilcoxon p 가 같고 부호만 뒤집히는가

기존 코드는 **읽기만 한다.** 각 함수는 AST 로 정의만 뽑아 격리된 네임스페이스에서
실행하므로, import 시 분석을 돌려버리는 파일(__main__ 가드 없는 것)도 안전하다.
아무 파일도 쓰지 않는다. 결과 해석은 docs/METRIC_SANITY.md.

실행:  python experiments/metric_sanity.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

EPS = 0.0          # T1/T2/T3 은 부동소수점 오차가 아니라 '정확히' 를 요구한다
TOL = 1e-9         # T4/T5 처럼 서로 다른 경로로 같은 값을 계산하는 경우만 허용


# ── 기존 구현을 부작용 없이 꺼내오기 ────────────────────────────────────
_CACHE: dict = {}


def grab(relpath: str, *names):
    """파일에서 지정한 함수 정의만 AST 로 뽑아 격리 네임스페이스에서 실행한다.

    모듈 최상위 코드는 실행되지 않는다 — 분석 스크립트를 import 하는 것과 다르다.
    """
    key = (relpath, names)
    if key in _CACHE:
        return _CACHE[key]
    src = (ROOT / relpath).read_text()
    tree = ast.parse(src)
    want = set(names)
    picked = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in want]
    missing = want - {n.name for n in picked}
    if missing:
        raise KeyError(f'{relpath}: {sorted(missing)} 없음')
    ns = {'np': np, 'numpy': np, 'stats': stats, 'wilcoxon': stats.wilcoxon,
          'float': float, 'int': int}
    try:
        import torch
        ns['torch'] = torch
    except ImportError:
        pass
    exec(compile(ast.Module(body=picked, type_ignores=[]), relpath, 'exec'), ns)
    out = tuple(ns[n] for n in names)
    _CACHE[key] = out
    return out


def _t(x):
    import torch
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


# ── 검사 대상 등록 ──────────────────────────────────────────────────────
# (표시이름, 파일, 호출 어댑터) — 어댑터는 (pred, lab, mask) → 스칼라.
POOLED_RMSE = [
    ('fusion_stats_paired_bootstrap.pooled_rmse', 'scripts/fusion_stats_paired_bootstrap.py',
     lambda p, l, m: grab('scripts/fusion_stats_paired_bootstrap.py', 'pooled_rmse')[0](p, l, m)),
    ('patient_cluster_test.pooled_rmse', 'scripts/patient_cluster_test.py',
     lambda p, l, m: grab('scripts/patient_cluster_test.py', 'pooled_rmse')[0](p, l, m)),
    ('recompute_late_fusion.pooled_metrics', 'scripts/recompute_late_fusion.py',
     lambda p, l, m: grab('scripts/recompute_late_fusion.py', 'pooled_metrics')[0](
         p, l, m.astype(bool))['rmse']),
    # 이름은 pooled_rmse 지만 (rmse, mae) 튜플을 준다. 이름-반환값 불일치는 그 자체로 함정이다.
    ('xgb_diversity_control.pooled_rmse[0]', 'scripts/xgb_diversity_control.py',
     lambda p, l, m: grab('scripts/xgb_diversity_control.py', 'pooled_rmse')[0](p, l, m)[0]),
    ('reliability_sensitivity.pooled[0]', 'scripts/reliability_sensitivity.py',
     lambda p, l, m: grab('scripts/reliability_sensitivity.py', 'pooled')[0](p, l, m)[0]),
    ('baseline_xgb.masked_rmse', 'baseline_xgb.py',
     lambda p, l, m: grab('baseline_xgb.py', 'masked_rmse')[0](
         p.copy(), l, m.astype(bool))[1]),
    ('make_case_heatmap.rmse', 'scripts/make_case_heatmap.py',
     lambda p, l, m: grab('scripts/make_case_heatmap.py', 'rmse')[0](p, l, m)),
    ('train.masked_mse (torch, sqrt 후)', 'train.py',
     lambda p, l, m: float(np.sqrt(grab('train.py', 'masked_mse')[0](
         _t(p), _t(l), _t(m)).item()))),
]

POOLED_MAE = [
    ('recompute_late_fusion.pooled_metrics', 'scripts/recompute_late_fusion.py',
     lambda p, l, m: grab('scripts/recompute_late_fusion.py', 'pooled_metrics')[0](
         p, l, m.astype(bool))['mae']),
    ('baseline_xgb.masked_mae', 'baseline_xgb.py',
     lambda p, l, m: float(grab('baseline_xgb.py', 'masked_mae')[0](
         p.copy(), l, m.astype(bool))[1])),
    ('make_case_heatmap.mae', 'scripts/make_case_heatmap.py',
     lambda p, l, m: grab('scripts/make_case_heatmap.py', 'mae')[0](p, l, m)),
    ('xgb_diversity_control.pooled_rmse[1]', 'scripts/xgb_diversity_control.py',
     lambda p, l, m: grab('scripts/xgb_diversity_control.py', 'pooled_rmse')[0](p, l, m)[1]),
    ('reliability_sensitivity.pooled[1]', 'scripts/reliability_sensitivity.py',
     lambda p, l, m: grab('scripts/reliability_sensitivity.py', 'pooled')[0](p, l, m)[1]),
    ('train.masked_mae (torch)', 'train.py',
     lambda p, l, m: float(grab('train.py', 'masked_mae')[0](_t(p), _t(l), _t(m)).item())),
]

# per-eye RMSE — 반환 형태가 제각각이라 어댑터에서 (n_eyes,) 배열로 맞춘다.
PER_EYE_RMSE = [
    ('fusion_stats_paired_bootstrap.per_eye_rmse', 'scripts/fusion_stats_paired_bootstrap.py',
     lambda p, l, m: grab('scripts/fusion_stats_paired_bootstrap.py', 'per_eye_rmse')[0](p, l, m)[0]),
    ('patient_cluster_test.per_eye_rmse', 'scripts/patient_cluster_test.py',
     lambda p, l, m: grab('scripts/patient_cluster_test.py', 'per_eye_rmse')[0](p, l, m)),
    ('recompute_late_fusion.eye_rmse', 'scripts/recompute_late_fusion.py',
     lambda p, l, m: grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](p, l, m.astype(bool))),
    ('verify_skeleton_numbers.eye_rmse', 'scripts/verify_skeleton_numbers.py',
     lambda p, l, m: grab('scripts/verify_skeleton_numbers.py', 'eye_rmse')[0](p, l, m.astype(bool))),
    ('build_final_model_comparison.eye_metric', 'scripts/build_final_model_comparison.py',
     lambda p, l, m: grab('scripts/build_final_model_comparison.py', 'eye_metric')[0](p, l, m, 'rmse')),
    ('fusion_consistency_matrix.eye_metric', 'scripts/fusion_consistency_matrix.py',
     lambda p, l, m: grab('scripts/fusion_consistency_matrix.py', 'eye_metric')[0](p, l, m, 'rmse')),
    ('compare_osflip.per_eye_rmse', 'scripts/compare_osflip.py',
     lambda p, l, m: grab('scripts/compare_osflip.py', 'per_eye_rmse')[0](p, l, m)),
    ('ablate_vert_cd.per_eye_rmse', 'scripts/ablate_vert_cd.py',
     lambda p, l, m: grab('scripts/ablate_vert_cd.py', 'per_eye_rmse')[0](p, l, m)),
    ('reliability_sensitivity.per_eye_rmse[0]', 'scripts/reliability_sensitivity.py',
     lambda p, l, m: grab('scripts/reliability_sensitivity.py', 'per_eye_rmse')[0](p, l, m)[0]),
    ('xgb_diversity_control.per_eye_rmse', 'scripts/xgb_diversity_control.py',
     lambda p, l, m: grab('scripts/xgb_diversity_control.py', 'per_eye_rmse')[0](p, l, m)),
]

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> bool:
    print(f'    {"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAILS.append(f'{name}  {detail}')
    return ok


def _call(entry, p, l, m):
    try:
        return entry[2](p, l, m)
    except Exception as e:                          # noqa: BLE001
        return f'ERROR: {type(e).__name__}: {e}'


# ── 합성 데이터 ─────────────────────────────────────────────────────────
def synth(n_eyes=40, n_pts=52, seed=0, full_mask=True):
    rng = np.random.default_rng(seed)
    lab = rng.uniform(0.0, 34.0, (n_eyes, n_pts))
    mask = np.ones((n_eyes, n_pts), dtype=bool)
    if not full_mask:                                # 안마다 유효 지점 수를 다르게
        for i in range(n_eyes):
            mask[i, rng.choice(n_pts, rng.integers(1, 8), replace=False)] = False
    return lab, mask, rng


# ── T1 예측 = 실측 ──────────────────────────────────────────────────────
def t1():
    print('\n[T1] 예측 = 실측 → 오차가 정확히 0')
    lab, mask, _ = synth()
    pred = lab.copy()
    for group, gname in ((POOLED_RMSE, 'pooled RMSE'), (POOLED_MAE, 'pooled MAE')):
        print(f'  {gname}')
        for e in group:
            v = _call(e, pred, lab, mask)
            check(e[0], isinstance(v, float) and v == 0.0, f'= {v!r}')
    print('  per-eye RMSE (전 안 정확히 0)')
    for e in PER_EYE_RMSE:
        v = _call(e, pred, lab, mask)
        ok = not isinstance(v, str) and np.all(np.asarray(v, float) == 0.0)
        check(e[0], ok, f'max={np.max(np.asarray(v, float)):.3g}' if not isinstance(v, str) else v)


# ── T2 상수 오프셋 ──────────────────────────────────────────────────────
def t2():
    print('\n[T2] 모든 예측에 상수 c → MAE 가 정확히 c 만큼 증가')
    lab, mask, rng = synth(seed=1)
    for c in (1.0, 3.5, 100.0):
        print(f'  (a) 완전예측 기준, c = {c}   → MAE 는 정확히 c 여야 한다')
        for e in POOLED_MAE:
            v = _call(e, lab + c, lab, mask)
            check(f'{e[0]}  c={c}', isinstance(v, float) and v == c, f'= {v!r}')
    # (b) 잔차가 있는 상태에서. "정확히 c 증가"는 **잔차 부호가 안 바뀔 때만** 성립한다.
    #     |r+c| = |r| + c  는  r 과 c 가 같은 부호일 때만 참이기 때문이다.
    #     그래서 전부 양수인 잔차를 쓴다. 부호가 섞이면 아래 (c)처럼 어긋나는 게 정답이다.
    resid = np.abs(rng.normal(0, 2.0, lab.shape))
    base = lab + resid
    print('  (b) 잔차가 전부 같은 부호(양수)인 예측 기준 → 어떤 c > 0 에도 정확히 c 증가')
    for c in (1.0, 7.25):
        for e in POOLED_MAE:
            m0, m1 = _call(e, base, lab, mask), _call(e, base + c, lab, mask)
            # (a) 와 달리 여기서는 독립적으로 누적된 두 평균의 '차'를 본다. 마지막
            # 자리 반올림이 남는 것이 정상이므로 몇 ulp 를 허용한다.
            ulp = 8 * float(np.spacing(c))
            ok = isinstance(m0, float) and isinstance(m1, float) and abs((m1 - m0) - c) <= ulp
            check(f'{e[0]}  c={c}', ok,
                  f'Δ={m1 - m0:.15f}  (오차 {abs((m1 - m0) - c):.1e}, 허용 {ulp:.1e})'
                  if isinstance(m1, float) else str(m1))
    # (c) 부호가 섞인 잔차: 어긋나는 것이 수학적으로 옳다. 여기서 PASS 가 뜨면 오히려 이상하다.
    mixed = lab + rng.normal(0, 2.0, lab.shape)
    e0 = POOLED_MAE[0]
    for c in (0.5, 20.0):
        d = _call(e0, mixed + c, lab, mask) - _call(e0, mixed, lab, mask)
        print(f'  (c) 부호 섞인 잔차, c = {c}: Δ = {d:.4f} ≠ c — 어긋나는 것이 정답이다.')
    print('      (c 가 최대 잔차보다 커도 Δ = c + mean(잔차) − mean|잔차| 이지 c 가 아니다.)')


# ── T3 마스크된 지점의 극단값 ───────────────────────────────────────────
def t3():
    print('\n[T3] 마스크된 지점에 극단값 → 결과 불변')
    lab, mask, rng = synth(seed=2, full_mask=False)
    pred = lab + rng.normal(0, 2.0, lab.shape)
    off = ~mask
    print(f'  마스크 off 지점 {off.sum()}개에 극단값을 주입한다.')
    for tag, val in (('1e9', 1e9), ('-1e9', -1e9), ('nan', np.nan), ('inf', np.inf)):
        print(f'  주입값 = {tag}')
        poison = pred.copy()
        poison[off] = val
        for group in (POOLED_RMSE, POOLED_MAE, PER_EYE_RMSE):
            for e in group:
                a, b = _call(e, pred, lab, mask), _call(e, poison, lab, mask)
                if isinstance(a, str) or isinstance(b, str):
                    check(f'{e[0]}  [{tag}]', False, str(b if isinstance(b, str) else a))
                    continue
                a, b = np.asarray(a, float), np.asarray(b, float)
                ok = a.shape == b.shape and np.array_equal(a, b)
                check(f'{e[0]}  [{tag}]', ok,
                      '' if ok else f'변함: {np.nanmax(np.abs(b - a)) if np.isfinite(b).any() else b}')


# ── T4 pooled ↔ per-eye 항등식 ─────────────────────────────────────────
def _identity(pred, lab, mask, label, tol=TOL):
    pooled = grab('scripts/fusion_stats_paired_bootstrap.py', 'pooled_rmse')[0](pred, lab, mask)
    eye = grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](pred, lab, mask.astype(bool))
    mu, sd = float(eye.mean()), float(eye.std())
    rhs = float(np.sqrt(mu ** 2 + sd ** 2))
    ok = abs(pooled - rhs) < tol
    check(f'RMSE_pooled = sqrt(mu^2+sigma^2)  [{label}]', ok,
          f'pooled={pooled:.10f}  sqrt={rhs:.10f}  Δ={pooled - rhs:+.2e}')
    # 원고는 pooled MAE = per-eye 평균 MAE 도 '정확히' 라고 적는다
    pm = grab('scripts/recompute_late_fusion.py', 'pooled_metrics')[0](pred, lab, mask.astype(bool))['mae']
    em = float(np.mean([np.mean(np.abs((pred[i] - lab[i])[mask[i]])) for i in range(len(pred))]))
    ok2 = abs(pm - em) < tol
    check(f'MAE_pooled = mean(per-eye MAE)   [{label}]', ok2,
          f'pooled={pm:.10f}  mean={em:.10f}  Δ={pm - em:+.2e}')
    return ok and ok2


def t4():
    print('\n[T4] pooled ↔ per-eye 항등식 (원고 03_methods.tex:200-205)')
    print('  원고는 "every eye contributes exactly 52 valid points" 를 전제로 단다.')
    lab, mask, rng = synth(seed=3, full_mask=True)
    pred = lab + rng.normal(0, 3.0, lab.shape)
    _identity(pred, lab, mask, '유효 지점 수 동일 (전제 성립)')
    lab2, mask2, rng2 = synth(seed=4, full_mask=False)
    pred2 = lab2 + rng2.normal(0, 3.0, lab2.shape)
    print('  전제를 깨면 (안마다 유효 지점 수가 다르면) 어떻게 되는지:')
    pooled = grab('scripts/fusion_stats_paired_bootstrap.py', 'pooled_rmse')[0](pred2, lab2, mask2)
    eye = grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](pred2, lab2, mask2)
    rhs = float(np.sqrt(eye.mean() ** 2 + eye.std() ** 2))
    print(f'    pooled={pooled:.6f}  sqrt(mu^2+sd^2)={rhs:.6f}  Δ={pooled - rhs:+.2e}'
          '   ← 어긋나는 게 정상. 전제가 없으면 성립하지 않는 항등식이다.')


# ── T5 모델 교환 대칭성 ─────────────────────────────────────────────────
def t5():
    print('\n[T5] 두 모델을 바꿔 넣으면 Wilcoxon p 는 같고 Δ 부호만 뒤집힌다')
    lab, mask, rng = synth(seed=5)
    a = lab + rng.normal(0, 2.0, lab.shape)
    b = lab + rng.normal(0, 2.4, lab.shape)
    ea = grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](a, lab, mask)
    eb = grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](b, lab, mask)
    impls = [
        ('recompute_late_fusion.paired_wilcoxon',
         grab('scripts/recompute_late_fusion.py', 'paired_wilcoxon')[0]),
        ('build_final_model_comparison.wil',
         grab('scripts/build_final_model_comparison.py', 'wil')[0]),
    ]
    for nm, fn in impls:
        p1, p2 = fn(ea, eb), fn(eb, ea)
        check(f'{nm}  p(a,b) == p(b,a)', p1 == p2, f'{p1} vs {p2}')
    fs = grab('scripts/fusion_stats_paired_bootstrap.py', 'per_eye_rmse')[0]
    r1, _ = fs(a, lab, mask)
    r2, _ = fs(b, lab, mask)
    p1 = stats.wilcoxon(r1, r2).pvalue
    p2 = stats.wilcoxon(r2, r1).pvalue
    check('scipy wilcoxon 자체의 교환 대칭성', p1 == p2, f'{p1:.12g} vs {p2:.12g}')
    d1, d2 = float((r1 - r2).mean()), float((r2 - r1).mean())
    check('평균 Δ 는 부호만 뒤집힌다', d1 == -d2, f'{d1:+.10f} vs {d2:+.10f}')
    w1 = int(np.sum(r1 < r2)); w2 = int(np.sum(r2 < r1))
    check('우세 안 수의 합 = n (동점 없음)', w1 + w2 == len(r1), f'{w1}+{w2} vs n={len(r1)}')
    # verify_skeleton_numbers.wil 은 부호 규약까지 반환한다 — 교환 시 뒤집혀야 한다
    wil_v = grab('scripts/verify_skeleton_numbers.py', 'wil')[0]
    v1, v2 = wil_v(r1, r2), wil_v(r2, r1)
    print(f'    verify_skeleton_numbers.wil(a,b) = {v1}')
    print(f'    verify_skeleton_numbers.wil(b,a) = {v2}')


# ── 구현 간 일치 + 실제 데이터 ──────────────────────────────────────────
def t6_agreement():
    print('\n[T6] 같은 양을 재는 구현들이 서로 같은 값을 주는가 (합성 데이터)')
    lab, mask, rng = synth(seed=6, full_mask=False)
    pred = lab + rng.normal(0, 3.0, lab.shape)
    for group, gname in ((POOLED_RMSE, 'pooled RMSE'), (POOLED_MAE, 'pooled MAE'),
                         (PER_EYE_RMSE, 'per-eye RMSE')):
        vals = {e[0]: _call(e, pred, lab, mask) for e in group}
        ref_name, ref = next(iter(vals.items()))
        print(f'  {gname}  (기준: {ref_name})')
        for nm, v in vals.items():
            if isinstance(v, str):
                check(nm, False, v); continue
            a, b = np.asarray(ref, float), np.asarray(v, float)
            ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=TOL, equal_nan=True)
            check(nm, ok, '' if ok else f'최대차 {np.nanmax(np.abs(a - b)):.3g}')


def t7_real():
    print('\n[T7] 실제 OOF 데이터에서 전제 확인')
    try:
        from oof_common import load_oof_npz
    except ImportError as e:
        print(f'  건너뜀: {e}'); return
    preds, labs, masks = [], [], []
    for k in range(5):
        f = ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz'
        if not f.exists():
            print(f'  건너뜀: {f.name} 없음'); return
        z = load_oof_npz(f)
        preds.append(z['pred']); labs.append(z['labels']); masks.append(z['mask'])
    P, L, M = np.concatenate(preds), np.concatenate(labs), np.concatenate(masks).astype(bool)
    cnt = M.sum(axis=1)
    print(f'  {P.shape[0]}안 × {P.shape[1]}지점.  안별 유효 지점 수: '
          f'min {cnt.min()}  max {cnt.max()}  유일값 {sorted(set(cnt.tolist()))}')
    check('모든 안이 정확히 52 지점 (T4 항등식의 전제)', bool((cnt == 52).all()),
          f'52가 아닌 안 {int((cnt != 52).sum())}개')
    nf_all = int((~np.isfinite(L)).sum())
    nf_on = int((~np.isfinite(L[M])).sum())
    check('라벨에 비유한값 없음 (마스크 안쪽)', nf_on == 0, f'마스크 안 {nf_on} / 전체 {nf_all}')
    check('예측에 비유한값 없음 (마스크 안쪽)', int((~np.isfinite(P[M])).sum()) == 0)
    check('마스크 바깥에도 비유한값 없음 (torch 손실이 요구한다)',
          int((~np.isfinite(P)).sum()) + int((~np.isfinite(L)).sum()) == 0)
    print(f'  저장 dtype: pred {P.dtype}, labels {L.dtype}')
    # float32 저장이므로 float64 기준 허용오차를 쓰면 실패한다. dtype 정밀도에 맞춘다.
    _identity(P, L, M, '실제 OOF (XGB, float32 정밀도 허용)',
              tol=10 * np.finfo(P.dtype).eps * 8.72)
    # 위에서 항등식이 어긋나면 float32 누적 때문인지 확인한다 — 로직 오류와 구분해야 한다.
    p64, l64 = P.astype(np.float64), L.astype(np.float64)
    pooled = grab('scripts/fusion_stats_paired_bootstrap.py', 'pooled_rmse')[0](p64, l64, M)
    eye = grab('scripts/recompute_late_fusion.py', 'eye_rmse')[0](p64, l64, M)
    rhs = float(np.sqrt(eye.mean() ** 2 + eye.std() ** 2))
    check('같은 항등식을 float64 로 다시 계산하면 정확히 일치', abs(pooled - rhs) == 0.0,
          f'pooled={pooled:.12f}  sqrt={rhs:.12f}  Δ={pooled - rhs:+.2e}')


def main():
    print('=' * 78)
    print('채점 코드 sanity check   ·   기존 코드는 읽기만, 아무 파일도 쓰지 않는다')
    print('=' * 78)
    for fn in (t1, t2, t3, t4, t5, t6_agreement, t7_real):
        fn()
    print('\n' + '=' * 78)
    if FAILS:
        print(f'FAIL {len(FAILS)}건:')
        for f in FAILS:
            print(f'  · {f}')
    else:
        print('전 항목 PASS')
    print('=' * 78)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
