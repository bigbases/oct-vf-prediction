#!/usr/bin/env python3
"""오차 구조 분석 — test 부호 반전의 기전과 "융합은 꼬리를 깎는다"의 증거.

A. 브랜치 균형 이동: OOF 와 test 사이에서 두 단일모달의 상대 강도가 얼마나 움직였는가.
   w 는 OOF 균형에 맞춰 적합되므로, 균형이 이동하면 test 에서 약해진 쪽에 과대 가중이
   걸린다. test 부호 반전이 여기서 설명된다.
   w_test_oracle 은 진단 전용이며 어떤 주장에도 쓰지 않는다(test 적합 = 오염).

B. 오차 분포 형태: pooled RMSE/MAE 비. 비가 크면 전형오차는 작고 꼬리가 두껍다.

C. 꼬리 대 전형: 안 단위 오차분포의 중앙값과 90분위를 fusion 과 CNN 에서 비교한다.
   "큰 오차를 내는 안을 골라서" 비교하면 평균회귀로 fusion 이 유리해지는 순환이 생기므로,
   어느 모델의 오차도 조건으로 걸지 않고 각 모델의 분포 통계만 비교한다.
   분위수 차이에 환자 클러스터 부트스트랩 CI 를 붙여 기술통계가 아닌 추론으로 만든다.
   (fusion_noninferiority.py 와 동일한 절차 — 10000회, seed 42)

D. 평균 기반 검정과 순위 기반 검정이 엇갈리는 칸. 엇갈림 자체가 꼬리 기전의 방증이다.

읽기 전용. 출력: runs/fusion_error_structure.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_eval_common import (  # noqa: E402
    NAMES, load_all, oof_raw, test_raw, eye_metric, fit_w,
)


def pooled(pred, lab, mask):
    d = (pred - lab)[mask.astype(bool)]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


N_BOOT = 10000
SEED = 42


def cluster_boot_quantile_ci(fe, ce, pid, q, rng):
    """환자 클러스터 부트스트랩으로 (분위수(fusion) - 분위수(CNN)) 의 95% CI."""
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        diffs[b] = np.percentile(fe[idx], q) - np.percentile(ce[idx], q)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fit_w_flat(x, c, lab, mask):
    """단일 배열에서 RMSE 최소 w (진단 전용)."""
    m = mask.astype(bool)
    xs, cs, ls = x[m], c[m], lab[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        v = float(np.sqrt(np.mean((w * xs + (1 - w) * cs - ls) ** 2)))
        if v < best[0]:
            best = (v, float(w))
    return best[1]


def main() -> None:
    oof, tst, test_lab, test_mask = load_all()

    balance, shape, tail = [], [], []
    for nm in NAMES:
        fo, xo, co, lo, mo, pid_oof = oof_raw(oof, nm)
        ft, xt, ct, w_oof = test_raw(oof, tst, nm)
        pid_test = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)

        # ---- A. 브랜치 균형 ----
        xo_eye = eye_metric(xo, lo, mo, 'rmse').mean()
        co_eye = eye_metric(co, lo, mo, 'rmse').mean()
        xt_eye = eye_metric(xt, test_lab, test_mask, 'rmse').mean()
        ct_eye = eye_metric(ct, test_lab, test_mask, 'rmse').mean()
        gap_oof, gap_test = xo_eye - co_eye, xt_eye - ct_eye

        w_oracle = fit_w_flat(xt, ct, test_lab, test_mask)
        f_oracle = w_oracle * xt + (1 - w_oracle) * ct
        balance.append(dict(
            backbone=nm,
            xgb_oof=float(xo_eye), xgb_test=float(xt_eye), xgb_shift=float(xt_eye - xo_eye),
            cnn_oof=float(co_eye), cnn_test=float(ct_eye), cnn_shift=float(ct_eye - co_eye),
            gap_oof=float(gap_oof), gap_test=float(gap_test),
            gap_shift=float(gap_test - gap_oof),
            w_oof_used=float(w_oof), w_test_oracle=float(w_oracle),
            fus_test_at_w_oof=float(eye_metric(ft, test_lab, test_mask, 'rmse').mean()),
            fus_test_at_w_oracle=float(eye_metric(f_oracle, test_lab, test_mask, 'rmse').mean()),
        ))

        # ---- B. 오차 분포 형태 ----
        for split, (fp, xp, cp, lb, mk) in (
            ('OOF', (fo, xo, co, lo, mo)),
            ('TEST', (ft, xt, ct, test_lab, test_mask)),
        ):
            row = dict(backbone=nm, split=split)
            for tag, p in (('xgb', xp), ('cnn', cp), ('fusion', fp)):
                r, a = pooled(p, lb, mk)
                row[f'{tag}_rmse'], row[f'{tag}_mae'] = r, a
                row[f'{tag}_ratio'] = r / a
            shape.append(row)

        # ---- C. 꼬리 대 전형 ----
        for split, (fp, cp, lb, mk, pd_) in (
            ('OOF', (fo, co, lo, mo, pid_oof)),
            ('TEST', (ft, ct, test_lab, test_mask, pid_test)),
        ):
            for metric in ('rmse', 'mae'):
                fe = eye_metric(fp, lb, mk, metric)
                ce = eye_metric(cp, lb, mk, metric)
                d = fe - ce
                rng = np.random.default_rng(SEED)
                med_lo, med_hi = cluster_boot_quantile_ci(fe, ce, pd_, 50, rng)
                rng = np.random.default_rng(SEED)
                p90_lo, p90_hi = cluster_boot_quantile_ci(fe, ce, pd_, 90, rng)
                tail.append(dict(
                    backbone=nm, split=split, metric=metric, n=int(fe.size),
                    median_fus=float(np.median(fe)), median_cnn=float(np.median(ce)),
                    median_delta=float(np.median(fe) - np.median(ce)),
                    median_ci_lo=med_lo, median_ci_hi=med_hi,
                    p90_fus=float(np.percentile(fe, 90)), p90_cnn=float(np.percentile(ce, 90)),
                    p90_delta=float(np.percentile(fe, 90) - np.percentile(ce, 90)),
                    p90_ci_lo=p90_lo, p90_ci_hi=p90_hi,
                    p90_superior=bool(p90_hi < 0),
                    mean_delta=float(np.mean(d)), median_of_delta=float(np.median(d)),
                    win_frac=float(np.mean(d < 0)),
                ))

    # ---- D. 평균 대 순위 엇갈림 ----
    disagree = []
    ni_p, cm_p = ROOT / 'runs/fusion_noninferiority.json', ROOT / 'runs/fusion_consistency_matrix.json'
    if ni_p.exists() and cm_p.exists():
        ni = {(r['backbone'], r['metric'], r['split']): r for r in json.loads(ni_p.read_text())['rows']}
        cm = {(r['backbone'], r['metric'], r['split']): r for r in json.loads(cm_p.read_text())['rows']}
        for key, nr in ni.items():
            cr = cm.get(key)
            if cr is None:
                continue
            mean_sup = nr['ci_hi'] < 0
            rank_sup = cr['vs_cnn'] == 'WIN'
            if mean_sup != rank_sup:
                disagree.append(dict(backbone=key[0], metric=key[1], split=key[2],
                                     ci_hi=nr['ci_hi'], p_wilcoxon=cr['p_cnn'],
                                     mean_superior=mean_sup, rank_superior=rank_sup))

    # ================= 출력 =================
    print('A. 브랜치 균형 이동 (안 단위 평균 RMSE, dB)')
    print(f"{'backbone':<14}{'XGB OOF→test':>14}{'CNN OOF→test':>14}{'격차 이동':>10}"
          f"{'w_oof':>7}{'w_test*':>9}")
    print('-' * 68)
    for b in balance:
        print(f"{b['backbone']:<14}{b['xgb_shift']:>+14.3f}{b['cnn_shift']:>+14.3f}"
              f"{b['gap_shift']:>+10.3f}{b['w_oof_used']:>7.2f}{b['w_test_oracle']:>9.2f}")
    print('-' * 68)
    print(f"  XGB 는 test 에서 평균 {np.mean([b['xgb_shift'] for b in balance]):+.3f} dB, "
          f"CNN 은 {np.mean([b['cnn_shift'] for b in balance]):+.3f} dB")
    print(f"  두 브랜치 격차 이동 평균 {np.mean([b['gap_shift'] for b in balance]):+.3f} dB "
          f"(양수 = test 에서 XGB 가 상대적으로 더 나빠짐)")
    print(f"  w_oof 평균 {np.mean([b['w_oof_used'] for b in balance]):.2f} vs "
          f"test-oracle 평균 {np.mean([b['w_test_oracle'] for b in balance]):.2f} "
          f"(oracle 은 진단 전용, 어떤 주장에도 쓰지 않음)")

    print('\nB. pooled RMSE/MAE 비 (크면 전형오차 작고 꼬리 두꺼움)')
    print(f"{'backbone':<14}{'split':<6}{'XGB':>7}{'CNN':>7}{'fusion':>8}")
    print('-' * 44)
    for s in shape:
        print(f"{s['backbone']:<14}{s['split']:<6}{s['xgb_ratio']:>7.3f}"
              f"{s['cnn_ratio']:>7.3f}{s['fusion_ratio']:>8.3f}")
    print('-' * 44)
    print(f"  XGB 평균 {np.mean([s['xgb_ratio'] for s in shape]):.3f} | "
          f"CNN 평균 {np.mean([s['cnn_ratio'] for s in shape]):.3f} | "
          f"fusion 평균 {np.mean([s['fusion_ratio'] for s in shape]):.3f}")

    print('\nC. 꼬리 대 전형 — 안 단위 오차분포 (fusion - CNN, 음수 = fusion 우세)')
    print(f"{'backbone':<14}{'split':<6}{'m':<5}{'중앙값차':>9}{'  95% CI':<18}"
          f"{'90분위차':>9}{'  95% CI':<18}{'승률':>6}")
    print('-' * 84)
    for t in tail:
        print(f"{t['backbone']:<14}{t['split']:<6}{t['metric'].upper():<5}"
              f"{t['median_delta']:>+9.3f}  [{t['median_ci_lo']:+.2f},{t['median_ci_hi']:+.2f}]"
              f"{'':<4}{t['p90_delta']:>+9.3f}  [{t['p90_ci_lo']:+.2f},{t['p90_ci_hi']:+.2f}]"
              f"{'*' if t['p90_superior'] else ' '}{t['win_frac']:>5.2f}")
    print('-' * 84)
    oo = [t for t in tail if t['split'] == 'OOF']
    print(f"  OOF 10칸 평균: 중앙값차 {np.mean([t['median_delta'] for t in oo]):+.3f}, "
          f"90분위차 {np.mean([t['p90_delta'] for t in oo]):+.3f}")
    print(f"  90분위 CI 상한 < 0 (= 꼬리에서 유의하게 우월): "
          f"{sum(t['p90_superior'] for t in tail)}/20칸 "
          f"(OOF {sum(t['p90_superior'] for t in oo)}/10, "
          f"test {sum(t['p90_superior'] for t in tail if t['split'] == 'TEST')}/10)")
    print(f"  중앙값 CI 상한 < 0 (= 전형적인 안에서 우월): "
          f"{sum(t['median_ci_hi'] < 0 for t in tail)}/20칸")

    print('\nD. 평균 기반(부트스트랩 CI)과 순위 기반(Wilcoxon)이 엇갈리는 칸')
    if not disagree:
        print('  없음')
    for d in disagree:
        print(f"  {d['backbone']} {d['split']} {d['metric'].upper()}: "
              f"CI 상한 {d['ci_hi']:+.4f} (평균 우월={d['mean_superior']}), "
              f"Wilcoxon p={d['p_wilcoxon']:.4f} (순위 우월={d['rank_superior']})")
    print(f"  20칸 중 {len(disagree)}칸에서 엇갈린다.")

    # OOF 10칸 평균 — 위 C 요약 출력(print)에는 있었지만 JSON 에는 없어서
    # 04_results.tex 의 "-0.05 / -1.43 / -0.15 dB" 가 근거 없이 떠 있었다(MISS).
    # 검증 게이트의 근거 풀은 runs/*.json 이므로 같은 식을 스칼라로 남긴다.
    tail_oof_mean = {
        k: float(np.mean([t[k] for t in oo]))
        for k in ('median_delta', 'p90_delta', 'median_of_delta')
    }
    tail_oof_mean['n_cells'] = len(oo)

    out = ROOT / 'runs/fusion_error_structure.json'
    out.write_text(json.dumps(dict(
        note=('A=브랜치 균형 이동, B=pooled RMSE/MAE 비, C=꼬리 대 전형(안 단위 분포통계), '
              'D=평균 대 순위 검정 엇갈림. w_test_oracle 은 test 적합이므로 진단 전용이며 '
              '어떤 주장에도 쓰지 않는다.'),
        balance=balance, shape=shape, tail=tail,
        tail_oof_mean=tail_oof_mean, disagree=disagree,
    ), ensure_ascii=False, indent=2))
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
