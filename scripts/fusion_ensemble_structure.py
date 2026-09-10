#!/usr/bin/env python3
"""앙상블 영상 브랜치에서 오차 구조와 감도 추세 — 가장 불리한 조건에서의 검정.

논문 §fusion ceiling 의 null (앙상블 OOF 에서 fusion vs 영상단독 p=0.46) 이 있는 자리에서
같은 질문을 다시 묻는다.

  전체 시야 평균이 침묵하는 조건에서도 (a) 융합이 오차분포의 상단 꼬리를 깎는가,
  (b) 이득이 저감도 구간에 몰려 있는가.

단일 백본 5종에서 확인된 두 현상이 앙상블에서도 성립하면, 그것은 "약한 영상 모델이라
그렇다"로 반박되지 않는다.

감도 구간과 최소 지점수는 fusion_sensitivity_trend.py 의 사전 지정값을 그대로 쓴다.
읽기 전용. 출력: runs/fusion_ensemble_structure.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_eval_common import ENSEMBLE, load_all, oof_raw, test_raw, eye_metric  # noqa: E402
from fusion_sensitivity_trend import (  # noqa: E402
    BIN_LABELS, MIN_PTS, eye_bin_rmse, slopes, N_BOOT, SEED,
)


def boot_stat_ci(fe, ce, pid, fn):
    """환자 클러스터 부트스트랩으로 fn(fe)-fn(ce) 의 95% CI."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        d[b] = fn(fe[idx]) - fn(ce[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_mean_ci(v, pid):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        m[b] = np.mean(v[idx])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    oof, tst, test_lab, test_mask = load_all(ensemble=True)
    fo, xo, co, lo, mo, pid_oof = oof_raw(oof, ENSEMBLE)
    ft, xt, ct, w = test_raw(oof, tst, ENSEMBLE)
    pid_test = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)

    struct, trend = [], []
    for split, (fp, cp, lb, mk, pd_) in (
        ('OOF', (fo, co, lo, mo, pid_oof)),
        ('TEST', (ft, ct, test_lab, test_mask, pid_test)),
    ):
        for metric in ('rmse', 'mae'):
            fe, ce = eye_metric(fp, lb, mk, metric), eye_metric(cp, lb, mk, metric)
            d = fe - ce
            mlo, mhi = boot_mean_ci(d, pd_)
            med_lo, med_hi = boot_stat_ci(fe, ce, pd_, np.median)
            p90 = lambda a: np.percentile(a, 90)  # noqa: E731
            p90_lo, p90_hi = boot_stat_ci(fe, ce, pd_, p90)
            struct.append(dict(
                split=split, metric=metric, n=int(fe.size),
                mean_fus=float(np.mean(fe)), mean_cnn=float(np.mean(ce)),
                mean_delta=float(np.mean(d)), mean_ci_lo=mlo, mean_ci_hi=mhi,
                p_wilcoxon=float(wilcoxon(fe, ce).pvalue),
                median_delta=float(np.median(fe) - np.median(ce)),
                median_ci_lo=med_lo, median_ci_hi=med_hi,
                p90_delta=float(np.percentile(fe, 90) - np.percentile(ce, 90)),
                p90_ci_lo=p90_lo, p90_ci_hi=p90_hi,
                p90_superior=bool(p90_hi < 0), median_superior=bool(med_hi < 0),
                win_frac=float(np.mean(d < 0)),
            ))

        fb, cb = eye_bin_rmse(fp, lb, mk), eye_bin_rmse(cp, lb, mk)
        delta = fb - cb
        sl = slopes(delta)
        ok = np.isfinite(sl)
        slo, shi = boot_mean_ci(sl[ok], pd_[ok])
        trend.append(dict(
            split=split, n_eyes_with_slope=int(ok.sum()),
            mean_slope=float(np.mean(sl[ok])), slope_ci_lo=slo, slope_ci_hi=shi,
            p_wilcoxon=float(wilcoxon(sl[ok]).pvalue),
            frac_positive=float(np.mean(sl[ok] > 0)),
            per_bin=[dict(bin=BIN_LABELS[b],
                          n_eyes=int(np.isfinite(delta[:, b]).sum()),
                          mean_delta=float(np.nanmean(delta[:, b])))
                     for b in range(4)],
        ))

    # ================= 출력 =================
    print(f'앙상블(5백본 평균) 영상 브랜치, w = {w:.2f} (전체 OOF 적합)')
    print('\n오차분포 — fusion 대 앙상블 영상단독 (음수 = fusion 우세)')
    print(f"{'split':<6}{'m':<6}{'평균차':>9}{'  p':>10}{'중앙값차':>10}{'  95% CI':<18}"
          f"{'90분위차':>10}{'  95% CI':<18}")
    print('-' * 84)
    for s in struct:
        print(f"{s['split']:<6}{s['metric'].upper():<6}{s['mean_delta']:>+9.3f}"
              f"{s['p_wilcoxon']:>10.3f}{s['median_delta']:>+10.3f}"
              f"  [{s['median_ci_lo']:+.2f},{s['median_ci_hi']:+.2f}]{'':<3}"
              f"{s['p90_delta']:>+10.3f}"
              f"  [{s['p90_ci_lo']:+.2f},{s['p90_ci_hi']:+.2f}]"
              f"{'*' if s['p90_superior'] else ''}")
    print('-' * 84)
    print(f"  90분위에서 유의 우월: {sum(s['p90_superior'] for s in struct)}/4칸 | "
          f"중앙값에서 유의 우월: {sum(s['median_superior'] for s in struct)}/4칸")

    print('\n감도 구간 추세 (양수 = 저감도로 갈수록 fusion 이 더 유리)')
    for t in trend:
        print(f"  {t['split']}: 기울기 {t['mean_slope']:+.4f} "
              f"[{t['slope_ci_lo']:+.3f},{t['slope_ci_hi']:+.3f}], "
              f"p={t['p_wilcoxon']:.3g}, n={t['n_eyes_with_slope']}안, "
              f"기울기>0 비율 {t['frac_positive']:.2f}")
        cells = '  '.join(f"{pb['bin']} {pb['mean_delta']:+.3f}" for pb in t['per_bin'])
        print(f'    구간별 delta: {cells}')

    out = ROOT / 'runs/fusion_ensemble_structure.json'
    out.write_text(json.dumps(dict(
        note=('앙상블(5백본 평균) 영상 브랜치에서의 오차구조·감도추세. 구간과 최소 지점수는 '
              'fusion_sensitivity_trend.py 의 사전 지정값을 그대로 사용. '
              'delta = fusion - 앙상블 영상단독, 음수 = fusion 우세.'),
        w_test=float(w), min_pts=MIN_PTS, bin_labels=BIN_LABELS,
        n_boot=N_BOOT, seed=SEED, structure=struct, trend=trend,
    ), ensure_ascii=False, indent=2))
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
