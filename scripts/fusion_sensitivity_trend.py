#!/usr/bin/env python3
"""감도 구간에 따른 융합 이득의 단조 추세 — 사전 지정 주 검정 1개.

사전 지정 사항 (돌리기 전에 고정, 결과를 보고 바꾸지 않는다):
  1. 구간은 참 감도(dB) 기준. 지수 1=[0,10) 2=[10,20) 3=[20,30) 4=[30,inf).
     지수가 커질수록 감도가 높다.
  2. 한 안의 한 구간은 유효 지점이 MIN_PTS 개 이상일 때만 센다.
  3. 안별로 delta(=fusion RMSE - CNN RMSE)를 구간 지수에 OLS 회귀해 기울기를 얻는다.
     유효 구간이 2개 미만인 안은 기울기가 정의되지 않으므로 제외한다.
  4. **주 검정 = PRIMARY 셀 하나**의 기울기에 대한 일표본 Wilcoxon 양측검정.
     나머지 셀은 보조·일관성 확인이며 확증적 주장에 쓰지 않는다.
  5. 방향 예측은 사후가 아니라 기전에서 연역된다. 융합이 큰 오차를 깎는다면
     delta 는 저감도(작은 지수)에서 더 음수여야 하므로 기울기는 **양수**여야 한다.
     예측이 빗나가면 그대로 보고하고 기전 서술을 철회한다.

주의: 층을 참 라벨로 나눈다. 예측을 조건으로 거는 것이 아니라 결과변수의 참값을
      조건으로 거는 층화 오차분석이므로 순환이 없다.

읽기 전용. 출력: runs/fusion_sensitivity_trend.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_eval_common import NAMES, load_all, oof_raw, test_raw  # noqa: E402

BIN_EDGES = [0.0, 10.0, 20.0, 30.0, np.inf]
BIN_LABELS = ['[0,10)', '[10,20)', '[20,30)', '[30,inf)']
MIN_PTS = 3
N_BOOT = 10000
SEED = 42
PRIMARY = ('IR-v2', 'OOF', 'rmse')  # 논문 대표 백본 + 검정력이 높은 split


def bin_index(lab: np.ndarray) -> np.ndarray:
    return np.digitize(lab, BIN_EDGES[1:-1], right=False)  # 0..3


def eye_bin_rmse(pred, lab, mask):
    """(n_eye, 4) 구간별 RMSE. 유효 지점 < MIN_PTS 이면 nan."""
    n = pred.shape[0]
    out = np.full((n, 4), np.nan)
    for i in range(n):
        m = mask[i].astype(bool)
        if not m.any():
            continue
        bi = bin_index(lab[i][m])
        err = (pred[i] - lab[i])[m]
        for b in range(4):
            sel = bi == b
            if sel.sum() >= MIN_PTS:
                out[i, b] = np.sqrt(np.mean(err[sel] ** 2))
    return out


def slopes(delta: np.ndarray):
    """구간 지수(1..4)에 대한 안별 OLS 기울기. 유효 구간 2개 미만이면 nan."""
    x_all = np.arange(1, 5, dtype=float)
    out = np.full(delta.shape[0], np.nan)
    for i in range(delta.shape[0]):
        ok = np.isfinite(delta[i])
        if ok.sum() < 2:
            continue
        x, y = x_all[ok], delta[i][ok]
        out[i] = np.polyfit(x, y, 1)[0]
    return out


def cluster_boot_mean_ci(v, pid, rng):
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        means[b] = np.mean(v[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    oof, tst, test_lab, test_mask = load_all()

    rows = []
    oof_lab = oof_mask = None
    for nm in NAMES:
        fo, xo, co, lo, mo, pid_oof = oof_raw(oof, nm)
        if oof_lab is None:
            oof_lab, oof_mask = lo, mo
        ft, xt, ct, _ = test_raw(oof, tst, nm)
        pid_test = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)

        for split, (fp, cp, lb, mk, pd_) in (
            ('OOF', (fo, co, lo, mo, pid_oof)),
            ('TEST', (ft, ct, test_lab, test_mask, pid_test)),
        ):
            fb = eye_bin_rmse(fp, lb, mk)
            cb = eye_bin_rmse(cp, lb, mk)
            delta = fb - cb
            sl = slopes(delta)
            ok = np.isfinite(sl)
            sl_ok, pid_ok = sl[ok], pd_[ok]

            p = float(wilcoxon(sl_ok).pvalue) if sl_ok.size >= 6 else float('nan')
            rng = np.random.default_rng(SEED)
            lo_ci, hi_ci = cluster_boot_mean_ci(sl_ok, pid_ok, rng)

            per_bin = []
            for b in range(4):
                d = delta[:, b]
                dv = d[np.isfinite(d)]
                per_bin.append(dict(
                    bin=BIN_LABELS[b], n_eyes=int(dv.size),
                    mean_delta=float(np.mean(dv)) if dv.size else float('nan'),
                    median_delta=float(np.median(dv)) if dv.size else float('nan'),
                ))

            rows.append(dict(
                backbone=nm, split=split, metric='rmse',
                n_eyes_with_slope=int(sl_ok.size), n_eyes_total=int(sl.size),
                mean_slope=float(np.mean(sl_ok)), median_slope=float(np.median(sl_ok)),
                slope_ci_lo=lo_ci, slope_ci_hi=hi_ci,
                p_wilcoxon=p,
                frac_positive=float(np.mean(sl_ok > 0)),
                per_bin=per_bin,
                is_primary=bool((nm, split, 'rmse') == PRIMARY),
            ))

    # 구간별 지점 분포 — tab:where 각주의 근거.
    # 2026-08-26 추가. 종전에는 이 비율이 원고에만 있고 어느 산출물에도 없어
    # 근거가 산문 문서 하나뿐이었다. 기존 키는 건드리지 않고 추가만 한다.
    # 라벨·마스크는 백본과 무관하다(load_fold 가 5백본 마스크를 모두 교집합한다).
    bin_dist = {}
    for split, (lb, mk) in (
        ('OOF', (oof_lab, oof_mask)), ('TEST', (test_lab, test_mask)),
    ):
        y = lb[mk.astype(bool)]
        bi = bin_index(y)
        bin_dist[split] = dict(
            n_points=int(y.size),
            n_by_bin=[int((bi == b).sum()) for b in range(4)],
            frac_by_bin=[float(np.mean(bi == b)) for b in range(4)],
        )

    # ================= 출력 =================
    print(f'구간: {"  ".join(BIN_LABELS)} dB (지수 1..4, 클수록 고감도), '
          f'구간 최소 지점수 {MIN_PTS}')
    print(f'주 검정 = {PRIMARY[0]} {PRIMARY[1]} {PRIMARY[2].upper()} 한 칸. 나머지는 보조.\n')

    print('안별 delta 기울기 (양수 = 저감도로 갈수록 fusion 이 더 유리)')
    print(f"{'backbone':<14}{'split':<6}{'n':>5}{'평균기울기':>11}{'  95% CI':<20}"
          f"{'p':>10}{'기울기>0':>9}")
    print('-' * 78)
    for r in rows:
        star = ' *주' if r['is_primary'] else ''
        print(f"{r['backbone']:<14}{r['split']:<6}{r['n_eyes_with_slope']:>5}"
              f"{r['mean_slope']:>+11.4f}  [{r['slope_ci_lo']:+.3f},{r['slope_ci_hi']:+.3f}]"
              f"{'':<3}{r['p_wilcoxon']:>10.2e}{r['frac_positive']:>9.2f}{star}")
    print('-' * 78)

    pr = next(r for r in rows if r['is_primary'])
    verdict = ('예측대로 양수' if pr['mean_slope'] > 0 else '예측과 반대(음수)')
    sig = '유의' if pr['p_wilcoxon'] < 0.05 else '비유의'
    print(f"  주 검정: 기울기 {pr['mean_slope']:+.4f} ({verdict}), p={pr['p_wilcoxon']:.2e} ({sig}), "
          f"n={pr['n_eyes_with_slope']}안")
    npos = sum(r['mean_slope'] > 0 for r in rows)
    nsig = sum(r['p_wilcoxon'] < 0.05 and r['mean_slope'] > 0 for r in rows)
    print(f"  보조 일관성: 기울기 양수 {npos}/{len(rows)}칸, 그중 유의 {nsig}칸")

    print('\n구간별 delta 평균 (fusion RMSE - CNN RMSE, 음수 = fusion 우세)')
    print(f"{'backbone':<14}{'split':<6}" + ''.join(f'{b:>12}' for b in BIN_LABELS))
    print('-' * 68)
    for r in rows:
        cells = ''.join(f"{pb['mean_delta']:>+12.3f}" for pb in r['per_bin'])
        print(f"{r['backbone']:<14}{r['split']:<6}{cells}")
    print('-' * 68)
    print('안 수:        ' + ''.join(f"{pb['n_eyes']:>12}" for pb in rows[0]['per_bin'])
          + '   (OOF IR-v2 기준)')

    out = ROOT / 'runs/fusion_sensitivity_trend.json'
    out.write_text(json.dumps(dict(
        note=('감도 구간별 융합 이득의 단조 추세. delta = 안·구간 단위 (fusion RMSE - CNN RMSE). '
              '기울기는 구간 지수 1..4 에 대한 OLS. 양수 = 저감도에서 fusion 이 더 유리. '
              '주 검정은 is_primary=true 인 한 칸이며 나머지는 보조.'),
        bin_edges=[float(e) for e in BIN_EDGES], bin_labels=BIN_LABELS,
        min_pts=MIN_PTS, n_boot=N_BOOT, seed=SEED,
        primary=dict(backbone=PRIMARY[0], split=PRIMARY[1], metric=PRIMARY[2]),
        bin_distribution=bin_dist,
        rows=rows,
    ), ensure_ascii=False, indent=2))
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
