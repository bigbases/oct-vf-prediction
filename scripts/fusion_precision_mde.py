#!/usr/bin/env python3
"""test fold 의 정밀도 진술 — 최소검출가능효과(MDE)와 필요 표본.

목적: "추가 실험을 하면 test 에서도 fusion > CNN 이 유의해지지 않겠나"에 대한 답.
n=37 의 검출 한계가 실제 효과의 2~3배이므로, 효과를 키우는 어떤 feature engineering
도 표본 크기 문제를 이기지 못한다.

서술 규약 — 사후검정력(observed power)으로 쓰지 말 것:
  관측 효과로 계산한 검정력은 p값의 단조 변환일 뿐이라 통계 리뷰어가 지적한다.
  sd 를 test 가 아니라 OOF(n=240)에서 추정하고 결과를 MDE 와 필요 표본으로만
  제시하면 사후 계산이 아니라 사전 정밀도 진술이 된다. 이 스크립트의 primary 열은
  전부 sd_oof 기반이며, sd_test 기반 값은 투명성을 위한 참고로만 병기한다.

읽기 전용. 저장된 npz 만 사용하고 학습은 없다.
출력: runs/fusion_precision_mde.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_eval_common import NAMES, load_all, oof_arrays, test_arrays  # noqa: E402

Z_ALPHA = 1.959963985  # 양측 0.05
Z_POWER = 0.841621234  # 검정력 0.80
Z_SUM = Z_ALPHA + Z_POWER
# Wilcoxon 부호순위의 정규분포 하 점근효율. t 기반 필요표본을 이 값으로 나눠 보정한다.
WILCOXON_ARE = 3.0 / np.pi
N_BOOT = 10000
SEED = 42


def cluster_design_effect(d: np.ndarray, pid: np.ndarray) -> tuple[float, float]:
    """환자 클러스터 부트스트랩 SE 와 단순 SE 의 비로 설계효과를 실측한다.

    한 환자의 두 눈은 독립이 아니므로 단순 SE 는 낙관적이다. 설계효과가 1보다 크면
    유효 표본은 안 수보다 작고, 필요 표본은 그만큼 늘어난다.
    """
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        means[b] = np.mean(d[idx])
    se_cluster = float(np.std(means, ddof=1))
    se_naive = float(np.std(d, ddof=1) / np.sqrt(d.size))
    deff = (se_cluster / se_naive) ** 2 if se_naive > 0 else float('nan')
    return se_cluster, deff


def mde(sd: float, n: int) -> float:
    """n 에서 양측 0.05, 검정력 0.80 으로 검출 가능한 최소 효과."""
    return Z_SUM * sd / np.sqrt(n)


def n_required(sd: float, delta: float) -> float:
    """delta 를 검정력 0.80 으로 잡는 데 필요한 표본."""
    if not np.isfinite(delta) or abs(delta) < 1e-12:
        return float('inf')
    return (Z_SUM * sd / abs(delta)) ** 2


def main() -> None:
    oof, tst, test_lab, test_mask = load_all()

    rows = []
    for nm in NAMES:
        for metric in ('rmse', 'mae'):
            f_o, x_o, c_o, pid_o = oof_arrays(oof, nm, metric)
            f_t, x_t, c_t, pid_t = test_arrays(oof, tst, test_lab, test_mask, nm, metric)

            for comparator, o_arr, t_arr in (('CNN', c_o, c_t), ('XGB', x_o, x_t)):
                d_o = f_o - o_arr           # 음수 = fusion 우세
                d_t = f_t - t_arr
                sd_o, sd_t = float(np.std(d_o, ddof=1)), float(np.std(d_t, ddof=1))
                n_o, n_t = int(d_o.size), int(d_t.size)
                delta_o = float(np.mean(d_o))

                _, deff = cluster_design_effect(d_o, pid_o)
                nreq = n_required(sd_o, delta_o)
                mde_naive = mde(sd_o, n_t)
                mde_clu = mde(sd_o, n_t / deff) if np.isfinite(deff) and deff > 0 else float('nan')
                rows.append(dict(
                    backbone=nm, metric=metric, comparator=comparator,
                    n_oof=n_o, n_test=n_t,
                    n_pat_oof=int(np.unique(pid_o).size), n_pat_test=int(np.unique(pid_t).size),
                    delta_oof=delta_o, delta_test=float(np.mean(d_t)),
                    sd_oof=sd_o, sd_test=sd_t,
                    se_oof=sd_o / np.sqrt(n_o), se_test=sd_t / np.sqrt(n_t),
                    design_effect_oof=deff, n_test_effective=n_t / deff,
                    mde_test_from_sd_oof=mde_naive,
                    mde_test_from_sd_test=mde(sd_t, n_t),
                    mde_test_cluster_adj=mde_clu,
                    mde_over_effect=mde_naive / abs(delta_o) if abs(delta_o) > 1e-12 else float('inf'),
                    n_required_80=nreq,
                    n_required_80_wilcoxon=nreq / WILCOXON_ARE,
                    n_required_80_cluster_adj=nreq * deff,
                    n_required_over_n_test=nreq / n_t,
                    n_required_cluster_over_n_test=nreq * deff / n_t,
                ))

    # ---- self-check: 기존 산출물과 delta 일치 확인 ----
    ni_path = ROOT / 'runs/fusion_noninferiority.json'
    checks = []
    if ni_path.exists():
        ni = json.loads(ni_path.read_text())
        want = {(r['backbone'], r['metric'], r['split']): r['delta'] for r in ni['rows']}
        for r in rows:
            if r['comparator'] != 'CNN':
                continue
            for split, got in (('OOF', r['delta_oof']), ('TEST', r['delta_test'])):
                ref = want.get((r['backbone'], r['metric'], split))
                if ref is None:
                    continue
                checks.append(abs(ref - got))
        if checks:
            worst = max(checks)
            status = 'OK' if worst < 1e-6 else 'MISMATCH'
            print(f'self-check vs fusion_noninferiority.json: {status} '
                  f'(최대 delta 차이 {worst:.2e}, {len(checks)}칸)')
            if status == 'MISMATCH':
                raise SystemExit('기존 산출물과 delta 가 불일치한다. 로더 절차를 먼저 맞춰야 한다.')
    print()

    cnn_rows = [r for r in rows if r['comparator'] == 'CNN']
    print('fusion vs 영상단독(CNN) — test fold 정밀도 [sd 는 OOF(n=240)에서 추정]')
    print(f"{'backbone':<14}{'metric':<7}{'OOF delta':>10}{'sd_oof':>8}"
          f"{'MDE(n=37)':>11}{'MDE/효과':>9}{'필요 n':>8}{'DEFF':>7}{'보정 필요 n':>12}")
    print('-' * 93)
    for r in cnn_rows:
        print(f"{r['backbone']:<14}{r['metric'].upper():<7}{r['delta_oof']:>10.3f}"
              f"{r['sd_oof']:>8.3f}{r['mde_test_from_sd_oof']:>11.3f}"
              f"{r['mde_over_effect']:>9.1f}{r['n_required_80']:>8.0f}"
              f"{r['design_effect_oof']:>7.2f}{r['n_required_80_cluster_adj']:>12.0f}")
    print('-' * 93)

    mdes = [r['mde_test_from_sd_oof'] for r in cnn_rows]
    effs = [abs(r['delta_oof']) for r in cnn_rows]
    nreqs = [r['n_required_80'] for r in cnn_rows]
    nreqs_c = [r['n_required_80_cluster_adj'] for r in cnn_rows]
    deffs = [r['design_effect_oof'] for r in cnn_rows]
    n_test = cnn_rows[0]['n_test']
    print(f"  test: {n_test}안 / {cnn_rows[0]['n_pat_test']}환자, "
          f"OOF: {cnn_rows[0]['n_oof']}안 / {cnn_rows[0]['n_pat_oof']}환자")
    print(f'  검출 한계 MDE: {min(mdes):.2f} ~ {max(mdes):.2f} dB')
    print(f'  실제 효과    : {min(effs):.2f} ~ {max(effs):.2f} dB')
    print(f'  필요 표본    : {min(nreqs):.0f} ~ {max(nreqs):.0f} 안 '
          f'(현재의 {min(nreqs) / n_test:.1f}~{max(nreqs) / n_test:.1f}배)')
    print(f'  설계효과 {min(deffs):.2f}~{max(deffs):.2f} 보정 후: '
          f'{min(nreqs_c):.0f} ~ {max(nreqs_c):.0f} 안 '
          f'(현재의 {min(nreqs_c) / n_test:.1f}~{max(nreqs_c) / n_test:.1f}배)')
    print(f'  Wilcoxon 보정 필요 표본: {min(nreqs) / WILCOXON_ARE:.0f} ~ '
          f'{max(nreqs) / WILCOXON_ARE:.0f} 안')

    out = ROOT / 'runs/fusion_precision_mde.json'
    out.write_text(json.dumps(dict(
        note=('test fold 정밀도. delta = 안 단위(fusion) - 안 단위(comparator), 음수=fusion 우세. '
              'MDE 는 양측 0.05, 검정력 0.80. primary 는 sd_oof 기반(사전 정밀도 진술). '
              'sd_test 기반은 참고. n_required_80_wilcoxon 은 t 기반 필요표본을 '
              'Wilcoxon 점근효율 3/pi 로 보정한 값.'),
        z_alpha=Z_ALPHA, z_power=Z_POWER, wilcoxon_are=float(WILCOXON_ARE),
        n_test=n_test, rows=rows,
    ), ensure_ascii=False, indent=2))
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
