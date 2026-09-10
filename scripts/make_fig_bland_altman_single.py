#!/usr/bin/env python3
"""Bland-Altman 단일 패널 — 안 단위 평균 감도의 예측-실측 일치도, 층 없음.

`make_fig_bland_altman.py` 의 단일 패널만 떼어내고 **표본을 240안으로 넓힌 것**이다.
그 스크립트는 4패널·2패널을 함께 내느라 층별 n 을 `runs/severity_region.json` 과
대조하고, 층 배정은 `cohort_md.csv` 조인을 타므로 조인 실패 5안이 빠져 235안이 된다.
단일 패널은 층을 쓰지 않는다. 축은 실측 평균 감도이고 층 경계가 필요 없다.
빠진 5안도 52점 전부 유효한 예측과 실측을 갖고 있으므로 뺄 이유가 없다.
그래서 여기서는 OOF 240안 전부를 그린다(선택 없음 = 선택 편향 없음).

일치한계는 회귀 기반이다(Bland & Altman 1999). |잔차|가 실측 감도에 따라 줄어
상수 한계는 저감도에서 너무 좁고 고감도에서 너무 넓다.

입력 (읽기 전용):
  runs/oof/xgb_90d_fold{0..4}_val.npz                                 XGB OOF
  runs/phasec_b0_inception_resnet_v2_5fold/val_preds_fold{0..4}.npz   CNN OOF
  runs/severity_region.json                                           w_xgb 대조
정본 트리에서 실행하면 패스 A, `step4_work/B/scripts/` 에서 실행하면 패스 B다
(`ROOT = parents[1]` 이 입력 전체를 그 트리로 재조준한다).

출력:
  build/figs/fig_bland_altman_single.png / .pdf
  build/figs/preview/...__print_size.png     판독 확인용. 원고에 넣지 않는다
실행: python scripts/make_fig_bland_altman_single.py
env: aaa (matplotlib + numpy + scipy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from fig_style import figure, zero_line, save_fig, GREYS, OKABE_ITO  # noqa: E402
from build_severity_region import MD, STRATA, nkey, stack_oof, opt_w  # noqa: E402

REF = ROOT / 'runs' / 'severity_region.json'
# 240안이 겹쳐 25~30 dB 구간이 뭉친다. 채운 원 + 반투명이라 겹친 정도가
# 농도로 읽힌다. 색은 하나뿐이다 — 집단을 인코딩하지 않는다. 중증도로 나누면
# 가로축과 중복이고 회색조 인쇄에서 그 구분이 사라진다.
#
# 색 선택: Okabe-Ito 파랑(#0072B2). 그림 2 (a) 도 같은 파랑을 쓰지만 거기서는
# summary branch 라는 계열을 인코딩한다. 이 그림은 계열이 하나뿐이라 색이 아무
# 변수도 인코딩하지 않고, 따라서 같은 색이 원고 안에서 다른 것을 가리키는 충돌이
# 생기지 않는다. 그렇다면 원고 전체 강조색을 파랑 하나로 통일하는 편이 낫다.
# 회귀선과 일치한계선은 검정·진회색이라 파란 점 위에서 명도로 뜬다.
DOT = OKABE_ITO[1]         # '#0072B2' Okabe-Ito blue
DOT_DARK = '#004A73'       # 같은 색상, 명도만 낮춘 테두리 (V 0.648배)
ALPHA = 0.50
VERM = OKABE_ITO[2]        # '#D55E00' 이전 판. 되돌리기용으로 남긴다
VERM_DARK = '#8A3D00'
# 대안 팔레트. arXiv:2511.20639 Figure 5 가 실제로 쓴 주황이다(저장소 슬라이드
# PDF 원본에서 추출). Okabe-Ito 보다 밝고 채도가 낮아 인상이 부드럽다. 밝은
# 만큼 회색조에서 흰 배경과의 명도 차가 68 → 42 단계로 줄어, 저감도 쪽
# 고립점이 흐려진다. 그래서 soft 는 두 가지로 보완한다:
#   alpha 0.50 → 0.65   면 자체를 진하게 (밀집 구간 계조는 조금 눌린다)
#   테두리 0.3 → 0.5 pt + 더 어두운 톤   고립점 윤곽을 회색조에서 살린다
# 팔레트마다 alpha 와 테두리 굵기가 다르므로 항목에 함께 담는다.
SOFT = '#E6996F'
SOFT_DARK = '#8A4A25'      # 같은 색상, 명도를 더 낮춘 테두리
PALETTES = {
    'blue': dict(face=DOT, edge=DOT_DARK, alpha=ALPHA, lw=0.3),
    'vermillion': dict(face=VERM, edge=VERM_DARK, alpha=ALPHA, lw=0.3),
    'soft': dict(face=SOFT, edge=SOFT_DARK, alpha=0.65, lw=0.5),
}
# 캡션에 적을 x 격자. 자료 범위는 0.0~32.4 dB 다.
CAPTION_X = [0, 5, 15, 25, 30, 32]


def eye_means() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """안별 (실측 평균, 예측 평균, MD 조인 성공 여부), 그리고 w_xgb."""
    ref = json.loads(REF.read_text(encoding='utf-8'))
    X, C, L, M, keys = stack_oof()
    w = opt_w(X, C, L, M)
    if w != ref['w_xgb']:
        raise SystemExit(f'w_xgb {w} != {REF.name} 의 {ref["w_xgb"]}')
    F = w * X + (1 - w) * C
    m = M.astype(bool)

    # 마스크 밖 지점은 평균에서 뺀다. 현재 코호트는 전 안이 52점 전부 유효하다.
    meas = np.array([L[i][m[i]].mean() for i in range(len(keys))])
    pred = np.array([F[i][m[i]].mean() for i in range(len(keys))])

    md = np.array([MD.get(nkey(*k), np.nan) for k in keys])
    joined = np.array([np.isfinite(v) and any(fn(v) for _, fn in STRATA)
                       for v in md])
    if joined.sum() != 235:
        raise SystemExit(f'층 배정 {int(joined.sum())} != 235')
    if len(keys) != 240:
        raise SystemExit(f'OOF 안 {len(keys)} != 240')
    return meas, pred, joined, w


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    """회귀 기반 편향선과 일치한계. half = 2.46 * E|resid|(x)."""
    r = st.linregress(x, y)
    resid = y - (r.intercept + r.slope * x)
    q = st.linregress(x, np.abs(resid))
    half = 2.46 * (q.intercept + q.slope * x)
    return {'n': int(x.size), 'a': float(r.intercept), 'b': float(r.slope),
            'r': float(r.rvalue), 'p': float(r.pvalue),
            'at5': float(r.intercept + r.slope * 5),
            'at30': float(r.intercept + r.slope * 30),
            'qa': float(q.intercept), 'qb': float(q.slope),
            'out': int((np.abs(resid) > half).sum())}


def build(x: np.ndarray, y: np.ndarray, s: dict, pal: dict):
    fig, ax = figure('single', ratio=0.80)
    zero_line(ax)
    ax.plot(x, y, linestyle='none', marker='o', markersize=3.0,
            markerfacecolor=pal['face'], markeredgecolor=pal['edge'],
            markeredgewidth=pal['lw'], alpha=pal['alpha'], zorder=3)
    gx = np.linspace(x.min(), x.max(), 200)
    line = s['a'] + s['b'] * gx
    half = 2.46 * (s['qa'] + s['qb'] * gx)
    ax.plot(gx, line, color='#000000', linewidth=1.0, zorder=4, label='Mean bias')
    for sgn in (+1, -1):
        ax.plot(gx, line + sgn * half, color=GREYS[3], linewidth=0.9,
                linestyle=(0, (6, 1.8)), zorder=4,
                label=r'$\pm$1.96 SD' if sgn > 0 else None)
    ax.plot([], [], color='#000000', linewidth=0.5, linestyle=(0, (2, 2)),
            label='Zero')
    ax.set_xlabel('Measured mean sensitivity (dB)')
    ax.set_ylabel(r'Predicted $-$ measured (dB)')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.85, handlelength=2.4)
    return fig


def row(tag: str, s: dict) -> str:
    return (f'{tag:14s} {s["n"]:4d} {s["a"]:+8.3f} {s["b"]:+8.4f} '
            f'{s["at5"]:+8.2f} {s["at30"]:+8.2f} {s["out"]:5d} {s["r"]:+7.2f}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'figs')
    ap.add_argument('--dot', choices=sorted(PALETTES), default='blue',
                    help='산점 색. blue=#0072B2, vermillion=#D55E00, soft=#E6996F')
    ap.add_argument('--name', default='fig_bland_altman_single',
                    help='출력 파일 이름(확장자 없이)')
    args = ap.parse_args()
    pal = PALETTES[args.dot]
    outdir = Path(args.out).resolve()

    meas, pred, joined, w = eye_means()
    y = pred - meas
    full, sub = fit(meas, y), fit(meas[joined], y[joined])

    print(f'가드 통과. late fusion IR-v2 w_xgb={w}. OOF 240안 / MD 조인 235안.\n')
    print(f'{"표본":14s} {"n":>4s} {"절편":>8s} {"기울기":>8s} '
          f'{"@5dB":>8s} {"@30dB":>8s} {"한계밖":>5s} {"r":>7s}')
    print(row('240안 (전체)', full))
    print(row('235안 (층조인)', sub))

    # 캡션에 쓸 값. 회귀 기반이라 편향도 한계도 x 의 함수다.
    print(f'\n240안 회귀 기반 일치한계 (dB)')
    print(f'{"x (실측 dB)":>12s} {"상한":>9s} {"편향":>9s} {"하한":>9s} {"폭":>9s}')
    for gx in CAPTION_X:
        b = full['a'] + full['b'] * gx
        h = 2.46 * (full['qa'] + full['qb'] * gx)
        print(f'{gx:>12.0f} {b + h:>+9.2f} {b:>+9.2f} {b - h:>+9.2f} {2 * h:>9.2f}')
    print()

    fig = build(meas, y, full, pal)
    save_fig(fig, args.name, outdir=outdir)
    prev_dir = outdir / 'preview'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev = prev_dir / f'{args.name}__print_size.png'
    fig.savefig(prev, dpi=96)
    print(f'  {prev}  (판독 확인용, 원고에 넣지 않는다)')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
