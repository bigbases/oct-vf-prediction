#!/usr/bin/env python3
"""fig_where_gain 의 2패널 확장 — (a) 구간별 계통 편향 (b) 중심화 전후 이득.

단일 패널판(`make_fig_where_gain.py`)은 "이득이 저감도에 몰린다"만 보인다.
리뷰어가 다음에 묻는 것은 "그 이득이 정보냐 편향 상쇄냐"다. (a) 가 두 브랜치의
구간별 계통 편향을 보이고, (b) 가 image branch 를 중심화했을 때 그 이득이
남는지를 보인다. 두 패널을 같이 놓아야 논증이 닫힌다.

데이터는 **재계산하지 않는다**. experiments/bias_structure/ 의 두 산출물을 읽는다:
  bias_by_bin.json         (a) 구간별 잔차 평균. 패스 B / OOF
  bias_by_bin_refit.json   (b) original / global_center_train_refitw. 패스 B / OOF

패스 B 기준이다 — 정본 runs/ 는 아직 패스 A 라 여기서 읽지 않는다. 그래서
기존 fig_where_gain(패스 A, OOF+held-out)과 (b) 의 원본 계열은 값이 조금 다르다
(OOF 기울기 +0.252 대 +0.230). 같은 그림에 섞지 않는다.

(b) 의 중심화는 학습 폴드에서 뽑은 스칼라 하나를 빼고 w 를 재적합한 것이다.
구간별 oracle 보정이 아니다.

출력:
  build/figs/fig_where_gain_2panel.png / .pdf          양단폭 180 mm, 가로 2패널
  build/figs/fig_where_gain_2panel_single.png / .pdf   단폭 88 mm, 세로 2패널
  build/figs/preview/...__print_size.png               판독 확인용. 원고에 넣지 않는다
실행: python scripts/make_fig_where_gain_2panel.py
env: aaa (matplotlib + numpy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from fig_style import (figure, zero_line, panel_label, save_fig, series,  # noqa: E402
                       faint, use_style, MM, GREYS, ANNOT)

BIAS = ROOT / 'experiments' / 'bias_structure' / 'bias_by_bin.json'
REFIT = ROOT / 'experiments' / 'bias_structure' / 'bias_by_bin_refit.json'

PASS, SPLIT = 'B', 'OOF'
ENSEMBLE = 'ENSEMBLE'
BACKBONES = ['IR-v2', 'Inception-v3', 'VGG16', 'Xception', 'DenseNet121']
# (b) 에 그릴 두 계열. 오른쪽 값은 보고된 기울기 — 엉뚱한 패스/변형을 집어오면
# 여기서 걸린다.
VARIANTS = [('original', 'As reported', +0.230),
            ('global_center_train_refitw', 'Image branch centred', -0.010)]

XS = np.arange(1, 5, dtype=float)
TICKS = [r'$[0,10)$', r'$[10,20)$', r'$[20,30)$', r'$[30,\infty)$']


def load() -> dict:
    bias = json.loads(BIAS.read_text(encoding='utf-8'))['passes'][PASS]
    refit = json.loads(REFIT.read_text(encoding='utf-8'))['passes'][PASS]

    ens = bias[ENSEMBLE][SPLIT]
    out = {'xgb': np.array(ens['xgb_residual_mean_by_bin']),
           'ens_bias': np.array(ens['cnn_residual_mean_by_bin']),
           'n_points': ens['n_points_by_bin'],
           'backbones': {nm: np.array(bias[nm][SPLIT]['cnn_residual_mean_by_bin'])
                         for nm in BACKBONES},
           'trend': []}

    # XGB 잔차는 백본과 무관해야 한다. 어긋나면 서로 다른 마스크를 섞은 것이다.
    for nm in BACKBONES:
        got = np.array(bias[nm][SPLIT]['xgb_residual_mean_by_bin'])
        if not np.allclose(got, out['xgb'], rtol=0, atol=1e-12):
            raise SystemExit(f'XGB 구간 잔차가 {nm} 에서 다르다 — 마스크 불일치')

    bad = []
    for key, label, expect in VARIANTS:
        t = refit[ENSEMBLE][SPLIT]['variants'][key]
        if round(t['mean_slope'], 3) != expect:
            bad.append(f'{key} 기울기 {t["mean_slope"]:.4f} != 보고값 {expect:+.3f}')
        out['trend'].append({
            'label': label,
            'y': np.array([pb['mean_delta'] for pb in t['per_bin']]),
            'n_bin': [pb['n_eyes'] for pb in t['per_bin']],
            'slope': t['mean_slope'],
            'ci': (t['slope_ci_lo'], t['slope_ci_hi']),
            'p': t['p_wilcoxon'],
            'n_eyes': t['n_eyes_with_slope'],
            'w': t['w_by_fold'],
        })
    if bad:
        raise SystemExit('산출물이 보고값과 어긋난다 — ' + '; '.join(bad))
    return out


# rcParams['axes.unicode_minus'] 는 눈금 포맷터에만 걸린다. f-string 이 찍는
# '-' 는 ASCII 하이픈(U+002D)이라 축 눈금의 유니코드 마이너스(U+2212)와
# 한 그림 안에서 길이·높이가 다르게 보인다. 텍스트에 쓰는 수치는 여기를 거친다.
MINUS = '\u2212'


def signed(v: float, nd: int = 3) -> str:
    return f'{v:+.{nd}f}'.replace('-', MINUS)


def sign_labels(ax, y_pos: float, y_neg: float, txt_pos: str, txt_neg: str) -> None:
    """부호 규약을 왼쪽 여백(x < 1, 자료 없음)에 적는다. 캡션에만 두면
    그림만 떼어 본 독자가 방향을 거꾸로 읽는다."""
    for y, txt in ((y_pos, txt_pos), (y_neg, txt_neg)):
        ax.annotate(txt, xy=(0.66, y), ha='left', va='center',
                    fontsize=6.8, color=ANNOT, style='italic')


def panel_a(ax, d: dict) -> None:
    zero_line(ax, linewidth=0.8)
    for nm in BACKBONES:                       # 배경 계열: 개별 백본 5종
        ax.plot(XS, d['backbones'][nm], **faint())
    for i, (y, lab) in enumerate(((d['ens_bias'], 'Image branch (ensemble)'),
                                  (d['xgb'], 'Summary branch (XGBoost)'))):
        s = series(i)
        ax.plot(XS, y, color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markerfacecolor='white',
                markeredgecolor=s['color'], linewidth=1.4, zorder=3, label=lab)
    ax.plot([], [], label='Individual backbones', **faint())

    ax.set_ylabel('Mean residual (dB)')
    # 부호 규약. 0선 바로 옆에 두면 백본 계열이 그 위를 지나 글자가 깨진다
    # (실측). 왼쪽 아래는 자료가 없다 — 첫 구간에서 모든 계열이 +13 dB 위다.
    sign_labels(ax, 2.5, -3.5, 'over-predicts', 'under-predicts')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.85)


def panel_b(ax, d: dict) -> None:
    zero_line(ax, linewidth=0.8)
    for i, t in enumerate(d['trend']):
        s = series(i)
        ax.plot(XS, t['y'], color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markerfacecolor='white',
                markeredgecolor=s['color'], linewidth=1.4, zorder=3,
                label=t['label'])
    ax.set_ylabel(r'Fusion $-$ image branch (dB)')
    # 중심화 계열이 0선에 붙어 있어 0선 옆에 적을 수 없다. 왼쪽 여백에 둔다.
    sign_labels(ax, 0.15, -0.30, 'image branch better', 'fusion better')

    # 머리글 'Slope (dB per bin, 95% CI):' 은 캡션으로 옮겼다. 값과 CI 는 남긴다.
    lines = [f'{t["label"]}  {signed(t["slope"])} '
             f'({signed(t["ci"][0])}, {signed(t["ci"][1])})' for t in d['trend']]
    ax.annotate('\n'.join(lines), xy=(0.015, 0.985), xycoords='axes fraction',
                ha='left', va='top', fontsize=6.3, color='#3d3d3d', linespacing=1.6)
    ax.legend(loc='lower right', fontsize=6.5, framealpha=0.85)


def finish(ax) -> None:
    ax.set_xticks(XS)
    ax.set_xticklabels(TICKS)
    ax.set_xlim(0.6, 4.4)
    ax.set_xlabel('Measured sensitivity bin (dB)')
    lo, hi = ax.get_ylim()
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


# MDPI 본문 폭은 약 170 mm다. fig_style 의 'double'(180 mm)은 이를 넘어
# 조판에서 축소되므로, 가로 2패널은 165 mm로 잡는다. 높이는 180 mm×0.42와
# 같은 약 76 mm를 유지한다(165×0.46).
WIDE_MM = 165.0
WIDE_RATIO = 0.46


def build(d: dict, layout: str):
    """layout='wide' 165 mm 가로 2패널 / 'tall' 단폭 세로 2패널."""
    if layout == 'wide':
        use_style()
        w = WIDE_MM * MM
        fig, axes = plt.subplots(1, 2, figsize=(w, w * WIDE_RATIO),
                                 layout='constrained')
        lab_x = -0.13
    else:
        fig, axes = figure('single', ratio=1.55, nrows=2)
        lab_x = -0.17
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    for ax in axes:
        finish(ax)
    for ax, t in zip(axes, ('(a)', '(b)')):
        panel_label(ax, t, x=lab_x, y=1.02)
    return fig


def emit(fig, name: str, outdir: Path, *, pdf: bool) -> None:
    save_fig(fig, name, outdir=outdir, pdf=pdf)
    prev_dir = ROOT / 'build' / 'figs' / 'preview'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev = prev_dir / f'{name}__print_size.png'
    fig.savefig(prev, dpi=96)
    print(f'  {prev.relative_to(ROOT)}  (판독 확인용, 원고에 넣지 않는다)')
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'figs')
    args = ap.parse_args()
    outdir = Path(args.out).resolve()
    pdf = outdir == (ROOT / 'build' / 'figs')

    d = load()
    print(f'가드 통과. 패스 {PASS} / {SPLIT}. 재계산 없음 — '
          f'{BIAS.name} + {REFIT.name} 을 읽었다.\n')

    print('(a) 구간별 잔차 평균 (dB, 양수 = 과대예측)')
    hdr = f'{"branch":24s} ' + ' '.join(f'{t:>10s}' for t in
                                        ['[0,10)', '[10,20)', '[20,30)', '[30,inf)'])
    print(hdr)
    print('-' * len(hdr))
    for nm in BACKBONES:
        print(f'{nm:24s} ' + ' '.join(f'{v:>+10.3f}' for v in d['backbones'][nm]))
    print(f'{"Image branch (ensemble)":24s} ' +
          ' '.join(f'{v:>+10.3f}' for v in d['ens_bias']))
    print(f'{"Summary branch (XGB)":24s} ' +
          ' '.join(f'{v:>+10.3f}' for v in d['xgb']))
    print(f'{"points / bin":24s} ' + ' '.join(f'{v:>10d}' for v in d['n_points']))

    print('\n(b) 구간별 fusion - image branch (dB, 음수 = fusion 우세)')
    print(hdr + f' {"slope":>8s} {"95% CI":>18s} {"p":>9s}')
    for t in d['trend']:
        print(f'{t["label"]:24s} ' + ' '.join(f'{v:>+10.3f}' for v in t['y']) +
              f' {t["slope"]:>+8.3f} ({t["ci"][0]:+.3f}, {t["ci"][1]:+.3f}) '
              f'{t["p"]:>9.2g}')
        print(f'{"":24s} ' + ' '.join(f'{v:>10d}' for v in t['n_bin']) +
              f'   eyes {t["n_eyes"]}, w {"/".join(f"{x:.2f}" for x in t["w"])}')
    print()

    for layout, name in (('wide', 'fig_where_gain_2panel'),
                         ('tall', 'fig_where_gain_2panel_single')):
        emit(build(d, layout), name, outdir, pdf=pdf)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
