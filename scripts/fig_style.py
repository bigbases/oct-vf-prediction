#!/usr/bin/env python3
"""Shared figure style for the manuscript figures (MDPI two-column).

Every scripts/make_fig_*.py builds its canvas with figure() and writes it with
save_fig(). The three design decisions behind that are spelled out below.

원고 그림 공용 스타일.

MDPI 2단 조판 기준. 그림 스크립트(`scripts/make_fig_*.py`)는 전부 이 모듈의
`figure()` 로 캔버스를 만들고 `save_fig()` 로 저장한다.

설계 근거 세 가지:

1. **크기를 tight bbox로 흔들지 않는다.** `bbox_inches='tight'` 는 저장 시점에
   캔버스를 잘라내서 실제 폭이 88 mm가 아니게 된다. 대신 `layout='constrained'`
   로 축을 캔버스 안에 맞추고 bbox 없이 저장한다. PNG 폭이 정확히 2079 px
   (= 88 mm @ 600 dpi)로 나온다.
2. **색만으로 계열을 구분하지 않는다.** 흑백 인쇄와 색각 이상 양쪽에서 살아남게
   선종류와 마커를 반드시 함께 바꾼다. 색 팔레트는 Okabe-Ito(색각 이상 대응).
3. **Palatino는 이 환경에 없다.** MDPI 본문은 Palatino 계열이지만 설치된 serif는
   STIXGeneral / Liberation Serif / DejaVu Serif 뿐이다. 폴백 목록 맨 앞에
   Palatino·P052를 두어 나중에 설치되면 자동으로 잡히게 했다.

출력: 없음 (모듈)
실행: python scripts/fig_style.py          # 데모 그림으로 단폭 판독성 확인
      python scripts/fig_style.py --out build/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'docs' / 'journal_manuscript'

MM = 1 / 25.4
# MDPI 2단: 단폭 \linewidth ~= 88 mm, 양단폭 \textwidth ~= 180 mm.
WIDTH = {'single': 88 * MM, 'double': 180 * MM}
DPI = 600

# 축소 후 판독 하한을 8/7 pt로 잡고 시작한다. 더 줄이면 인쇄에서 무너진다.
RC = {
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Palatino Linotype', 'P052', 'URW Palladio L',
                   'STIXGeneral', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 8,

    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 1.0,
    'lines.markersize': 3.0,
    'lines.markeredgewidth': 0.6,

    'grid.linewidth': 0.4,
    'grid.color': '#cccccc',
    'grid.alpha': 0.8,
    'legend.frameon': False,
    'legend.handlelength': 2.4,
    'legend.borderaxespad': 0.3,
    'legend.labelspacing': 0.35,

    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.facecolor': 'white',
    # 벡터 저장 시 글자를 곡선으로 바꾸지 않고 TrueType으로 묻는다.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': True,
}

# 색각 이상 대응(Okabe-Ito) + 회색조에서도 명도가 갈리는 순서.
OKABE_ITO = ['#000000', '#0072B2', '#D55E00', '#009E73', '#CC79A7',
             '#E69F00', '#56B4E9']
GREYS = ['#000000', '#7a7a7a', '#b4b4b4', '#3d3d3d', '#969696']
# 주석·부호 라벨 공통 색. 인쇄에서 뭉개지지 않는 하한이 대략 50% 회색이라
# 이보다 밝게 두지 않는다. 그림 3·5 가 같이 쓴다.
ANNOT = '#4a4a4a'
# 회색조 인쇄에서 5계열이 갈려야 한다. 대시 길이(0 / 6 / 1 / 2.5 / 8)와
# 점 개수(- / - / - / 1 / 2)를 둘 다 벌려 두 축으로 구분되게 한다.
DASHES = [
    '-',                                        # 실선
    (0, (6, 1.8)),                              # 긴 파선
    (0, (1, 1.6)),                              # 점선
    (0, (2.5, 1.2, 0.6, 1.2)),                  # 짧은 파선 - 점
    (0, (8, 1.6, 0.8, 1.6, 0.8, 1.6)),          # 긴 파선 - 점 - 점
]
MARKERS = ['o', 's', '^', 'D', 'v']


def use_style() -> None:
    """rcParams를 이 모듈 기준으로 갈아끼운다. figure()가 자동으로 부른다."""
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(RC)


def series(i: int, *, grey: bool = False) -> dict:
    """계열 i의 색·선종류·마커. 셋을 항상 함께 바꿔 흑백에서도 갈린다."""
    palette = GREYS if grey else OKABE_ITO
    return {'color': palette[i % len(palette)],
            'linestyle': DASHES[i % len(DASHES)],
            'marker': MARKERS[i % len(MARKERS)]}


def faint(color: str = '#000000') -> dict:
    """배경으로 깔리는 보조 계열(예: primary 아닌 백본들)."""
    return {'color': color, 'linestyle': '-', 'linewidth': 0.5, 'alpha': 0.30,
            'zorder': 1.5}


def figure(width: str = 'single', ratio: float = 0.72, *,
           height: float | None = None, **kw):
    """단폭(기본) 또는 양단폭 캔버스. ratio는 높이/폭."""
    use_style()
    w = WIDTH[width]
    h = height if height is not None else w * ratio
    return plt.subplots(figsize=(w, h), layout='constrained', **kw)


def band(ax, x0: float, x1: float, *, label: str | None = None, **kw):
    """교차 구간 등의 음영. 회색 채움이라 색과 무관하게 인쇄된다."""
    opts = {'color': '#000000', 'alpha': 0.12, 'linewidth': 0, 'zorder': 0}
    opts.update(kw)
    return ax.axvspan(x0, x1, label=label, **opts)


def zero_line(ax, y: float = 0.0, **kw):
    opts = {'color': '#000000', 'linewidth': 0.5, 'linestyle': (0, (2, 2)),
            'zorder': 1}
    opts.update(kw)
    return ax.axhline(y, **opts)


def panel_label(ax, text: str, *, x: float = -0.16, y: float = 1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left')


def save_fig(fig, name: str, *, outdir: Path | str = FIG_DIR, pdf: bool = True,
             tight: bool = False) -> list[Path]:
    """PNG(600 dpi) + PDF 저장. 경로·dpi·bbox를 여기서만 정한다.

    tight=False가 기본이다 — 캔버스 폭을 조판 폭 그대로 유지해야 축소율이
    예측 가능하다. 라벨이 잘리면 ratio나 constrained layout으로 해결한다.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kw = {'bbox_inches': 'tight', 'pad_inches': 0.01} if tight else {}
    written = []
    png = outdir / f'{name}.png'
    fig.savefig(png, dpi=DPI, **kw)
    written.append(png)
    if pdf:
        p = outdir / f'{name}.pdf'
        fig.savefig(p, **kw)
        written.append(p)
    for p in written:
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        print(f'  {rel}  ({p.stat().st_size / 1024:.0f} KB)')
    return written


def _demo(outdir: Path) -> None:
    """단폭 축소 판독성 확인용. 데이터는 전부 합성이다 — 결과가 아니다."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 401)

    fig, ax = figure('single')
    for i in range(5):                       # 보조 계열(백본 5개 자리)
        ax.plot(x, 3.0 * np.exp(-x / 45) - 1.2 + 0.25 * rng.standard_normal(),
                **faint())
    for i, lab in enumerate(['synthetic A', 'synthetic B']):
        y = (3.0 - 0.9 * i) * np.exp(-x / (45 + 12 * i)) - 1.2
        st = series(i)
        ax.plot(x, y, color=st['color'], linestyle=st['linestyle'], zorder=3)
        ax.plot(x[::50], y[::50], linestyle='none', marker=st['marker'],
                color=st['color'], markerfacecolor='white', label=lab, zorder=3)
    band(ax, 84, 86, label='crossing band')
    zero_line(ax)
    ax.set_xlabel('Percentile')
    ax.set_ylabel(r'$\Delta$ absolute error (dB)')
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right')
    ax.set_title('DEMO ONLY: synthetic data, not a result')
    print('데모 저장:')
    save_fig(fig, 'fig_style_demo', outdir=outdir)
    # 축소 판독 확인: 같은 캔버스를 96 dpi로 한 장 더 뽑으면 화면에서 본
    # 크기가 인쇄 시 88 mm와 같아진다(88 mm @ 96 dpi = 333 px).
    preview = Path(outdir) / 'fig_style_demo_at_print_size.png'
    fig.savefig(preview, dpi=96)
    print(f'  {preview.relative_to(ROOT)}  (인쇄 실물 크기 미리보기)')
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'figs' / 'style_demo')
    args = ap.parse_args()
    use_style()
    fam = plt.rcParams['font.serif']
    from matplotlib.font_manager import findfont, FontProperties
    got = findfont(FontProperties(family='serif'))
    print(f'serif 폴백 1순위 후보: {fam[0]} / 실제 선택: {Path(got).name}')
    _demo(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
