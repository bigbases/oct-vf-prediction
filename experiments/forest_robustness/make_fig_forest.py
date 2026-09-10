#!/usr/bin/env python3
"""Figure 2: forest plot of the robustness checks of Section 4.2.

One sign convention only: x = mean over eyes of RMSE(fusion) - RMSE(summary),
so negative favours fusion. Out-of-fold and held-out are drawn in separate
blocks, because pooling them would make the interval widths reflect sample size
rather than the conditions.

Figure 6: 강건성 검사 forest plot.

원고 §4.2 의 아홉 항목은 표 3~5 와 본문 숫자로만 흩어져 있다. 이 그림은 그것을
대체하지 않고 요약한다 — 각 조건에서 fusion 이 summary branch(XGB) 를 얼마나
이기는지를 한 축 위에 모은다.

부호 규약은 하나뿐이다: x = mean_eyes[ RMSE(fusion) - RMSE(summary) ], 음수가
fusion 우세. 축 라벨에 방향을 적는다. 원고 본문은 이득을 양수로 쓰는 곳과
차이를 음수로 쓰는 곳이 섞여 있으나 그림은 섞지 않는다.

out-of-fold(240안)와 held-out(37안)은 구획을 나눈다. 같은 구획에 놓으면
신뢰구간 폭이 조건 차이가 아니라 표본 크기 차이를 반영한다.

입력: experiments/forest_robustness/forest_rows.json (compute_rows.py 산출)
출력: build/figs/fig_forest_robustness.{png,pdf} + preview
실행: python experiments/forest_robustness/make_fig_forest.py [--width 165]
env: aaa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from fig_style import use_style, save_fig, MM, GREYS, OKABE_ITO  # noqa: E402

SRC = REPO / 'experiments/forest_robustness/forest_rows.json'

# 색: Figure 2 와 같은 계열(검정 + #0072B2). 파랑은 primary 조건 하나에만 쓴다 —
# 나머지 열두 행은 전부 primary 의 변형이고, 독자가 기준선을 즉시 찾아야 한다.
# 회색조 인쇄 대비로 마커 모양도 함께 바꾼다(다이아 vs 원). 조건 '종류'(백본/
# 시드/마스크)는 색으로 나누지 않는다 — 라벨이 이미 말하고, 색을 넷으로 쪼개면
# 파랑이 뜻하는 바가 그림 2 와 충돌한다.
REF, BASE = OKABE_ITO[1], '#000000'

# compute_rows.py 의 label -> 그림에 찍을 짧은 라벨. 순서가 곧 행 순서다.
OOF_ROWS = [
    ('Primary: Inception-ResNet-v2', 'Primary (Inception-ResNet-v2)', True),
    ('Inception-v3',                 'Inception-v3',                  False),
    ('VGG16',                        'VGG16',                         False),
    ('Xception',                     'Xception',                      False),
    ('DenseNet121',                  'DenseNet121',                   False),
    ('Five-backbone ensemble',       'Five-backbone ensemble',        False),
    ('Seed 43',                      'Seed 43',                       False),
    ('Seed 44',                      'Seed 44',                       False),
    ('No vertical C/D (25 params)',  'No vertical C/D (25 params)',   False),
    ('Floor labels removed',         'Floor labels removed',          False),
    ('FL < 20%',                     'Fixation loss $<$ 20%',         False),
    ('FL < 20% and FP < 15%',        'FL $<$ 20% and FP $<$ 15%',     False),
    ('Fellow-eye features (46 params)', 'Fellow-eye features (46 params)', False),
]
TEST_ROWS = [
    ('Held-out: primary',  'Primary (Inception-ResNet-v2)', True),
    ('Held-out: ensemble', 'Five-backbone ensemble',        False),
]
GAP = 2.8          # 두 구획 사이 빈 행 수 (구분선 + 구획 머리글 자리)
NCOL_X = 0.27      # n 열을 찍을 데이터 x 좌표. 최대 CI 상한(+0.01)보다 오른쪽


def draw(rows_json: dict, width_mm: float):
    by = {r['label']: r for r in rows_json['rows']}
    use_style()
    n_slots = len(OOF_ROWS) + GAP + len(TEST_ROWS) + 1.2
    h_mm = 5.2 * n_slots + 22            # 행당 5.2 mm + 축·머리글 여백
    fig, ax = plt.subplots(figsize=(width_mm * MM, h_mm * MM), layout='constrained')

    ticks, labels, greyed = [], [], []
    ax.axvline(0.0, color=BASE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)

    def block(items, top_y, title):
        yy = top_y
        ax.text(0.0, yy + 0.95, title, transform=ax.get_yaxis_transform(),
                ha='left', va='bottom', fontsize=7.5, style='italic',
                clip_on=False)
        for key, lab, is_ref in items:
            ticks.append(yy); labels.append(lab); greyed.append(key is None)
            if key is None:                       # 예측이 저장돼 있지 않은 행
                ax.text(-1.90, yy, 'no stored predictions', ha='left',
                        va='center', fontsize=6.8, style='italic',
                        color=GREYS[1], zorder=3)
                ax.text(NCOL_X, yy, '--', ha='right', va='center', fontsize=7,
                        color=GREYS[1])
                yy -= 1.0
                continue
            r = by[key]
            c = REF if is_ref else BASE
            ax.plot([r['ci_lo'], r['ci_hi']], [yy, yy], color=c, linewidth=0.9,
                    solid_capstyle='butt', zorder=3)
            for xe in (r['ci_lo'], r['ci_hi']):   # 끝단 캡
                ax.plot([xe, xe], [yy - 0.16, yy + 0.16], color=c, linewidth=0.9,
                        zorder=3)
            ax.plot([r['delta']], [yy], linestyle='none',
                    marker='D' if is_ref else 'o',
                    markersize=3.4 if is_ref else 3.0, markerfacecolor=c,
                    markeredgecolor=c, zorder=4)
            ax.text(NCOL_X, yy, str(r['n_eyes']), ha='right', va='center',
                    fontsize=7, color=BASE)
            yy -= 1.0
        return yy

    ax.text(NCOL_X, 0.95, 'eyes', ha='right', va='bottom', fontsize=7.5,
            style='italic', color=GREYS[1])
    y = block(OOF_ROWS, 0.0, 'Out-of-fold')
    ax.axhline(y + 1.0 - GAP / 2, color=GREYS[2], linewidth=0.5, zorder=0)
    y = block(TEST_ROWS, y + 1.0 - GAP, 'Held-out')

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for t, g in zip(ax.get_yticklabels(), greyed):
        if g:
            t.set_color(GREYS[1])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y + 0.6, 1.9)
    ax.set_xlim(-1.95, 0.30)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_xlabel(r'Per-eye RMSE difference, fusion $-$ summary (dB)'
                  '\n' r'$\leftarrow$ fusion better        summary better $\rightarrow$')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_bounds(-1.95, 0.0)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=165.0, help='mm')
    ap.add_argument('--out', type=Path, default=REPO / 'build' / 'figs')
    ap.add_argument('--name', default='fig_forest_robustness')
    args = ap.parse_args()

    d = json.loads(SRC.read_text(encoding='utf-8'))
    bad = [k for k, v in d['checks'].items() if isinstance(v, dict) and not v['ok']]
    if bad:
        raise SystemExit(f'가드 실패, 그리지 않는다: {bad}')

    fig = draw(d, args.width)
    fig.canvas.draw()
    # 행 라벨이 잘리는지 실측: 가장 긴 y 틱 라벨의 왼쪽 끝이 캔버스 안인가.
    left = min(t.get_window_extent().x0 for t in fig.axes[0].get_yticklabels())
    ax_l = fig.axes[0].get_window_extent().x0
    print(f'폭 {args.width:.0f} mm: 라벨 왼쪽 끝 {left:.1f} px (캔버스 0), '
          f'축 왼쪽 {ax_l:.1f} px, 축 폭 '
          f'{fig.axes[0].get_window_extent().width / fig.dpi * 25.4:.1f} mm '
          f'-> {"잘림" if left < 0 else "여유 %.1f mm" % (left / fig.dpi * 25.4)}')
    save_fig(fig, args.name, outdir=args.out)
    prev = Path(args.out) / 'preview'
    prev.mkdir(parents=True, exist_ok=True)
    p = prev / f'{args.name}__print_size.png'
    fig.savefig(p, dpi=96)
    print(f'  {p}  (판독 확인용, 원고에 넣지 않는다)')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
