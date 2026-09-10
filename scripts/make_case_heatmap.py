#!/usr/bin/env python3
"""그림 3: 대표 증례 24-2 히트맵 — 실측 vs 요약 브랜치 vs 영상 브랜치 vs late fusion.

`scripts/fig_style.py` 규격(serif, 8/7 pt, 양단폭 180 mm, 600 dpi)을 따른다.
다른 세 그림과 나란히 놓았을 때 폰트와 눈금 크기가 같아야 하기 때문이다.

패널 제목은 원고 본문 용어를 쓴다("Summary branch" / "Image branch"). `XGB` ·
`CNN` · `52-pt masked` · `w_xgb` 는 코드 용어라 그림에 넣지 않는다 — 캡션이
같은 말을 이미 한다.

컬러맵은 viridis 다. cividis 와 실물 크기로 대조한 뒤 고른 것이라 근거를
남긴다.

  cividis 가 나은 축: 회색조에서 5 dB 차의 최소 명도차가 0.102 로 viridis
  (0.065)보다 57% 넓고, 청-황 단일 축이라 protan/deutan 양쪽에서 순서가
  보존된다. 둘 다 명도는 단조다 — viridis 가 순서를 뒤집는다는 말은 틀렸다.

  viridis 를 고른 축: 이 그림이 실제로 읽히는 방식. 24~29 dB 구간이 cividis
  에서는 전부 같은 겨자색 덩어리가 되는데 viridis 는 청록에서 연두로 갈린다
  (Image branch · Late fusion 패널). 그리고 Measured field 의 0 dB 두 칸이
  viridis 에서는 진보라로 튀고 cividis 에서는 남색이라 주변과 덜 대비된다.
  그 두 칸이 이 그림의 요지(어느 브랜치도 절대 암점을 재현하지 못한다)다.

  즉 회색조 여유를 내주고 유채색 판독을 샀다. 칸마다 숫자가 찍혀 있어
  회색조 인쇄에서도 값 자체는 읽히므로 감당되는 거래다.

`--cmap` 으로 바꿔 비교할 수 있다. 칸 안 숫자 색은 배경 명도에서 자동으로
정하므로 컬러맵을 바꿔도 대비가 유지된다(viridis 의 흑/백 전환점은 22 dB,
cividis 는 18 dB — 손으로 박으면 반드시 틀린다).

가중치 W = 0.47 은 IR-v2 단일 백본 late fusion 값이다(원고 §Four ways).
5백본 앙상블(w = 0.33)이 아니다 — 이 그림은 단일 백본 증례다.

입력 (읽기 전용):
  runs/oof/xgb_90d_fold{0..4}_val.npz
  runs/phasec_b0_inception_resnet_v2_5fold/val_preds_fold{0..4}.npz

출력:
  <out>/fig_case_representative.png
  build/figs/preview/  판독 확인용(인쇄 실물 크기 + 회색조). 원고에 넣지 않는다
실행: python scripts/make_case_heatmap.py PID OS
      python scripts/make_case_heatmap.py PID OS --cmap cividis --name fig_case_cividis
env: aaa (matplotlib + numpy + PIL)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402
from fig_style import figure, save_fig  # noqa: E402

W = 0.47
XGB = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
CNN = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
VMIN, VMAX = 0, 34


# 24-2 좌표 (build_severity_region.py와 동일)
def build_coords():
    rows = [(+21, [-9, -3, 3, 9]), (+15, [-15, -9, -3, 3, 9, 15]),
            (+9, [-21, -15, -9, -3, 3, 9, 15, 21]), (+3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]),
            (-3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]), (-9, [-21, -15, -9, -3, 3, 9, 15, 21]),
            (-15, [-15, -9, -3, 3, 9, 15]), (-21, [-9, -3, 3, 9])]
    return [(x, y) for y, xs in rows for x in xs]


COORDS54 = build_coords()
BLIND = {25, 34}
COORDS52 = [c for i, c in enumerate(COORDS54) if i not in BLIND]
XS = [-27, -21, -15, -9, -3, 3, 9, 15, 21]
YS = [21, 15, 9, 3, -3, -9, -15, -21]


def to_grid(vec52, mask52=None):
    g = np.full((8, 9), np.nan)
    for v, (x, y) in zip(vec52, COORDS52):
        g[YS.index(y), XS.index(x)] = v
    if mask52 is not None:
        for i, (mm, (x, y)) in enumerate(zip(mask52, COORDS52)):
            if not mm:
                g[YS.index(y), XS.index(x)] = np.nan
    return g


def find_eye(pid, eye):
    for k in range(5):
        xz = load_oof_npz(XGB[k]); cz = load_oof_npz(CNN / f'val_preds_fold{k}.npz')
        cidx = {kk: j for j, kk in enumerate(cz['keys'])}
        for i, key in enumerate(xz['keys']):
            if key[0] == pid and key[1] == eye and key in cidx:
                j = cidx[key]
                m = (xz['mask'][i] & cz['mask'][j]).astype(bool)
                return dict(lab=xz['labels'][i], xgb=xz['pred'][i], cnn=cz['pred'][j],
                            fus=W * xz['pred'][i] + (1 - W) * cz['pred'][j], mask=m, key=key, fold=k)
    raise SystemExit(f'not found: {pid} {eye}')


def rmse(p, l, m):
    d = (p - l)[m]; return float(np.sqrt(np.mean(d ** 2)))
def mae(p, l, m):
    d = (p - l)[m]; return float(np.mean(np.abs(d)))


def text_color(cmap, value):
    """칸 배경 명도(BT.601)를 보고 흰/검정을 고른다.

    임계값을 dB로 못 박으면 컬러맵을 바꿀 때마다 손으로 다시 맞춰야 하고,
    회색조 변환본에서 글자가 배경에 묻히는 사고가 난다.
    """
    r, g, b = cmap((value - VMIN) / (VMAX - VMIN))[:3]
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'


def build(d, cmap_name):
    m = d['mask']
    panels = [('Measured field', d['lab'], None)]
    for title, vec in (('Summary branch', d['xgb']), ('Image branch', d['cnn']),
                       ('Late fusion', d['fus'])):
        panels.append((title, vec, (rmse(vec, d['lab'], m), mae(vec, d['lab'], m))))

    # 폭 180 mm 안에 패널 4개 + 색상바. 패널당 약 41 mm, 칸 하나가 약 4.6 mm 라
    # 두 자리 숫자를 6.5 pt(약 2.3 mm)로 넣어도 칸 안에 여유가 있다.
    fig, axes = figure('double', ratio=0.245, ncols=4)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('#e8e8e8')      # 마스크된 지점. 감도 0과 구분돼야 한다
    for ax, (title, vec, err) in zip(axes, panels):
        g = to_grid(vec, m)
        im = ax.imshow(g, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect='equal')
        for r in range(8):
            for c in range(9):
                if not np.isnan(g[r, c]):
                    ax.text(c, r, f'{g[r, c]:.0f}', ha='center', va='center',
                            fontsize=6.5, color=text_color(cmap, g[r, c]))
        # 실측 패널만 한 줄이면 제목이 아래로 내려앉아 나머지 셋과 어긋난다.
        # 빈 둘째 줄을 넣어 네 제목의 첫 줄을 같은 높이에 맞춘다.
        ax.set_title(f'{title}\n ' if err is None
                     else f'{title}\nRMSE {err[0]:.2f} / MAE {err[1]:.2f} dB')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.012)
    cbar.set_label('VF sensitivity (dB)')
    cbar.outline.set_linewidth(0.6)
    return fig


def previews(fig, name):
    """인쇄 실물 크기 래스터와 그 회색조 변환본. --out 을 따라가지 않는다."""
    from PIL import Image
    out = ROOT / 'build' / 'figs' / 'preview'
    out.mkdir(parents=True, exist_ok=True)
    # 180 mm @ 96 dpi = 680 px. 화면에서 본 크기가 인쇄 폭과 같아진다.
    p = out / f'{name}__print_size.png'
    fig.savefig(p, dpi=96)
    q = out / f'{name}__grey.png'
    Image.open(p).convert('L').save(q)
    for f in (p, q):
        print(f'  {f.relative_to(ROOT)}')
    return p, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pid'); ap.add_argument('eye')
    ap.add_argument('--out', type=Path, default=ROOT / 'docs' / 'journal_manuscript')
    ap.add_argument('--name', default='fig_case_representative')
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--pdf', action='store_true')
    args = ap.parse_args()

    d = find_eye(args.pid, args.eye)
    m = d['mask']
    print(f'fold={d["fold"]}  n_valid={int(m.sum())}  cmap={args.cmap}')
    for n, v in (('summary', d['xgb']), ('image', d['cnn']), ('fusion', d['fus'])):
        print(f'  {n:8s} RMSE {rmse(v, d["lab"], m):6.3f}  MAE {mae(v, d["lab"], m):6.3f}')

    fig = build(d, args.cmap)
    save_fig(fig, args.name, outdir=Path(args.out).resolve(), pdf=args.pdf)
    previews(fig, args.name)
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
