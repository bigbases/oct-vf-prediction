#!/usr/bin/env python3
"""그림 1: 파이프라인 도식.

2 Cirrus 큐브 → 두께맵 2종(비식별 대표안의 실제 crop) + 요약 26지표
→ CNN / 52×XGB → late fusion (w·XGB + (1-w)·CNN) → 24-2 52지점 감도.

배치는 가운데선 M 하나로 잡는다. 위 줄은 M+18, 아래 줄은 M-18, fusion 과
출력은 M. 큐브에서 나가는 네 화살표가 M 에 대해 거울대칭이 되는 건 이 규칙
덕분이라, 상자를 옮길 때 y 를 손으로 박지 말고 TOP/BOT/M 을 쓴다.

env: aaa (matplotlib + PIL + numpy). 사용:
  python scripts/make_fig_pipeline.py PID EYE [out.png]
"""
from __future__ import annotations
import sys
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa
from make_case_heatmap import to_grid, find_eye  # noqa

if len(sys.argv) < 3:
    raise SystemExit('usage: make_fig_pipeline.py PID EYE [out.png]')
PID, EYE = sys.argv[1], sys.argv[2]
OUT = (
    Path(sys.argv[3])
    if len(sys.argv) > 3
    else ROOT / 'docs/journal_manuscript/fig_pipeline.png'
)

BOX = dict(boxstyle='round,pad=0.012', mutation_aspect=1.0)
GRAY = '#f2f2f2'
EDGE = '#555555'
NOTE = '#333333'

PT = 0.3019       # 1 pt 를 데이터 y 단위로 환산 (figsize 높이 4.6 in / 100)
_CENTERED: list = []   # (머리글, 본문, 상자) — recenter_boxes 가 쓴다

# 큐브 → 표현 배선. 'x' = 화살표 4개(교차), 'bus' = 세로 버스 1개.
# 두 안 모두 "큐브 둘 다 표현 둘 다에 들어간다"를 말한다. 'x' 가 더 명시적이다.
ROUTING = 'x'

M = 51.0          # 가운데선
TOP, BOT = M + 18, M - 18
HDR_Y = 87.0      # 위쪽 열 제목의 공통 baseline
IMG_CX = 31.75    # 두께맵 프레임 가로 중심


def _crop_gca_thickness(img):
    # image_preprocessing._crop_gca_thickness와 동일 (torchvision 의존 없이 복제)
    arr = np.array(img.convert('RGB')).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue = (b > r + 40) & (b > g + 40) & (b > 80)
    rows = np.where(blue.sum(axis=1) > 30)[0]
    cols = np.where(blue.sum(axis=0) > 30)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def _crop_rnfl_thickness(img, top_margin=16):
    # image_preprocessing._crop_rnfl_thickness와 동일
    W, H = img.size
    sq = H - top_margin
    arr = np.array(img.convert('RGB')).astype(float)
    max_c = arr.max(axis=2); min_c = arr.min(axis=2)
    sat = (max_c - min_c) / (max_c + 1e-5)
    cols = np.where(((sat > 0.2) & (max_c > 30)).any(axis=0))[0]
    if len(cols) == 0:
        return img
    right = int(cols[-1]) + 1
    return img.crop((max(0, right - sq), top_margin, right, H))


def model_input():
    """대표안의 실제 모델 입력(161×322)을 재현."""
    rp = sorted(glob.glob(str(ROOT / f'cirrus_out/by_region/RNFL_{EYE.lower()}_thickness_map/*{PID}*')))[0]
    gp = sorted(glob.glob(str(ROOT / f'cirrus_out/by_region/GCA_{EYE.lower()}_thickness_map/*{PID}*')))[0]
    rnfl = _crop_rnfl_thickness(Image.open(rp).convert('RGB')).resize((161, 161))
    gca = _crop_gca_thickness(Image.open(gp).convert('RGB')).resize((161, 161))
    cat = np.concatenate([np.asarray(gca), np.asarray(rnfl)], axis=1)
    return cat


def box(ax, x, y, w, h, body, head=None, fc=GRAY, fontsize=8.5,
        linespacing=1.45, head_gap=0.6):
    """머리글(bold) + 본문. head=None 이면 본문만 가운데 놓는다.

    linespacing 기본값은 1.2 였다. 정보가 많은 상자에서 줄이 붙어 보여 1.45 로
    올렸고, 상자 높이도 같이 키웠다(안 그러면 글자가 테두리를 넘는다).
    머리글을 bold 로 두는 이유는 판형 때문이다 — 자연폭 262 mm 를 180 mm 로
    줄여 실으므로 8.5 pt 가 지면에서 약 5.8 pt 로 인쇄된다. 그 크기에서는
    줄간격만으로는 위계가 안 보이고 무게 차이는 보인다.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=EDGE, lw=0.9, **BOX))
    cx, cy = x + w / 2, y + h / 2
    if head is None:
        ax.text(cx, cy, body, ha='center', va='center', fontsize=fontsize,
                linespacing=linespacing)
        return
    # pt → 데이터 y 단위로 어림해서 일단 쌓는다. 어림값은 실제 렌더 높이보다
    # 크므로(줄간격이 마지막 줄 아래에도 붙는 것으로 계산된다) 이대로 두면
    # 글자 덩어리가 상자 위쪽으로 치우친다. _CENTERED 에 등록해 두고
    # recenter_boxes() 에서 실측 높이로 다시 맞춘다.
    hh = fontsize * 1.2 * PT
    bh = (body.count('\n') + 1) * fontsize * linespacing * PT
    gap = head_gap * fontsize * PT
    top = cy + (hh + gap + bh) / 2
    t1 = ax.text(cx, top, head, ha='center', va='top', fontsize=fontsize,
                 weight='bold')
    t2 = ax.text(cx, top - hh - gap, body, ha='center', va='top',
                 fontsize=fontsize, linespacing=linespacing)
    _CENTERED.append((t1, t2, (x, y, w, h)))


def recenter_boxes(ax, pad_px: int = 1) -> None:
    """머리글+본문 덩어리를 상자 세로 중앙에 맞춘다.

    글꼴 metric 으로 계산한 텍스트 bbox 는 마지막 줄 아래에 여유가 더 붙는다.
    그 bbox 를 중앙에 맞추면 잉크가 0.6 데이터단위쯤 위로 뜬다(실측). 그래서
    content_bbox 와 같은 방식으로 한 번 래스터화해서 상자 안 잉크를 직접 재고
    그만큼 내린다. 글자를 고치거나 글꼴이 바뀌어도 따라온다.
    """
    if not _CENTERED:
        return
    fig = ax.figure
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].min(axis=2)
    H = buf.shape[0]
    inv = ax.transData.inverted()
    for t1, t2, (bx, by, bw, bh) in _CENTERED:
        (x0, y0), (x1, y1) = ax.transData.transform([(bx, by), (bx + bw, by + bh)])
        # 글자만 세야 한다. 문턱을 40 으로 두면 검은 글자(0)는 잡히고 테두리
        # #555(85)와 그 안티에일리어싱은 빠진다. 여백을 크게 잡아 잘라내면
        # 글자가 테두리에 가까운 상자(CNN)에서 잉크 위끝이 잘려 오차가 난다.
        sub = buf[int(H - y1) + pad_px:int(H - y0) - pad_px,
                  int(x0) + pad_px:int(x1) - pad_px]
        rows = np.where((sub < 40).any(axis=1))[0]
        if len(rows) == 0:
            continue
        ink_disp_y = H - (int(H - y1) + pad_px + (rows[0] + rows[-1]) / 2)
        dy = (by + bh / 2) - inv.transform((0, ink_disp_y))[1]
        for t in (t1, t2):
            t.set_y(t.get_position()[1] + dy)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=11, lw=1.0, color=EDGE,
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.add_line(Line2D(xs, ys, lw=1.0, color=EDGE, solid_capstyle='round'))


def content_bbox(fig, pad_in=0.05):
    """잉크가 실제로 닿은 사각형을 재서 Bbox(인치)로 돌려준다.

    이 그림의 축은 `add_axes([0,0,1,1])` + `axis('off')` 라 도형과 글자가
    캔버스의 일부만 쓴다. `bbox_inches='tight'` 는 축 프레임(=캔버스 전체)을
    기준으로 잘라서 아래쪽 16%가 빈 채로 저장됐다. \\textwidth 로 넣으면 그
    빈 띠까지 폭에 맞춰 확대돼 그림이 필요 이상으로 커진다.

    좌표를 손으로 박지 않고 한 번 래스터화해서 재는 이유는, 도형을 옮기거나
    글자를 고쳐도 잘림이 따라오게 하기 위해서다.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].min(axis=2)
    ink = buf < 250
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    H, W = ink.shape
    fw, fh = fig.get_size_inches()
    x0 = cols[0] / W * fw - pad_in
    x1 = (cols[-1] + 1) / W * fw + pad_in
    # 버퍼는 위→아래, Bbox 는 아래→위 기준이라 행을 뒤집는다.
    y0 = (H - 1 - rows[-1]) / H * fh - pad_in
    y1 = (H - rows[0]) / H * fh + pad_in
    return Bbox.from_extents(max(0.0, x0), max(0.0, y0), min(fw, x1), min(fh, y1))


def main():
    cat = model_input()
    d = find_eye(PID, EYE)

    # dpi 를 저장 해상도와 맞춰 둔다. recenter_boxes 가 화면 버퍼를 재는데,
    # 기본 100 dpi 에서는 글꼴 힌팅 때문에 글자 덩어리 높이가 300 dpi 저장본보다
    # 8% 크게 잡혀 보정값이 어긋난다(실측).
    fig = plt.figure(figsize=(10.5, 4.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # -- column 1: acquisition ------------------------------------------
    box(ax, 2, TOP - 6, 15, 12, '200×200', head='Optic disc cube')
    box(ax, 2, BOT - 6, 15, 12, '512×128', head='Macular cube')

    # -- column 2: the two representations ------------------------------
    # image representation (top): the actual 161×322 model input.
    # 축 폭 0.19 는 프레임 오른쪽 선에 정확히 닿아 이미지가 오른쪽으로 쏠려
    # 보였다. 프레임 중심 IMG_CX 에 맞춰 0.17 폭으로 다시 앉힌다.
    axm = fig.add_axes([0.2325, (TOP - 11) / 100, 0.17, 0.22])
    axm.imshow(cat); axm.axis('off')
    # 상자 밖이지만 이것도 흐름 위의 한 노드 이름이다. 짝인 '26 summary
    # parameters' 가 bold 라 여기만 regular 로 두면 위계가 어긋난다.
    ax.text(IMG_CX, HDR_Y, 'Two thickness maps (GCA, RNFL)', ha='center',
            va='baseline', fontsize=8.5, weight='bold')
    # 크기 줄. '200×200' / '512×128' / '52 sensitivities (dB)' 와 같은 자리라
    # 같은 8.5 pt regular. 캡션과 겹치던 것은 'cropped, resized' 쪽이었다.
    ax.text(IMG_CX, HDR_Y - 3.2, '161 × 322', ha='center', va='baseline',
            fontsize=8.5)
    ax.add_patch(FancyBboxPatch((21.5, TOP - 13), 20.5, 26, fc='none', ec=EDGE,
                                lw=0.9, boxstyle='round,pad=0.012'))

    # tabular representation (bottom). 높이 20 → 22 (줄간 1.55 를 담기 위해).
    # 'GCL' 은 §3.2/약어표의 GCIPL 과 어긋나 고쳤다.
    box(ax, 21.5, BOT - 11, 20.5, 22,
        'avg/min GCIPL + 6 sectors\navg RNFL + 4 quadrants\n'
        '+ 12 clock hours, vert. C/D', head='26 summary parameters',
        fontsize=7.8, linespacing=1.55)

    # -- cube → representation -------------------------------------------
    if ROUTING == 'x':
        # M 에 대해 정확히 거울대칭. 긴 둘은 M 위에서 교차한다.
        arrow(ax, 17, TOP + 3, 21.5, TOP + 6)      # disc → maps
        arrow(ax, 17, TOP - 3, 21.5, BOT + 6)      # disc → params
        arrow(ax, 17, BOT + 3, 21.5, TOP - 6)      # macula → maps
        arrow(ax, 17, BOT - 3, 21.5, BOT - 6)      # macula → params
    else:
        xb = 19.3                                   # 세로 버스
        line(ax, [17, xb], [TOP, TOP])
        line(ax, [17, xb], [BOT, BOT])
        line(ax, [xb, xb], [BOT, TOP])
        for y in (TOP, BOT):
            ax.plot([xb], [y], 'o', ms=3.0, color=EDGE, zorder=4)
        arrow(ax, xb, TOP, 21.5, TOP)
        arrow(ax, xb, BOT, 21.5, BOT)

    # -- column 3: branches ---------------------------------------------
    # 48-67(폭 19) 에서 46-64(폭 18) 로 당겼다. fusion 상자를 키울 자리.
    box(ax, 46, TOP - 9, 18, 18,
        'Inception-ResNet-v2\nGAP + dense head\n→ 52 outputs', head='CNN',
        fc='#e8eef8')
    box(ax, 46, BOT - 9, 18, 18,
        '52 pointwise regressors\n→ 52 outputs', head='XGBoost', fc='#e8f4e8')
    arrow(ax, 42, TOP, 46, TOP)
    arrow(ax, 42, BOT, 46, BOT)

    # -- column 4: fusion ------------------------------------------------
    # 폭 12 → 16, 높이 16 → 18. 원본은 이 상자만 다른 상자의 2/3 폭이었다.
    # 'one scalar w, fitted on out-of-fold' 주석은 뺐다 — 상자가 이미 w 를
    # 보이고, 적합 방식은 캡션과 §3.7 이 말한다.
    # 상자가 16 으로 넓어졌으니 수식은 한 줄로 둔다. 두 줄로 끊으면 둘째 줄이
    # '+' 로 시작해 별개의 식처럼 읽힌다. 8 pt 한 줄 폭 12.35 < 내부 폭 16.
    # 높이 18 은 CNN(60-78)과 XGBoost(24-42) 사이를 정확히 메운다.
    box(ax, 68.5, M - 9, 16, 18,
        '$\\hat{y}=w\\,\\hat{y}_{\\mathrm{XGB}}+(1{-}w)\\,\\hat{y}_{\\mathrm{CNN}}$',
        head='Late fusion', fc='#f8ecec', fontsize=8.0, linespacing=1.7)
    arrow(ax, 64, TOP - 4, 68.5, M + 6)
    arrow(ax, 64, BOT + 4, 68.5, M - 6)

    # -- output: predicted 24-2 grid -------------------------------------
    axo = fig.add_axes([0.875, (M - 22) / 100, 0.115, 0.44])
    g = to_grid(d['fus'], d['mask'])
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#ffffff')
    axo.imshow(g, cmap=cmap, vmin=0, vmax=34, aspect='equal')
    axo.set_xticks([]); axo.set_yticks([])
    for s in axo.spines.values():
        s.set_color(EDGE)
    # aspect='equal' 이라 축 상자가 실제로는 세로로 줄어드는데, 제목을 데이터
    # 좌표에 박아두면 16 단위(≈17 mm)나 뜬다. set_title 은 줄어든 축을 따라간다.
    # 한 Text 는 무게가 하나뿐이라 이름(bold)과 단위 설명(regular)을 나눈다.
    # transAxes 도 set_title 과 같이 줄어든 축 상자를 따라간다.
    axo.text(0.5, 1.03, '52 sensitivities (dB)', transform=axo.transAxes,
             ha='center', va='bottom', fontsize=8.5)
    axo.set_title('24-2 field', fontsize=8.5, pad=15, weight='bold')
    arrow(ax, 85, M, 87.3, M)

    recenter_boxes(ax)
    bb = content_bbox(fig)
    fig.savefig(OUT, dpi=300, bbox_inches=bb, facecolor='white')
    print(f'saved {OUT}  ({bb.width * 25.4:.0f} x {bb.height * 25.4:.0f} mm 자연폭, '
          f'가로세로비 {bb.width / bb.height:.2f})')

    # 판독 확인용. 원고에 넣지 않는다. 180 mm @ 96 dpi = 680 px.
    prev = ROOT / 'build' / 'figs' / 'preview'
    prev.mkdir(parents=True, exist_ok=True)
    p = prev / 'fig_pipeline__print_size.png'
    fig.savefig(p, dpi=96 * (180 / 25.4) / bb.width, bbox_inches=bb,
                facecolor='white')
    Image.open(p).convert('L').save(prev / 'fig_pipeline__grey.png')
    for f in (p, prev / 'fig_pipeline__grey.png'):
        print(f'  {f.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
