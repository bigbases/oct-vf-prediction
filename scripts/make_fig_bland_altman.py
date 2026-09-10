#!/usr/bin/env python3
"""그림 4: Bland-Altman — 안 단위 평균 감도의 예측-실측 일치도, 중증도별.

원고 §Results "Where the model helps" 의 근거 그림. tab:strata 는 층별 RMSE만
주는데, 리뷰어가 묻는 것은 "층이 깊어지면 오차가 커진다"가 아니라 **어느 방향으로
치우치는가**다. RMSE는 부호를 지운다. 이 그림이 그 부호를 보인다.

입력 (읽기 전용):
  runs/oof/xgb_90d_fold{0..4}_val.npz                  XGB OOF 예측
  runs/phasec_b0_inception_resnet_v2_5fold/val_preds_fold{0..4}.npz   CNN OOF 예측
  cohort_md.csv                                        중증도 층 배정용 MD
  runs/severity_region.json                            대조용 정본 수치
층 배정·키·마스크는 전부 `build_severity_region` 에서 그대로 import 한다.
같은 함수를 쓰므로 층 집합이 tab:strata 와 **구성상** 같고, 추가로 층별 n과
fusion pooled RMSE를 위 JSON과 대조해 어긋나면 그림을 그리지 않고 중단한다.

안 단위 집계다. 지점 단위(235안 × 52점 = 12,220점)로 그리면 점이 뭉쳐
일치한계가 안 단위 해석과 달라진다. x축은 MD가 아니라 두 감도의 평균이다
(MD를 쓰면 층 배정 축과 표시 축이 같아져 순환이 된다).

OOF 240안 중 5안은 cohort_md.csv 조인 실패로 층이 없다(결측이 아니라 키
불일치 — CANONICAL_SPEC.md §2.1). 그래서 235안이고 tab:strata 와 같다.

2패널/4패널 둘 다 나온다. 4패널의 moderate 는 29안뿐이라 ±1.96 SD 가
불안정하다(SD 자체의 상대오차가 약 13%). 어느 쪽을 원고에 넣을지는 미정.

출력:
  build/figs/fig_bland_altman_2panel.png / .pdf     normal+early 147 / moderate+advanced 88
  build/figs/fig_bland_altman_4panel.png / .pdf     93 / 54 / 29 / 59
  build/figs/preview/...__print_size.png            판독 확인용. 원고에 넣지 않는다
실행: python scripts/make_fig_bland_altman.py
      python scripts/make_fig_bland_altman.py --out docs/journal_manuscript
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
from matplotlib.lines import Line2D  # noqa: E402

from fig_style import figure, zero_line, panel_label, save_fig, GREYS  # noqa: E402
from build_severity_region import MD, STRATA, nkey, stack_oof, opt_w, pooled  # noqa: E402

REF = ROOT / 'runs' / 'severity_region.json'
TITLE = {'normal': 'Normal', 'early': 'Early',
         'moderate': 'Moderate', 'advanced': 'Advanced'}
# 2패널 묶음. 경계는 MD -6 dB — tab:strata 의 네 층을 쪼개지 않고 반으로 가른다.
PAIRS = [('Normal and early', ['normal', 'early']),
         ('Moderate and advanced', ['moderate', 'advanced'])]


def load() -> tuple[dict, float]:
    """층별 (x, y). x = 두 평균의 평균, y = 예측 - 실측. 안 단위."""
    ref = json.loads(REF.read_text(encoding='utf-8'))
    X, C, L, M, keys = stack_oof()
    w = opt_w(X, C, L, M)
    if w != ref['w_xgb']:
        raise SystemExit(f'w_xgb {w} != {REF.name} 의 {ref["w_xgb"]}')
    F = w * X + (1 - w) * C
    m = M.astype(bool)

    md_vals = np.array([MD.get(nkey(*k), np.nan) for k in keys])
    out, bad = {}, []
    for name, fn in STRATA:
        rows = np.array([i for i in range(len(keys))
                         if np.isfinite(md_vals[i]) and fn(md_vals[i])])
        r = ref['severity'][name]
        if len(rows) != r['n_eyes']:
            bad.append(f'{name} n {len(rows)} != {r["n_eyes"]}')
        sel = np.zeros(M.shape, bool)
        sel[rows] = True
        rmse, _ = pooled(F, L, m & sel)
        if rmse != r['fusion']['rmse']:
            bad.append(f'{name} fusion RMSE {rmse} != {r["fusion"]["rmse"]}')
        # 마스크 밖 지점은 평균에서 뺀다. 현재 코호트는 전 안이 52점 전부
        # 유효해서 결과가 같지만, 마스크를 무시하면 조용히 틀리는 날이 온다.
        pm = np.array([F[i][m[i]].mean() for i in rows])
        lm = np.array([L[i][m[i]].mean() for i in rows])
        out[name] = {'x': (pm + lm) / 2, 'meas': lm, 'y': pm - lm, 'n': len(rows)}
    if bad:
        raise SystemExit(f'재계산값이 {REF.name} 과 어긋난다 — ' + '; '.join(bad))
    if sum(d['n'] for d in out.values()) != 235:
        raise SystemExit(f'층 합계 {sum(d["n"] for d in out.values())} != 235')
    return out, w


def merge(data: dict, names: list[str]) -> dict:
    return {'x': np.concatenate([data[n]['x'] for n in names]),
            'meas': np.concatenate([data[n]['meas'] for n in names]),
            'y': np.concatenate([data[n]['y'] for n in names]),
            'n': sum(data[n]['n'] for n in names)}


def stats(d: dict) -> dict:
    """평균 편향과 일치한계. SD는 표본 SD(ddof=1)."""
    bias, sd = float(d['y'].mean()), float(d['y'].std(ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    return {'bias': bias, 'sd': sd, 'lo': lo, 'hi': hi,
            'out': int(((d['y'] < lo) | (d['y'] > hi)).sum()),
            'r': float(np.corrcoef(d['x'], d['y'])[0, 1])}


def draw(ax, d: dict, title: str, *, legend: bool) -> dict:
    s = stats(d)
    zero_line(ax)
    ax.plot(d['x'], d['y'], linestyle='none', marker='o', markersize=2.4,
            markerfacecolor='none', markeredgecolor='#000000',
            markeredgewidth=0.5, alpha=0.75, zorder=3)
    ax.axhline(s['bias'], color='#000000', linewidth=1.0, zorder=4)
    for y in (s['lo'], s['hi']):
        ax.axhline(y, color=GREYS[3], linewidth=0.9, linestyle=(0, (6, 1.8)),
                   zorder=4)
    ax.set_title(f'{title} (n = {d["n"]})', fontsize=8)
    # 세 선의 값은 왼쪽 끝에, 이름표는 오른쪽 끝에. 산점은 x 중앙에 몰려 있어
    # 양 끝이 비고, 값과 이름표를 갈라 두면 어느 패널에서도 안 겹친다.
    for y, txt in ((s['bias'], f'{s["bias"]:+.2f}'),
                   (s['hi'], f'{s["hi"]:+.2f}'), (s['lo'], f'{s["lo"]:+.2f}')):
        ax.annotate(txt, xy=(0, y), xycoords=('axes fraction', 'data'),
                    xytext=(2, 2), textcoords='offset points',
                    ha='left', va='bottom', fontsize=6.5, color='#404040')
    if legend:
        # 선 이름을 선 옆에 붙이면 산점과 겹친다(실측). 첫 패널에 범례로 한 번만
        # 둔다. 나머지 패널은 왼쪽 끝 숫자만으로 읽힌다.
        ax.legend(handles=[
            Line2D([], [], color='#000000', linewidth=1.0, label='Mean bias'),
            Line2D([], [], color=GREYS[3], linewidth=0.9,
                   linestyle=(0, (6, 1.8)), label=r'$\pm$1.96 SD'),
            Line2D([], [], color='#000000', linewidth=0.5,
                   linestyle=(0, (2, 2)), label='Zero'),
        ], loc='upper left', fontsize=6, framealpha=0.85, handlelength=2.4)
    return s


def build_single(data: dict):
    """단일 패널. x = 실측, 일치한계는 회귀 기반(Bland & Altman 1999).

    층 분할을 안 쓴다. 층은 MD로 나뉘고 MD는 실측 평균감도와 강하게 붙어 있어
    x축이 이미 같은 일을 한다(2패널에서 (a) x=20-31, (b) x=3-25로 거의 안 겹쳤다).

    상수 일치한계도 안 쓴다. |잔차|가 x에 따라 준다(기울기 -0.082/dB, fusion).
    상수 한계는 저감도에서 너무 좁고 고감도에서 너무 넓다.
    """
    x = np.concatenate([d['meas'] for d in data.values()])
    y = np.concatenate([d['y'] for d in data.values()])
    r = st.linregress(x, y)
    q = st.linregress(x, np.abs(y - (r.intercept + r.slope * x)))

    fig, ax = figure('single', ratio=0.80)
    zero_line(ax)
    ax.plot(x, y, linestyle='none', marker='o', markersize=2.4,
            markerfacecolor='none', markeredgecolor='#000000',
            markeredgewidth=0.5, alpha=0.75, zorder=3)
    gx = np.linspace(x.min(), x.max(), 200)
    fit, half = r.intercept + r.slope * gx, 2.46 * (q.intercept + q.slope * gx)
    ax.plot(gx, fit, color='#000000', linewidth=1.0, zorder=4, label='Mean bias')
    for sgn in (+1, -1):
        ax.plot(gx, fit + sgn * half, color=GREYS[3], linewidth=0.9,
                linestyle=(0, (6, 1.8)), zorder=4,
                label=r'$\pm$1.96 SD' if sgn > 0 else None)
    ax.plot([], [], color='#000000', linewidth=0.5, linestyle=(0, (2, 2)),
            label='Zero')
    ax.set_xlabel('Measured mean sensitivity (dB)')
    ax.set_ylabel(r'Predicted $-$ measured (dB)')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.85, handlelength=2.4)
    return fig, {'single': {'bias': float(r.intercept), 'sd': float(r.slope),
                            'lo': float(r.intercept + r.slope * 5),
                            'hi': float(r.intercept + r.slope * 30),
                            'out': int((np.abs(y - (r.intercept + r.slope * x))
                                        > 2.46 * (q.intercept + q.slope * x)).sum()),
                            'r': float(r.rvalue)}}


def build(data: dict, groups: list, *, ncols: int, ratio: float, name: str):
    nrows = -(-len(groups) // ncols)
    fig, axes = figure('double', ratio=ratio, nrows=nrows, ncols=ncols,
                       sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    out = {}
    for i, (title, d) in enumerate(groups):
        ax = axes[i]
        out[title] = draw(ax, d, title, legend=(i == 0))
        panel_label(ax, f'({chr(97 + i)})', x=-0.10, y=1.06)
    # 축 라벨은 패널마다 반복하지 않는다. 양단폭을 2열로 쪼개면 패널 하나가
    # 88 mm 아래라 "Mean of predicted and measured sensitivity (dB)" 가 축 폭을
    # 넘어 옆 패널을 침범한다(실측).
    fig.supxlabel('Mean of predicted and measured sensitivity (dB)', fontsize=8)
    fig.supylabel(r'Predicted $-$ measured (dB)', fontsize=8)
    return fig, out


def emit(fig, name: str, outdir: Path) -> None:
    save_fig(fig, name, outdir=outdir)
    prev_dir = Path(outdir) / 'preview'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev = prev_dir / f'{name}__print_size.png'
    fig.savefig(prev, dpi=96)
    print(f'  {prev.relative_to(ROOT)}  (판독 확인용, 원고에 넣지 않는다)')
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'figs')
    args = ap.parse_args()

    data, w = load()
    print(f'가드 통과. late fusion IR-v2 w_xgb={w}, '
          f'{sum(d["n"] for d in data.values())}안.\n')

    four = [(TITLE[n], data[n]) for n, _ in STRATA]
    two = [(t, merge(data, ns)) for t, ns in PAIRS]

    hdr = f'{"group":22s} {"n":>4s} {"bias":>7s} {"SD":>6s} ' \
          f'{"LoA":>16s} {"밖":>4s} {"r(x,y)":>7s}'
    print(hdr)
    figs = [('fig_bland_altman_4panel', four, 2, 0.78),
            ('fig_bland_altman_2panel', two, 2, 0.44)]
    for name, groups, ncols, ratio in figs:
        fig, st = build(data, groups, ncols=ncols, ratio=ratio, name=name)
        print(f'-- {name}')
        for title, s in st.items():
            n = dict(groups)[title]['n']
            print(f'{title:22s} {n:4d} {s["bias"]:+7.2f} {s["sd"]:6.2f} '
                  f'{s["lo"]:+7.2f}~{s["hi"]:+7.2f} {s["out"]:4d} {s["r"]:+7.2f}')
        emit(fig, name, args.out)

    fig, sg = build_single(data)
    r = sg['single']
    print('-- fig_bland_altman_single  (x = 실측, 회귀 기반 일치한계)')
    print(f'  bias(x) = {r["bias"]:+.2f} {r["sd"]:+.3f}*x  '
          f'→ x=5 에서 {r["lo"]:+.2f}, x=30 에서 {r["hi"]:+.2f} dB. '
          f'한계 밖 {r["out"]}안 / 235')
    emit(fig, 'fig_bland_altman_single', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
