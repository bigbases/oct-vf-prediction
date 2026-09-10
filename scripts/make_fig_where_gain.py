#!/usr/bin/env python3
"""그림 4: 감도 구간별 fusion - 영상 브랜치. 원고 `tab:where` 를 대체한다.

§Where the gain lives 의 주장은 "전체 시야 평균이 침묵하는 것은 이득이 없어서가
아니라 이득과 손실이 상쇄되기 때문"이다. 표는 그 상쇄를 여덟 개의 숫자로
적었을 뿐이라 부호가 바뀌는 지점이 눈에 들어오지 않는다. 그림은 0선을 한 번
가로지르는 두 계열로 그것을 보인다.

영상 브랜치는 **5백본 앙상블**이다 — 전체 시야 평균이 p = 0.46(OOF) /
0.64(held-out)로 읽히는, 가장 불리한 조건. 음수 = fusion 우세.

입력 (읽기 전용):
  runs/fusion_ensemble_structure.json    정본. `trend` 블록
  runs/oof/xgb_90d_fold{0..4}_{val,test}.npz
  runs/phasec_b0_{backbone}_5fold/{val,test}_preds_fold{0..4}.npz

구간 경계·MIN_PTS·가중치는 전부 `fusion_sensitivity_trend` / `fusion_eval_common`
에서 그대로 import 한다. 그린 값(구간별 평균 delta, 구간별 n, 기울기)을 raw npz
에서 다시 계산해 위 JSON과 대조하고, 어긋나면 그림을 그리지 않고 중단한다.
부트스트랩 CI만 JSON에서 그대로 읽는다(10,000회 재표집을 그림마다 돌리지 않는다).

구간별 n이 계열마다 다르다. 한 안이 어떤 구간에 유효 지점 3개(MIN_PTS) 미만이면
그 구간에서 빠지기 때문이다 — 결측이 아니라 정의다. held-out 의 [0,10) 은
12안뿐이라 캡션에 구간별 n을 적는다.

출력:
  build/figs/fig_where_gain.png / .pdf
  build/figs/preview/fig_where_gain__print_size.png   판독 확인용. 원고에 넣지 않는다
실행: python scripts/make_fig_where_gain.py
      python scripts/make_fig_where_gain.py --out docs/journal_manuscript
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

from fig_style import figure, zero_line, save_fig, series  # noqa: E402
from fusion_eval_common import ENSEMBLE, load_all, oof_raw, test_raw  # noqa: E402
from fusion_sensitivity_trend import BIN_LABELS, eye_bin_rmse, slopes  # noqa: E402

REF = ROOT / 'runs' / 'fusion_ensemble_structure.json'
# JSON 의 split 이름 → 원고 표기. 본문은 "out-of-fold" / "held-out" 으로 쓴다.
SPLITS = [('OOF', 'Out-of-fold'), ('TEST', 'Held-out')]
# x 눈금은 구간 지수 1..4 다. 기울기가 이 지수에 대한 회귀이므로 축과 기울기의
# 단위가 같아진다("dB per bin"). 감도 중앙값을 축으로 쓰면 마지막 구간이
# [30,inf) 라 중앙값이 정의되지 않는다.
XS = np.arange(1, 5, dtype=float)
TICKS = [r'$[0,10)$', r'$[10,20)$', r'$[20,30)$', r'$[30,\infty)$']


def recompute() -> dict:
    """raw npz 에서 구간별 delta·n·기울기를 다시 계산한다."""
    oof, tst, test_lab, test_mask = load_all(ensemble=True)
    fo, _, co, lo, mo, _ = oof_raw(oof, ENSEMBLE)
    ft, _, ct, w = test_raw(oof, tst, ENSEMBLE)
    out = {'w': float(w)}
    for split, (fp, cp, lb, mk) in (('OOF', (fo, co, lo, mo)),
                                    ('TEST', (ft, ct, test_lab, test_mask))):
        d = eye_bin_rmse(fp, lb, mk) - eye_bin_rmse(cp, lb, mk)
        sl = slopes(d)
        ok = np.isfinite(sl)
        out[split] = {'delta': np.nanmean(d, axis=0),
                      'n_bin': np.isfinite(d).sum(axis=0),
                      'slope': float(np.mean(sl[ok])),
                      'n_eyes': int(ok.sum())}
    return out


def load() -> dict:
    """정본 JSON 과 대조한 뒤 그림에 쓸 값만 돌려준다."""
    ref = json.loads(REF.read_text(encoding='utf-8'))
    got = recompute()
    bad = []
    if not np.isclose(got['w'], ref['w_test'], rtol=0, atol=1e-12):
        bad.append(f'w {got["w"]} != {ref["w_test"]}')
    if ref['bin_labels'] != BIN_LABELS:
        bad.append(f'구간 라벨 {ref["bin_labels"]} != {BIN_LABELS}')

    data = {}
    for split, label in SPLITS:
        r = next(t for t in ref['trend'] if t['split'] == split)
        g = got[split]
        for b in range(4):
            if int(g['n_bin'][b]) != r['per_bin'][b]['n_eyes']:
                bad.append(f'{split} {BIN_LABELS[b]} n {g["n_bin"][b]} '
                           f'!= {r["per_bin"][b]["n_eyes"]}')
            if not np.isclose(g['delta'][b], r['per_bin'][b]['mean_delta'],
                              rtol=1e-9, atol=0):
                bad.append(f'{split} {BIN_LABELS[b]} delta {g["delta"][b]:.6f} '
                           f'!= {r["per_bin"][b]["mean_delta"]:.6f}')
        if not np.isclose(g['slope'], r['mean_slope'], rtol=1e-9, atol=0):
            bad.append(f'{split} 기울기 {g["slope"]:.6f} != {r["mean_slope"]:.6f}')
        if g['n_eyes'] != r['n_eyes_with_slope']:
            bad.append(f'{split} 기울기 산출 안 {g["n_eyes"]} '
                       f'!= {r["n_eyes_with_slope"]}')
        data[split] = {'label': label, 'y': g['delta'], 'n_bin': g['n_bin'],
                       'n_eyes': g['n_eyes'], 'slope': r['mean_slope'],
                       'ci': (r['slope_ci_lo'], r['slope_ci_hi']),
                       'p': r['p_wilcoxon']}
    if bad:
        raise SystemExit(f'재계산값이 {REF.name} 과 어긋난다 — ' + '; '.join(bad))
    return data


def build(data: dict):
    fig, ax = figure('single', ratio=0.78)
    # 0선이 이 그림의 전부다 — 두 계열이 그것을 한 번 가로지르는 것이 주장이다.
    # 그래서 기본 zero_line 보다 굵게 긋는다.
    zero_line(ax, linewidth=0.8)
    for i, (split, _) in enumerate(SPLITS):
        d = data[split]
        s = series(i)
        ax.plot(XS, d['y'], color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markerfacecolor='white',
                markeredgecolor=s['color'], zorder=3,
                label=f'{d["label"]} ({d["n_eyes"]} eyes)')

    ax.set_xticks(XS)
    ax.set_xticklabels(TICKS)
    ax.set_xlim(0.6, 4.4)
    ax.set_xlabel('Measured sensitivity bin (dB)')
    ax.set_ylabel(r'Fusion $-$ image branch (dB)')

    lo, hi = ax.get_ylim()
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    # 부호 규약을 0선 바로 옆에 적는다. "음수 = fusion 우세"가 캡션에만 있으면
    # 그림만 떼어 본 독자가 방향을 거꾸로 읽는다. 왼쪽 여백(x < 1)은 자료가
    # 없고 0선에 붙어 있어 위/아래가 곧 부호로 읽힌다. 오른쪽 아래는 범례라
    # 쓸 수 없다.
    for dy, va, txt in ((3, 'bottom', 'image branch better'),
                        (-3, 'top', 'fusion better')):
        ax.annotate(txt, xy=(0.66, 0.0), xytext=(0, dy),
                    textcoords='offset points', ha='left', va=va,
                    fontsize=6, color='#7a7a7a', style='italic')

    # 기울기는 좌상단에. 자료가 좌하 → 우상으로 흐르므로 이 모서리가 비어 있다.
    lines = ['Slope (dB per bin, 95% CI):']
    for split, _ in SPLITS:
        d = data[split]
        lines.append(f'  {d["label"]}  {d["slope"]:+.3f} '
                     f'({d["ci"][0]:+.3f}, {d["ci"][1]:+.3f})')
    ax.annotate('\n'.join(lines), xy=(0.015, 0.985), xycoords='axes fraction',
                ha='left', va='top', fontsize=6, color='#3d3d3d', linespacing=1.5)
    ax.legend(loc='lower right', fontsize=6.5, framealpha=0.85)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'figs')
    # 원고 트리(docs/journal_manuscript)에는 다른 네 그림과 맞춰 PNG만 둔다.
    ap.add_argument('--pdf', action='store_true', default=None,
                    help='PDF도 저장 (--out 기본값에서는 자동으로 켜진다)')
    args = ap.parse_args()
    if args.pdf is None:
        args.pdf = Path(args.out).resolve() == (ROOT / 'build' / 'figs')

    data = load()
    print(f'가드 통과. 5백본 앙상블 영상 브랜치, {REF.name} 과 일치.\n')
    hdr = (f'{"split":12s} {"eyes":>5s} ' +
           ' '.join(f'{t:>10s}' for t in BIN_LABELS) +
           f' {"slope":>8s} {"95% CI":>18s} {"p":>9s}')
    print(hdr)
    print('-' * len(hdr))
    for split, _ in SPLITS:
        d = data[split]
        cells = ' '.join(f'{v:>+10.3f}' for v in d['y'])
        print(f'{d["label"]:12s} {d["n_eyes"]:5d} {cells} {d["slope"]:>+8.3f} '
              f'({d["ci"][0]:+.3f}, {d["ci"][1]:+.3f}) {d["p"]:>9.2g}')
        print(f'{"":12s} {"n/bin":>5s} ' +
              ' '.join(f'{int(v):>10d}' for v in d['n_bin']))
    print()

    fig = build(data)
    save_fig(fig, 'fig_where_gain', outdir=Path(args.out).resolve(),
             pdf=args.pdf)
    # 판독 확인용 래스터는 --out 과 무관하게 늘 build/figs/preview/ 로 보낸다.
    # --out docs/journal_manuscript 로 뽑을 때 원고 트리에 preview/ 가 생기면
    # 그대로 Overleaf 로 올라간다.
    prev_dir = ROOT / 'build' / 'figs' / 'preview'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev = prev_dir / 'fig_where_gain__print_size.png'
    fig.savefig(prev, dpi=96)
    print(f'  {prev.relative_to(ROOT)}  (판독 확인용, 원고에 넣지 않는다)')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
