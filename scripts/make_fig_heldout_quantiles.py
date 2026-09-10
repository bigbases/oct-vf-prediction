#!/usr/bin/env python3
"""Figure 3: held-out residual quantiles - where fusion loses and where it wins.

Quantiles are computed over point-level residuals, not eye-level ones. The
script aborts if its recomputed values disagree with the stored canonical
numbers.

그림 3: held-out 잔차 분위수 — fusion이 어디서 지고 어디서 이기는가.

원고 §Discussion "What the held-out set does not support" 의 근거 그림.
tab:ladder 의 held-out MAE 역전(6.033 → 6.293 dB)을 본 리뷰어가 반드시 묻는
자리라, 손실이 중앙에, 이득이 꼬리에 몰려 있다는 것을 분포로 보인다.

입력 (읽기 전용):
  runs/oof/xgb_90d_fold{0..4}_test.npz          XGB held-out 예측
  runs/phasec_b0_{backbone}_5fold/test_preds_fold{0..4}.npz   CNN held-out 예측
  runs/heldout_mae_reversal.json                대조용 정본 수치
분위수는 안 단위가 아니라 **점 단위** 잔차 위에서 계산된다(37안 × 52점 = 1924).
verify_heldout_mae_reversal.py 와 같은 방식이며, 재계산값이 위 JSON의 7분위수·
crossing·MAE·RMSE와 어긋나면 중단한다.

2026-08-27 확정안: 차이 곡선(fusion - image), 백본 5개 중첩, p50-p100.
교차가 p84-p86이고 p50 아래는 차이가 ±0.12 dB라 0선과 구분되지 않는다.
탈락안 셋은 `--variants` 로만 나온다(기본은 안 나온다).

출력:
  build/figs/fig_heldout_quantiles.png       원고에 넣을 것 (600 dpi, 88 mm)
  build/figs/fig_heldout_quantiles.pdf       원고에 넣을 것 (벡터)
  build/figs/preview/...__print_size.png      판독 확인용. 원고에 넣지 않는다
  build/figs/variants/...                     --variants 일 때만
실행: python scripts/make_fig_heldout_quantiles.py
      python scripts/make_fig_heldout_quantiles.py --variants
      python scripts/make_fig_heldout_quantiles.py --out docs/journal_manuscript
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
from fig_style import figure, series, faint, band, zero_line, save_fig, WIDTH, ANNOT  # noqa

# verify_heldout_mae_reversal.py 와 동일. w는 각 백본의 OOF 선택값이고
# inception_v3 의 예측은 clip_mse 디렉터리에 있다(디렉터리명이 백본명이 아님).
BACKBONES = {
    'inception_resnet_v2': ('runs/phasec_b0_inception_resnet_v2_5fold', 0.47),
    'inception_v3':        ('runs/phasec_b0_clip_mse_5fold',            0.60),
    'vgg16':               ('runs/phasec_b0_vgg16_5fold',               0.63),
    'xception':            ('runs/phasec_b0_xception_5fold',            0.43),
    'densenet121':         ('runs/phasec_b0_densenet121_5fold',         0.50),
}
PRIMARY = 'inception_resnet_v2'
LABEL = {'inception_resnet_v2': 'Inception-ResNet-v2', 'inception_v3': 'Inception-v3',
         'vgg16': 'VGG-16', 'xception': 'Xception', 'densenet121': 'DenseNet-121'}
REF = ROOT / 'runs' / 'heldout_mae_reversal.json'


def _key(d):
    return [f'{p}|{e}|{v}' for p, e, v in zip(d['patient_id'], d['eye'], d['vf_date'])]


def load_residuals() -> dict:
    """백본별 점 단위 절대잔차 (image, fusion). 정본 JSON과 대조 후 반환한다."""
    ref = json.loads(REF.read_text())
    xf = [np.load(ROOT / f'runs/oof/xgb_90d_fold{i}_test.npz', allow_pickle=True)
          for i in range(5)]
    kx = _key(xf[0])
    out = {}
    for name, (cnn_dir, w) in BACKBONES.items():
        cf = [np.load(ROOT / cnn_dir / f'test_preds_fold{i}.npz', allow_pickle=True)
              for i in range(5)]
        kc = _key(cf[0])
        # 두 branch의 held-out 안 집합이 1안 다르므로 교집합에서만 평가한다.
        common = [k for k in kc if k in set(kx)]
        ic = [kc.index(k) for k in common]
        ix = [kx.index(k) for k in common]
        cnn = np.mean([c['pred'] for c in cf], 0)[ic]
        xgb = np.mean([x['pred'] for x in xf], 0)[ix]
        y = cf[0]['labels'][ic]
        m = cf[0]['mask'][ic] & xf[0]['mask'][ix]
        fus = w * xgb + (1 - w) * cnn
        ec, ef = np.abs(cnn - y)[m], np.abs(fus - y)[m]
        _guard(name, ec, ef, cnn, fus, y, m, ref[name])
        out[name] = {'image': ec, 'fusion': ef, 'n_eyes': len(common),
                     'crossing': ref[name]['crossing_percentile']}
    return out


def _guard(name, ec, ef, cnn, fus, y, m, r) -> None:
    """재계산값이 정본 JSON과 어긋나면 그림을 그리지 않고 중단한다."""
    q = lambda a, p: float(np.percentile(a, p))
    bad = []
    for p in (10, 25, 50, 75, 90, 95, 99):
        got, exp = round(q(ef, p) - q(ec, p), 3), r['quantile_delta'][f'p{p}']
        if abs(got - exp) > 1e-9:
            bad.append(f'p{p} delta {got} != {exp}')
    cross = next((p for p in range(50, 100) if q(ef, p) < q(ec, p)), None)
    if cross != r['crossing_percentile']:
        bad.append(f'crossing {cross} != {r["crossing_percentile"]}')
    for lab, pred, key in (('cnn', cnn, 'cnn'), ('fusion', fus, 'fusion')):
        e = (pred - y)[m]
        for stat, val in (('mae', np.abs(e).mean()), ('rmse', np.sqrt((e ** 2).mean()))):
            got, exp = round(float(val), 3), r[key][stat]
            if got != exp:
                bad.append(f'{lab} {stat} {got} != {exp}')
    if bad:
        raise SystemExit(f'{name}: 재계산값이 {REF.name} 과 어긋난다 — ' + '; '.join(bad))


def panel_a(data, grid):
    """(a) 두 branch의 절대오차 곡선. 보조 백본은 옅게 깔고 primary만 진하게."""
    fig, ax = figure('single', ratio=0.74)
    for name, d in data.items():
        if name == PRIMARY:
            continue
        for br, ls in (('image', '-'), ('fusion', (0, (4, 1.5)))):
            ax.plot(grid, np.percentile(d[br], grid), **{**faint(), 'linestyle': ls})
    lo, hi = min(d['crossing'] for d in data.values()), max(d['crossing'] for d in data.values())
    band(ax, lo, hi)
    d = data[PRIMARY]
    for i, br in enumerate(('image', 'fusion')):
        st = series(i)
        ax.plot(grid, np.percentile(d[br], grid), color=st['color'],
                linestyle=st['linestyle'], zorder=3,
                label='Image branch' if br == 'image' else 'Fusion')
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Percentile of pointwise absolute error')
    ax.set_ylabel('Absolute error (dB)')
    ax.annotate(f'crossing\np{lo}-p{hi}', xy=(hi, 0.5), xycoords=('data', 'axes fraction'),
                xytext=(-2, 0), textcoords='offset points', ha='right', va='center',
                fontsize=6.5, color=ANNOT)
    ax.legend(loc='upper left')
    return fig


def panel_b(data, grid, *, xlim, grey_others: bool = False, legend_loc='lower left'):
    """(b)(c) 차이 곡선 fusion - image. 0선 교차가 그대로 crossing이다.

    grey_others=True 면 보조 백본을 무채색으로 깔아 primary와 교차 구간만 남긴다.
    """
    fig, ax = figure('single', ratio=0.74)
    lo, hi = min(d['crossing'] for d in data.values()), max(d['crossing'] for d in data.values())
    band(ax, lo, hi)
    zero_line(ax)
    delta = lambda d: np.percentile(d['fusion'], grid) - np.percentile(d['image'], grid)
    if grey_others:
        for i, (name, d) in enumerate(n for n in data.items() if n[0] != PRIMARY):
            ax.plot(grid, delta(d), **{**faint(), 'alpha': 0.45, 'linewidth': 0.6},
                    label='Other backbones (n = 4)' if i == 0 else None)
        st = series(0)
        ax.plot(grid, delta(data[PRIMARY]), color=st['color'], linestyle=st['linestyle'],
                linewidth=1.3, zorder=3, label=LABEL[PRIMARY] + ' (primary)')
    else:
        for i, (name, d) in enumerate(data.items()):
            st = series(i)
            prim = name == PRIMARY
            # 보조 계열 0.9 — 0.8에서는 96 dpi 화면 래스터에서 대시 패턴이
            # 뭉개져 회색조 판독이 깨진다(실측). 1.0은 primary 위계를 흐린다.
            ax.plot(grid, delta(d), color=st['color'], linestyle=st['linestyle'],
                    linewidth=1.3 if prim else 0.9, alpha=1.0 if prim else 0.85,
                    zorder=3 if prim else 2,
                    label=LABEL[name] + (' (primary)' if prim else ''))
    ax.set_xlim(*xlim)
    ax.set_xlabel('Percentile of pointwise absolute error')
    # 긴 라벨은 단폭에서 축 높이의 80%를 먹는다. 세로 라벨은 짧게 두고
    # "absolute error"는 x축 라벨과 캡션이 이미 말하고 있다.
    ax.set_ylabel(r'Fusion $-$ image (dB)')
    # 부호 설명은 0선 바로 위 왼쪽 끝에 붙인다. 오른쪽 끝이 비어 있다는 기존
    # 전제는 틀렸다 — VGG-16 과 primary 가 p99 위에서 0을 넘어 올라와 'worse'
    # 꼬리와 겹친다(확정안 실측). p50-p70 은 모든 계열이 +0.54 dB 위라 비어 있다.
    # 흰 bbox 는 전체 구간 탈락안(variant_b)에서 곡선이 0선에 붙는 경우의 보험.
    ax.annotate('above 0: fusion worse', xy=(xlim[0], 0), xycoords='data',
                xytext=(3, 3), textcoords='offset points',
                ha='left', va='bottom', fontsize=6.5, color=ANNOT,
                bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none',
                          alpha=0.75))
    ax.legend(loc=legend_loc, fontsize=6)
    return fig


def _emit(fig, name: str, outdir: Path) -> None:
    """최종본은 outdir에, 판독 확인용 실물 크기 래스터는 preview/ 에 둔다."""
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
    ap.add_argument('--variants', action='store_true',
                    help='탈락안 셋도 variants/ 에 뽑는다')
    args = ap.parse_args()

    data = load_residuals()
    n_pts = len(data[PRIMARY]['image'])
    print(f'가드 통과. {len(data)}개 백본, {data[PRIMARY]["n_eyes"]}안 / {n_pts}점.')

    full = np.linspace(0, 100, 501)
    upper = np.linspace(50, 100, 501)

    print('확정안:')
    _emit(panel_b(data, upper, xlim=(50, 100), legend_loc='lower left'),
          'fig_heldout_quantiles', args.out)

    if args.variants:
        vdir = Path(args.out) / 'variants'
        print('탈락안:')
        _emit(panel_a(data, full), 'variant_a_two_abs_error_curves', vdir)
        _emit(panel_b(data, full, xlim=(0, 100), legend_loc='lower left'),
              'variant_b_full_percentile_range', vdir)
        _emit(panel_b(data, upper, xlim=(50, 100), grey_others=True,
                      legend_loc='lower left'),
              'variant_c_grey_other_backbones', vdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
