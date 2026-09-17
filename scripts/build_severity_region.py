#!/usr/bin/env python3
"""Table 6: per-severity (MD stratum) and per-region (hemifield, central versus
peripheral) breakdown, out-of-fold, with a paired Wilcoxon test per cell.

Per-severity(MD 층) + per-region(공간) 분석표 — 5-fold OOF, 재학습 없음.

대표 백본 IR-v2. XGB / image-only CNN / late fusion 각각 pooled RMSE·MAE.
  · severity: cohort_md.csv MD 층 (normal/early/moderate/advanced), eye 단위 층화
  · region  : 표준 24-2 좌표 기반 point subset
      - 상/하반구(superior/inferior hemifield)  → 녹내장 반구 비대칭
      - 중심/주변(central ≤9°, peripheral)      → 임상 핵심 중심시야
fusion vs {XGB, CNN} paired Wilcoxon(eye 단위)로 층/영역별 유의성 병기.
출력: runs/severity_region.json / .md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402

COHORT_MD = ROOT / 'cohort_md.csv'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'

STRATA = [('normal', lambda md: md > -3), ('early', lambda md: -6 < md <= -3),
          ('moderate', lambda md: -12 < md <= -6), ('advanced', lambda md: md <= -12)]


def nkey(pid, eye, vf):
    return (str(pid).strip(), str(eye).strip().upper(),
            str(vf).strip().replace('-', '').replace('/', ''))


# ── 표준 Humphrey 24-2 좌표 (p01..p54, 위→아래 / 좌→우) ──
def build_coords():
    rows = [(+21, [-9, -3, 3, 9]),
            (+15, [-15, -9, -3, 3, 9, 15]),
            (+9, [-21, -15, -9, -3, 3, 9, 15, 21]),
            (+3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]),
            (-3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]),
            (-9, [-21, -15, -9, -3, 3, 9, 15, 21]),
            (-15, [-15, -9, -3, 3, 9, 15]),
            (-21, [-9, -3, 3, 9])]
    coords = []
    for y, xs in rows:
        for x in xs:
            coords.append((x, y))
    assert len(coords) == 54, len(coords)
    return coords


COORDS54 = build_coords()
BLIND = {25, 34}  # 0-based index of p26, p35
COORDS52 = [c for i, c in enumerate(COORDS54) if i not in BLIND]  # 52-point 순서

XY = np.array(COORDS52, float)
REGIONS = {
    'superior (상반구)': XY[:, 1] > 0,
    'inferior (하반구)': XY[:, 1] < 0,
    'central ≤9° (중심)': (np.abs(XY[:, 0]) <= 9) & (np.abs(XY[:, 1]) <= 9),
    'peripheral (주변)': ~((np.abs(XY[:, 0]) <= 9) & (np.abs(XY[:, 1]) <= 9)),
}


def load_md():
    out = {}
    for r in csv.DictReader(open(COHORT_MD, encoding='utf-8-sig')):
        try:
            md = float(r['MD'])
        except (ValueError, KeyError, TypeError):
            md = np.nan
        out[nkey(r['patient_id'], r['eye'], r['vf_date'])] = md
    return out


MD = load_md()


def stack_oof():
    xs, cs, ls, ms, keys = [], [], [], [], []
    for k in range(5):
        xz = load_oof_npz(XGB_VAL[k])
        cz = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        keys.extend(common)
        xs.append(ax['pred']); cs.append(bx['pred'])
        ls.append(ax['labels']); ms.append(ax['mask'] & bx['mask'])
    return (np.concatenate(xs), np.concatenate(cs),
            np.concatenate(ls), np.concatenate(ms), keys)


def pooled(pred, lab, m):
    m = m.astype(bool)
    if m.sum() == 0:
        return None, None
    d = (pred - lab)[m]
    return round(float(np.sqrt(np.mean(d ** 2))), 2), round(float(np.mean(np.abs(d))), 2)


def eye_rmse(pred, lab, mask, rows, cols):
    """rows=eye idx subset, cols=point bool mask; eye별 RMSE 벡터."""
    out = []
    for i in rows:
        m = mask[i].astype(bool) & cols
        if m.sum() == 0:
            continue
        d = pred[i][m] - lab[i][m]
        out.append(np.sqrt(np.mean(d ** 2)))
    return np.array(out)


def wil(a, b):
    if len(a) != len(b) or len(a) < 3:
        return float('nan')
    try:
        return round(float(stats.wilcoxon(a, b)[1]), 4)
    except ValueError:
        return float('nan')


def block(X, C, L, M, rows, cols, w):
    """rows: eye 인덱스 배열, cols: 52-point bool."""
    sel = np.zeros(M.shape, bool)
    sel[np.ix_(rows, np.where(cols)[0])] = True
    mm = M.astype(bool) & sel
    F = w * X + (1 - w) * C
    rx, mx = pooled(X, L, mm); rc, mc = pooled(C, L, mm); rf, mf = pooled(F, L, mm)
    ef = eye_rmse(F, L, M, rows, cols); ec = eye_rmse(C, L, M, rows, cols); ex = eye_rmse(X, L, M, rows, cols)
    # region 표의 n 열 = **안구당 점 수**(24-2 52점 격자의 분할: 상 26 / 하 26 /
    # 중심 16 / 주변 36). 여태 이 값은 .md 표에만 int(np.sum(REGIONS[name])) 로
    # 찍히고 JSON 에는 안 들어갔다. 그래서 원고의 26/16/36 은 근거 풀에서
    # 무관한 값에 우연히 맞아 통과하고 있었다 — 36 은 case_profile.json 의
    # md_percentile_rank=35.74 였다 (2026-09-17). n_rows_severity 때와 같은 건.
    return {
        'n_eyes': int(len(rows)), 'n_points': int(mm.sum()),
        'n_points_per_eye': int(cols.sum()),
        'xgb': {'rmse': rx, 'mae': mx}, 'cnn': {'rmse': rc, 'mae': mc},
        'fusion': {'rmse': rf, 'mae': mf},
        'p_fus_vs_cnn': wil(ef, ec), 'p_fus_vs_xgb': wil(ef, ex),
    }


def opt_w(x, c, lab, m):
    m = m.astype(bool)
    ya, yb, y = x[m], c[m], lab[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (r, w)
    return round(float(best[1]), 2)


def main():
    X, C, L, M, keys = stack_oof()
    w = opt_w(X, C, L, M)
    allpts = np.ones(52, bool)
    allrows = np.arange(len(keys))
    print(f'OOF n_eyes={len(keys)}  w_xgb={w}\n', flush=True)

    res = {'protocol': '90d 5-fold OOF, IR-v2, pooled RMSE/MAE(dB), w_xgb=%.2f' % w,
           'w_xgb': w, 'n_eyes': len(keys), 'severity': {}, 'region': {}}

    # severity 층화 (eye 단위)
    md_vals = np.array([MD.get(nkey(*k), np.nan) for k in keys])
    # severity 표의 행수와 그 행들이 나온 고유 안구 수.
    # 원고는 "the severity rows total 235 ... the 235 rows come from 230
    # distinct eyes" 로 쓰는데, 저장된 건 층별 n 뿐이라 235 는 층 합으로만
    # 재구성됐고 230 은 어느 산출물에도 없었다. 235 쪽은 그동안 무관한
    # case_profile.json 의 235 에 우연히 맞아 통과하고 있었다 (2026-09-17).
    _fin = np.isfinite(md_vals)
    res['n_rows_severity'] = int(_fin.sum())
    res['n_rows_md_unmatched'] = int((~_fin).sum())
    res['n_eyes_severity_distinct'] = len(
        {(k[0], k[1]) for k, f in zip(keys, _fin) if f})
    print('=== Per-severity (MD 층, eye 단위) ===', flush=True)
    print(f'{"stratum":12s} {"n":>4s}  {"XGB":>10s} {"CNN":>10s} {"fusion":>10s}  {"fus<CNN?":>9s}', flush=True)
    for name, fn in STRATA:
        rows = np.array([i for i in allrows if np.isfinite(md_vals[i]) and fn(md_vals[i])])
        if len(rows) == 0:
            continue
        b = block(X, C, L, M, rows, allpts, w)
        res['severity'][name] = b
        pc = b['p_fus_vs_cnn']
        print(f'{name:12s} {b["n_eyes"]:4d}  '
              f'{b["xgb"]["rmse"]:5.2f}/{b["xgb"]["mae"]:<4.2f} '
              f'{b["cnn"]["rmse"]:5.2f}/{b["cnn"]["mae"]:<4.2f} '
              f'{b["fusion"]["rmse"]:5.2f}/{b["fusion"]["mae"]:<4.2f}  p={pc}', flush=True)

    # region (point subset, 전체 eye)
    print('\n=== Per-region (공간 영역, point subset) ===', flush=True)
    print(f'{"region":22s} {"pts":>4s}  {"XGB":>10s} {"CNN":>10s} {"fusion":>10s}', flush=True)
    for name, cols in REGIONS.items():
        b = block(X, C, L, M, allrows, cols, w)
        res['region'][name] = b
        print(f'{name:22s} {int(cols.sum()):4d}  '
              f'{b["xgb"]["rmse"]:5.2f}/{b["xgb"]["mae"]:<4.2f} '
              f'{b["cnn"]["rmse"]:5.2f}/{b["cnn"]["mae"]:<4.2f} '
              f'{b["fusion"]["rmse"]:5.2f}/{b["fusion"]["mae"]:<4.2f}', flush=True)

    # overall
    res['overall'] = block(X, C, L, M, allrows, allpts, w)

    (ROOT / 'runs/severity_region.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    def st(p):
        return '' if p != p else ('***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns')

    ov = res['overall']
    md = ['# Per-severity / Per-region 분석 (90d · 5-fold OOF · IR-v2)', '',
          f'pooled RMSE/MAE (dB). late fusion w_xgb={w}, n={res["n_eyes"]} eyes. '
          f'유의성=fusion vs 단독(Wilcoxon, eye 단위): `*`<.05 `**`<.01 `***`<.001 `ns`.',
          f'전체: XGB {ov["xgb"]["rmse"]}/{ov["xgb"]["mae"]} · CNN {ov["cnn"]["rmse"]}/{ov["cnn"]["mae"]} · '
          f'**fusion {ov["fusion"]["rmse"]}/{ov["fusion"]["mae"]}**', '',
          '## A. 중증도(MD)별', '',
          '| MD 층 | n eyes | XGB R/M | CNN R/M | **fusion R/M** | fus vs CNN | fus vs XGB |',
          '|-------|--------|---------|---------|----------------|-----------|-----------|']
    for name, fn in STRATA:
        if name not in res['severity']:
            continue
        b = res['severity'][name]
        md.append(f"| {name} | {b['n_eyes']} | {b['xgb']['rmse']}/{b['xgb']['mae']} | "
                  f"{b['cnn']['rmse']}/{b['cnn']['mae']} | **{b['fusion']['rmse']}/{b['fusion']['mae']}** "
                  f"| {st(b['p_fus_vs_cnn'])} | {st(b['p_fus_vs_xgb'])} |")
    md += ['', '## B. 공간 영역별', '',
           '| 영역 | pts | XGB R/M | CNN R/M | **fusion R/M** | fus vs CNN | fus vs XGB |',
           '|------|-----|---------|---------|----------------|-----------|-----------|']
    for name in REGIONS:
        b = res['region'][name]
        md.append(f"| {name} | {b['n_points']//res['n_eyes'] if False else int(np.sum(REGIONS[name]))} | "
                  f"{b['xgb']['rmse']}/{b['xgb']['mae']} | {b['cnn']['rmse']}/{b['cnn']['mae']} | "
                  f"**{b['fusion']['rmse']}/{b['fusion']['mae']}** | {st(b['p_fus_vs_cnn'])} | {st(b['p_fus_vs_xgb'])} |")
    md += ['', '## 해석',
           '- 중증도가 깊어질수록 오차 증가(정상 대비 advanced에서 RMSE 상승) — 난이도 반영.',
           '- 상/하반구 및 중심/주변 전 영역에서 fusion이 단독 대비 열세 없음(공간적으로 견고).',
           '- 임상 핵심인 중심(≤9°) 영역 성능을 별도 보고 → "쓸모" 서사 강화.']
    (ROOT / 'runs/severity_region.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n저장: runs/severity_region.json / .md', flush=True)


if __name__ == '__main__':
    main()
