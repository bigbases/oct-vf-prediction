#!/usr/bin/env python3
"""좌우(OD/OS) 규약 일관성 검사. 읽기 전용 — 어떤 파일도 쓰지 않는다.

찾는 것: "규약을 바꾸면 성능이 달라지나"가 아니라 **"코드 안에서 규약이 일치하나"**.
규약 오류는 값 하나하나가 그럴듯해서 값 대조로는 안 잡힌다. 집단(OD/OS)을 갈라
분포를 봐야 나온다. 배경과 2026-08-26 판정은 docs/LATERALITY_AUDIT.md.

검사 A (핵심) — tabular 프레임 항등식
    사분면 값은 해당 3개 시계시간의 평균이다. 각 행에서 |사분면 − mean(3시간)| 을
    재면 어느 시간 삼각과 짝인지 반올림 오차 수준으로 드러난다. OD 행과 OS 행이
    다른 삼각을 가리키면 같은 컬럼이 두 눈에서 다른 해부학 위치를 뜻한다.
    (2026-08-26 결과: rnfl_q_t/q_n 이 어긋남 — 수정하지 않기로 결정.)

검사 B — 미러 대조
    라벨 52점·시계시간 12개의 집단 평균 프로파일을 OD/OS 로 나눠, OS 를 그대로 쓴
    경우와 좌우 미러한 경우 중 어느 쪽이 OD 와 더 닮았는지 본다. 미러 쪽이 이기면
    그 블록이 정규화되지 않은 것이다.

검사 C (핵심) — OD/OS 성능 분리
    분지별 per-eye RMSE/MAE 와 지점별(52) MAE 를 OD/OS 로 나눈다. OD/OS 는 중증도가
    다를 수 있으므로 raw 격차만 보면 안 된다. RMSE ~ 1 + 안평균감도 + I(OS) 로
    보정한 OS 계수가 판정 기준이다.

실행:
    python scripts/check_laterality.py
    python scripts/check_laterality.py --csv ml_final_180d_excl_empty_flip.csv \
        --csv-nonflip ml_final_180d_excl_empty.csv
    python scripts/check_laterality.py --skip-preds      # 검사 A/B 만
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

# ── 규약 상수 ──────────────────────────────────────────────────────────
QUAD = ['rnfl_q_s', 'rnfl_q_t', 'rnfl_q_i', 'rnfl_q_n']
HOURS = [f'rnfl_h{i:02d}' for i in range(1, 13)]
# 사분면 ↔ 시계시간 삼각. 이름은 OD 프레임 기준 위치이지 해부학 단정이 아니다.
TRIADS = {'h08-10': (8, 9, 10), 'h02-04': (2, 3, 4), 'h11-01': (11, 12, 1), 'h05-07': (5, 6, 7)}
# 미러 = 6시·12시만 고정, 나머지 10개는 전부 재배치 (make_rnfl_flip.py:17-19)
MIRROR_H = {1: 11, 2: 10, 3: 9, 4: 8, 5: 7, 6: 6, 7: 5, 8: 4, 9: 3, 10: 2, 11: 1, 12: 12}
GCA_SEC = ['s_sup_t', 's_sup', 's_sup_n', 's_inf_n', 's_inf', 's_inf_t']

# 24-2 표준 좌표 (p01..p54, 위→아래 / 좌→우). scripts/build_severity_region.py 와 동일.
VF_ROWS = [(+21, [-9, -3, 3, 9]), (+15, [-15, -9, -3, 3, 9, 15]),
           (+9, [-21, -15, -9, -3, 3, 9, 15, 21]),
           (+3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]),
           (-3, [-27, -21, -15, -9, -3, 3, 9, 15, 21]),
           (-9, [-21, -15, -9, -3, 3, 9, 15, 21]), (-15, [-15, -9, -3, 3, 9, 15]),
           (-21, [-9, -3, 3, 9])]
PT54 = [f'p{i:02d}' for i in range(1, 55)]
BLIND54 = {25, 34}                      # 0-based p26, p35
IDX52 = [i for i in range(54) if i not in BLIND54]
# 라벨 미러 = FLIP_ROWS 행 단위 반전 (scripts/apply_vf_neg1_to_zero.py:19-27)
FLIP_ROWS = [(1, 4), (5, 10), (11, 18), (19, 27), (28, 36), (37, 44), (45, 50), (51, 54)]

BACKBONES = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    # 주의: 디렉터리 이름은 clip_mse 지만 실제 체크포인트는 Inception-v3 다.
    'Inception-v3': 'runs/phasec_b0_clip_mse_5fold',
    'VGG16': 'runs/phasec_b0_vgg16_5fold',
    'Xception': 'runs/phasec_b0_xception_5fold',
    'DenseNet121': 'runs/phasec_b0_densenet121_5fold',
}
OSFLIP_DIR = 'runs/phasec_b0_inception_resnet_v2_5fold_osflip'
XGB_TMPL = 'runs/oof/xgb_90d_fold{k}_{s}.npz'
W_GRID = np.linspace(0, 1, 101)


def _mirror52() -> np.ndarray:
    """52점 벡터의 좌우 미러 사상. 상대가 맹점이면 -1(비교 불가)."""
    rev = np.arange(54)
    for a, b in FLIP_ROWS:
        rev[a - 1:b] = np.arange(a - 1, b)[::-1]
    pos = {g: j for j, g in enumerate(IDX52)}
    return np.array([pos.get(rev[g], -1) for g in IDX52])


MIRROR52 = _mirror52()


def fnum(row, col):
    v = row.get(col, '')
    try:
        return float(v) if v not in ('', None) else np.nan
    except ValueError:
        return np.nan


# 직접-미러 상관 차이가 이보다 작으면 동률로 본다. 표본 노이즈로 순서가 뒤집힌다.
TIE = 0.05


def corr(a, b, min_n=4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= min_n else float('nan')


def read_csv(path: Path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


# ── 검사 A: tabular 프레임 항등식 ───────────────────────────────────────
def check_a(rows, tag: str, normalised: bool) -> dict:
    print(f'\n[A] tabular 프레임 항등식 — {tag}')
    print('    |사분면 − mean(시계시간 3개)| 중앙값(µm). 가장 작은 칸이 그 사분면의 짝이다.')
    names = list(TRIADS)
    out = {}
    for eye in ('OD', 'OS'):
        rs = [r for r in rows if r['eye'].upper() == eye]
        if not rs:
            continue
        print(f'    {eye}  n={len(rs)}')
        print('      ' + f'{"":9s} ' + ' '.join(f'{n:>9s}' for n in names))
        for q in QUAD:
            med = {}
            for n in names:
                d = [abs(fnum(r, q) - np.mean([fnum(r, f'rnfl_h{i:02d}') for i in TRIADS[n]]))
                     for r in rs]
                d = [x for x in d if np.isfinite(x)]
                med[n] = float(np.median(d)) if d else float('nan')
            pair = min(names, key=lambda n: med[n])
            out[(eye, q)] = pair
            print(f'      {q:9s} ' + ' '.join(f'{med[n]:9.2f}' for n in names) + f'   → {pair}')
    bad = [q for q in QUAD if out.get(('OD', q)) != out.get(('OS', q))]
    if not normalised:
        # 정규화 전에는 시계시간 번호가 화면 위치를 뜻하므로 두 눈이 달라야 정상이다.
        # (추출 단계에서 사분면은 좌우를 인식하고 시계시간은 인식하지 않는다.)
        print(f'    (정규화 전 파일 — 두 눈이 다른 삼각을 가리키는 게 정상. 불일치: '
              f'{", ".join(bad) if bad else "없음"})')
    elif bad:
        print(f'    ⚠️ 불일치: {", ".join(bad)} — OD 행과 OS 행이 서로 다른 시간 삼각을 가리킨다.')
        print('       정규화 후에는 시계시간도 해부학 위치를 뜻하므로 두 눈이 같아야 한다.')
        print('       같은 컬럼이 두 눈에서 다른 해부학 위치를 뜻한다는 뜻이다.')
    else:
        print('    OK — 네 사분면 모두 두 눈이 같은 시간 삼각을 가리킨다.')
    # 분포 지문: 스왑되면 SD 가 짝 컬럼과 맞바뀐다
    print('    분포 지문 (컬럼별 SD, µm):')
    for q in ('rnfl_q_t', 'rnfl_q_n'):
        sd = {}
        for eye in ('OD', 'OS'):
            v = np.array([fnum(r, q) for r in rows if r['eye'].upper() == eye])
            sd[eye] = float(np.nanstd(v))
        print(f'      {q:9s} OD {sd.get("OD", float("nan")):5.1f}   OS {sd.get("OS", float("nan")):5.1f}')
    return {'pairs': {f'{e}.{q}': v for (e, q), v in out.items()}, 'mismatched': bad}


# ── 검사 B: 미러 대조 ──────────────────────────────────────────────────
def _vf_profile(rows, eye):
    rs = [r for r in rows if r['eye'].upper() == eye]
    A = np.full((len(rs), 52), np.nan)
    for i, r in enumerate(rs):
        for j, g in enumerate(IDX52):
            v = fnum(r, PT54[g])
            A[i, j] = v if np.isfinite(v) and v >= 0 else np.nan
    return np.nanmean(A, axis=0)


def _mirror_line(name, a, b, bm):
    d, m = corr(a, b), corr(a, bm)
    verdict = ('⚠️ 미러 우세 — 정규화 안 됨' if m > d + TIE
               else '동률 — 판정 불가' if abs(d - m) <= TIE else 'OK (직접 우세)')
    print(f'      {name:22s} 직접 {d:6.3f}   미러 {m:6.3f}   {verdict}')
    return {'direct': d, 'mirror': m}


def check_b(rows, tag: str, normalised: bool) -> dict:
    print(f'\n[B] 미러 대조 — {tag}')
    print('    집단 평균 프로파일. 미러 쪽이 이기면 그 블록이 OD 프레임으로 정규화되지 않은 것이다.')
    if not normalised:
        print('    (정규화 전 파일 — 라벨·시계시간은 미러가 이기는 게 정상. 양성 대조로 쓴다.)')
    res = {}
    v = MIRROR52 >= 0
    lo, ls = _vf_profile(rows, 'OD'), _vf_profile(rows, 'OS')
    res['vf52'] = _mirror_line('VF 52점 라벨', lo[v], ls[v], ls[MIRROR52[v]])
    ho = np.array([np.nanmean([fnum(r, h) for r in rows if r['eye'].upper() == 'OD']) for h in HOURS])
    hs = np.array([np.nanmean([fnum(r, h) for r in rows if r['eye'].upper() == 'OS']) for h in HOURS])
    res['hours'] = _mirror_line('RNFL 시계시간 12개', ho, hs, np.array([hs[MIRROR_H[i + 1] - 1] for i in range(12)]))
    qo = np.array([np.nanmean([fnum(r, q) for r in rows if r['eye'].upper() == 'OD']) for q in QUAD])
    qs = np.array([np.nanmean([fnum(r, q) for r in rows if r['eye'].upper() == 'OS']) for q in QUAD])
    res['quadrants'] = _mirror_line('RNFL 사분면 4개', qo, qs, qs[[0, 3, 2, 1]])
    print('      ※ 사분면 4개는 t/n 값이 서로 가까워 이 대조로는 판정력이 없다. 검사 A 를 쓴다.')
    print('      ※ GCA 섹터는 섹터끼리 상관이 높아 어떤 상관 검사로도 t/n 이 갈리지 않는다.')
    print('         2026-08-26 판정 불가로 남김 (docs/LATERALITY_AUDIT.md §3).')
    return res


# ── 검사 C: OD/OS 성능 분리 ────────────────────────────────────────────
def load_split(k: int, split: str):
    s = 'val' if split == 'OOF' else 'test'
    xz = load_oof_npz(ROOT / XGB_TMPL.format(k=k, s=s))
    czs = {nm: load_oof_npz(ROOT / d / f'{s}_preds_fold{k}.npz') for nm, d in BACKBONES.items()}
    osf = ROOT / OSFLIP_DIR / f'{s}_preds_fold{k}.npz'
    if osf.exists():
        czs['IR-v2-osflip'] = load_oof_npz(osf)
    keys = set(xz['keys'])
    for cz in czs.values():
        keys &= set(cz['keys'])
    keys = sorted(keys)

    def pick(z):
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        ii = [idx[kk] for kk in keys]
        return z['pred'][ii], z['labels'][ii], z['mask'][ii]

    xp, lab, xm = pick(xz)
    cnns, cm = {}, None
    for nm, cz in czs.items():
        cp, _, m = pick(cz)
        cnns[nm] = cp
        if nm in BACKBONES:                      # osflip 은 마스크 교집합에서 제외
            cm = m if cm is None else (cm & m)
    return dict(xgb=xp, cnns=cnns, ir=cnns['IR-v2'],
                ens=np.mean([cnns[n] for n in BACKBONES], axis=0),
                lab=lab, mask=(xm & cm).astype(bool),
                eye=np.array([str(kk[1]) for kk in keys], dtype=object))


def fit_gw(folds, ckey: str) -> float:
    x = np.concatenate([f['xgb'][f['mask']] for f in folds])
    c = np.concatenate([f[ckey][f['mask']] for f in folds])
    l = np.concatenate([f['lab'][f['mask']] for f in folds])
    return float(W_GRID[int(np.argmin([np.mean((w * x + (1 - w) * c - l) ** 2) for w in W_GRID]))])


def check_c(n_folds: int) -> dict:
    print('\n[C] OD/OS 성능 분리')
    print('    정렬 규약은 scripts/export_robustness_gains.py 와 동일')
    print('    (공통 키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합). fusion 은 nested global-w.')
    OOF = {k: load_split(k, 'OOF') for k in range(n_folds)}
    oth = {k: [OOF[i] for i in range(n_folds) if i != k] for k in range(n_folds)}
    w_ir = {k: fit_gw(oth[k], 'ir') for k in range(n_folds)}
    w_en = {k: fit_gw(oth[k], 'ens') for k in range(n_folds)}
    print(f'    nested w  IR-v2 {[round(w_ir[k], 2) for k in range(n_folds)]}'
          f'   5-ens {[round(w_en[k], 2) for k in range(n_folds)]}')
    res = {}
    for split in ('OOF', 'TEST'):
        D = OOF if split == 'OOF' else {k: load_split(k, 'TEST') for k in range(n_folds)}
        res[split] = _report_split(D, split, w_ir, w_en, n_folds)
    return res


def _branches(f, k, w_ir, w_en):
    br = {'XGB (summary)': f['xgb'], 'CNN IR-v2 (image)': f['ir'], 'CNN 5-ens (image)': f['ens'],
          'Fusion XGB+IR-v2': w_ir[k] * f['xgb'] + (1 - w_ir[k]) * f['ir'],
          'Fusion XGB+5-ens': w_en[k] * f['xgb'] + (1 - w_en[k]) * f['ens']}
    if 'IR-v2-osflip' in f['cnns']:
        br['CNN IR-v2 osflip'] = f['cnns']['IR-v2-osflip']
    return br


def _report_split(D, split, w_ir, w_en, n_folds) -> dict:
    resid, eyerm, ptmae, ms, eyes = {}, {}, {}, [], []
    for k in range(n_folds):
        f = D[k]
        br = _branches(f, k, w_ir, w_en)
        for i in range(len(f['eye'])):
            m = f['mask'][i]
            if not m.any():
                continue
            eyes.append(f['eye'][i])
            ms.append(float(np.mean(f['lab'][i][m])))
            for nm, p in br.items():
                d = (p[i] - f['lab'][i])[m]
                resid.setdefault(nm, {}).setdefault(f['eye'][i], []).append(d)
                eyerm.setdefault(nm, []).append(float(np.sqrt(np.mean(d ** 2))))
                ptmae.setdefault(nm, {}).setdefault(f['eye'][i], []).append(
                    np.where(m, np.abs(p[i] - f['lab'][i]), np.nan))
    eyes = np.array(eyes); ms = np.array(ms)
    od, os_ = eyes == 'OD', eyes == 'OS'
    print(f'\n    ── {split} ──  OD {od.sum()}안  OS {os_.sum()}안')
    print(f'    안 평균감도(dB): OD {ms[od].mean():.2f}  OS {ms[os_].mean():.2f}  '
          f'MWU p={stats.mannwhitneyu(ms[od], ms[os_]).pvalue:.3f}'
          '   ← 격차가 있으면 raw RMSE 차이는 규약 근거가 못 된다')
    print(f'    {"branch":20s} {"RMSE_OD":>8s} {"RMSE_OS":>8s} {"Δ":>7s} '
          f'{"MAE_OD":>7s} {"MAE_OS":>7s} {"Δ":>7s} {"β(OS)|감도":>10s} {"p":>8s}')
    out = {}
    for nm in resid:
        r, mae = {}, {}
        for e in ('OD', 'OS'):
            d = np.concatenate(resid[nm][e])
            r[e] = float(np.sqrt(np.mean(d ** 2))); mae[e] = float(np.mean(np.abs(d)))
        v = np.array(eyerm[nm])
        X = np.column_stack([np.ones_like(ms), ms, os_.astype(float)])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        res_ = v - X @ beta
        dof = len(v) - 3
        se = np.sqrt((res_ @ res_ / dof) * np.linalg.inv(X.T @ X)[2, 2])
        p = float(2 * stats.t.sf(abs(beta[2] / se), dof))
        out[nm] = {'rmse_od': r['OD'], 'rmse_os': r['OS'], 'mae_od': mae['OD'], 'mae_os': mae['OS'],
                   'beta_os_adj': float(beta[2]), 'p_adj': p}
        print(f'    {nm:20s} {r["OD"]:8.3f} {r["OS"]:8.3f} {r["OS"] - r["OD"]:+7.3f} '
              f'{mae["OD"]:7.3f} {mae["OS"]:7.3f} {mae["OS"] - mae["OD"]:+7.3f} '
              f'{beta[2]:+10.3f} {p:8.3f}')
    flagged = [nm for nm, d in out.items() if abs(d['beta_os_adj']) > 0.5 and d['p_adj'] < 0.05]
    print('    ⚠️ 감도 보정 후에도 OS 계수가 큰 분지: ' + (', '.join(flagged) if flagged else '없음'))

    print(f'\n    지점별(52) MAE 프로파일 — 미러 쪽이 {TIE:.2f} 넘게 이기면 그 분지가 어긋난 것이다')
    print(f'    {"branch":20s} {"corr(OD,OS)":>12s} {"corr(OD,mirrOS)":>16s} {"최대 절대 ΔMAE":>14s} {"지점":>6s}')
    v = MIRROR52 >= 0
    for nm in ptmae:
        a = np.nanmean(np.array(ptmae[nm]['OD']), axis=0)
        b = np.nanmean(np.array(ptmae[nm]['OS']), axis=0)
        c1, c2 = corr(a[v], b[v]), corr(a[v], b[MIRROR52[v]])
        d = b - a
        j = int(np.nanargmax(np.abs(d)))
        out[nm]['pt_corr_direct'], out[nm]['pt_corr_mirror'] = c1, c2
        print(f'    {nm:20s} {c1:12.3f} {c2:16.3f} {d[j]:+14.3f} {"p" + str(IDX52[j] + 1):>6s}'
              + ('   ⚠️ 미러 우세' if c2 > c1 + TIE else '   (동률)' if abs(c1 - c2) <= TIE else ''))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default='ml_final_90d_excl_empty_flip.csv',
                    help='정규화(flip) 후 CSV — 학습이 실제로 쓰는 파일')
    ap.add_argument('--csv-nonflip', default='ml_final_90d_excl_empty.csv',
                    help='정규화 전 CSV. 있으면 대조로 함께 출력')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--skip-preds', action='store_true', help='검사 C 생략 (A/B 만)')
    a = ap.parse_args()

    print('=' * 78)
    print('좌우(OD/OS) 규약 일관성 검사   ·   읽기 전용')
    print('=' * 78)
    for name, is_main in ((a.csv_nonflip, False), (a.csv, True)):
        p = ROOT / name
        if not p.exists():
            print(f'\n(건너뜀: {name} 없음)')
            continue
        rows = read_csv(p)
        tag = f'{name}{"  ← 정본" if is_main else "  (정규화 전, 대조)"}'
        check_a(rows, tag, normalised=is_main)
        check_b(rows, tag, normalised=is_main)
    if not a.skip_preds:
        try:
            check_c(a.folds)
        except FileNotFoundError as e:
            print(f'\n[C] 건너뜀 — 예측 번들 없음: {e}')
    print('\n판정 기준 요약:')
    print('  · 검사 A 는 정본(flip) 파일에서만 판정한다. 거기서 불일치가 뜨면 그 컬럼은')
    print('    두 눈에서 다른 해부학 위치를 뜻한다 (확정적). 정규화 전 파일의 불일치는 정상.')
    print('  · 검사 B 도 정본 파일에서 판정한다. 미러가 이기면 정규화되지 않았다 (확정적).')
    print('  · 검사 C 의 raw Δ 는 중증도 교란을 받는다. β(OS)|감도 만 근거로 쓴다.')


if __name__ == '__main__':
    main()
