#!/usr/bin/env python3
"""GCA crop 이탈안 민감도 분석 — 평가 마스크만 조정한다.

배경: image_preprocessing._crop_gca_thickness 는 파란 화소의 bounding box 로
경계를 잡는다. 검출되는 파랑은 리포트에 그려진 고정 도형이 아니라 두께맵
컬러맵의 파랑/청록 구역(=얇은 GCIPL)이라, crop 크기가 자료에 의존한다.
분석 240안에서 crop 면적과 MD 의 상관은 r=+0.255 (p=7.6e-05) 다.

이 스크립트가 하는 일: 저장된 예측에서 crop 이 주군집을 벗어난 안을 평가에서
빼고 원고의 주 주장을 다시 계산한다. 재학습 없음. w 재적합 없음 — OOF 는
fold별 nested w, TEST 는 전체 OOF 적합 w 를 원래대로 쓴다. 이탈안을 뺀 뒤
w 를 다시 맞추면 '평가 마스크만 조정' 이 아니게 된다.

구간·MIN_PTS·기울기·마스크·백본 목록은 전부 섀도 트리의
scripts/fusion_sensitivity_trend.py 와 scripts/fusion_eval_common.py 에서
import 한다. 새로 정의하지 않는다.

패스 A(정본 runs/ 와 동일, 원고 숫자의 출처)와 패스 B(라테랄리티 수정본)를
모두 돌린다. bias_by_bin.py 와 같은 방식이다.

읽기 전용. 정본 runs/ 와 원고를 건드리지 않는다.
출력: experiments/gca_crop/crop_sensitivity.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREES = {'A': REPO / 'experiments/laterality_qfix/step4_work/A',
         'B': REPO / 'experiments/laterality_qfix/step4_work/B'}
SIZES = REPO / 'experiments/gca_crop/crop_sizes.json'
SEED = 42
N_BOOT = 10000
THRESHOLDS = [300, 320, 335]      # min(crop_W, crop_H) < thr 이면 이탈안
CENTERED_FOR = ('IR-v2', 'ENSEMBLE')   # 중심화 반사실을 돌릴 백본


# ---------------------------------------------------------------- 공통 도구
def load_modules(tree: Path):
    for m in ('fusion_eval_common', 'fusion_sensitivity_trend', 'oof_common',
              'fig_style'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tree / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        fst = importlib.import_module('fusion_sensitivity_trend')
        assert fec.ROOT == tree, f'ROOT 불일치: {fec.ROOT} != {tree}'
        return fec, fst
    finally:
        sys.path.pop(0)


def boot_ci(v, pid, rng):
    """환자 단위 cluster bootstrap 평균 95% CI.

    fusion_sensitivity_trend.cluster_boot_mean_ci 와 수치가 같다. 군집을 다시
    뽑아 이어붙인 벡터의 평균은 (뽑힌 군집 합의 합)/(뽑힌 군집 크기의 합) 과
    항등이고, rng.integers 의 소비 순서도 같아 난수까지 일치한다. 반복문을
    행렬 한 번으로 바꾼 것뿐이다 — self-check 에서 원본 값과 대조한다.
    """
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    ng = len(groups)
    idx = rng.integers(0, ng, (N_BOOT, ng))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled(pred, lab, mask, keep):
    """유지 대상 안의 마스크된 지점을 통째로 풀링한 RMSE / MAE."""
    m = mask.astype(bool) & keep[:, None]
    d = (pred - lab)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def wilcox(a, b):
    if a.size < 6 or np.allclose(a, b):
        return float('nan')
    return float(wilcoxon(a, b).pvalue)


# ------------------------------------------------------- 이탈안 정의 (1)
def outlier_flags():
    sz = json.loads(SIZES.read_text(encoding='utf-8'))
    out = {}
    for split, rec in sz.items():
        out[split] = {k: (v['gca_w'], v['gca_h']) for k, v in rec.items()}
    return out


def keep_vector(keys, sizes, thr):
    """thr=None 이면 전체 유지."""
    if thr is None:
        return np.ones(len(keys), bool)
    return np.array([min(sizes['|'.join(k)]) >= thr for k in keys])


# ------------------------------------------------------------- §4.4 추세
def trend(fst, fus, cnn, lab, mask, pid, keep, rng_seed=SEED):
    d = (fst.eye_bin_rmse(fus, lab, mask) - fst.eye_bin_rmse(cnn, lab, mask))[keep]
    sl = fst.slopes(d)
    ok = np.isfinite(sl)
    sl_ok, pid_ok = sl[ok], pid[keep][ok]
    lo, hi = boot_ci(sl_ok, pid_ok, np.random.default_rng(rng_seed))
    per_bin = []
    for b in range(4):
        col = d[:, b]; v = col[np.isfinite(col)]
        per_bin.append({'bin': fst.BIN_LABELS[b], 'n_eyes': int(v.size),
                        'mean_delta': float(v.mean()) if v.size else float('nan')})
    return {'n_eyes_with_slope': int(sl_ok.size),
            'mean_slope': float(sl_ok.mean()),
            'median_slope': float(np.median(sl_ok)),
            'slope_ci_lo': lo, 'slope_ci_hi': hi,
            'p_wilcoxon': float(wilcoxon(sl_ok).pvalue) if sl_ok.size >= 6 else float('nan'),
            'frac_positive': float(np.mean(sl_ok > 0)),
            'per_bin': per_bin}


def bin_residual_means(fst, pred, lab, mask, keep):
    m = mask.astype(bool) & keep[:, None]
    bi = fst.bin_index(lab[m])
    r = (pred - lab)[m]
    return np.array([float(r[bi == b].mean()) if (bi == b).any() else np.nan
                     for b in range(4)])


def center_by_bin(fst, cnn, lab, bmeans):
    out = cnn.copy()
    bi = fst.bin_index(lab)
    for b in range(4):
        if np.isfinite(bmeans[b]):
            out[bi == b] -= bmeans[b]
    return out


# ------------------------------------------------------------------ 본체
def split_blocks(fec, fst, oof, tst, test_lab, test_mask, name):
    """한 백본의 OOF / TEST 원자료. w 는 원래 규약대로 고정."""
    xs, cs, ls, ms, pids, ws, bounds = [], [], [], [], [], [], []
    n = 0
    for k in range(5):
        w = fec.fit_w([oof[i] for i in range(5) if i != k], name)
        f = oof[k]
        xs.append(f['xgb']); cs.append(f['cnns'][name])
        ls.append(f['lab']); ms.append(f['mask'])
        pids += [str(kk[0]) for kk in f['keys']]
        ws.append(w); bounds.append((n, n + f['xgb'].shape[0])); n += f['xgb'].shape[0]
    xo, co = np.concatenate(xs), np.concatenate(cs)
    fo = np.empty_like(xo)
    for w, (a, b) in zip(ws, bounds):
        fo[a:b] = w * xo[a:b] + (1 - w) * co[a:b]
    w_all = fec.fit_w([oof[k] for k in range(5)], name)
    fo_glob = w_all * xo + (1 - w_all) * co

    xt = np.mean([tst[k]['xgb'] for k in range(5)], 0)
    ct = np.mean([tst[k]['cnns'][name] for k in range(5)], 0)
    ft = w_all * xt + (1 - w_all) * ct
    return {
        'OOF': dict(xgb=xo, cnn=co, fus_nested=fo, fus_global=fo_glob,
                    lab=np.concatenate(ls), mask=np.concatenate(ms),
                    pid=np.array(pids, dtype=object),
                    keys=[k for j in range(5) for k in oof[j]['keys']],
                    w_nested=[round(w, 4) for w in ws], w_global=float(w_all)),
        'TEST': dict(xgb=xt, cnn=ct, fus_nested=ft, fus_global=ft,
                     lab=test_lab, mask=test_mask.astype(bool),
                     pid=np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object),
                     keys=list(tst[0]['keys']),
                     w_nested=None, w_global=float(w_all)),
    }


def run_tree(pk, tree, sizes):
    fec, fst = load_modules(tree)
    oof, tst, test_lab, test_mask = fec.load_all(ensemble=True)
    names = fec.NAMES + [fec.ENSEMBLE]
    sets = {'full': None, **{f'thr{t}': t for t in THRESHOLDS}}

    res = {}
    for nm in names:
        blocks = split_blocks(fec, fst, oof, tst, test_lab, test_mask, nm)
        entry = {'w_oof_nested': blocks['OOF']['w_nested'],
                 'w_global': blocks['OOF']['w_global']}
        for split, B in blocks.items():
            keys, sz = B['keys'], sizes[split]
            sblk = {}
            for sname, thr in sets.items():
                keep = keep_vector(keys, sz, thr)
                cell = {'n_eyes': int(keep.sum()), 'n_dropped': int((~keep).sum())}
                for wm in ('nested', 'global'):
                    if split == 'TEST' and wm == 'nested':
                        continue
                    fus = B[f'fus_{wm}']
                    pr, pm = {}, {}
                    for tag, arr in (('summary', B['xgb']), ('image', B['cnn']),
                                     ('fusion', fus)):
                        pr[tag], pm[tag] = pooled(arr, B['lab'], B['mask'], keep)
                    ex = fec.eye_metric(B['xgb'], B['lab'], B['mask'], 'rmse')[keep]
                    ec = fec.eye_metric(B['cnn'], B['lab'], B['mask'], 'rmse')[keep]
                    ef = fec.eye_metric(fus, B['lab'], B['mask'], 'rmse')[keep]
                    ax = fec.eye_metric(B['xgb'], B['lab'], B['mask'], 'mae')[keep]
                    ac = fec.eye_metric(B['cnn'], B['lab'], B['mask'], 'mae')[keep]
                    af = fec.eye_metric(fus, B['lab'], B['mask'], 'mae')[keep]
                    cell[wm] = {
                        'pooled_rmse': pr, 'pooled_mae': pm,
                        'per_eye_rmse_mean': {'summary': float(ex.mean()),
                                              'image': float(ec.mean()),
                                              'fusion': float(ef.mean())},
                        'p_fusion_vs_summary_rmse': wilcox(ef, ex),
                        'p_fusion_vs_image_rmse': wilcox(ef, ec),
                        'p_fusion_vs_summary_mae': wilcox(af, ax),
                        'p_fusion_vs_image_mae': wilcox(af, ac),
                        'mean_diff_fusion_minus_image_rmse': float((ef - ec).mean()),
                        'mean_diff_summary_minus_fusion_rmse': float((ex - ef).mean()),
                        'n_eyes_fusion_better_than_image': int((ef < ec).sum()),
                    }
                # §4.4 — nested(OOF) / global(TEST) 규약 그대로
                fus_tr = B['fus_nested']
                variants = {'original': trend(fst, fus_tr, B['cnn'], B['lab'],
                                              B['mask'], B['pid'], keep)}
                if nm in CENTERED_FOR:
                    bm = bin_residual_means(fst, B['cnn'], B['lab'], B['mask'], keep)
                    g = float(((B['cnn'] - B['lab'])[B['mask'] & keep[:, None]]).mean())
                    for vname, cc in (('bin_centered',
                                       center_by_bin(fst, B['cnn'], B['lab'], bm)),
                                      ('global_centered', B['cnn'] - g)):
                        if split == 'OOF':
                            ff = np.empty_like(cc)
                            for w, (a, b) in zip(B['w_nested'],
                                                 _bounds(oof)):
                                ff[a:b] = w * B['xgb'][a:b] + (1 - w) * cc[a:b]
                        else:
                            wv = B['w_global']
                            ff = wv * B['xgb'] + (1 - wv) * cc
                        variants[vname] = trend(fst, ff, cc, B['lab'], B['mask'],
                                                B['pid'], keep)
                    variants['_cnn_residual_mean_by_bin'] = bm.tolist()
                    variants['_cnn_residual_mean_global'] = g
                cell['trend'] = variants
                sblk[sname] = cell
            entry[split] = sblk
        res[nm] = entry
        o = entry['OOF']
        print(f"  [{pk}] {nm:<13} full fus {o['full']['global']['pooled_rmse']['fusion']:.3f} "
              f"-> thr335 {o['thr335']['global']['pooled_rmse']['fusion']:.3f} | "
              f"slope {o['full']['trend']['original']['mean_slope']:+.4f} "
              f"-> {o['thr335']['trend']['original']['mean_slope']:+.4f}", flush=True)
    return res


# ------------------------------------------------- (3) 방향 확인
def _ols_outlier_coef(y, md, out):
    """rmse ~ 1 + MD + is_outlier. 이탈 계수를 돌려준다."""
    X = np.column_stack([np.ones_like(y), md, out.astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2]), beta


def _match_by_md(md_out, md_ctl, caliper=1.0):
    """이탈안마다 MD 가 가장 가까운 비이탈안을 비복원 1:1 로 붙인다.
    caliper 밖이면 짝을 짓지 않는다. MD 가 나쁜 쪽부터 처리해 희소한 구간이
    먼저 짝을 고르게 한다."""
    free = list(range(len(md_ctl)))
    pairs = []
    for i in np.argsort(md_out):                 # MD 낮은(나쁜) 순
        if not free:
            break
        j = min(free, key=lambda t: abs(md_ctl[t] - md_out[i]))
        if abs(md_ctl[j] - md_out[i]) <= caliper:
            pairs.append((int(i), int(j)))
            free.remove(j)
    return pairs


def direction_block(pk, tree, sizes, thr=335):
    """이탈안이 특별히 나쁜가, 아니면 중증도로 설명되는가."""
    from scipy.stats import mannwhitneyu
    sys.path.insert(0, str(REPO / 'scripts'))
    from build_severity_region import MD, nkey        # noqa: E402
    fec, fst = load_modules(tree)
    oof, tst, test_lab, test_mask = fec.load_all(ensemble=True)
    out = {'threshold': thr}
    for nm in CENTERED_FOR:
        B = split_blocks(fec, fst, oof, tst, test_lab, test_mask, nm)['OOF']
        keys = B['keys']
        keep = keep_vector(keys, sizes['OOF'], thr)
        md = np.array([MD.get(nkey(*k), np.nan) for k in keys])
        pid = B['pid']
        blk = {'n_kept': int(keep.sum()), 'n_outlier': int((~keep).sum()),
               'n_md_missing_outlier': int(np.isnan(md[~keep]).sum()),
               'n_md_missing_kept': int(np.isnan(md[keep]).sum()),
               'md_median_outlier': float(np.nanmedian(md[~keep])),
               'md_median_kept': float(np.nanmedian(md[keep])),
               'branches': {}}
        for tag, arr in (('summary', B['xgb']), ('image', B['cnn']),
                         ('fusion', B['fus_nested'])):
            e = fec.eye_metric(arr, B['lab'], B['mask'], 'rmse')
            raw = {'mean_outlier': float(e[~keep].mean()),
                   'mean_kept': float(e[keep].mean()),
                   'median_outlier': float(np.median(e[~keep])),
                   'median_kept': float(np.median(e[keep])),
                   'diff_mean': float(e[~keep].mean() - e[keep].mean()),
                   'p_mannwhitney': float(mannwhitneyu(e[~keep], e[keep]).pvalue)}
            ok = np.isfinite(md)
            io_, ik = (~keep) & ok, keep & ok
            pairs = _match_by_md(md[io_], md[ik])
            eo, ek = e[io_], e[ik]
            do = np.array([eo[i] - ek[j] for i, j in pairs])
            mo = np.array([md[io_][i] - md[ik][j] for i, j in pairs])
            matched = {'n_pairs': len(pairs),
                       'mean_md_gap': float(np.abs(mo).mean()) if len(pairs) else float('nan'),
                       'mean_diff': float(do.mean()) if len(pairs) else float('nan'),
                       'p_wilcoxon': (float(wilcoxon(do).pvalue)
                                      if len(pairs) >= 6 else float('nan'))}
            coef, _ = _ols_outlier_coef(e[ok], md[ok], ~keep[ok])
            rng = np.random.default_rng(SEED)
            groups = [np.flatnonzero(pid[ok] == p) for p in np.unique(pid[ok])]
            ng = len(groups)
            bs = np.empty(N_BOOT)
            yy, mm, oo = e[ok], md[ok], (~keep[ok])
            for bi in range(N_BOOT):
                idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
                if oo[idx].sum() < 2 or oo[idx].all():
                    bs[bi] = np.nan
                    continue
                bs[bi] = _ols_outlier_coef(yy[idx], mm[idx], oo[idx])[0]
            bs = bs[np.isfinite(bs)]
            adj = {'coef_outlier': coef,
                   'ci_lo': float(np.percentile(bs, 2.5)),
                   'ci_hi': float(np.percentile(bs, 97.5)),
                   'n_boot_valid': int(bs.size)}
            blk['branches'][tag] = {'raw': raw, 'md_matched': matched,
                                    'md_adjusted': adj}
        out[nm] = blk
        b = blk['branches']['fusion']
        print(f"  [{pk}] {nm:<9} fusion 안별RMSE 이탈 {b['raw']['mean_outlier']:.2f} "
              f"vs 유지 {b['raw']['mean_kept']:.2f} (p={b['raw']['p_mannwhitney']:.3f}) | "
              f"MD매칭 {b['md_matched']['n_pairs']}쌍 차 {b['md_matched']['mean_diff']:+.2f} "
              f"(p={b['md_matched']['p_wilcoxon']:.3f}) | MD보정 계수 "
              f"{b['md_adjusted']['coef_outlier']:+.2f} "
              f"[{b['md_adjusted']['ci_lo']:+.2f},{b['md_adjusted']['ci_hi']:+.2f}]", flush=True)
    return out


def _bounds(oof):
    n, out = 0, []
    for k in range(5):
        s = oof[k]['xgb'].shape[0]
        out.append((n, n + s)); n += s
    return out


def main() -> int:
    sizes = outlier_flags()
    fec, _ = load_modules(TREES['A'])
    oof, tst, _, _ = fec.load_all()
    keysets = {'OOF': [k for j in range(5) for k in oof[j]['keys']],
               'TEST': list(tst[0]['keys'])}

    # (1) 이탈안 정의
    excl = {}
    for split, keys in keysets.items():
        excl[split] = {}
        for t in THRESHOLDS:
            keep = keep_vector(keys, sizes[split], t)
            excl[split][f'thr{t}'] = {
                'n_total': len(keys), 'n_excluded': int((~keep).sum()),
                'eyes': ['|'.join(k) for k, kp in zip(keys, keep) if not kp],
            }
        print(f'{split}: ' + '  '.join(
            f"thr{t} 제외 {excl[split][f'thr{t}']['n_excluded']}/{len(keys)}"
            for t in THRESHOLDS))

    res = {'note': ('GCA crop 이탈안 민감도. 재학습 없음, w 재적합 없음, '
                    '평가 마스크(안 단위)만 조정. 이탈안 기준 = min(crop_W, crop_H) < thr.'),
           'seed': SEED, 'n_boot': N_BOOT, 'thresholds': THRESHOLDS,
           'exclusions': excl, 'passes': {}}
    for pk, tree in TREES.items():
        print(f'=== 패스 {pk} ===', flush=True)
        res['passes'][pk] = run_tree(pk, tree, sizes)

    print('=== (3) 방향 확인 (thr335, OOF) ===', flush=True)
    res['direction'] = {pk: direction_block(pk, tree, sizes)
                        for pk, tree in TREES.items()}

    p = REPO / 'experiments/gca_crop/crop_sensitivity.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n저장: {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
