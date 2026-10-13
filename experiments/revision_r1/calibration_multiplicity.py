#!/usr/bin/env python3
"""R1-5 (calibration, centred ensemble as comparator) and R1-6 (Holm control for the
sensitivity-bin comparisons). Plan: docs/revision_r1/prereg_R1-5_R1-6.md.

Reads only the submitted pass-B predictions through fusion_eval_common.load_all (the
loader behind the frozen fusion results). No model is trained or refitted. Guards: the
branch-level numbers that are already frozen must be reproduced before anything new is
computed; a mismatch stops the script.

Output: calibration_multiplicity.json next to this file (no patient keys).
  HVF_PASS_B=<pass-B tree> python calibration_multiplicity.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))

import fusion_eval_common as FE          # noqa: E402
import fusion_sensitivity_trend as FST   # noqa: E402

assert FE.ROOT.resolve() == PASS_B.resolve(), FE.ROOT
N_BOOT, SEED = 5000, 42
EDGES = [0.0, 10.0, 20.0, 30.0, np.inf]
BINS = ['[0,10)', '[10,20)', '[20,30)', '[30,inf)']
ENS = FE.ENSEMBLE
BACKBONES = FE.NAMES
OUT = HERE / 'calibration_multiplicity.json'


# ------------------------------------------------------------------ assemble branches
def const_global(folds):
    r = np.concatenate([(f['cnns'][ENS] - f['lab'])[f['mask'].astype(bool)] for f in folds])
    return float(r.mean())


def assemble():
    oof, tst, test_lab, test_mask = FE.load_all(ensemble=True)
    B = {s: {} for s in ('OOF', 'TEST')}
    meta = {'w_fusion_nested': [], 'w_ensfusion_nested': [], 'centring_const_by_fold': []}
    parts = {n: [] for n in ['Summary'] + BACKBONES + ['Ensemble', 'Centred ensemble', 'Fusion',
                                                        'Ensemble fusion']}
    lab, mask, pid = [], [], []
    for k in range(5):
        tr = [oof[i] for i in range(5) if i != k]
        f = oof[k]
        w1 = FE.fit_w(tr, 'IR-v2')
        w2 = FE.fit_w(tr, ENS)
        c = const_global(tr)
        meta['w_fusion_nested'].append(w1)
        meta['w_ensfusion_nested'].append(w2)
        meta['centring_const_by_fold'].append(c)
        parts['Summary'].append(f['xgb'])
        for n in BACKBONES:
            parts[n].append(f['cnns'][n])
        parts['Ensemble'].append(f['cnns'][ENS])
        parts['Centred ensemble'].append(f['cnns'][ENS] - c)
        parts['Fusion'].append(w1 * f['xgb'] + (1 - w1) * f['cnns']['IR-v2'])
        parts['Ensemble fusion'].append(w2 * f['xgb'] + (1 - w2) * f['cnns'][ENS])
        lab.append(f['lab']); mask.append(f['mask'].astype(bool))
        pid += [str(kk[0]) for kk in f['keys']]
    B['OOF'] = {n: np.concatenate(v) for n, v in parts.items()}
    oof_lab, oof_mask, oof_pid = np.concatenate(lab), np.concatenate(mask), np.array(pid, dtype=object)

    allf = [oof[k] for k in range(5)]
    w1, w2, c = FE.fit_w(allf, 'IR-v2'), FE.fit_w(allf, ENS), const_global(allf)
    meta.update(w_fusion_test=w1, w_ensfusion_test=w2, centring_const_test=c)
    tx = np.mean([tst[k]['xgb'] for k in range(5)], 0)
    tc = {n: np.mean([tst[k]['cnns'][n] for k in range(5)], 0) for n in BACKBONES + [ENS]}
    B['TEST'] = {'Summary': tx, **{n: tc[n] for n in BACKBONES}, 'Ensemble': tc[ENS],
                 'Centred ensemble': tc[ENS] - c,
                 'Fusion': w1 * tx + (1 - w1) * tc['IR-v2'],
                 'Ensemble fusion': w2 * tx + (1 - w2) * tc[ENS]}
    test_pid = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)
    data = {'OOF': (oof_lab, oof_mask, oof_pid), 'TEST': (test_lab, test_mask.astype(bool), test_pid)}
    return B, data, meta


# ---------------------------------------------------------------------------- guards
def pooled_rmse(p, l, m):
    d = (p - l)[m]
    return float(np.sqrt(np.mean(d ** 2)))


def guards(B, data):
    want = [('OOF', 'Summary', 8.6603), ('OOF', 'IR-v2', 8.5397), ('OOF', 'Ensemble', 8.1147),
            ('OOF', 'Ensemble fusion', 7.9629), ('OOF', 'Centred ensemble', 7.9801),
            ('TEST', 'Summary', 8.9193), ('TEST', 'IR-v2', 8.7323), ('TEST', 'Fusion', 8.4105)]
    out = []
    for s, n, v in want:
        lab, m, _ = data[s]
        got = pooled_rmse(B[s][n], lab, m)
        out.append({'split': s, 'branch': n, 'got': round(got, 4), 'want': v, 'ok': abs(got - v) <= 5e-4})
    # per-bin delta of the submitted trend analysis (IR-v2, OOF, bin [0,10))
    lab, m, _ = data['OOF']
    fb = FST.eye_bin_rmse(B['OOF']['Fusion'], lab, m)
    cb = FST.eye_bin_rmse(B['OOF']['IR-v2'], lab, m)
    d0 = fb[:, 0] - cb[:, 0]
    d0 = d0[np.isfinite(d0)]
    out.append({'split': 'OOF', 'branch': 'per_bin[0,10) fusion-IRv2 mean delta', 'got': round(float(d0.mean()), 4),
                'want': -0.6202, 'ok': abs(d0.mean() - (-0.6202)) <= 5e-4 and d0.size == 103})
    bad = [g for g in out if not g['ok']]
    for g in out:
        print(f"guard {g['split']:4s} {g['branch']:40s} got {g['got']} want {g['want']} {'ok' if g['ok'] else 'FAIL'}")
    if bad:
        raise SystemExit('guard failed; nothing computed')
    return out


# ------------------------------------------------------------- clustered bootstrap
def patient_index(pid):
    u, inv = np.unique(pid, return_inverse=True)
    return u, inv


def boot_weights(n_pat, rng):
    """(N_BOOT, n_pat) multiplicity of each patient in each resample."""
    idx = rng.integers(0, n_pat, size=(N_BOOT, n_pat))
    W = np.zeros((N_BOOT, n_pat))
    for b in range(N_BOOT):
        W[b] = np.bincount(idx[b], minlength=n_pat)
    return W


def calib_stats(S):
    """S: (..., 5) sums n, Σŷ, Σy, Σŷ², Σŷy -> bias, slope, intercept."""
    n, sp, sy, spp, spy = (S[..., i] for i in range(5))
    mp, my = sp / n, sy / n
    cov = spy / n - mp * my
    var = spp / n - mp ** 2
    b = cov / var
    a = my - b * mp
    return mp - my, b, a


def suff(pred, lab, sel, pinv, n_pat):
    """Per-patient sufficient statistics over selected cells."""
    S = np.zeros((n_pat, 5))
    rows = np.repeat(pinv, pred.shape[1]).reshape(pred.shape)[sel]
    p, y = pred[sel].astype(float), lab[sel].astype(float)
    for i, v in enumerate((np.ones_like(p), p, y, p * p, p * y)):
        S[:, i] = np.bincount(rows, weights=v, minlength=n_pat)
    return S


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else [None, None]


def calibration(B, data, W):
    out = {}
    for s in ('OOF', 'TEST'):
        lab, m, pid = data[s]
        _, pinv = patient_index(pid)
        n_pat = pinv.max() + 1
        Ws = W[s]
        out[s] = {}
        for name, pred in B[s].items():
            rec = {}
            strata = [('all', m)]
            strata += [(f'measured {BINS[i]}', m & (lab >= EDGES[i]) & (lab < EDGES[i + 1])) for i in range(4)]
            strata += [(f'predicted {BINS[i]}', m & (pred >= EDGES[i]) & (pred < EDGES[i + 1])) for i in range(4)]
            for label, sel in strata:
                S = suff(pred, lab, sel, pinv, n_pat)
                tot = S.sum(0)
                if tot[0] < 10:
                    rec[label] = {'n_cells': int(tot[0])}
                    continue
                bias, slope, icpt = calib_stats(tot)
                bb = Ws @ S
                with np.errstate(invalid='ignore', divide='ignore'):
                    bbias, bslope, bicpt = calib_stats(bb)
                r = {'n_cells': int(tot[0]), 'mean_bias': float(bias), 'mean_bias_ci': ci(bbias)}
                if not label.startswith('measured'):
                    r.update(slope=float(slope), slope_ci=ci(bslope), intercept=float(icpt), intercept_ci=ci(bicpt))
                rec[label] = r
            out[s][name] = rec
    return out


# -------------------------------------------------------------------- paired tests
def eye_metric(p, l, m, kind):
    return FE.eye_metric(p, l, m, kind)


def paired(a, b, pid, Wp):
    """a, b per-eye values; delta = a - b. Clustered CI of the mean delta."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    _, pinv = patient_index(pid[ok])
    n_pat = pinv.max() + 1
    sd = np.bincount(pinv, weights=d, minlength=n_pat)
    sn = np.bincount(pinv, minlength=n_pat).astype(float)
    Wb = Wp(n_pat)
    with np.errstate(invalid='ignore', divide='ignore'):
        bm = (Wb @ sd) / (Wb @ sn)
    try:
        p = float(stats.wilcoxon(a[ok], b[ok])[1])
    except ValueError:
        p = float('nan')
    return {'n_eyes': int(ok.sum()), 'n_patients': int(n_pat), 'mean_delta': float(d.mean()),
            'median_delta': float(np.median(d)), 'ci': ci(bm), 'p_wilcoxon': p,
            'n_a_better': int(np.sum(d < 0))}


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    B, data, meta = assemble()
    g = guards(B, data)
    rng = np.random.default_rng(SEED)
    W = {}
    for s in ('OOF', 'TEST'):
        _, pinv = patient_index(data[s][2])
        W[s] = boot_weights(pinv.max() + 1, rng)

    def Wp_factory():
        r = np.random.default_rng(SEED)
        cache = {}

        def get(n):
            if n not in cache:
                cache[n] = boot_weights(n, r)
            return cache[n]
        return get
    Wp = Wp_factory()

    cal = calibration(B, data, W)

    # R1-5: fusion and ensemble fusion vs centred ensemble
    cen = {}
    for s in ('OOF', 'TEST'):
        lab, m, pid = data[s]
        rows = []
        for fus in ('Fusion', 'Ensemble fusion'):
            for kind in ('rmse', 'mae'):
                a = eye_metric(B[s][fus], lab, m, kind)
                b = eye_metric(B[s]['Centred ensemble'], lab, m, kind)
                r = paired(a, b, pid, Wp)
                r.update(a=fus, b='Centred ensemble', metric=kind,
                         convention='delta = per-eye a - b; negative favours a')
                rows.append(r)
        for r, adj in zip(rows, holm([r['p_wilcoxon'] for r in rows])):
            r['p_holm'] = adj
        cen[s] = rows

    # R1-6: per-bin fusion vs image, Holm within split over 8 tests
    pb = {}
    for s in ('OOF', 'TEST'):
        lab, m, pid = data[s]
        rows = []
        for fus, img in (('Fusion', 'IR-v2'), ('Ensemble fusion', 'Ensemble')):
            fb = FST.eye_bin_rmse(B[s][fus], lab, m)
            cb = FST.eye_bin_rmse(B[s][img], lab, m)
            for i in range(4):
                r = paired(fb[:, i], cb[:, i], pid, Wp)
                r.update(fusion=fus, image=img, bin=BINS[i], metric='per-eye RMSE within bin',
                         convention='delta = fusion - image; negative favours fusion')
                rows.append(r)
        for r, adj in zip(rows, holm([r['p_wilcoxon'] for r in rows])):
            r['p_holm'] = adj
        pb[s] = rows

    doc = {
        'note': 'R1-5 calibration and centred-ensemble comparisons, R1-6 Holm-adjusted bin-level '
                'comparisons. Plan: docs/revision_r1/prereg_R1-5_R1-6.md. Pass B predictions, '
                'no refitting.',
        'pass': 'B', 'n_boot': N_BOOT, 'seed': SEED, 'bin_edges': [0, 10, 20, 30, 'inf'],
        'min_pts_per_bin': FST.MIN_PTS,
        'weights_and_constants': meta,
        'guards': g,
        'calibration': cal,
        'centred_ensemble_comparisons': cen,
        'bin_level_comparisons': pb,
    }
    OUT.write_text(json.dumps(doc, indent=1, default=float) + '\n')
    print(f'wrote {OUT.name}')


if __name__ == '__main__':
    main()
