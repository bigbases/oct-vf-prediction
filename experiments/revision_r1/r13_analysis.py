#!/usr/bin/env python3
"""R1-3 analysis (plan: docs/revision_r1/prereg_R1-3_R1-4.md §2): the submitted analyses
re-run with every image branch in the right-eye (mirrored) frame.

The submitted computations are reused, not rewritten: fusion_eval_common (loading, weight
fitting, per-eye metrics) and fusion_sensitivity_trend (bin RMSE, slopes, clustered CI).
Only the map from backbone name to prediction directory changes.

Guards, run first on the submitted (unmirrored) directories, must reproduce frozen values:
  forest_rows.json        fusion - summary per-eye RMSE delta and p, five backbones
  ensemble_decisive       ensemble fusion OOF pooled RMSE 7.963 (nested w)
  fusion_sensitivity_trend / bias_by_bin   IR-v2 and ensemble OOF slopes

Endpoint 4 (mirrored vs unmirrored) uses the submitted run as the unmirrored reference for
IR-v2 and Inception-v3, and the matched current-code rerun for VGG16, Xception, DenseNet121
(the submitted runs of those three are not reproducible; decision log of the plan).

Output: r13_analysis.json (no patient keys). Missing runs are reported, not guessed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))
sys.path.insert(0, str(HERE))

import fusion_eval_common as FE          # noqa: E402
import fusion_sensitivity_trend as FST   # noqa: E402
import calibration_multiplicity as CM    # noqa: E402

# A clone keeps the frozen results at paper/results_frozen; the authors' working tree keeps
# this repository under release/ (same fallback as centred_irv2.py).
FROZEN = next(p for p in (REPO / 'paper/results_frozen', REPO / 'release/paper/results_frozen') if p.exists())
SUBMITTED = dict(FE.BB)
MIRRORED = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold_osflip',
    'Inception-v3': HERE / 'runs/r13_osflip_inception_v3',
    'VGG16': HERE / 'runs/r13_osflip_vgg16',
    'Xception': HERE / 'runs/r13_osflip_xception',
    'DenseNet121': HERE / 'runs/r13_osflip_densenet121',
}
UNMIRRORED_REF = {
    'IR-v2': SUBMITTED['IR-v2'], 'Inception-v3': SUBMITTED['Inception-v3'],
    'VGG16': HERE / 'runs/r13_noflip_vgg16', 'Xception': HERE / 'runs/r13_noflip_xception',
    'DenseNet121': HERE / 'runs/r13_noflip_densenet121',
}


def complete(d):
    d = PASS_B / d
    return all((d / f'{s}_preds_fold{k}.npz').exists() for k in range(5) for s in ('val', 'test'))


_cache = {}


def Wboot(n):
    if n not in _cache:
        _cache[n] = CM.boot_weights(n, np.random.default_rng(CM.SEED))
    return _cache[n]


def endpoints(bbmap):
    FE.BB.clear(); FE.BB.update(bbmap)
    oof, tst, tl, tm = FE.load_all(ensemble=True)
    tm = tm.astype(bool)
    res = {'n_eyes_oof': int(sum(len(oof[k]['keys']) for k in range(5))), 'n_eyes_test': len(tst[0]['keys']),
           'fusion_vs_summary': [], 'ceiling': [], 'trend': []}
    # 1. per backbone, Table 3 convention (w fitted on all OOF folds, applied to OOF and held-out)
    for nm in FE.NAMES:
        w = FE.fit_w([oof[k] for k in range(5)], nm)
        lab = np.concatenate([oof[k]['lab'] for k in range(5)])
        msk = np.concatenate([oof[k]['mask'] for k in range(5)]).astype(bool)
        x = np.concatenate([oof[k]['xgb'] for k in range(5)])
        c = np.concatenate([oof[k]['cnns'][nm] for k in range(5)])
        pid = np.array([str(kk[0]) for k in range(5) for kk in oof[k]['keys']], dtype=object)
        f = w * x + (1 - w) * c
        for kind in ('rmse', 'mae'):
            r = CM.paired(FE.eye_metric(f, lab, msk, kind), FE.eye_metric(x, lab, msk, kind), pid, Wboot)
            r.update(backbone=nm, split='OOF', metric=kind, w=w,
                     pooled_rmse_fusion=CM.pooled_rmse(f, lab, msk), pooled_rmse_cnn=CM.pooled_rmse(c, lab, msk))
            res['fusion_vs_summary'].append(r)
            tf, tx, tc, tpid = FE.test_arrays(oof, tst, tl, tm, nm, kind)
            r = CM.paired(tf, tx, tpid, Wboot)
            r.update(backbone=nm, split='TEST', metric=kind, w=w)
            res['fusion_vs_summary'].append(r)
    # 2. ensemble ceiling, nested w (OOF) / all-OOF w (held-out)
    for kind in ('rmse', 'mae'):
        f, x, c, pid = FE.oof_arrays(oof, FE.ENSEMBLE, kind)
        r = CM.paired(f, c, pid, Wboot); r.update(split='OOF', metric=kind, comparison='ensemble fusion - ensemble')
        res['ceiling'].append(r)
        tf, tx, tc, tpid = FE.test_arrays(oof, tst, tl, tm, FE.ENSEMBLE, kind)
        r = CM.paired(tf, tc, tpid, Wboot); r.update(split='TEST', metric=kind, comparison='ensemble fusion - ensemble')
        res['ceiling'].append(r)
    fo, xo, co, lo, mo, _ = FE.oof_raw(oof, FE.ENSEMBLE)
    res['ensemble_fusion_oof_pooled_rmse'] = CM.pooled_rmse(fo, lo, mo.astype(bool))
    # 3. sensitivity-bin trend, as fusion_sensitivity_trend.py
    for nm in FE.NAMES + [FE.ENSEMBLE]:
        fo, xo, co, lo, mo, pid_o = FE.oof_raw(oof, nm)
        ft, xt, ct, _ = FE.test_raw(oof, tst, nm)
        pid_t = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)
        for split, (fp, cp, lb, mk, pd_) in (('OOF', (fo, co, lo, mo, pid_o)), ('TEST', (ft, ct, tl, tm, pid_t))):
            delta = FST.eye_bin_rmse(fp, lb, mk) - FST.eye_bin_rmse(cp, lb, mk)
            sl = FST.slopes(delta); ok = np.isfinite(sl)
            lo_ci, hi_ci = FST.cluster_boot_mean_ci(sl[ok], pd_[ok], np.random.default_rng(FST.SEED))
            res['trend'].append({'backbone': nm, 'split': split, 'n_eyes_with_slope': int(ok.sum()),
                                 'mean_slope': float(sl[ok].mean()), 'ci': [lo_ci, hi_ci],
                                 'p_wilcoxon': float(wilcoxon(sl[ok]).pvalue),
                                 'per_bin_mean_delta': [float(np.nanmean(delta[:, b])) for b in range(4)]})
    return res, oof, tst


def guards(sub):
    fr = {r['label'].replace('Primary: Inception-ResNet-v2', 'IR-v2'): r
          for r in json.load(open(FROZEN / 'forest_rows.json'))['rows'] if r['group'] in ('primary', 'backbone')}
    out = []
    for r in sub['fusion_vs_summary']:
        if r['split'] == 'OOF' and r['metric'] == 'rmse':
            ref = fr[r['backbone']]
            ok = abs(r['mean_delta'] - ref['delta']) < 1e-6 and abs(r['p_wilcoxon'] - ref['p_wilcoxon']) / ref['p_wilcoxon'] < 1e-6
            out.append({'what': f"fusion-summary {r['backbone']}", 'got': r['mean_delta'], 'want': ref['delta'], 'ok': bool(ok)})
    out.append({'what': 'ensemble fusion OOF pooled RMSE', 'got': sub['ensemble_fusion_oof_pooled_rmse'], 'want': 7.9629,
                'ok': abs(sub['ensemble_fusion_oof_pooled_rmse'] - 7.9629) < 5e-4})
    tr = {(t['backbone'], t['split']): t for t in sub['trend']}
    for key, want in ((('IR-v2', 'OOF'), 0.3366640013158698), (('ENSEMBLE', 'OOF'), 0.23037689511302756)):
        out.append({'what': f'trend slope {key}', 'got': tr[key]['mean_slope'], 'want': want,
                    'ok': abs(tr[key]['mean_slope'] - want) < 1e-6})
    for g in out:
        print(f"guard {g['what']:40s} got {g['got']:.6f} want {g['want']:.6f} {'ok' if g['ok'] else 'FAIL'}")
    if not all(g['ok'] for g in out):
        raise SystemExit('guard failed; nothing computed')
    return out


def cnn_vs(ref_map, new_map):
    """Endpoint 4: per-eye RMSE of the mirrored CNN minus the unmirrored reference."""
    out = []
    for nm in FE.NAMES:
        if not (complete(ref_map[nm]) and complete(new_map[nm])):
            out.append({'backbone': nm, 'status': 'run not complete'}); continue
        a = dict(SUBMITTED); a.update({n: new_map[n] if n == nm else SUBMITTED[n] for n in FE.NAMES})
        b = dict(SUBMITTED); b.update({n: ref_map[n] if n == nm else SUBMITTED[n] for n in FE.NAMES})
        vals = {}
        for tag, m in (('new', a), ('ref', b)):
            FE.BB.clear(); FE.BB.update(m)
            oof, tst, tl, tm = FE.load_all()
            lab = np.concatenate([oof[k]['lab'] for k in range(5)])
            msk = np.concatenate([oof[k]['mask'] for k in range(5)]).astype(bool)
            c = np.concatenate([oof[k]['cnns'][nm] for k in range(5)])
            pid = np.array([str(kk[0]) for k in range(5) for kk in oof[k]['keys']], dtype=object)
            vals[tag] = (FE.eye_metric(c, lab, msk, 'rmse'), pid, CM.pooled_rmse(c, lab, msk))
        r = CM.paired(vals['new'][0], vals['ref'][0], vals['new'][1], Wboot)
        r.update(backbone=nm, split='OOF', metric='rmse', pooled_new=vals['new'][2], pooled_ref=vals['ref'][2])
        out.append(r)
    return out


def finite(o):
    """NaN -> null (a p-value is undefined when every paired difference is zero), so the file is strict JSON."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def main():
    sub, *_ = endpoints(SUBMITTED)
    g = guards(sub)
    doc = {'note': 'R1-3: submitted analyses in the mirrored (right-eye) frame. Plan: '
                   'docs/revision_r1/prereg_R1-3_R1-4.md section 2.', 'guards': g, 'submitted': sub}
    missing = [nm for nm, d in MIRRORED.items() if not complete(d)]
    doc['mirrored_runs_missing'] = missing
    if not missing:
        mir, *_ = endpoints(MIRRORED)
        doc['mirrored'] = mir
    doc['mirrored_vs_unmirrored_cnn'] = cnn_vs(UNMIRRORED_REF, MIRRORED)
    doc['rerun_vs_submitted_cnn'] = cnn_vs(SUBMITTED, UNMIRRORED_REF)
    FE.BB.clear(); FE.BB.update(SUBMITTED)
    (HERE / 'r13_analysis.json').write_text(json.dumps(finite(doc), indent=1, default=float, allow_nan=False) + '\n')
    print('mirrored runs missing:', missing or 'none')


if __name__ == '__main__':
    import phi_safe
    phi_safe.install()
    main()
