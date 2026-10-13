#!/usr/bin/env python3
"""R1-4 analysis (plan: docs/revision_r1/prereg_R1-3_R1-4.md §3).

Arms: CNN 0 submitted IR-v2 / 1 exclusion / 2 censored; XGBoost 0 submitted / 1 exclusion.
Eyes, folds and the location mask are those of the submitted analysis
(fusion_eval_common.load_all: 240 OOF eyes, 37 held-out). The mask stored by the exclusion
run is not used. Arms whose predictions do not exist yet are skipped and reported as such.

Guards (arm 0 must reproduce neg1_mask_sensitivity.json, non-floor cells):
  OOF  fusion 6.3327, CNN 6.7669, XGB 7.1605;  held-out  fusion 6.4404, CNN 6.0834, XGB 7.6697
and, on all cells, the submitted main comparison (fusion - summary, per-eye RMSE, OOF) of
forest_rows.json: delta, 95% CI and Wilcoxon p.

Comparisons marked added=True are not in the plan: fusion vs summary within each floor
rule (arm 0 on non-floor cells as the reference), added on 2026-10-13 at the authors'
request after the arm-1 results had been seen.

Output: r14_analysis.json (no patient keys).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))
sys.path.insert(0, str(HERE))

import fusion_eval_common as FE                       # noqa: E402
from oof_common import load_oof_npz                   # noqa: E402
from floor_cells import floor_cells                   # noqa: E402
import calibration_multiplicity as CM                 # noqa: E402
import phi_safe                                       # noqa: E402

W = 0.47
CNN_DIRS = {0: PASS_B / 'runs/phasec_b0_inception_resnet_v2_5fold',
            1: HERE / 'runs/r14_exclude_irv2', 2: HERE / 'runs/r14_censored_irv2'}
XGB_DIRS = {0: PASS_B / 'runs/oof', 1: HERE / 'runs/r14_xgb/arm1'}
FROZEN = REPO / 'release/paper/results_frozen'
GUARD = {('OOF', 'fusion'): 6.3327, ('OOF', 'cnn'): 6.7669, ('OOF', 'xgb'): 7.1605,
         ('TEST', 'fusion'): 6.4404, ('TEST', 'cnn'): 6.0834, ('TEST', 'xgb'): 7.6697}


def have(d, kind):
    pat = 'xgb_90d_fold{k}_{s}.npz' if kind == 'xgb' else '{s}_preds_fold{k}.npz'
    return all((d / pat.format(k=k, s=s)).exists() for k in range(5) for s in ('val', 'test'))


def aligned(d, kind, keys, split, k):
    s = 'val' if split == 'OOF' else 'test'
    z = load_oof_npz(d / (f'xgb_90d_fold{k}_{s}.npz' if kind == 'xgb' else f'{s}_preds_fold{k}.npz'))
    idx = {kk: i for i, kk in enumerate(z['keys'])}
    return z['pred'][phi_safe.lookup(idx, keys, f'{d.name} {s} fold {k}')]


def assemble():
    oof, tst, test_lab, test_mask = FE.load_all()
    floor, _ = floor_cells(current=PASS_B / 'ml_final_90d_excl_empty_flip.csv')
    cnn_arms = [a for a, d in CNN_DIRS.items() if have(d, 'cnn')]
    xgb_arms = [a for a, d in XGB_DIRS.items() if have(d, 'xgb')]
    D = {}
    for split in ('OOF', 'TEST'):
        if split == 'OOF':
            keys = [kk for k in range(5) for kk in oof[k]['keys']]
            lab = np.concatenate([oof[k]['lab'] for k in range(5)])
            mask = np.concatenate([oof[k]['mask'] for k in range(5)]).astype(bool)
            get = lambda d, kind: np.concatenate([aligned(d, kind, oof[k]['keys'], 'OOF', k) for k in range(5)])
        else:
            keys = list(tst[0]['keys'])
            lab, mask = test_lab, test_mask.astype(bool)
            get = lambda d, kind: np.mean([aligned(d, kind, keys, 'TEST', k) for k in range(5)], 0)
        fl = np.stack(phi_safe.lookup(floor, [tuple(kk) for kk in keys], 'floor cells')) & mask
        pid = np.array([str(kk[0]) for kk in keys], dtype=object)
        D[split] = dict(lab=lab, mask=mask, floor=fl, pid=pid,
                        cnn={a: get(CNN_DIRS[a], 'cnn') for a in cnn_arms},
                        xgb={a: get(XGB_DIRS[a], 'xgb') for a in xgb_arms})
    return D, cnn_arms, xgb_arms


def pooled(p, l, m):
    d = (p - l)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def main():
    D, cnn_arms, xgb_arms = assemble()
    out = {'note': 'R1-4 floor-label analysis; non-floor = cells not recorded below 0 dB.',
           'arms_available': {'cnn': cnn_arms, 'xgb': xgb_arms}, 'w_fixed': W, 'guards': [], 'splits': {}}
    for split, d in D.items():
        nf = d['mask'] & ~d['floor']
        g = {'cnn': d['cnn'][0], 'xgb': d['xgb'][0], 'fusion': W * d['xgb'][0] + (1 - W) * d['cnn'][0]}
        for name, p in g.items():
            got = pooled(p, d['lab'], nf)[0]
            ok = abs(got - GUARD[(split, name)]) <= 5e-4
            out['guards'].append({'split': split, 'branch': name, 'got': round(got, 4),
                                  'want': GUARD[(split, name)], 'ok': ok})
            print(f"guard {split} {name}: {got:.4f} vs {GUARD[(split, name)]} {'ok' if ok else 'FAIL'}")
    rng_cache = {}

    def Wboot(n):
        if n not in rng_cache:
            rng_cache[n] = CM.boot_weights(n, np.random.default_rng(CM.SEED))
        return rng_cache[n]

    ref = next(r for r in json.load(open(FROZEN / 'forest_rows.json'))['rows'] if r['group'] == 'primary')
    d = D['OOF']
    fus = W * d['xgb'][0] + (1 - W) * d['cnn'][0]
    r = CM.paired(FE.eye_metric(fus, d['lab'], d['mask'], 'rmse'), FE.eye_metric(d['xgb'][0], d['lab'], d['mask'], 'rmse'),
                  d['pid'], Wboot)
    for name, got, want in (('delta', r['mean_delta'], ref['delta']), ('ci_lo', r['ci'][0], ref['ci_lo']),
                            ('ci_hi', r['ci'][1], ref['ci_hi']), ('p_wilcoxon', r['p_wilcoxon'], ref['p_wilcoxon'])):
        ok = abs(got - want) <= 1e-6 * max(1.0, abs(want)) if name != 'p_wilcoxon' else abs(got / want - 1) <= 1e-6
        out['guards'].append({'split': 'OOF', 'branch': f'main comparison all cells {name}', 'got': got, 'want': want, 'ok': ok})
        print(f"guard OOF main comparison {name}: {got:.6g} vs {want:.6g} {'ok' if ok else 'FAIL'}")
    if not all(x['ok'] for x in out['guards']):
        raise SystemExit('guard failed; nothing computed')

    for split, d in D.items():
        lab, m, fl, pid = d['lab'], d['mask'], d['floor'], d['pid']
        nf = m & ~fl
        branches = {f'cnn{a}': d['cnn'][a] for a in cnn_arms}
        branches.update({f'xgb{a}': d['xgb'][a] for a in xgb_arms})
        for a in cnn_arms:
            for b in xgb_arms:
                branches[f'fusion_c{a}x{b}'] = W * d['xgb'][b] + (1 - W) * d['cnn'][a]
        rec = {'n_eyes': int(m.shape[0]), 'cells': int(m.sum()), 'floor_cells': int(fl.sum()), 'branches': {}}
        low = m & (lab < 10.0)
        for name, p in branches.items():
            r_nf, a_nf = pooled(p, lab, nf)
            r_all, a_all = pooled(p, lab, m)
            rec['branches'][name] = {
                'nonfloor_rmse': r_nf, 'nonfloor_mae': a_nf, 'all_rmse': r_all, 'all_mae': a_all,
                'lowest_bin_bias_rule_applied': float((p - lab)[low].mean()),
                'lowest_bin_bias_floor_excluded': float((p - lab)[low & ~fl].mean()),
                'mean_prediction_at_floor_cells': float(p[fl].mean()),
            }
        comps = []
        pairs = [('fusion_c1x1', 'fusion_c0x0', True, False), ('cnn1', 'cnn0', False, False),
                 ('xgb1', 'xgb0', False, False), ('cnn2', 'cnn0', False, False), ('cnn2', 'cnn1', False, False),
                 ('fusion_c2x1', 'fusion_c0x0', False, False),
                 ('fusion_c0x0', 'xgb0', False, True), ('fusion_c1x1', 'xgb1', False, True),
                 ('fusion_c2x1', 'xgb1', False, True)]
        for a, b, primary, added in pairs:
            if a in branches and b in branches:
                for kind in ('rmse', 'mae'):
                    ea = FE.eye_metric(branches[a], lab, nf, kind)
                    eb = FE.eye_metric(branches[b], lab, nf, kind)
                    r = CM.paired(ea, eb, pid, Wboot)
                    r.update(a=a, b=b, metric=f'per-eye {kind} on non-floor cells',
                             primary=bool(primary and kind == 'rmse' and split == 'OOF'), added=added)
                    comps.append(r)
            else:
                comps.append({'a': a, 'b': b, 'status': 'arm not available yet', 'added': added})
        rec['comparisons'] = comps
        out['splits'][split] = rec
    (HERE / 'r14_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('arms available:', out['arms_available'])


if __name__ == '__main__':
    phi_safe.install()
    main()
