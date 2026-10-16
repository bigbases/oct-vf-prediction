#!/usr/bin/env python3
"""R1-5 addendum: the centred single-backbone image branch (Inception-ResNet-v2) as a
comparator. Plan: docs/revision_r1/prereg_R1-5_R1-6.md, addendum 2026-10-16.

Same inputs, centring rule, bootstrap resamples and tests as calibration_multiplicity.py,
whose functions are reused. Guards, each stopping the script on a mismatch: the branch-level
guards of calibration_multiplicity.py; the centring rule, applied to the ensemble,
reproduces the frozen centring constants; every calibration and comparison value of the
frozen calibration_multiplicity.json is reproduced exactly before the new branch is added.
No model is trained or refitted.

Output: centred_irv2.json next to this file (no patient keys).
  HVF_PASS_B=<pass-B tree> python centred_irv2.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import calibration_multiplicity as CM

HERE = Path(__file__).resolve().parent
FROZEN = Path(os.environ.get('HVF_FROZEN', next(
    p for p in (CM.REPO / 'paper/results_frozen', CM.REPO / 'release/paper/results_frozen') if p.exists())))
OUT = HERE / 'centred_irv2.json'
IRV2, NEW = 'IR-v2', 'Centred IR-v2'
PAIRS = (('Fusion', 'Centred ensemble'), ('Ensemble fusion', 'Centred ensemble'), ('Fusion', NEW))
SPLITS = ('OOF', 'TEST')


def const_global(folds, name):
    """calibration_multiplicity.const_global for any image branch."""
    r = np.concatenate([(f['cnns'][name] - f['lab'])[f['mask'].astype(bool)] for f in folds])
    return float(r.mean())


def plain(x):
    return json.loads(json.dumps(x, default=float))


def add_branch(B, meta):
    raw, _, _, _ = CM.FE.load_all(ensemble=True)
    oof = [raw[k] for k in range(5)]
    c_ens = [const_global([oof[i] for i in range(5) if i != k], CM.ENS) for k in range(5)]
    c_ens_test = const_global(oof, CM.ENS)
    assert c_ens == meta['centring_const_by_fold'] and c_ens_test == meta['centring_const_test'], \
        'centring rule does not reproduce the ensemble constants'
    c = [const_global([oof[i] for i in range(5) if i != k], IRV2) for k in range(5)]
    c_test = const_global(oof, IRV2)
    irv2 = np.concatenate([oof[k]['cnns'][IRV2] for k in range(5)])
    assert np.array_equal(irv2, B['OOF'][IRV2]), 'fold order differs from calibration_multiplicity'
    B['OOF'][NEW] = np.concatenate([oof[k]['cnns'][IRV2] - c[k] for k in range(5)])
    B['TEST'][NEW] = B['TEST'][IRV2] - c_test
    return {'centring_const_irv2_by_fold': c, 'centring_const_irv2_test': c_test}


def wp_factory():
    """calibration_multiplicity.main's resampler: one draw per patient count, in call order."""
    r = np.random.default_rng(CM.SEED)
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = CM.boot_weights(n, r)
        return cache[n]
    return get


def pooled(B, data):
    out = {}
    for s in SPLITS:
        lab, m, _ = data[s]
        out[s] = {}
        for n in ('Summary', IRV2, NEW, 'Fusion'):
            d = (B[s][n] - lab)[m].astype(float)
            out[s][n] = {'rmse': float(np.sqrt(np.mean(d ** 2))), 'mae': float(np.mean(np.abs(d)))}
    return out


def main():
    frozen = json.loads((FROZEN / 'calibration_multiplicity.json').read_bytes())
    B, data, meta = CM.assemble()
    g = CM.guards(B, data)
    consts = add_branch(B, meta)

    rng = np.random.default_rng(CM.SEED)
    W = {s: CM.boot_weights(CM.patient_index(data[s][2])[1].max() + 1, rng) for s in SPLITS}
    cal = plain(CM.calibration(B, data, W))
    for s in SPLITS:
        for br, rec in frozen['calibration'][s].items():
            assert cal[s][br] == rec, f'calibration {s} {br} not reproduced'
    g.append({'split': 'both', 'branch': 'calibration_multiplicity.calibration, all branches and strata',
              'ok': True, 'rule': 'exact equality'})

    Wp = wp_factory()
    cmp = {}
    for s in SPLITS:
        lab, m, pid = data[s]
        rows = []
        for a, b in PAIRS:
            for kind in ('rmse', 'mae'):
                r = CM.paired(CM.eye_metric(B[s][a], lab, m, kind), CM.eye_metric(B[s][b], lab, m, kind), pid, Wp)
                r.update(a=a, b=b, metric=kind, convention='delta = per-eye a - b; negative favours a')
                rows.append(plain(r))
        for r, old in zip(rows, frozen['centred_ensemble_comparisons'][s]):
            assert r == {k: v for k, v in old.items() if k != 'p_holm'}, f'comparison {s} {r["a"]} not reproduced'
        for r, adj in zip(rows, CM.holm([r['p_wilcoxon'] for r in rows])):
            r['p_holm'] = adj
        cmp[s] = rows
    g.append({'split': 'both', 'branch': 'calibration_multiplicity.centred_ensemble_comparisons, '
              'all values except p_holm', 'ok': True, 'rule': 'exact equality'})

    doc = {
        'note': 'R1-5 addendum: Inception-ResNet-v2 image branch centred by the rule of the centred '
                'ensemble, its calibration, and fusion vs the centred branch. Plan: '
                'docs/revision_r1/prereg_R1-5_R1-6.md, addendum 2026-10-16. Pass B predictions, no '
                'refitting; same bootstrap resamples as calibration_multiplicity.json.',
        'pass': 'B', 'n_boot': CM.N_BOOT, 'seed': CM.SEED,
        'centring': consts,
        'guards': g,
        'calibration': {s: {NEW: cal[s][NEW]} for s in SPLITS},
        'comparisons': cmp,
        'holm_family': 'six tests per split, in list order: fusion and ensemble fusion vs the centred '
                       'ensemble (as frozen in calibration_multiplicity.json, where the family is these '
                       'four) and fusion vs the centred Inception-ResNet-v2 branch, each on RMSE and MAE',
        'pooled': pooled(B, data),
    }
    OUT.write_text(json.dumps(doc, indent=1, default=float) + '\n')
    print(f'wrote {OUT.name}')


if __name__ == '__main__':
    import phi_safe
    phi_safe.install()
    main()
