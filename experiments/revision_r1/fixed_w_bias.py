#!/usr/bin/env python3
"""Mean residual of the IR-v2 fusion at the fixed weight of Table 3 (w fitted once on all
out-of-fold folds, = 0.47), out-of-fold - the quantity the limitations paragraph of the
submitted manuscript describes (printed there as "+0.8 dB", which no artefact holds; see
docs/revision_r1_manuscript_changes.md M2). calibration_multiplicity.json holds the
out-of-fold fusion only with nested w.

Same branches, cells and clustered bootstrap as calibration_multiplicity.py (imported).
Guards: Summary and nested-w Fusion out-of-fold mean bias equal calibration_multiplicity.json.
Output: fixed_w_bias.json (no patient keys).
"""
import json
from pathlib import Path

import numpy as np

import calibration_multiplicity as CM

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fixed_w_bias.json'


def main():
    B, data, meta = CM.assemble()
    oof, *_ = CM.FE.load_all(ensemble=True)
    w = meta['w_fusion_test']
    x = np.concatenate([oof[k]['xgb'] for k in range(5)])
    c = np.concatenate([oof[k]['cnns']['IR-v2'] for k in range(5)])
    lab, m, pid = data['OOF']
    branches = {'Summary': B['OOF']['Summary'], 'Fusion nested w': B['OOF']['Fusion'],
                'Fusion fixed w': w * x + (1 - w) * c}
    _, pinv = CM.patient_index(pid)
    n_pat = pinv.max() + 1
    Wb = CM.boot_weights(n_pat, np.random.default_rng(CM.SEED))
    rec, boot = {}, {}
    for name, p in branches.items():
        S = CM.suff(p, lab, m, pinv, n_pat)
        bias = CM.calib_stats(S.sum(0))[0]
        bb = CM.calib_stats(Wb @ S)[0]
        rec[name] = {'mean_bias': float(bias), 'mean_bias_ci': CM.ci(bb)}
        boot[name] = bb
    frozen = json.load(open(HERE / 'calibration_multiplicity.json'))['calibration']['OOF']
    guards = []
    for name, ref in (('Summary', 'Summary'), ('Fusion nested w', 'Fusion')):
        want = frozen[ref]['all']['mean_bias']
        ok = abs(rec[name]['mean_bias'] - want) <= 1e-6
        guards.append({'branch': name, 'got': rec[name]['mean_bias'], 'want': want, 'ok': ok})
        print(f"guard OOF {name} mean bias {rec[name]['mean_bias']:.6f} vs {want:.6f} {'ok' if ok else 'FAIL'}")
    if not all(g['ok'] for g in guards):
        raise SystemExit('guard failed; nothing written')
    out = {'note': 'IR-v2 fusion mean residual (prediction - label, dB) out-of-fold at the fixed '
                   'Table 3 weight; differences are fusion - Summary. Same cells and clustered '
                   'bootstrap (5000, seed 42) as calibration_multiplicity.json.',
           'split': 'OOF', 'w_fixed': w, 'n_cells': int(m.sum()), 'guards': guards, 'branches': rec,
           'fixed_minus_summary': {'mean': rec['Fusion fixed w']['mean_bias'] - rec['Summary']['mean_bias'],
                                   'ci': CM.ci(boot['Fusion fixed w'] - boot['Summary'])},
           'nested_minus_summary': {'mean': rec['Fusion nested w']['mean_bias'] - rec['Summary']['mean_bias'],
                                    'ci': CM.ci(boot['Fusion nested w'] - boot['Summary'])}}
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('w_fixed', 'branches', 'fixed_minus_summary', 'nested_minus_summary')}, indent=1))


if __name__ == '__main__':
    import phi_safe
    phi_safe.install()
    main()
