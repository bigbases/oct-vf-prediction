#!/usr/bin/env python3
"""Compact tables from calibration_multiplicity.json (read-only)."""
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'calibration_multiplicity.json'))
f = lambda x, n=2: '—' if x is None else f'{x:+.{n}f}'
c = lambda v, n=2: f'[{v[0]:+.{n}f}, {v[1]:+.{n}f}]' if v and v[0] is not None else ''
order = ['Summary', 'IR-v2', 'Inception-v3', 'VGG16', 'Xception', 'DenseNet121', 'Ensemble', 'Centred ensemble', 'Fusion', 'Ensemble fusion']
for s in ('OOF', 'TEST'):
    print(f'\n## calibration, all locations — {s}')
    print('| branch | mean bias | slope | intercept |')
    for n in order:
        r = d['calibration'][s][n]['all']
        print(f"| {n} | {f(r['mean_bias'])} {c(r['mean_bias_ci'])} | {r['slope']:.3f} {c(r['slope_ci'],3)} | {f(r['intercept'])} {c(r['intercept_ci'])} |")
    print(f'\n## mean bias by measured bin — {s}')
    print('| branch | ' + ' | '.join(['[0,10)', '[10,20)', '[20,30)', '[30,inf)']) + ' |')
    for n in order:
        rr = d['calibration'][s][n]
        print(f'| {n} | ' + ' | '.join(f(rr[f"measured {b}"].get("mean_bias"), 1) for b in ['[0,10)', '[10,20)', '[20,30)', '[30,inf)']) + ' |')
    print(f'\n## slope by predicted bin — {s}')
    print('| branch | ' + ' | '.join(['[0,10)', '[10,20)', '[20,30)', '[30,inf)']) + ' |')
    for n in order:
        rr = d['calibration'][s][n]
        cells = []
        for b in ['[0,10)', '[10,20)', '[20,30)', '[30,inf)']:
            x = rr[f'predicted {b}']
            cells.append(f"{x['slope']:.2f} (n={x['n_cells']})" if 'slope' in x else f"n={x['n_cells']}")
        print(f'| {n} | ' + ' | '.join(cells) + ' |')
for s in ('OOF', 'TEST'):
    print(f'\n## vs centred ensemble — {s}')
    print('| a | metric | mean Δ (a−b) | 95% CI | p | p Holm | a better |')
    for r in d['centred_ensemble_comparisons'][s]:
        print(f"| {r['a']} | {r['metric']} | {r['mean_delta']:+.3f} | {c(r['ci'],3)} | {r['p_wilcoxon']:.2g} | {r['p_holm']:.2g} | {r['n_a_better']}/{r['n_eyes']} |")
for s in ('OOF', 'TEST'):
    print(f'\n## bin-level fusion − image — {s}')
    print('| comparison | bin | eyes | mean Δ | 95% CI | p | p Holm |')
    for r in d['bin_level_comparisons'][s]:
        print(f"| {r['fusion']} − {r['image']} | {r['bin']} | {r['n_eyes']} | {r['mean_delta']:+.3f} | {c(r['ci'],3)} | {r['p_wilcoxon']:.2g} | {r['p_holm']:.2g} |")
w = d['weights_and_constants']
print('\nweights/constants:', {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in w.items()})
