"""Is the GCA sector laterality convention applied consistently in BOTH eyes?

Measurement only -- reads the cached _measurements.npy from validate.py and recomputes
nothing. Splits the same paired (mask, Cirrus) values by eye.

A convention that is merely misnamed relative to anatomy holds equally in OD and OS:
the columns still point at one fixed screen side per eye, so both halves fit well. A
mirroring defect of the rnfl_q_t kind holds in one eye and collapses in the other.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import validate as V

recs = list(np.load(os.path.join(HERE, '_measurements.npy'), allow_pickle=True))


def pairs(rec_subset, kind, variant, **kw):
    if kind == 'gca':
        return V.collect_gca(rec_subset, variant, kw['convention'])
    return V.collect_rnfl(rec_subset, variant, kw['band'])


def row(name, pred, ref):
    p, r = np.asarray(pred, float), np.asarray(ref, float)
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    if len(p) < 10:
        return f'{name:10s} n={len(p):4d}  (insufficient)'
    b, a = np.polyfit(r, p, 1)
    d = p - r
    return (f'{name:10s} {len(p):4d} {np.corrcoef(p, r)[0, 1]:6.3f} {b:7.3f} '
            f'{a:+8.2f} {np.abs(d).mean():7.2f} {d.mean():+8.2f} {r.std(ddof=1):7.2f}')


HEAD = f'{"param":10s} {"n":>4s} {"r":>6s} {"slope":>7s} {"icpt":>8s} {"MAE":>7s} {"bias":>8s} {"refSD":>7s}'


def block(title, params, get):
    print('\n' + title)
    for eye in ('OD', 'OS'):
        sub = [r for r in recs if r['eye'] == eye]
        acc = get(sub)
        print(f'-- {eye}  ({len(sub)} eyes)')
        print('   ' + HEAD)
        for k in params:
            print('   ' + row(k, *acc[k]))


# GCA -- scale is fitted per split so a k difference cannot masquerade as a fit failure
def gca_get(conv):
    def f(sub):
        acc = V.collect_gca(sub, 'b', conv)
        k = V.gca_global_scale(acc)
        print(f'   global scale k = {k:.1f} um/unit')
        return {kk: ([v * k for v in pv], rv) for kk, (pv, rv) in acc.items()}
    return f


block('=== GCA 6 sectors, variant (b) detected fovea centre, OCR label convention ===',
      V.SECTORS + ['avg_gcl'], gca_get('ocr'))
block('=== GCA 6 sectors, same but ANATOMICAL label convention (contrast) ===',
      V.SECTORS + ['avg_gcl'], gca_get('anatomical'))
block('=== RNFL quadrants, variant (b) detected disc centre, band 0.20 mm ===',
      V.QUADS + ['avg_rnfl'], lambda sub: V.collect_rnfl(sub, 'b', 0.20))
