"""Is the GCA sector naming a screen rule or an eye-dependent (anatomical-style) rule?

Measurement and code inspection only. Reads the cached _measurements.npy; recomputes
no images.

(1) what ocr_oct_values.parse_sectors does, and what validate.sector_screen_key
    returns for OS under the "ocr" convention.
(2) distribution contrast: for each hypothesis about which screen wedge is
    anatomically superotemporal / superonasal, are the OD and OS distributions the
    same population?
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import validate as V

recs = list(np.load(os.path.join(HERE, '_measurements.npy'), allow_pickle=True))
KEYS = ['s_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n']

print('=== (1) code ===')
# ocr_oct_values imports pytesseract, which is not installed here, so the signature
# and the branch are read from the source text rather than by import.
import re
src = open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'ocr_oct_values.py'), encoding='utf-8').read()
blk = src[src.index('def parse_sectors('):src.index('# \u2500\u2500 3. RNFL')]
print('ocr_oct_values.py: ' + blk.splitlines()[0])
print('  takes eye:', "eye" in blk.splitlines()[0])
for ln in blk.splitlines():
    if 'eye.upper()' in ln or 'angles_deg' in ln or 'keys ' in ln:
        print('   |', ln.strip())
for eye, ang in (('OD', [90, 30, 330, 270, 210, 150]), ('OS', [90, 150, 210, 270, 330, 30])):
    print(f'  {eye} screen angle per key: ' + '  '.join(f'{k}={a}' for k, a in zip(KEYS, ang)))
print('  -> the two lists differ, so the label depends on the eye: 30<->150, 330<->210')
print()
for eye in ('OD', 'OS'):
    print(f'  validate.sector_screen_key({eye}, "ocr"): ' +
          '  '.join(f'{k}->{V.sector_screen_key(k, eye, "ocr")}' for k in KEYS))
    print(f'  validate.sector_screen_key({eye}, "anatomical"): ' +
          '  '.join(f'{k}->{V.sector_screen_key(k, eye, "anatomical")}' for k in KEYS))

# ── (2) distributions ────────────────────────────────────────
def ref_vals(eye, key):
    return np.array([r['ref']['sectors'][key] for r in recs
                     if r['eye'] == eye and np.isfinite(r['ref']['sectors'][key])], float)


def mask_vals(eye, wedge, k=236.2):
    out = []
    for r in recs:
        if r['eye'] != eye:
            continue
        m = (r.get('gca') or {}).get('b')
        if m and np.isfinite(m[wedge]):
            out.append(m[wedge] * k)
    return np.array(out, float)


def welch(a, b):
    s = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = (a.mean() - b.mean()) / s
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return t, d


def contrast(label, a, b, na, nb):
    t, d = welch(a, b)
    print(f'  {label:44s} {na} {a.mean():6.2f}+-{a.std(ddof=1):5.2f} (n={len(a)})   '
          f'{nb} {b.mean():6.2f}+-{b.std(ddof=1):5.2f} (n={len(b)})   '
          f'diff {a.mean() - b.mean():+6.2f}  Welch t={t:+6.2f}  d={d:+.2f}')


print('\n=== (2a) mask pixel values by SCREEN wedge (variant b, common scale) ===')
print('    upper-left = a150, upper-right = a030')
for eye in ('OD', 'OS'):
    L, R = mask_vals(eye, 'a150'), mask_vals(eye, 'a030')
    t, d = welch(L, R)
    print(f'  {eye}: upper-left {L.mean():6.2f}+-{L.std(ddof=1):5.2f}   '
          f'upper-right {R.mean():6.2f}+-{R.std(ddof=1):5.2f}   '
          f'L-R {L.mean() - R.mean():+6.2f}  d={d:+.2f}')
print('  mirror check (same anatomy should land on opposite screen sides):')
contrast('OD upper-right  vs  OS upper-left', mask_vals('OD', 'a030'), mask_vals('OS', 'a150'),
         'OD-UR', 'OS-UL')
contrast('OD upper-right  vs  OS upper-right', mask_vals('OD', 'a030'), mask_vals('OS', 'a030'),
         'OD-UR', 'OS-UR')

print('\n=== (2b) reference COLUMN values, OD vs OS ===')
print('    hypothesis A: columns are named eye-dependently and correctly')
print('    hypothesis B: same eye-dependent branch, T and N globally swapped')
print('    both predict OD and OS draw the same anatomy -> matching distributions')
for key in ('s_sup_t', 's_sup_n', 's_sup', 's_inf'):
    contrast(f'column {key}', ref_vals('OD', key), ref_vals('OS', key), 'OD', 'OS')

print('\n=== (2c) screen-rule counterfactual ===')
print('    if the naming ignored the eye, os_s_sup_t would have been read at 30 deg')
print('    (screen upper-right) like OD. Compare the column actually stored for OS')
print('    against what a screen rule would have stored:')
contrast('OD s_sup_t vs OS s_sup_t (as stored, eye-branched)',
         ref_vals('OD', 's_sup_t'), ref_vals('OS', 's_sup_t'), 'OD', 'OS')
contrast('OD s_sup_t vs OS s_sup_n (= OS upper-right cell)',
         ref_vals('OD', 's_sup_t'), ref_vals('OS', 's_sup_n'), 'OD', 'OS')
