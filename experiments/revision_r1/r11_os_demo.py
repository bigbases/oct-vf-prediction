#!/usr/bin/env python3
"""Show, on one left-eye sample, the map-derived XGBoost features before and after they are
put in the right-eye frame, next to the printed parameters of the same row (which the
label file already holds in the right-eye frame). Read-only; prints no patient key.

  python r11_os_demo.py sample_04        (a sample id from samples/)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r11_xgb as R                       # noqa: E402

S, IP, sm, CFG = R.S, R.IP, R.sm, R.CFG

RNFL_HOURS = [f'h{k:02d}' for k in range(1, 13)]
GCA_REG = ['[0,60)', '[60,120)', '[120,180)', '[180,240)', '[240,300)', '[300,360)', 'centre', 'outside']
# names in the right-eye frame (OD fundus view: nasal = image right)
GCA_NAME_OD = ['sup-nasal', 'superior', 'sup-temporal', 'inf-temporal', 'inferior', 'inf-nasal', 'centre', 'outside']


def main(sample):
    priv = Path.home() / 'hvf_private' / 'revision_r1_sample_map.csv'
    key = next(tuple(r[1:]) for r in csv.reader(open(priv)) if r[0] == sample)
    assert key[1] == 'OS', 'pick a left-eye sample'
    rows = list(csv.DictReader(open(R.CSV, encoding='utf-8-sig')))
    row = next(r for r in rows if (r['patient_id'], r['eye'], r['vf_date']) == key)
    index = IP.build_image_index()
    lut = S.load_lut(rows, index)
    paths = index[key]
    rn = IP._crop_rnfl_thickness(Image.open(paths['rnfl_os_thickness']).convert('RGB'))
    gc = IP._crop_gca_thickness(Image.open(paths['gca_os_thickness']).convert('RGB'))

    raw_r = np.array(S.sector_average(rn, 'rnfl', *lut)[1]['wedge_means'])
    raw_g = np.array(S.sector_average(gc, 'gca', *lut)[1]['wedge_means'])
    conv_r = R.map20(rn, 'rnfl', 'OS', lut)
    conv_g = R.map20(gc, 'gca', 'OS', lut)
    printed_h = np.array([float(row[f'rnfl_{h}']) for h in RNFL_HOURS])

    print(f'{sample} (left eye). map20, RNFL clock-hour wedges, decoded um')
    print('  wedge index = screen clock position of the OS image as acquired')
    print(f'  {"pos":>4} {"OS screen":>10} {"-> right-eye frame":>19} {"printed (label file)":>21}')
    for k, h in enumerate(RNFL_HOURS):
        # wedge index 0 = hour 12; hour hh is index hh % 12
        i = (k + 1) % 12
        print(f'  {h:>4} {raw_r[i]:>10.1f} {conv_r[i]:>19.1f} {printed_h[k]:>21.1f}')
    r_raw = np.corrcoef([raw_r[(k + 1) % 12] for k in range(12)], printed_h)[0, 1]
    r_conv = np.corrcoef([conv_r[(k + 1) % 12] for k in range(12)], printed_h)[0, 1]
    print(f'  correlation with the printed clock hours over 12 positions: before {r_raw:+.2f}, after {r_conv:+.2f}')
    print('  (map20 averages each wedge to the map edge, printed values sit on the 3.46 mm circle,'
          ' so levels differ; the shape across positions is what must agree)')
    print(f'\n{sample}. map20, GCA regions, decoded units (scale unknown)')
    print(f'  {"screen region":>14} {"OS screen":>10} {"-> right-eye":>13}  right-eye name')
    for i, reg in enumerate(GCA_REG):
        print(f'  {reg:>14} {raw_g[i]:>10.1f} {conv_g[i]:>13.1f}  {GCA_NAME_OD[i]}')


if __name__ == '__main__':
    main(sys.argv[1])
