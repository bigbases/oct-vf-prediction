#!/usr/bin/env python3
"""QC of the fovea / annulus fit on every GCA map (read-only).

detect_gca_center() always returns the best-voting centre, so a wrong fit is not a
'failure'. Fit quality here = fraction of the drawn ring pixels (purple core, R >= 50) that
lie within 3 px of the fitted inner or outer ellipse. Writes cache/gca_fit_qc.json keyed by
content key (no patient key) and prints the distribution.
"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S


def qc(job):
    kind, ck, arr = job
    a = arr.astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    purple = (B > G + 15) & (R > G + 5) & (R >= S.PURPLE_MIN_R) & ((R + G + B) / 3 < 170)
    c = S.sm.detect_gca_center(arr, S.CFG)
    if c is None or purple.sum() < 50:
        return ck, {'fit': None, 'n_purple': int(purple.sum())}
    H, W = purple.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rho = np.hypot((xx - c[0]) / c[2], (yy - c[1]) / c[3])
    near = (np.abs(rho - 0.25) * c[2] < 3) | (np.abs(rho - 1.0) * c[2] < 3)
    off = ((c[0] - (W / 2 - 0.5)) / (W / 6.0), (c[1] - (H / 2 - 0.5)) / (H / 6.0))
    return ck, {'fit': round(float((purple & near).sum() / purple.sum()), 3), 'n_purple': int(purple.sum()),
                'centre_offset_mm': [round(off[0], 2), round(off[1], 2)], 'shape': [H, W]}


if __name__ == '__main__':
    rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
    index = S.IP.build_image_index()
    jobs = [(k, ck, np.asarray(crop)) for _, k, ck, crop in S.thickness_jobs(rows, index) if k == 'gca']
    with Pool(6) as p:
        res = dict(p.map(qc, jobs, chunksize=4))
    json.dump(res, open(S.CACHE / 'gca_fit_qc.json', 'w'), indent=1)
    f = np.array([v['fit'] for v in res.values() if v['fit'] is not None])
    print(f'GCA maps {len(res)}; fit fraction pct 1/5/10/50: {np.percentile(f, [1, 5, 10, 50]).round(3).tolist()}')
    for t in (0.3, 0.5, 0.7):
        print(f'  fit < {t}: {int((f < t).sum())}')
    print('  no fit / too few ring pixels:', sum(v['fit'] is None for v in res.values()))
    shapes = {}
    for v in res.values():
        if v.get('shape'): shapes[tuple(v['shape'])] = shapes.get(tuple(v['shape']), 0) + 1
    print('  crop shapes (H, W): count', sorted(shapes.items(), key=lambda x: -x[1])[:6])
