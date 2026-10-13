#!/usr/bin/env python3
"""Centre of every GCA annulus fit, relative to the crop (fractions of width / height), so
that the refit start can be confined to where successful fits put the fovea.
Writes cache/fit_centres.json (content key -> fit). Read-only otherwise."""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
import annulus_fit as AF


def one(job):
    ck, arr = job
    f = AF.fit_annulus(arr, S.CFG)
    H, W, _ = arr.shape
    return ck, None if f is None else {'fx': f[0] / W, 'fy': f[1] / H, 'support': f[4], 'W': W, 'H': H}


if __name__ == '__main__':
    rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
    index = S.IP.build_image_index()
    jobs = [(ck, np.asarray(crop)) for _, k, ck, crop in S.thickness_jobs(rows, index) if k == 'gca']
    with Pool(6) as p:
        res = dict(p.map(one, jobs, chunksize=4))
    json.dump(res, open(S.CACHE / 'fit_centres.json', 'w'), indent=1)
    adj = S.ADJUDICATION
    ok = np.array([[v['fx'], v['fy']] for k, v in res.items() if v and k not in adj])
    print('successful fits', len(ok), ' fx pct 0.5/50/99.5', np.percentile(ok[:, 0], [0.5, 50, 99.5]).round(3).tolist(),
          ' fy', np.percentile(ok[:, 1], [0.5, 50, 99.5]).round(3).tolist())
