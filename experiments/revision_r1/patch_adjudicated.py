#!/usr/bin/env python3
"""Re-render the cache entries of the adjudicated GCA maps (cache/fit_adjudication.json) and
refresh the index summary. Other entries are untouched."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
doc = json.load(open(S.CACHE / 'index.json'))
adj = S.ADJUDICATION
rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
lut = S.load_lut(rows, index)
done = set()
for _, kind, ck, crop in S.thickness_jobs(rows, index):
    if kind == 'gca' and ck in adj and ck not in done:
        img, diag, _ = S.sector_average(crop, kind, *lut)
        img.save(S.CACHE / f'{ck}.png')
        doc['maps'][ck] = diag
        done.add(ck)
d = list(doc['maps'].values())
sm = doc['summary']
sm['centre_fallback'] = {k: sum(x['centre_fallback'] for x in d if x['kind'] == k) for k in ('gca', 'rnfl')}
sm['gca_status'] = {st: sum(1 for x in d if x['kind'] == 'gca' and x.get('gca_status') == st)
                    for st in ('fit', 'refit', 'map_mean')}
sm['grey_on_gca_max'] = max(x['frac']['grey'] for x in d if x['kind'] == 'gca')
sm['roundtrip_max_err'] = max(x['roundtrip_max_err'] for x in d)
sm['wedges_filled_from_map_mean'] = sum(x['wedges_filled_from_map_mean'] for x in d)
json.dump(doc, open(S.CACHE / 'index.json', 'w'), indent=1)
print(f'patched {len(done)} maps; GCA status {sm["gca_status"]}; grey on GCA max {sm["grey_on_gca_max"]}; round trip max {sm["roundtrip_max_err"]}')
