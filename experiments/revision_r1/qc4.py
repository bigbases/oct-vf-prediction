#!/usr/bin/env python3
"""Four review sheets after the second QC round, plus the QC table over every map.

  qc_01, qc_02  GCA maps whose annulus fit failed (decentred scans): the sector-averaged GCA
                panel must equal the original
  qc_03         a map whose ring is drawn in a light colour (old purple rule missed it)
  qc_04         the earlier sample_02 (RNFL data gap, GCA ring)
Prints the QC table; writes samples/qc_table.json. No patient key printed; the sheet-to-eye
map goes to the private directory.
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
from cache_key import content_key

rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
lut_rgb, lut_val = S.load_lut(rows, index)
idx = json.load(open(S.CACHE / 'index.json'))
maps, summ = idx['maps'], idx['summary']

def gck(r):
    p = index.get((r['patient_id'], r['eye'], r['vf_date']), {}).get(f"gca_{r['eye'].lower()}_thickness")
    return content_key(S.IP._crop_gca_thickness(Image.open(p).convert('RGB'))) if p else None

adjd = json.load(open(S.CACHE / 'fit_adjudication.json'))['maps']
bad_rows, light_rows, seen = [], [], set()
for r in rows:
    k = gck(r)
    if not k or k in seen:
        continue
    seen.add(k)
    m = maps[k]
    if k in adjd and adjd[k]['class'] == 'a':
        bad_rows.append((adjd[k]['tile'], r))
    elif m.get('gca_status') == 'fit' and m['fit_support'] >= 0.5:
        p = index[(r['patient_id'], r['eye'], r['vf_date'])][f"gca_{r['eye'].lower()}_thickness"]
        a = np.asarray(S.IP._crop_gca_thickness(Image.open(p).convert('RGB'))).astype(float)
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        npur = int(((B > G + 15) & (R > G + 5) & (R >= S.PURPLE_MIN_R) & ((R + G + B) / 3 < 170)).sum())
        light_rows.append((npur, r))
bad_rows = [r for _, r in sorted(bad_rows, key=lambda x: x[0])]
light_rows = [r for _, r in sorted(light_rows, key=lambda x: x[0])]
priv = Path.home() / 'hvf_private' / 'revision_r1_sample_map.csv'
s02 = next(tuple(x[1:]) for x in csv.reader(open(priv)) if x[0] == 'sample_02')
r02 = next(r for r in rows if (r['patient_id'], r['eye'], r['vf_date']) == s02)
picks = [('qc_01', bad_rows[0], 'GCA (a) bad scan -> map mean'),
         ('qc_02', bad_rows[1], 'GCA (a) bad scan -> map mean'),
         ('qc_03', light_rows[0], 'light-coloured ring'),
         ('qc_04', r02, 'earlier sample_02')]
diag = {lab: S.render_sample(lab, r, index, lut_rgb, lut_val, note) for lab, r, note in picks}

# eyes affected by a failed fit (for the evaluation exclusion) and their MD
md = {}
for r in csv.DictReader(open(S.REPO / 'cohort_md.csv', encoding='utf-8-sig')):
    try:
        md[(r['patient_id'], r['eye'], r['vf_date'])] = float(r.get('MD') or 'nan')
    except ValueError:
        pass
def md_of(keys):
    ev = [r for r in rows if gck(r) in keys]
    v = np.array([md.get((r['patient_id'], r['eye'], r['vf_date']), np.nan) for r in ev])
    return {'eye_visits': len(ev), 'oof': sum(r['cv_fold'] != 'test' for r in ev),
            'test': sum(r['cv_fold'] == 'test' for r in ev),
            'md_median': float(np.nanmedian(v)) if np.isfinite(v).any() else None,
            'md_min': float(np.nanmin(v)) if np.isfinite(v).any() else None,
            'md_max': float(np.nanmax(v)) if np.isfinite(v).any() else None}
cls_a = {k for k, e in adjd.items() if e['class'] == 'a'}
map_mean = {k for k, m in maps.items() if m['kind'] == 'gca' and m.get('gca_status') == 'map_mean'}
all_md = np.array([md.get((r['patient_id'], r['eye'], r['vf_date']), np.nan) for r in rows])
import collections
table = {'summary': summ,
         'adjudication_of_16': dict(collections.Counter(f"{e['class']}/{e['verdict']}" for e in adjd.values())),
         'class_a_bad_scan': md_of(cls_a),
         'map_mean_all (a + rejected refits)': md_of(map_mean),
         'md_cohort_median': float(np.nanmedian(all_md)),
         'sheets': diag}
json.dump(table, open(S.SAMPLES / 'qc_table.json', 'w'), indent=1, default=float)
with open(priv, 'a', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    for lab, r, _ in picks:
        w.writerow([lab, r['patient_id'], r['eye'], r['vf_date']])
print(json.dumps({k: v for k, v in table.items() if k != 'sheets'}, indent=1, default=float))
