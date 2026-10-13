#!/usr/bin/env python3
"""R1-1, XGBoost arm: printed parameters vs the same kind of values measured on the map.
Plan: docs/revision_r1/prereg_R1-1.md. Fitting code is the submitted one (export_xgb_oof.py,
CPU), imported from the pass-B tree; only the feature matrix changes.

Feature sets (2026-10-13 decision: primary map23, secondary map26 vs printed26)
  printed26   the submitted 26 printed parameters            (guard: = submitted predictions)
  printed25   printed26 without vertical C/D                 (guard: = forest_robustness/xgb25)
  map23       primary. The 20 region means of the sector-averaged map (12 RNFL clock-hour
              wedges, 8 GCA regions: 6 annulus sectors, centre, outside; read from the cache
              index) + 3 disc measures from the map's disc overlay: disc area, cup area (mm^2)
              and vertical cup-to-disc ratio. The sector-averaged map keeps the disc pixels, so
              without these the CNN would see information the XGBoost input lacks.
  map26       secondary. The 26 printed quantities measured on the map: Cirrus geometry
              (RNFL 3.46 mm circle, 0.20 mm band -> 12 clock hours, 4 quadrants, average; GCA
              annulus -> 6 sectors, average, minimum of 360 one-degree spokes) + the map's
              vertical C/D.
GCA maps without a usable annulus (adjudicated bad scans and bad fits the refit could not
recover; cache/fit_adjudication.json) give their map mean for every GCA feature - the same
information as their sector-averaged image. Those eyes stay in training and evaluation.

Map features are put in the right-eye frame: left-eye maps are mirrored before measuring
(for map20 the mirror is a fixed permutation of the region indices). GCA values stay in
decoded units; trees are invariant to that scale.

  python r11_xgb.py features         # map23 + map26 for every row -> runs/r11_xgb/features.npz
  python r11_xgb.py fit <set> ...    # 5-fold OOF + held-out predictions per set
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PASS_B / 'scripts'))

import sector_maps as S                    # noqa: E402  (scipy before torch: see sector_maps)
import export_xgb_oof as X                 # noqa: E402
from oof_common import row_key, save_oof_npz, load_oof_npz  # noqa: E402
from cache_key import content_key  # noqa: E402
import annulus_fit as AF  # noqa: E402
import phi_safe  # noqa: E402
from xgboost import XGBRegressor           # noqa: E402

sm, IP, CFG = S.sm, S.IP, S.CFG
CSV = PASS_B / 'ml_final_90d_excl_empty_flip.csv'
OUT = HERE / 'runs' / 'r11_xgb'

# region-index permutation that mirrors a map left-right (screen angle t -> 180 - t)
RNFL_MIRROR = [(-k) % 12 for k in range(12)]          # hour 12 <-> 12, 3 <-> 9, ...
GCA_MIRROR = [2, 1, 0, 5, 4, 3, 6, 7]                  # [0,60)<->[120,180), [180,240)<->[300,360)


CUP_GREY_MIN = 160   # inside the disc overlay the cup is the lighter grey (disc 127-169, p10-p90)


def map20(crop, kind, eye, lut, index_doc=None):
    """Region means of the sector-averaged map, right-eye order. From the cache index when given."""
    if index_doc is not None:
        d = index_doc[content_key(crop)]
    else:
        d = S.sector_average(crop, kind, *lut)[1]
    # a GCA map without a usable annulus carries one value (its map mean) in all 8 regions,
    # exactly what its sector-averaged image shows
    v = np.array(d['wedge_means'], float)
    if eye == 'OS':
        v = v[RNFL_MIRROR] if kind == 'rnfl' else v[GCA_MIRROR]
    return v


def disc3(crop):
    """Disc area, cup area (mm^2) and vertical C/D from the RNFL map's disc overlay."""
    from scipy import ndimage
    a = np.asarray(crop.convert('RGB')).astype(float)
    H, W, _ = a.shape
    mx, mn = a.max(2), a.min(2)
    grey = ((mx - mn) / (mx + 1e-5) < S.GREY_SAT) & (mx > S.BLACK_MAX) & (mx < S.GREY_MAX)
    lab, n = ndimage.label(grey)
    if n == 0:
        return [np.nan] * 3
    k = int(np.argmax(ndimage.sum(grey, lab, range(1, n + 1)))) + 1
    disc = ndimage.binary_fill_holes(lab == k)
    cup = disc & (a.mean(2) > CUP_GREY_MIN)
    mm2 = (CFG['rnfl']['map_extent_mm'] / W) * (CFG['rnfl']['map_extent_mm'] / H)
    yd = np.where(disc.any(1))[0]
    yc = np.where(cup.any(1))[0]
    vcd = (yc[-1] - yc[0] + 1) / (yd[-1] - yd[0] + 1) if yc.size else 0.0
    return [float(disc.sum() * mm2), float(cup.sum() * mm2), float(vcd)]


def map25_part(crop, kind, eye, lut, index_doc):
    """The printed quantities measured on the map with the Cirrus geometry, right-eye frame.
    GCA ellipses (fit or refit) come from the sector-map cache, so both arms use the same
    geometry; a map-mean GCA map gives its map mean for all 8 GCA quantities."""
    d = index_doc[content_key(crop)]
    arr = np.asarray(crop.convert('RGB'))
    H, W, _ = arr.shape
    if kind == 'gca' and d.get('gca_status') == 'map_mean':
        return [float(d['wedge_means'][0])] * 8
    if eye == 'OS':
        arr = arr[:, ::-1].copy()
    vals, ramp = sm.decode(arr, *lut, S.MAX_DIST)
    if kind == 'rnfl':
        c = sm.detect_disc_center(arr) or (W / 2.0 - 0.5, H / 2.0 - 0.5)
    else:
        cx, cy, ox, oy, ix, iy = d['centre']
        if eye == 'OS':
            cx = W - 1 - cx
        c = (cx, cy, ox, oy, ix, iy)
    # thickness pixels only: the overlay classes of sector_maps (frame, disc, gaps, contours)
    valid = np.ones((H, W), bool)
    for m_ in S.overlay_masks(arr, kind, ramp, c if kind == 'gca' else None).values():
        valid &= ~m_
    mean = lambda mask: float(vals[mask & valid].mean()) if (mask & valid).sum() >= S.MIN_WEDGE_PX else np.nan
    if kind == 'rnfl':
        mk = sm.rnfl_masks((H, W), c, W / CFG['rnfl']['map_extent_mm'], 'OD', CFG)
        return [mean(mk[f'h{k:02d}']) for k in range(1, 13)] + [mean(mk[q]) for q in ('q_s', 'q_t', 'q_i', 'q_n')] + [mean(mk['avg'])]
    # GCA annulus between the fitted inner and outer ellipses; sectors in the outer
    # ellipse's elliptical angle, named in the right-eye frame
    yy, xx = np.mgrid[0:H, 0:W]
    rho_out = np.hypot((xx - c[0]) / c[2], (yy - c[1]) / c[3])
    rho_in = np.hypot((xx - c[0]) / c[4], (yy - c[1]) / c[5])
    ann = (rho_out <= 1.0) & (rho_in >= 1.0)
    th = np.degrees(np.arctan2(-(yy - c[1]) / c[3], (xx - c[0]) / c[2])) % 360.0
    secs = []
    for region in ('s_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n'):
        lo, hi = sm.map_region_to_angle(region, 'OD', CFG)
        secs.append(mean(ann & sm._in_wedge(th, lo % 360.0, (lo % 360.0) + (hi - lo))))
    # minimum: lowest of 360 one-degree spoke means across the annulus
    spoke = np.floor(th).astype(int)
    a2 = ann & valid
    sp = [vals[a2 & (spoke == a)].mean() for a in range(360) if (a2 & (spoke == a)).sum() >= 3]
    return secs + [mean(ann), float(min(sp)) if sp else np.nan]


def features():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    index = IP.build_image_index()
    lut = S.load_lut(rows, index)
    idx_doc = json.load(open(S.CACHE / 'index.json'))['maps']
    F23 = np.full((len(rows), 23), np.nan); F26 = np.full((len(rows), 26), np.nan)
    for i, r in enumerate(rows):
        paths = index.get(row_key(r), {})
        lo = r['eye'].lower()
        g, n = paths.get(f'gca_{lo}_thickness'), paths.get(f'rnfl_{lo}_thickness')
        if n:
            crop = IP._crop_rnfl_thickness(Image.open(n).convert('RGB'))
            d3 = disc3(crop)
            F23[i, :12] = map20(crop, 'rnfl', r['eye'], lut, idx_doc)
            F23[i, 20:] = d3
            F26[i, :17] = map25_part(crop, 'rnfl', r['eye'], lut, idx_doc)
            F26[i, 25] = d3[2]
        if g:
            crop = IP._crop_gca_thickness(Image.open(g).convert('RGB'))
            F23[i, 12:20] = map20(crop, 'gca', r['eye'], lut, idx_doc)
            F26[i, 17:25] = map25_part(crop, 'gca', r['eye'], lut, idx_doc)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'features.npz', keys=np.array([row_key(r) for r in rows], dtype=object),
                        map23=F23, map26=F26)
    # sanity: map-derived vertical C/D against the printed one
    pv = np.array([float(r[f"{r['eye'].lower()}_vert_cd"]) if r.get(f"{r['eye'].lower()}_vert_cd", '') else np.nan
                   for r in rows])
    ok = np.isfinite(pv) & np.isfinite(F23[:, 22])
    print(f'rows {len(rows)}; rows with a missing map23 value {int(np.isnan(F23).any(1).sum())}, '
          f'map26 {int(np.isnan(F26).any(1).sum())}; r(map vCD, printed vert_cd) = '
          f'{np.corrcoef(F23[ok, 22], pv[ok])[0, 1]:.3f} over {int(ok.sum())} rows')


def feature_matrix(rows, fset, feats):
    if fset in ('printed26', 'printed25'):
        Xs = np.stack([X.row_to_xy(r)[0] for r in rows])
        if fset == 'printed25':
            names = X.FEATURES_OD
            Xs = np.delete(Xs, names.index('od_vert_cd'), axis=1)  # same position for OD and OS
        return Xs
    idx = {k: i for i, k in enumerate(map(tuple, feats['keys']))}
    return np.stack([feats[fset][i] for i in phi_safe.lookup(idx, [row_key(r) for r in rows], 'features.npz')]).astype(np.float32)


def fit(fsets):
    rows = X.load_rows(CSV)
    feats = np.load(OUT / 'features.npz', allow_pickle=True) if any(f.startswith('map') for f in fsets) else None
    kw = X._xgb_kwargs('cpu')
    for fset in fsets:
        for k in map(str, range(5)):
            rows_tr = [r for r in rows if r['cv_fold'] not in (k, 'test')]
            Xtr = feature_matrix(rows_tr, fset, feats)
            ytr = [X.row_to_xy(r) for r in rows_tr]
            Y_tr = np.stack([t[1] for t in ytr]); M_tr = np.stack([t[2] for t in ytr])
            cm = X.impute_train_stats(Xtr)
            for split, rows_out in (('val', [r for r in rows if r['cv_fold'] == k]),
                                    ('test', [r for r in rows if r['cv_fold'] == 'test'])):
                Xo = X.impute(feature_matrix(rows_out, fset, feats), cm)
                yo = [X.row_to_xy(r) for r in rows_out]
                preds = np.full((len(rows_out), X.N_PT), np.nan, dtype=np.float32)
                for pi in range(X.N_PT):
                    valid = M_tr[:, pi]
                    if valid.sum() < 10:
                        continue
                    mdl = XGBRegressor(**kw)
                    mdl.fit(X.impute(Xtr, cm)[valid], Y_tr[valid, pi])
                    preds[:, pi] = mdl.predict(Xo)
                path = OUT / fset / f'xgb_90d_fold{k}_{split}.npz'
                save_oof_npz(path, [row_key(r) for r in rows_out], preds,
                             np.stack([t[1] for t in yo]), np.stack([t[2] for t in yo]),
                             meta={'model': 'xgb_tabular', 'feature_set': fset, 'n_features': Xtr.shape[1],
                                   'val_fold': k, 'split': split, 'device': 'cpu'})
                ref = {'printed26': PASS_B / 'runs/oof' / f'xgb_90d_fold{k}_{split}.npz',
                       'printed25': REPO / 'experiments/forest_robustness/xgb25' / f'xgb_25feat_no_vertcd_fold{k}_{split}.npz'}.get(fset)
                if ref is not None:
                    a, b = load_oof_npz(ref), load_oof_npz(path)
                    assert a['keys'] == b['keys']
                    d = float(np.nanmax(np.abs(a['pred'] - b['pred'])))
                    print(f'guard {fset} fold {k} {split}: max |pred - submitted| = {d:.2e}', flush=True)
                    if d > 1e-5:
                        raise SystemExit(f'{fset} does not reproduce its submitted predictions')
        print(f'{fset}: done', flush=True)


if __name__ == '__main__':
    phi_safe.install()
    if sys.argv[1] == 'features':
        features()
    else:
        fit(sys.argv[2:])
