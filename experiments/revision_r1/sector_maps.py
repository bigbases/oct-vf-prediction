#!/usr/bin/env python3
"""R1-1: sector-averaged thickness maps for the CNN arm of the representation contrast.

For every thickness map the CNN sees, take exactly the crop the model takes
(image_preprocessing._crop_gca_thickness / _crop_rnfl_thickness), then

  1. invert colour to value with the colour ramp of the RNFL panel's own colorbar
     (350/175/0 um; experiments/sector_mask/sector_masks.extract_lut + decode). The GCA
     panel has no colorbar; it uses the same ramp, so its decoded values are proportional
     to thickness with an unknown scale. The scale cancels: the map is re-rendered with
     the same ramp it was decoded with;
  2. find the centre: optic disc on the RNFL map (centroid of the grey disc overlay),
     fovea on the GCA map (vote over the two annulus ellipses Cirrus draws). Both
     detectors are the ones validated in experiments/sector_mask on 333 eyes;
  3. split the whole map into regions around that centre, so that every pixel belongs to
     one region. RNFL: 12 clock-hour wedges of 30 deg. GCA: the Cirrus GCA geometry -
     the elliptical annulus (inner 1.2 x 1.0 mm, outer 4.8 x 4.0 mm) cut into its 6
     sectors of 60 deg (boundaries 0, 60, ..., 300 deg in the elliptical angle), plus the
     centre inside the annulus and everything outside it as two more regions (8 in all),
     so that the GCA map keeps at least the information of the printed sectors;
  4. average the decoded values over the valid pixels of each region (pixels that match
     the ramp; the disc overlay, drawn contours and frame do not) and fill the whole
     region, every pixel, with that average;
  5. render back to RGB with the same ramp (nearest ramp entry).

The result has the size of the crop and replaces it before the transform, so resizing,
jitter and normalisation are unchanged. Wedge geometry is in the screen frame and is
symmetric left-right, so the same construction serves both eyes; images are not mirrored
(as submitted).

Cache: cache/<content key>.png, where the key is a hash of the cropped pixels - no
patient key is written. cache/index.json holds per-map diagnostics.

  python sector_maps.py build      # all maps of the label file
  python sector_maps.py samples    # six original / sector-averaged pairs into samples/
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'sector_mask'))

# scipy (via sector_masks) before torch (via image_preprocessing): the other order fails
# with GLIBCXX_3.4.29 not found on this host.
import sector_masks as sm          # noqa: E402
import annulus_fit as AF           # noqa: E402
import image_preprocessing as IP   # noqa: E402

CFG = sm.load_config()
CSV = REPO / 'ml_final_90d_excl_empty_flip.csv'
CACHE = HERE / 'cache'
SAMPLES = HERE / 'samples'
MAX_DIST = CFG['decode']['lut_match_max_dist']
MIN_WEDGE_PX = 30
# The crops keep the panel frame: on RNFL a cyan border about 8 px deep (plus a sliver of
# the colorbar at the left edge), on GCA a 3 px line. Frame colours sit close to the
# ramp and would decode as thickness, so a border band is left out of every mean
# (measured on 30 eyes: ring 9 of RNFL and ring 4 of GCA are the first fully valid map
# rows). In the sector-averaged map the whole band is painted one fixed colour (BORDER_RGB):
# kept original, its tissue pixels showed through at the map edge (the superior RNFL bundle
# as a yellow streak on the top edge, 2026-10-13 QC).
BORDER_FRAC = {'rnfl': 9 / 271, 'gca': 4 / 343}
# per-kind median band colour of every original crop (channel-wise; GCA 280 maps, 1,599,316
# band pixels; RNFL 277 maps, 2,612,664), measured 2026-10-13 before any R1-1 result
BORDER_RGB = {'gca': (86, 133, 195), 'rnfl': (53, 169, 224)}
GCA_BLACK_CENTRE_IS_TISSUE = True   # set from the 2026-10-13 location check


from cache_key import content_key  # noqa: E402  (shared with train_r11.py)


def load_lut(rows, index):
    for r in rows:
        p = index.get((r['patient_id'], r['eye'], r['vf_date']), {}).get(f"rnfl_{r['eye'].lower()}_thickness")
        if p:
            lut_rgb, lut_val = sm.extract_lut(p, CFG['rnfl']['colormap_max_um'])
            if lut_rgb is not None:
                return lut_rgb, lut_val
    raise SystemExit('no colorbar found')


def lut_stability(rows, index, lut_rgb, n=40):
    """Max |RGB| difference between the reference ramp and ramps read from other panels."""
    diffs = []
    for r in rows[1:]:
        p = index.get((r['patient_id'], r['eye'], r['vf_date']), {}).get(f"rnfl_{r['eye'].lower()}_thickness")
        if not p:
            continue
        rgb, _ = sm.extract_lut(p, CFG['rnfl']['colormap_max_um'])
        if rgb is None:
            continue
        x = np.linspace(0, 1, len(lut_rgb))
        xi = np.linspace(0, 1, len(rgb))
        res = np.stack([np.interp(x, xi, rgb[:, c]) for c in range(3)], 1)
        diffs.append(float(np.abs(res - lut_rgb).max()))
        if len(diffs) >= n:
            break
    return diffs


def screen_angle(H, W, cx, cy, sx=1.0, sy=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.degrees(np.arctan2(-(yy - cy) / sy, (xx - cx) / sx)) % 360.0


def wedge_index(kind, H, W, centre):
    if kind == 'rnfl':
        th = screen_angle(H, W, centre[0], centre[1])
        # clock hour k is centred at 90 - 30k deg; index 0..11 with 0 = hour 12
        return (np.floor(((105.0 - th) % 360.0) / 30.0).astype(int)) % 12, 12
    # GCA: 0..5 annulus sectors (elliptical angle bins starting at 0 deg), 6 centre, 7 outside
    cx, cy, ox, oy, ix, iy = gca_ellipses(centre)
    yy, xx = np.mgrid[0:H, 0:W]
    rho_out = np.hypot((xx - cx) / ox, (yy - cy) / oy)
    rho_in = np.hypot((xx - cx) / ix, (yy - cy) / iy)
    th = screen_angle(H, W, cx, cy, ox, oy)
    idx = np.floor(th / 60.0).astype(int) % 6
    idx[rho_in < 1.0] = 6
    idx[rho_out > 1.0] = 7
    return idx, 8


def gca_ellipses(centre):
    """(cx, cy, outer_ax, outer_ay, inner_ax, inner_ay). A 4-tuple centre has the Cirrus
    inner/outer ratio (0.25 on both axes); a refit gives its inner axes explicitly."""
    if len(centre) >= 6:
        return tuple(float(v) for v in centre[:6])
    cx, cy, ox, oy = centre[:4]
    r = CFG['gca']['inner_semi_mm']['x'] / CFG['gca']['outer_semi_mm']['x']
    return float(cx), float(cy), float(ox), float(oy), r * float(ox), r * float(oy)


def render(values, lut_rgb, lut_val):
    i = np.abs(values[..., None] - lut_val[None, None, :]).argmin(-1)
    return lut_rgb[i].round().clip(0, 255).astype(np.uint8)


# Pixel classes that are not thickness. Each is kept as the original pixel in the output
# and left out of every mean, so that original and sector-averaged maps differ only in the
# thickness inside each region. Measured on 40 eyes (2026-10-13): the ramp ends in black
# (0 um = RGB 0,0,3), so black data gaps decoded as 0 um and entered the means; 58% of the
# purple GCA contour pixels and 42-64% of the frame band were also within ramp distance.
GREY_SAT, GREY_MAX = 0.12, 250      # optic-disc overlay and cup: desaturated, not white
BLACK_MAX = 40                       # data gaps
PURPLE_FRINGE_PX = 1                 # antialiasing around the drawn contours
PURPLE_MIN_R = 50
# The ring pixels are blends of the drawn line and the tissue under it, so their colour
# tracks the local thickness along the ellipses (within-map r = +0.93 between ring colour
# R-B and the adjacent tissue thickness, 40/40 maps, 2026-10-13). In the sector-averaged map
# the ring keeps its position but is painted one colour: the median ring colour of 80 GCA
# maps (189,175 ring pixels).
RING_RGB = (101, 113, 155)
# GCA annulus fit (annulus_fit.py). A fit whose ellipses run along a drawn line at fewer
# than FIT_SUPPORT_MIN of their points is a detection failure. Fixed on 2026-10-13 from the
# QC of all GCA maps, before any R1-1 result: visually correct fits had support 0.49-0.83,
# the six visibly wrong ones (decentred scans) 0.11-0.21. A failed map keeps its original
# pixels in the sector-averaged condition too; the eye is dropped from R1-1 evaluation.
FIT_SUPPORT_MIN = 0.35
# Maps flagged by FIT_SUPPORT_MIN were adjudicated visually (cache/fit_adjudication.json,
# before any R1-1 result): thin advanced maps have low line contrast, so a low support does
# not by itself mean a wrong fit. The adjudication overrides the threshold for those maps.
def _load_adjudication():
    f = Path(__file__).resolve().parent / 'cache' / 'fit_adjudication.json'
    return json.load(open(f))['maps'] if f.exists() else {}
ADJUDICATION = _load_adjudication()
RING_BAND_PX = 1.5


KEEP = ('border', 'grey', 'black', 'contour')   # not filled with the region value; border and
                                                  # contour are then painted one fixed colour
CONTOUR_BAND_PX = 2.5                             # around the fitted annulus ellipses


def overlay_masks(arr, kind, valid_ramp, centre=None):
    """Mutually exclusive non-thickness classes on one crop (H x W booleans).

    keep classes (KEEP): out of every mean and not filled with the region value; grey and
    black stay the original pixel, border and contour are painted one fixed colour -
      border   panel frame band (and the colorbar sliver on RNFL crops), painted BORDER_RGB;
      grey     optic-disc overlay and cup (desaturated, not white) - RNFL maps only;
      black    data gaps; on GCA maps black inside the centre ellipse is the floor of the
               foveal pit and stays tissue (93% of GCA black lies outside the annulus);
      contour  GCA annulus lines: purple core, plus off-ramp pixels within 2.5 px of the
               fitted ellipses (the dark blends of the line).
    out of the mean but filled with the region's value (no original detail leaks):
      fringe   one pixel around grey / black / contour (blends with the tissue next to them);
      off_ramp any other pixel further than MAX_DIST from the ramp.
    """
    from scipy import ndimage
    a = arr.astype(float)
    H, W, _ = a.shape
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-5)
    b = int(np.ceil(BORDER_FRAC[kind] * W))
    border = np.ones((H, W), bool)
    border[b:H - b, b:W - b] = False
    # The grey rule is for the RNFL disc overlay. On GCA maps the desaturated pixels are
    # mostly light-grey stretches of the drawn ring (91% on the rings, 2026-10-13); they fall
    # in the ring band or, off the ramp, in off_ramp - never decoded as thickness.
    grey = (sat < GREY_SAT) & (mx > BLACK_MAX) & (mx < GREY_MAX) & ~border if kind == 'rnfl' \
        else np.zeros((H, W), bool)
    black = (mx < BLACK_MAX) & ~border
    contour = np.zeros((H, W), bool)
    if kind == 'gca' and centre is not None:
        cx, cy, ox, oy, ix, iy = gca_ellipses(centre)
        yy, xx = np.mgrid[0:H, 0:W]
        rho_out = np.hypot((xx - cx) / ox, (yy - cy) / oy)
        rho_in = np.hypot((xx - cx) / ix, (yy - cy) / iy)
        if GCA_BLACK_CENTRE_IS_TISSUE:
            black &= rho_in >= 1.0
        # the drawn annulus lines: a band around the fitted ellipses, whatever their colour
        # (some maps draw them in a light blue-violet that sits close to the ramp)
        contour = ((np.abs(rho_in - 1.0) * ix < RING_BAND_PX) | (np.abs(rho_out - 1.0) * ox < RING_BAND_PX)) \
            & ~border & ~grey & ~black
    keep = border | grey | black | contour
    fringe = ndimage.binary_dilation(grey | black | contour, iterations=PURPLE_FRINGE_PX) & ~keep
    off_ramp = ~valid_ramp & ~keep & ~fringe
    return {'border': border, 'grey': grey, 'black': black, 'contour': contour,
            'fringe': fringe, 'off_ramp': off_ramp}


def _map_mean_gca(arr, vals, valid_ramp, lut_rgb, lut_val, support, adj, return_tissue):
    """GCA map without a usable annulus (bad scan, or a bad fit the refit could not
    recover): the frame band takes BORDER_RGB as on every sector map, no-data black stays
    original; every other pixel takes the map-wide mean of the thickness pixels. No ring is drawn - its position is unknown. This
    keeps less information than a sector map, i.e. it works against the sector condition."""
    H, W, _ = arr.shape
    a = arr.astype(float)
    b = int(np.ceil(BORDER_FRAC['gca'] * W))
    border = np.ones((H, W), bool)
    border[b:H - b, b:W - b] = False
    black = (a.max(2) < BLACK_MAX) & ~border
    keep = border | black
    tissue = ~keep & valid_ramp
    m = float(vals[tissue].mean())
    rgb = arr.copy()
    rgb[~keep] = render(np.full((H, W), m), lut_rgb, lut_val)[~keep]
    rgb[border] = BORDER_RGB['gca']
    back, bvalid = sm.decode(rgb, lut_rgb, lut_val, MAX_DIST)
    rt = float(np.max(np.abs(back[~keep] - m))) if bvalid[~keep].all() else float('inf')
    frac = {'border': border.mean(), 'grey': 0.0, 'black': black.mean(), 'contour': 0.0, 'fringe': 0.0,
            'off_ramp': (~keep & ~valid_ramp).mean()}
    diag = {'kind': 'gca', 'shape': [H, W], 'n_px': int(H * W),
            'fit_support': round(float(support), 3), 'gca_status': 'map_mean',
            'adjudication': None if adj is None else {k: adj[k] for k in ('class', 'verdict', 'reason')},
            'centre': None, 'frac': {k: round(float(v), 5) for k, v in frac.items()},
            'frac_tissue': round(float(tissue.mean()), 4), 'frac_kept_original': round(float(keep.mean()), 4),
            'ramp_matched_but_overlay_px': {'border': 0, 'grey': 0, 'black': 0, 'contour': 0, 'fringe': 0},
            'centre_fallback': True, 'wedge_tissue_px_min': int(tissue.sum()),
            'wedges_filled_from_map_mean': 0, 'filled_regions': [], 'wedge_means': [round(m, 3)] * 8,
            'roundtrip_max_err': round(rt, 4)}
    if return_tissue:
        return Image.fromarray(rgb), diag, None, (tissue, np.full((H, W), -1))
    return Image.fromarray(rgb), diag, None


def sector_average(crop: Image.Image, kind: str, lut_rgb, lut_val, return_tissue=False):
    arr = np.asarray(crop.convert('RGB'))
    H, W, _ = arr.shape
    vals, valid_ramp = sm.decode(arr, lut_rgb, lut_val, MAX_DIST)
    fallback = False
    support = None
    status = None
    if kind == 'rnfl':
        c = sm.detect_disc_center(arr)
        if c is None:
            c, fallback = (W / 2.0 - 0.5, H / 2.0 - 0.5), True
        centre = (c[0], c[1])
    else:
        f = AF.fit_annulus(arr, CFG)
        support = f[4] if f is not None else 0.0
        status = 'fit'
        adj = ADJUDICATION.get(content_key(crop))
        if adj is not None:
            status = {'normal': 'fit', 'refit': 'refit'}.get(adj['verdict'], 'map_mean')
        elif f is None or f[4] < FIT_SUPPORT_MIN:
            status = 'map_mean'     # not expected: every flagged map is adjudicated
        if status == 'refit':
            e = adj['ellipse']
            ppm = max(W, H) / CFG['gca']['map_extent_mm']
            centre = (e['cx'], e['cy'], e['ox'], e['oy'],
                      e['inner_scale'] * CFG['gca']['inner_semi_mm']['x'] * ppm,
                      e['inner_scale'] * CFG['gca']['inner_semi_mm']['y'] * ppm)
        elif status == 'fit':
            centre = gca_ellipses(f[:4])
        else:
            centre = None
        fallback = status == 'map_mean'
    if kind == 'gca' and status == 'map_mean':
        return _map_mean_gca(arr, vals, valid_ramp, lut_rgb, lut_val, support, adj, return_tissue)
    ov = overlay_masks(arr, kind, valid_ramp, centre)
    keep = np.zeros((H, W), bool)
    non = np.zeros((H, W), bool)
    for k_, m_ in ov.items():
        non |= m_
        if k_ in KEEP:
            keep |= m_
    tissue = ~non
    idx, n = wedge_index(kind, H, W, centre)
    all_mean = float(vals[tissue].mean()) if tissue.any() else float('nan')
    means, npx, filled, filled_idx = [], [], 0, []
    out = np.full((H, W), np.nan)
    for k in range(n):
        sel = idx == k
        m = sel & tissue
        npx.append(int(m.sum()))
        if m.sum() >= MIN_WEDGE_PX:
            v = float(vals[m].mean())
        else:
            v, filled = all_mean, filled + 1
            filled_idx.append(k)
        means.append(v)
        out[sel] = v
    rgb = arr.copy()
    fill = ~keep
    rgb[fill] = render(out, lut_rgb, lut_val)[fill]
    rgb[ov['contour']] = RING_RGB
    rgb[ov['border']] = BORDER_RGB[kind]
    # round trip on the replaced pixels: decoding must return each region's value
    rfill = fill & ~ov['contour']
    back, bvalid = sm.decode(rgb, lut_rgb, lut_val, MAX_DIST)
    rt = float(np.max(np.abs(back[rfill] - out[rfill]))) if bvalid[rfill].all() else float('inf')
    # pixels that the ramp alone would have accepted but that belong to an overlay class:
    # these are the ones that were mixed into the means before the classes existed
    caught = {k: int((m & valid_ramp).sum()) for k, m in ov.items() if k != 'off_ramp'}
    frac_keep = round(float(keep.mean()), 4)
    diag = {'kind': kind, 'shape': [H, W], 'n_px': int(H * W),
            'fit_support': round(float(support), 3) if kind == 'gca' else None,
            'gca_status': status, 'centre': [round(float(v), 3) for v in centre],
            'frac': {k: round(float(m.mean()), 5) for k, m in ov.items()},
            'frac_tissue': round(float(tissue.mean()), 4), 'frac_kept_original': frac_keep,
            'ramp_matched_but_overlay_px': caught,
            'centre_fallback': fallback, 'wedge_tissue_px_min': int(min(npx)),
            'wedges_filled_from_map_mean': filled, 'filled_regions': filled_idx, 'wedge_means': [round(x, 3) for x in means],
            'roundtrip_max_err': round(rt, 4)}
    if return_tissue:
        return Image.fromarray(rgb), diag, centre, (tissue, idx)
    return Image.fromarray(rgb), diag, centre


def thickness_jobs(rows, index):
    """(kind, crop) for every thickness map the CNN reads, deduplicated by content."""
    seen = set()
    for r in rows:
        key = (r['patient_id'], r['eye'], r['vf_date'])
        paths = index.get(key, {})
        lo = r['eye'].lower()
        for kind, rk, crop_fn in (('gca', f'gca_{lo}_thickness', IP._crop_gca_thickness),
                                  ('rnfl', f'rnfl_{lo}_thickness', IP._crop_rnfl_thickness)):
            if rk not in paths:
                continue
            crop = crop_fn(Image.open(paths[rk]).convert('RGB'))
            ck = content_key(crop)
            if ck in seen:
                continue
            seen.add(ck)
            yield key, kind, ck, crop


_LUT = None


def _one(job):
    kind, ck, arr = job
    img, diag, _ = sector_average(Image.fromarray(arr), kind, *_LUT)
    img.save(CACHE / f'{ck}.png')
    return ck, diag


def _q(v):
    v = np.asarray(v, float)
    return {'min': round(float(v.min()), 4), 'median': round(float(np.median(v)), 4),
            'max': round(float(v.max()), 4)}


def build():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    index = IP.build_image_index()
    lut_rgb, lut_val = load_lut(rows, index)
    stab = lut_stability(rows, index, lut_rgb)
    CACHE.mkdir(exist_ok=True)
    jobs = [(kind, ck, np.asarray(crop)) for _, kind, ck, crop in thickness_jobs(rows, index)]
    global _LUT
    _LUT = (lut_rgb, lut_val)
    # The fovea search is a grid vote and takes seconds per map; six processes keep the
    # build near ten minutes without starving the training job's data loader.
    from multiprocessing import Pool
    with Pool(int(os.environ.get('R1_BUILD_PROCS', '6'))) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    out = dict(res)
    d = list(out.values())
    summary = {
        'maps': len(d),
        'by_kind': {k: sum(x['kind'] == k for x in d) for k in ('gca', 'rnfl')},
        'lut_entries': int(len(lut_val)), 'lut_step_um': round(float(lut_val[0] - lut_val[1]), 4),
        'lut_stability_max_rgb_diff': {'n_panels': len(stab), 'max': max(stab) if stab else None,
                                       'median': float(np.median(stab)) if stab else None},
        'centre_fallback': {k: sum(x['centre_fallback'] for x in d if x['kind'] == k) for k in ('gca', 'rnfl')},
        'gca_fit_support': _q([x['fit_support'] for x in d if x['kind'] == 'gca']),
        'gca_flagged (support < %.2f)' % FIT_SUPPORT_MIN: sum(1 for x in d if x['kind'] == 'gca' and x['fit_support'] < FIT_SUPPORT_MIN),
        'gca_status': {st: sum(1 for x in d if x['kind'] == 'gca' and x.get('gca_status') == st)
                       for st in ('fit', 'refit', 'map_mean')},
        'tissue_frac': {k: _q([x['frac_tissue'] for x in d if x['kind'] == k]) for k in ('gca', 'rnfl')},
        'excluded_frac_by_class': {k: {c: _q([x['frac'][c] for x in d if x['kind'] == k])
                                       for c in ('border', 'grey', 'black', 'contour', 'fringe', 'off_ramp')}
                                   for k in ('gca', 'rnfl')},
        'ramp_matched_but_overlay_px': {k: {c: int(sum(x['ramp_matched_but_overlay_px'][c] for x in d if x['kind'] == k))
                                            for c in ('border', 'grey', 'black', 'contour', 'fringe')}
                                        for k in ('gca', 'rnfl')},
        'wedges_filled_from_map_mean': sum(x['wedges_filled_from_map_mean'] for x in d),
        'roundtrip_max_err': max(x['roundtrip_max_err'] for x in d),
    }
    json.dump({'summary': summary, 'maps': out}, open(CACHE / 'index.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


def _overlay(crop, kind, centre):
    im = crop.convert('RGB').copy()
    dr = ImageDraw.Draw(im)
    W, H = im.size
    if centre is None:                       # GCA map-mean map: no geometry to draw
        dr.text((6, 6), 'no usable annulus: map mean', fill=(255, 255, 255))
        return im
    R = 2 * max(W, H)
    if kind == 'rnfl':
        cx, cy = centre[0], centre[1]
        for b in [(105.0 - 30.0 * k) % 360.0 for k in range(12)]:
            t = np.radians(b)
            dx, dy = np.cos(t), np.sin(t)
            dr.line([(cx, cy), (cx + R * dx, cy - R * dy)], fill=(255, 255, 255), width=1)
    else:
        cx, cy, ox, oy, ix, iy = gca_ellipses(centre)
        for b in [60.0 * k for k in range(6)]:
            t = np.radians(b)
            # boundary in elliptical angle space -> screen; rays run across the annulus only
            dr.line([(cx + np.cos(t) * ix, cy - np.sin(t) * iy), (cx + np.cos(t) * ox, cy - np.sin(t) * oy)],
                    fill=(255, 255, 255), width=1)
        for ax, ay in ((ix, iy), (ox, oy)):
            dr.ellipse([cx - ax, cy - ay, cx + ax, cy + ay], outline=(255, 255, 255), width=1)
    dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(0, 0, 0), width=2)
    return im


def render_sample(label, r, index, lut_rgb, lut_val, note=''):
    """One sample sheet: GCA row and RNFL row, six panels each. Returns the per-map diagnostics."""
    key = (r['patient_id'], r['eye'], r['vf_date'])
    paths = index[key]
    lo = r['eye'].lower()
    tiles, diags = [], {}
    S_ = 300
    for kind, rk, fn in (('gca', f'gca_{lo}_thickness', IP._crop_gca_thickness),
                         ('rnfl', f'rnfl_{lo}_thickness', IP._crop_rnfl_thickness)):
        crop = fn(Image.open(paths[rk]).convert('RGB'))
        sec, diag, centre, (tissue, idx) = sector_average(crop, kind, lut_rgb, lut_val, return_tissue=True)
        orig = np.asarray(crop.convert('RGB'))
        exc = orig.copy()
        exc[~tissue] = (255, 255, 255)
        exc[(np.asarray(sec) == orig).all(2) & ~tissue] = (255, 0, 255)
        sec_marked = sec.copy()
        if diag['filled_regions']:
            # outline the regions that took the map mean (too few thickness pixels)
            from scipy import ndimage
            sa = np.asarray(sec_marked).copy()
            for k in diag['filled_regions']:
                reg = idx == k
                edge = reg & ~ndimage.binary_erosion(reg, iterations=2)
                sa[edge] = (255, 0, 0)
            sec_marked = Image.fromarray(sa)
        row_t = [_overlay(crop, kind, centre).resize((S_, S_)), crop.resize((S_, S_)),
                 Image.fromarray(exc).resize((S_, S_), Image.NEAREST), sec_marked.resize((S_, S_)),
                 crop.resize((161, 161)).resize((S_, S_), Image.NEAREST),
                 sec.resize((161, 161)).resize((S_, S_), Image.NEAREST)]
        tiles.append(row_t)
        diags[kind] = diag
    pad, top = 6, 22
    canvas = Image.new('RGB', (6 * S_ + 7 * pad, 2 * (S_ + top) + pad + 14), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    heads = ['centre + regions', 'original crop', 'excl: magenta kept, white filled',
             'sector-averaged (red = map mean)', 'original @161 (model)', 'sector-avg @161 (model)']
    for rr, (kind, row_t) in enumerate(zip(('GCA', 'RNFL'), tiles)):
        y = rr * (S_ + top) + top
        for cc, t in enumerate(row_t):
            x = pad + cc * (S_ + pad)
            canvas.paste(t, (x, y))
            dr.text((x + 2, y - top + 4), f'{kind}: {heads[cc]}', fill=(0, 0, 0))
    exc_txt = '  '.join(f"{k.upper()} excluded {100 * (1 - diags[k]['frac_tissue']):.1f}% ("
                        + ', '.join(f"{c} {100 * v:.1f}" for c, v in diags[k]['frac'].items() if v > 0) + ')'
                        + (f" map-mean regions {diags[k]['filled_regions']}" if diags[k]['filled_regions'] else '')
                        for k in ('gca', 'rnfl'))
    dr.text((pad, 2 * (S_ + top) - 2), f'{label}  eye {r["eye"]}  {note}', fill=(0, 0, 0))
    dr.text((pad, 2 * (S_ + top) + 10), exc_txt, fill=(0, 0, 0))
    canvas.save(SAMPLES / f'{label}.png')
    return {k: {'frac': diags[k]['frac'], 'frac_tissue': diags[k]['frac_tissue'],
                'filled_regions': diags[k]['filled_regions'],
                'ramp_matched_but_overlay_px': diags[k]['ramp_matched_but_overlay_px']} for k in diags}


SAMPLE_DIAG = {}


def samples(n=6):
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    oof = [r for r in rows if r['cv_fold'] != 'test']
    pts = [p for p in IP.PT]

    def msens(r):
        v = [float(r[p]) for p in pts if r.get(p, '') not in ('', '-1')]
        return float(np.mean(v)) if v else np.nan
    oof = [r for r in oof if np.isfinite(msens(r))]
    oof.sort(key=msens)
    picks = [(f'sample_{i:02d}', r, f'mean measured sensitivity {msens(r):.1f} dB')
             for i, r in enumerate([oof[int(q * (len(oof) - 1))] for q in np.linspace(0.04, 0.96, n)], start=1)]
    index = IP.build_image_index()
    lut_rgb, lut_val = load_lut(rows, index)
    # the RNFL maps with the largest data gaps (thickness fraction < 0.6 in the cache index),
    # found by content key so that no patient key is needed here
    idx_doc = json.load(open(CACHE / 'index.json'))['maps']
    gap_keys = {k for k, v in idx_doc.items() if v['kind'] == 'rnfl' and v['frac_tissue'] < 0.6}
    extra, seen = [], set()
    for r in rows:
        p = index.get((r['patient_id'], r['eye'], r['vf_date']), {}).get(f"rnfl_{r['eye'].lower()}_thickness")
        if not p:
            continue
        ck = content_key(IP._crop_rnfl_thickness(Image.open(p).convert('RGB')))
        if ck in gap_keys and ck not in seen:
            seen.add(ck)
            extra.append(r)
    picks += [(f'sample_{n + i:02d}', r, f'large RNFL data gap; mean measured sensitivity {msens(r):.1f} dB')
              for i, r in enumerate(extra, start=1)]
    SAMPLES.mkdir(exist_ok=True)
    priv = Path.home() / 'hvf_private' / 'revision_r1_sample_map.csv'
    lines = []
    for label, r, note in picks:
        SAMPLE_DIAG[label] = render_sample(label, r, index, lut_rgb, lut_val, note)
        lines.append((label, r['patient_id'], r['eye'], r['vf_date']))
    with open(priv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'patient_id', 'eye', 'vf_date'])
        w.writerows(lines)
    os.chmod(priv, 0o600)
    json.dump(SAMPLE_DIAG, open(SAMPLES / 'samples_diag.json', 'w'), indent=1)
    print(f'wrote {len(picks)} samples ({len(extra)} large-gap) to {SAMPLES.relative_to(REPO)}; '
          'key map in the private directory')


if __name__ == '__main__':
    {'build': build, 'samples': samples}[sys.argv[1]]()
