"""Cirrus sector-mask reconstruction on the preprocessed OCT thickness maps.

Step 1 of the sector-averaging contrast experiment: build masks that reproduce the
Cirrus regional summary parameters, and test whether they actually do.

Read-only with respect to runs/, the canonical CSVs and the manuscript. Everything
this module writes lives under experiments/sector_mask/.

Frames
------
Clock hours (rnfl_h01..h12 in clockhours.csv / rnfl_detail.csv) are *screen* names:
ocr_rnfl_detail.py:145 assigns them by screen angle 90-30k around each eye's own
circle, identically for OD and OS. Quadrants (rnfl_q_s/t/i/n) and GCA sectors are
*anatomical* names: the OCR reads od_T at the left of the OD circle and os_T at the
right of the OS circle (ocr_rnfl_detail.py:80-89). The images are never flipped
(image_preprocessing.py:305, flip_os_images defaults to False). So a mask laid on an
OS image must use screen angles for clock hours and eye-dependent angles for
quadrants/sectors. See map_region_to_angle().

The GCA sector labels are the one case the code could not settle by reading, so both
readings were validated on 333 eyes and the data decided: oct_values.csv's *_t / *_n
sector columns follow ocr_oct_values.py:157 (OD s_sup_t read at screen 30 deg, i.e.
image right), which is the *opposite* side from the RNFL quadrant columns, where OD
temporal is image left (ocr_rnfl_detail.py:80-89, cross-checked by the quadrant =
mean-of-3-clock-hours identity in make_rnfl_flip.py). Under the RNFL convention the
four GCA T/N sectors collapse to r ~ 0.53-0.64; under the OCR convention they reach
r ~ 0.87-0.94. Both maps are the same eye in the same fundus orientation, so the two
cannot both be anatomically right -- see docs/LATERALITY_AUDIT.md.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'config.json')


def load_config(path=CONFIG_PATH):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 1. Colour lookup table, taken from the RNFL panel's own colorbar
# ─────────────────────────────────────────────────────────────
def _bar_columns(im):
    """Columns of the vertical colorbar: tall, non-white, and flat horizontally."""
    H, W, _ = im.shape
    nonwhite = im.sum(2) < 730
    hdiff = np.abs(np.diff(im.astype(float), axis=1)).mean(axis=(0, 2))
    cand = [x for x in range(W - 1) if nonwhite[:, x].sum() > 0.85 * H and hdiff[x] < 1.0]
    if not cand:
        return None
    runs, cur = [], [cand[0]]
    for x in cand[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    runs = [r for r in runs if len(r) >= 8]
    return runs[0] if runs else None


def extract_lut(rnfl_panel_path, max_um=350.0):
    """(N,3) float RGB ramp and matching (N,) micron values, top of bar -> max_um.

    The bar's own vertical extent is taken from its border column, which is the only
    part that stays non-white all the way to the top (the ramp itself ends in white).
    """
    im = np.array(Image.open(rnfl_panel_path).convert('RGB')).astype(int)
    bar = _bar_columns(im)
    if bar is None:
        return None, None
    border = bar[0] - 6
    nonwhite = im.sum(2) < 745
    cands = [x for x in range(max(0, bar[0] - 10), bar[0])
             if nonwhite[:, x].sum() > 0.85 * im.shape[0]]
    if cands:
        border = cands[0]
    ys = np.where(nonwhite[:, border])[0]
    y0, y1 = int(ys[0]), int(ys[-1])
    inner = bar[2:-2] if len(bar) > 6 else bar
    lut_rgb = im[y0:y1 + 1, inner, :].astype(float).mean(1)
    lut_val = max_um * np.linspace(1.0, 0.0, lut_rgb.shape[0])
    return lut_rgb, lut_val


def decode(patch_rgb, lut_rgb, lut_val, max_dist):
    """Nearest-colour inversion. Returns (values, valid_mask).

    Pixels further than max_dist from the ramp (grey disc overlay, panel frame,
    vessel annotation, greyscale LSO background) are marked invalid.
    """
    flat = patch_rgb.reshape(-1, 3).astype(np.float32)
    lut = lut_rgb.astype(np.float32)
    best_i = np.zeros(flat.shape[0], dtype=np.int64)
    best_d = np.full(flat.shape[0], np.inf, dtype=np.float32)
    step = 20000
    for s in range(0, flat.shape[0], step):
        chunk = flat[s:s + step]
        d = ((chunk[:, None, :] - lut[None, :, :]) ** 2).sum(2)
        i = d.argmin(1)
        best_i[s:s + step] = i
        best_d[s:s + step] = np.sqrt(d[np.arange(len(chunk)), i])
    vals = lut_val[best_i].reshape(patch_rgb.shape[:2])
    valid = (best_d <= max_dist).reshape(patch_rgb.shape[:2])
    return vals, valid


# ─────────────────────────────────────────────────────────────
# 2. Crops -- byte-identical to what image_preprocessing.py feeds the model
# ─────────────────────────────────────────────────────────────
def crop_rnfl(img):
    """Mirror of image_preprocessing._crop_rnfl_thickness (top_margin=16)."""
    W, H = img.size
    sq = H - 16
    arr = np.array(img.convert('RGB')).astype(float)
    mx = arr.max(2)
    mn = arr.min(2)
    sat = (mx - mn) / (mx + 1e-5)
    colored = (sat > 0.2) & (mx > 30)
    cols = np.where(colored.any(axis=0))[0]
    if len(cols) == 0:
        return None
    right = int(cols[-1]) + 1
    left = max(0, right - sq)
    return img.crop((left, 16, right, H))


def crop_gca(img):
    """Mirror of image_preprocessing._crop_gca_thickness."""
    arr = np.array(img.convert('RGB')).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue = (b > r + 40) & (b > g + 40) & (b > 80)
    rows = np.where(blue.sum(axis=1) > 30)[0]
    cols = np.where(blue.sum(axis=0) > 30)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


# ─────────────────────────────────────────────────────────────
# 3. Per-eye centring (variant b)
# ─────────────────────────────────────────────────────────────
def detect_disc_center(crop_rgb):
    """Centroid of the grey optic-disc overlay on the RNFL thickness map.

    Cirrus paints the detected disc margin as a flat grey blob. It is the only
    desaturated region inside the map, so a saturation threshold plus the largest
    connected component picks it out. Returns (cx, cy) or None.
    """
    a = crop_rgb.astype(float)
    mx = a.max(2)
    mn = a.min(2)
    sat = (mx - mn) / (mx + 1e-5)
    grey = (sat < 0.12) & (mx > 60) & (mx < 240)
    H, W = grey.shape
    grey[:4, :] = grey[-4:, :] = grey[:, :4] = grey[:, -4:] = False
    lab, n = ndimage.label(grey)
    if n == 0:
        return None
    sizes = ndimage.sum(grey, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 0.005 * H * W:
        return None
    cy, cx = ndimage.center_of_mass(grey, lab, k)
    return float(cx), float(cy)


def gca_contour_mask(crop_rgb):
    """The two drawn annulus contours: dark and reddish, unlike the foveal pit."""
    a = crop_rgb.astype(float)
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = (R + G + B) / 3.0
    return (lum < 140) & (R > G)


def detect_gca_center(crop_rgb, cfg, span_mm=0.7, step_px=0.5, tol_px=1.6):
    """Fovea centre from the annulus ellipses Cirrus draws on the GCA map.

    The contours are broken by the colour they sit on, so a per-ray outer-contour
    trace picks up outliers and diverges. Instead the ellipse *sizes* are known --
    outer 4.8x4.0 mm, inner 1.2x1.0 mm on a 6 mm map -- so only the centre is
    unknown, and it is found by voting: for each candidate centre, score how many
    contour pixels fall on the two predicted ellipses. Missing arcs only lower the
    score, they do not bias it.

    Returns (cx, cy, semi_x_px, semi_y_px) or None.
    """
    line = gca_contour_mask(crop_rgb)
    if line.sum() < 150:
        return None
    H, W = line.shape
    ext = cfg['gca']['map_extent_mm']
    ppmx, ppmy = W / ext, H / ext
    semis = [(cfg['gca']['outer_semi_mm']['x'], cfg['gca']['outer_semi_mm']['y']),
             (cfg['gca']['inner_semi_mm']['x'], cfg['gca']['inner_semi_mm']['y'])]
    ys, xs = np.where(line)
    xs = xs.astype(float)
    ys = ys.astype(float)
    cx0, cy0 = W / 2.0 - 0.5, H / 2.0 - 0.5
    offs = np.arange(-span_mm * ppmx, span_mm * ppmx + 1e-9, step_px)
    best = None
    for dy in offs:
        for dx in offs:
            cx, cy = cx0 + dx, cy0 + dy
            score = 0.0
            for sxmm, symm in semis:
                sx, sy = sxmm * ppmx, symm * ppmy
                rho = np.hypot((xs - cx) / sx, (ys - cy) / sy)
                d = np.abs(rho - 1.0) * np.hypot(sx, sy) / np.sqrt(2.0)
                score += np.exp(-(d / tol_px) ** 2).sum()
            if best is None or score > best[0]:
                best = (score, cx, cy)
    _, cx, cy = best
    return float(cx), float(cy), float(semis[0][0] * ppmx), float(semis[0][1] * ppmy)


# ─────────────────────────────────────────────────────────────
# 4. Region name -> screen angle
# ─────────────────────────────────────────────────────────────
def map_region_to_angle(region, eye, cfg):
    """Screen-frame angle window (deg, CCW from +x, y up) for a named region.

    Clock hours: screen names, so identical for OD and OS -- no laterality term.
    Quadrants and GCA sectors: anatomical names, so temporal/nasal swap sides between
    eyes. In fundus orientation the nasal retina is on the image right for OD and on
    the image left for OS (confirmed on the GCA panel, where the optic disc sits right
    of the fovea for OD and left of it for OS).
    """
    if region.startswith('h'):
        k = int(region[1:])
        c = (90 - 30 * k) % 360
        return c - 15.0, c + 15.0

    if region in ('q_s', 'q_i', 'q_t', 'q_n'):
        if region == 'q_s':
            return 45.0, 135.0
        if region == 'q_i':
            return 225.0, 315.0
        nasal_right = (eye == 'OD')
        right = (315.0, 405.0)
        left = (135.0, 225.0)
        if region == 'q_n':
            return right if nasal_right else left
        return left if nasal_right else right

    if region.startswith('s_'):
        half = cfg['gca']['sector_width_deg'] / 2.0
        vertical = {'s_sup': 90.0, 's_inf': 270.0}
        if region in vertical:
            c = vertical[region]
            return c - half, c + half
        # upper/lower x nasal/temporal
        upper = region.startswith('s_sup')
        temporal = region.endswith('_t')
        # Anatomical: temporal retina is image-right for OS (the GCA panel shows the
        # disc right of the fovea for OD, left of it for OS). ocr_oct_values.py:157
        # instead puts OD s_sup_t at 30 deg, i.e. image-right for OD -- the opposite.
        # Which one the CSV columns actually follow is decided empirically, so the
        # convention is a config switch and both are validated.
        temporal_right = (eye == 'OS') if cfg['gca'].get('label_convention',
                                                         'anatomical') == 'anatomical' \
            else (eye == 'OD')
        on_right = (temporal == temporal_right)
        if upper:
            c = 30.0 if on_right else 150.0
        else:
            c = 330.0 if on_right else 210.0
        return c - half, c + half

    raise ValueError(f'unknown region {region!r}')


def _in_wedge(theta_deg, lo, hi):
    t = (theta_deg - lo) % 360.0
    return t <= (hi - lo) % 360.0 if (hi - lo) % 360.0 != 0 else np.ones_like(t, bool)


# ─────────────────────────────────────────────────────────────
# 5. Mask construction
# ─────────────────────────────────────────────────────────────
def rnfl_masks(shape, center, px_per_mm, eye, cfg, band_width_mm=None):
    """Clock-hour, quadrant and whole-circle masks on the RNFL calculation circle."""
    H, W = shape
    r_c = cfg['rnfl']['calc_circle_diameter_mm'] / 2.0 * px_per_mm
    bw = (band_width_mm if band_width_mm is not None else cfg['rnfl']['band_width_mm']) * px_per_mm
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = -(yy - cy)
    r = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx)) % 360.0
    ring = (r >= r_c - bw / 2.0) & (r <= r_c + bw / 2.0)

    out = {}
    for k in range(1, 13):
        lo, hi = map_region_to_angle(f'h{k:02d}', eye, cfg)
        out[f'h{k:02d}'] = ring & _in_wedge(th, lo % 360.0, hi % 360.0 if hi % 360.0 else 360.0)
    for q in ('q_s', 'q_t', 'q_i', 'q_n'):
        lo, hi = map_region_to_angle(q, eye, cfg)
        out[q] = ring & _in_wedge(th, lo % 360.0, hi)
    out['avg'] = ring
    return out


def gca_masks(shape, center, semi_px, eye, cfg):
    """Six-sector elliptical annulus masks on the GCA thickness map."""
    H, W = shape
    cx, cy = center
    sx_out, sy_out = semi_px
    ratio = cfg['gca']['inner_semi_mm']['x'] / cfg['gca']['outer_semi_mm']['x']
    ratio_y = cfg['gca']['inner_semi_mm']['y'] / cfg['gca']['outer_semi_mm']['y']
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - cx) / sx_out
    v = -(yy - cy) / sy_out
    rho = np.hypot(u, v)
    annulus = (rho <= 1.0) & (rho >= max(ratio, ratio_y))

    if cfg['gca']['sector_angle_space'] == 'elliptical':
        th = np.degrees(np.arctan2(v, u)) % 360.0
    else:
        th = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0

    out = {}
    for region in ('s_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n'):
        lo, hi = map_region_to_angle(region, eye, cfg)
        out[region] = annulus & _in_wedge(th, lo % 360.0, (lo % 360.0) + (hi - lo))
    out['avg'] = annulus
    return out
