#!/usr/bin/env python3
"""Fovea / annulus fit for the GCA map that does not depend on the ring's colour and allows
a decentred fovea.

Why: experiments/sector_mask detect_gca_center() searches +-0.7 mm around the image centre
and votes with dark-reddish line pixels. On decentred scans (fovea at an edge or corner) it
returns a wrong centre without failing, and on maps whose ring is drawn in a light blue-violet
it sees little of the line (2026-10-13 QC: 6 visible misfits among the 16 lowest-fit maps).

Here: line pixels = pixels that differ from their 5x5 median by more than LINE_T RGB units
(a thin drawn line does; smooth thickness does not). The two ellipses have known size
(outer 4.8 x 4.0 mm, inner 1.2 x 1.0 mm on a 6 mm map); the centre is found by a coarse vote
over +-2.5 mm (3 px steps) refined at 0.5 px. Support = share of points along the fitted
ellipses (inside the crop) that have a line pixel within 1.5 px.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

LINE_T = 30.0
SPAN_MM = 2.5


def line_mask(arr):
    a = arr.astype(np.float32)
    med = ndimage.median_filter(a, size=(5, 5, 1))
    return np.sqrt(((a - med) ** 2).sum(2)) > LINE_T


def _score(xs, ys, cx, cy, semis, tol):
    s = 0.0
    for sx, sy in semis:
        rho = np.hypot((xs - cx) / sx, (ys - cy) / sy)
        d = np.abs(rho - 1.0) * np.hypot(sx, sy) / np.sqrt(2.0)
        s += np.exp(-(d / tol) ** 2).sum()
    return s


def fit_annulus(arr, cfg, line=None):
    """(cx, cy, outer_sx, outer_sy, support) in pixels."""
    H, W, _ = arr.shape
    line = line_mask(arr) if line is None else line
    # ignore the frame band, where the panel border makes long straight 'lines'
    b = int(np.ceil(0.02 * W)) + 2
    line[:b] = line[-b:] = False
    line[:, :b] = line[:, -b:] = False
    ext = cfg['gca']['map_extent_mm']
    ppx, ppy = W / ext, H / ext
    semis = [(cfg['gca']['outer_semi_mm']['x'] * ppx, cfg['gca']['outer_semi_mm']['y'] * ppy),
             (cfg['gca']['inner_semi_mm']['x'] * ppx, cfg['gca']['inner_semi_mm']['y'] * ppy)]
    ys, xs = np.where(line)
    if xs.size < 100:
        return None
    # subsample for the coarse pass
    rng = np.random.default_rng(0)
    pick = rng.choice(xs.size, size=min(xs.size, 2500), replace=False)
    xc, yc = xs[pick].astype(float), ys[pick].astype(float)
    cx0, cy0 = W / 2.0 - 0.5, H / 2.0 - 0.5
    best = None
    for dy in np.arange(-SPAN_MM * ppy, SPAN_MM * ppy + 1e-9, 3.0):
        for dx in np.arange(-SPAN_MM * ppx, SPAN_MM * ppx + 1e-9, 3.0):
            sc = _score(xc, yc, cx0 + dx, cy0 + dy, semis, 2.5)
            if best is None or sc > best[0]:
                best = (sc, cx0 + dx, cy0 + dy)
    _, bx, by = best
    xf, yf = xs.astype(float), ys.astype(float)
    best = None
    for dy in np.arange(-4.0, 4.01, 0.5):
        for dx in np.arange(-4.0, 4.01, 0.5):
            sc = _score(xf, yf, bx + dx, by + dy, semis, 1.6)
            if best is None or sc > best[0]:
                best = (sc, bx + dx, by + dy)
    _, cx, cy = best
    # support along the fitted ellipses, inside the crop
    near = ndimage.binary_dilation(line, iterations=2)
    hit = tot = 0
    for sx, sy in semis:
        t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        px, py = np.round(cx + sx * np.cos(t)).astype(int), np.round(cy + sy * np.sin(t)).astype(int)
        ok = (px >= b) & (px < W - b) & (py >= b) & (py < H - b)
        hit += int(near[py[ok], px[ok]].sum()); tot += int(ok.sum())
    support = hit / tot if tot else 0.0
    return float(cx), float(cy), float(semis[0][0]), float(semis[0][1]), float(support)


# ---------------------------------------------------------------------------------------
# Refit for maps whose first fit is wrong while fovea and ring are visible (2026-10-13 QC,
# author instructions): start at the fovea - searched only where successful fits put it, so
# that a dark damaged area is not taken for the fovea - size from the successful fits, then
# refine centre and size so that the drawn ring contrasts most with its surroundings, darker
# or lighter (on thin advanced maps the background is dark and the ring can be the lighter).
# Colour-agnostic: the ring is drawn in different colours on different maps.
# ---------------------------------------------------------------------------------------
def _lum_and_valid(arr, border_frac=0.02):
    a = arr.astype(np.float32)
    lum = a.mean(2)
    H, W = lum.shape
    valid = a.max(2) >= 40                      # no-data black is not tissue
    b = int(np.ceil(border_frac * W)) + 2
    valid[:b] = valid[-b:] = False
    valid[:, :b] = valid[:, -b:] = False
    return lum, valid


def fovea_start(arr, region=None, sigma=6.0):
    """Darkest point of the luminance map smoothed over valid pixels only, searched inside
    region = (fx0, fx1, fy0, fy1) in fractions of the crop when given."""
    lum, valid = _lum_and_valid(arr)
    w = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
    sm = ndimage.gaussian_filter(np.where(valid, lum, 0.0).astype(np.float32), sigma) / np.maximum(w, 1e-3)
    sm[w < 0.6] = np.inf                        # too close to no-data or the frame
    if region is not None:
        H, W = sm.shape
        keep = np.zeros_like(sm, bool)
        keep[int(region[2] * H):int(np.ceil(region[3] * H)), int(region[0] * W):int(np.ceil(region[1] * W))] = True
        sm[~keep] = np.inf
    y, x = np.unravel_index(np.argmin(sm), sm.shape)
    return float(x), float(y)


def inner_ring_start(arr, cfg, region=None):
    """Centre of the small (inner) ellipse by a vote over line pixels, inner ellipse only,
    over centres inside region (fractions of the crop) when given."""
    H, W, _ = arr.shape
    line = line_mask(arr)
    b = int(np.ceil(0.02 * W)) + 2
    line[:b] = line[-b:] = False
    line[:, :b] = line[:, -b:] = False
    ys, xs = np.where(line)
    if xs.size < 50:
        return None
    ppx, ppy = W / cfg['gca']['map_extent_mm'], H / cfg['gca']['map_extent_mm']
    semis = [(cfg['gca']['inner_semi_mm']['x'] * ppx, cfg['gca']['inner_semi_mm']['y'] * ppy)]
    best = None
    x_lo, x_hi, y_lo, y_hi = (8, W - 8, 8, H - 8) if region is None else \
        (region[0] * W, region[1] * W, region[2] * H, region[3] * H)
    for cy in np.arange(y_lo, y_hi, 3.0):
        for cx in np.arange(x_lo, x_hi, 3.0):
            sc = _score(xs.astype(float), ys.astype(float), cx, cy, semis, 1.6)
            if best is None or sc > best[0]:
                best = (sc, cx, cy)
    return best[1], best[2]


def ring_contrast(lum, valid, cx, cy, sx, sy, off=4.0, n=360):
    """Mean over both ellipses of |(luminance 4 px off the ellipse, both sides) - (luminance
    on it)|, valid points only. Higher = the ellipses sit on a line that stands out from its
    surroundings, darker or lighter. Averaging both sides makes a one-sided edge count half."""
    H, W = lum.shape
    vals = []
    for f in (1.0, 0.25):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ex, ey = f * sx * np.cos(t), f * sy * np.sin(t)
        nx, ny = ex / (f * sx) ** 2, ey / (f * sy) ** 2
        nn = np.hypot(nx, ny) + 1e-9
        nx, ny = nx / nn, ny / nn
        px, py = cx + ex, cy - ey
        on = np.stack([px, py], 1)
        pts = [on, on + off * np.stack([nx, -ny], 1), on - off * np.stack([nx, -ny], 1)]
        ix = [np.clip(np.round(q[:, 0]).astype(int), 0, W - 1) for q in pts]
        iy = [np.clip(np.round(q[:, 1]).astype(int), 0, H - 1) for q in pts]
        inside = np.ones(n, bool)
        for q, a, b in zip(pts, ix, iy):
            inside &= (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H) & valid[b, a]
        if inside.sum() < 0.2 * n:
            continue
        c = np.abs(0.5 * (lum[iy[1], ix[1]] + lum[iy[2], ix[2]]) - lum[iy[0], ix[0]])
        vals.append(c[inside])
    return float(np.concatenate(vals).mean()) if vals else -np.inf


def refit_annulus(arr, cfg, size_px, starts, span=25.0):
    """Refine centre and scale around each start; returns (cx, cy, sx, sy, contrast)."""
    lum, valid = _lum_and_valid(arr)
    sx0, sy0 = size_px
    best = None
    for (x0, y0) in starts:
        for dy in np.arange(-span, span + 1e-9, 2.0):
            for dx in np.arange(-span, span + 1e-9, 2.0):
                for s in np.arange(0.85, 1.2001, 0.05):
                    c = ring_contrast(lum, valid, x0 + dx, y0 + dy, s * sx0, s * sy0)
                    if best is None or c > best[0]:
                        best = (c, x0 + dx, y0 + dy, s)
    _, bx, by, bs = best
    for dy in np.arange(-2.0, 2.01, 0.5):
        for dx in np.arange(-2.0, 2.01, 0.5):
            for s in np.arange(bs - 0.04, bs + 0.0401, 0.01):
                c = ring_contrast(lum, valid, bx + dx, by + dy, s * sx0, s * sy0)
                if c > best[0]:
                    best = (c, bx + dx, by + dy, s)
    c, cx, cy, s = best
    return float(cx), float(cy), float(s * sx0), float(s * sy0), float(c)


def ellipse_contrast(lum, valid, cx, cy, ax, ay, off=3.0, n=360):
    """|mean of both sides - on| along ONE ellipse (semi-axes ax, ay), valid points only."""
    H, W = lum.shape
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ex, ey = ax * np.cos(t), ay * np.sin(t)
    nx, ny = ex / ax ** 2, ey / ay ** 2
    nn = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / nn, ny / nn
    pts = [(cx + ex, cy - ey), (cx + ex + off * nx, cy - ey - off * ny), (cx + ex - off * nx, cy - ey + off * ny)]
    ix = [np.round(x).astype(int) for x, _ in pts]
    iy = [np.round(y).astype(int) for _, y in pts]
    inside = np.ones(n, bool)
    for a, b in zip(ix, iy):
        inside &= (a >= 0) & (a < W) & (b >= 0) & (b < H)
    if inside.sum() < 0.25 * n:
        return -np.inf, 0.0
    a0, b0 = [np.clip(v, 0, W - 1) for v in ix], [np.clip(v, 0, H - 1) for v in iy]
    ok = inside & valid[b0[0], a0[0]] & valid[b0[1], a0[1]] & valid[b0[2], a0[2]]
    if ok.sum() < 0.2 * n:
        return -np.inf, 0.0
    c = np.abs(0.5 * (lum[b0[1], a0[1]] + lum[b0[2], a0[2]]) - lum[b0[0], a0[0]])
    return float(c[ok].mean()), float(ok.mean())


def refit_annulus_v2(arr, cfg, starts):
    """Two-stage refit for crops whose scale differs from the nominal one.
    1) centre and size of the INNER ellipse alone (isotropic scale 0.8-1.6 of nominal),
       searched +-15 px around each start; 2) with that centre (+-3 px), the OUTER ellipse's
       horizontal and vertical semi-axes independently (0.8-1.6 of nominal).
    Returns (cx, cy, outer_ax, outer_ay, inner_scale, contrast_inner, contrast_outer)."""
    lum, valid = _lum_and_valid(arr)
    H, W = lum.shape
    # The map is 6 x 6 mm. On crops cut along one side (portrait or landscape boxes from the
    # crop detector) the uncut side carries the true scale, so both axes use the longer side.
    ppx = ppy = max(W, H) / cfg['gca']['map_extent_mm']
    ix0, iy0 = cfg['gca']['inner_semi_mm']['x'] * ppx, cfg['gca']['inner_semi_mm']['y'] * ppy
    ox0, oy0 = cfg['gca']['outer_semi_mm']['x'] * ppx, cfg['gca']['outer_semi_mm']['y'] * ppy
    best = None
    for (x0, y0) in starts:
        for dy in np.arange(-15, 15.01, 1.0):
            for dx in np.arange(-15, 15.01, 1.0):
                for si in np.arange(0.8, 1.601, 0.05):
                    c, cov = ellipse_contrast(lum, valid, x0 + dx, y0 + dy, si * ix0, si * iy0)
                    if cov > 0.6 and (best is None or c > best[0]):
                        best = (c, x0 + dx, y0 + dy, si)
    if best is None:
        return None
    ci, cx, cy, si = best
    bo = None
    for dy in np.arange(-3, 3.01, 1.0):
        for dx in np.arange(-3, 3.01, 1.0):
            for sx in np.arange(0.8, 1.601, 0.025):
                for sy in np.arange(0.8, 1.601, 0.025):
                    c, cov = ellipse_contrast(lum, valid, cx + dx, cy + dy, sx * ox0, sy * oy0)
                    if bo is None or c > bo[0]:
                        bo = (c, cx + dx, cy + dy, sx, sy)
    co, cx2, cy2, sx, sy = bo
    return float(cx2), float(cy2), float(sx * ox0), float(sy * oy0), float(si), float(ci), float(co)
