"""Step 1 gate: do the reconstructed masks reproduce the Cirrus summary parameters?

Writes experiments/sector_mask/validation.json, a per-clock-hour signed-bias table and
mask PNGs. Reads ml_dataset.csv / rnfl_detail.csv / oct_values.csv and the extracted
report panels; writes nothing outside experiments/sector_mask/.

Regions are measured under *screen* names (h01..h12, q_up/right/down/left, GCA wedges
by screen centre angle) and only mapped onto the CSV's anatomical names at aggregation
time, so the laterality convention is a lookup and not baked into the pixels.

Usage:
  python experiments/sector_mask/validate.py [--limit N] [--sweep] [--reuse]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import sector_masks as sm
from paths import rewrite_data_path

CFG = sm.load_config()
HOURS = [f'h{k:02d}' for k in range(1, 13)]
QUADS = ['q_s', 'q_t', 'q_i', 'q_n']
SECTORS = ['s_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n']
SCREEN_QUADS = {'up': 90.0, 'left': 180.0, 'down': 270.0, 'right': 0.0}
SCREEN_WEDGES = [30, 90, 150, 210, 270, 330]


# ── screen <-> anatomical lookup ──────────────────────────────
def quad_screen_key(q, eye):
    if q == 'q_s':
        return 'up'
    if q == 'q_i':
        return 'down'
    nasal_right = (eye == 'OD')
    if q == 'q_n':
        return 'right' if nasal_right else 'left'
    return 'left' if nasal_right else 'right'


def sector_screen_key(s, eye, convention):
    cfg = {'gca': dict(CFG['gca'], label_convention=convention)}
    lo, hi = sm.map_region_to_angle(s, eye, cfg)
    return f'a{int(round(((lo + hi) / 2.0) % 360)):03d}'


# ── cohort ────────────────────────────────────────────────────
def build_cohort():
    ml = list(csv.DictReader(open(os.path.join(ROOT, 'ml_dataset.csv'), encoding='utf-8-sig')))
    detail = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'rnfl_detail.csv'), encoding='utf-8-sig')):
        detail[(r['patient_id'], r['eye'], r['vf_date'])] = r
    octv = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'oct_values.csv'), encoding='utf-8-sig')):
        octv[(r['patient_id'], r['eye'])] = r

    def num(d, k):
        try:
            return float(d.get(k, ''))
        except (TypeError, ValueError):
            return np.nan

    seen, cohort = set(), []
    for row in ml:
        pid, eye = row['patient_id'], row['eye']
        gca = rewrite_data_path(row.get('gca_dir', ''))
        rnfl = rewrite_data_path(row.get('rnfl_dir', ''))
        key = (pid, eye, gca, rnfl)
        if key in seen:
            continue
        seen.add(key)
        lo = eye.lower()
        ov = octv.get((pid, eye), {})
        det = detail.get((pid, eye, row['vf_date']), {})
        cohort.append({
            'patient_id': pid, 'eye': eye,
            'gca_map': os.path.join(gca, f'{lo}_thickness_map.png') if gca else '',
            'rnfl_map': os.path.join(rnfl, f'{lo}_thickness_map.png') if rnfl else '',
            'ref_hours': {h: num(det, f'rnfl_{h}') for h in HOURS},
            'ref_quads': {q: num(det, f'rnfl_{q}') for q in QUADS},
            'ref_avg_rnfl': num(ov, f'{lo}_avg_rnfl'),
            'ref_sectors': {s: num(ov, f'{lo}_{s}') for s in SECTORS},
            'ref_avg_gcl': num(ov, f'avg_gcl_{lo}'),
        })
    return cohort


# ── masks in screen frame ─────────────────────────────────────
def rnfl_screen_masks(shape, center, px_per_mm, band_mm):
    H, W = shape
    r_c = CFG['rnfl']['calc_circle_diameter_mm'] / 2.0 * px_per_mm
    bw = band_mm * px_per_mm
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0
    ring = (r >= r_c - bw / 2.0) & (r <= r_c + bw / 2.0)
    out = {}
    for k in range(1, 13):
        c = (90 - 30 * k) % 360
        out[f'h{k:02d}'] = ring & (((th - (c - 15)) % 360.0) < 30.0)
    for name, c in SCREEN_QUADS.items():
        out[name] = ring & (((th - (c - 45)) % 360.0) < 90.0)
    out['avg'] = ring
    return out


def gca_screen_masks(shape, center, semi_px):
    H, W = shape
    cx, cy = center
    sx, sy = semi_px
    ratio = max(CFG['gca']['inner_semi_mm']['x'] / CFG['gca']['outer_semi_mm']['x'],
                CFG['gca']['inner_semi_mm']['y'] / CFG['gca']['outer_semi_mm']['y'])
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - cx) / sx
    v = -(yy - cy) / sy
    rho = np.hypot(u, v)
    annulus = (rho <= 1.0) & (rho >= ratio)
    th = np.degrees(np.arctan2(v, u)) % 360.0
    out = {f'a{c:03d}': annulus & (((th - (c - 30)) % 360.0) < 60.0) for c in SCREEN_WEDGES}
    out['avg'] = annulus
    return out


def _means(vals, valid, masks):
    res = {}
    for name, m in masks.items():
        mv = m & valid
        frac = mv.sum() / max(m.sum(), 1)
        res[name] = float(vals[mv].mean()) if frac >= CFG['decode']['min_valid_fraction'] else np.nan
    return res


# ── per-eye measurement ───────────────────────────────────────
def measure_rnfl(path, lut_rgb, lut_val, bands):
    crop = sm.crop_rnfl(Image.open(path).convert('RGB'))
    if crop is None:
        return None
    arr = np.array(crop)
    H, W, _ = arr.shape
    vals, valid = sm.decode(arr, lut_rgb, lut_val, CFG['decode']['lut_match_max_dist'])
    ppm = W / CFG['rnfl']['map_extent_mm']
    disc = sm.detect_disc_center(arr)
    out = {'shape': [H, W], 'px_per_mm': ppm,
           'disc_offset_mm': None if disc is None else
           [(disc[0] - (W / 2 - 0.5)) / ppm, (disc[1] - (H / 2 - 0.5)) / ppm]}
    centers = {'a': (W / 2.0 - 0.5, H / 2.0 - 0.5), 'b': disc}
    for v, ctr in centers.items():
        out[v] = None if ctr is None else {
            f'{b:.2f}': _means(vals, valid, rnfl_screen_masks((H, W), ctr, ppm, b)) for b in bands}
    return out


def measure_gca(path, lut_rgb, lut_val):
    crop = sm.crop_gca(Image.open(path).convert('RGB'))
    if crop is None:
        return None
    arr = np.array(crop)
    H, W, _ = arr.shape
    vals, valid = sm.decode(arr, lut_rgb, lut_val, CFG['decode']['lut_match_max_dist'])
    ppm = W / CFG['gca']['map_extent_mm']
    sx = CFG['gca']['outer_semi_mm']['x'] * ppm
    sy = CFG['gca']['outer_semi_mm']['y'] * (H / CFG['gca']['map_extent_mm'])
    fit = sm.detect_gca_center(arr, CFG)
    out = {'shape': [H, W], 'px_per_mm': ppm,
           'fovea_offset_mm': None if fit is None else
           [(fit[0] - (W / 2 - 0.5)) / ppm, (fit[1] - (H / 2 - 0.5)) / ppm]}
    variants = {'a': ((W / 2.0 - 0.5, H / 2.0 - 0.5), (sx, sy)),
                'b': None if fit is None else ((fit[0], fit[1]), (fit[2], fit[3]))}
    for v, spec in variants.items():
        if spec is None:
            out[v] = None
            continue
        # GCA panel carries no colorbar, so keep the raw LUT fraction (0..1) and let a
        # single global scale be fitted later, once, across every eye and sector.
        m = _means(vals, valid, gca_screen_masks((H, W), *spec))
        out[v] = {k: (val / CFG['rnfl']['colormap_max_um']) for k, val in m.items()}
    return out


# ── statistics ────────────────────────────────────────────────
def stats(pred, ref):
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok], ref[ok]
    if len(p) < 10 or r.std() < 1e-9:
        return {'n': int(len(p))}
    d = p - r
    b, a = np.polyfit(r, p, 1)
    return {'n': int(len(p)),
            'pearson_r': float(np.corrcoef(p, r)[0, 1]),
            'slope': float(b), 'intercept': float(a),
            'mae': float(np.abs(d).mean()), 'bias': float(d.mean()),
            'loa_lower': float(d.mean() - 1.96 * d.std(ddof=1)),
            'loa_upper': float(d.mean() + 1.96 * d.std(ddof=1)),
            'ref_sd': float(r.std(ddof=1)),
            'mae_over_ref_sd': float(np.abs(d).mean() / r.std(ddof=1)),
            'ref_mean': float(r.mean()), 'pred_mean': float(p.mean())}


def verdict(s):
    c = CFG['criteria']
    if 'pearson_r' not in s:
        return 'INSUFFICIENT_N'
    checks = {'r': s['pearson_r'] >= c['pearson_r_min'],
              'slope': c['slope_range'][0] <= s['slope'] <= c['slope_range'][1],
              'intercept': abs(s['intercept']) <= c['intercept_abs_max_um'],
              'mae_sd': s['mae_over_ref_sd'] <= c['mae_over_sd_max']}
    s['checks'] = checks
    return 'PASS' if all(checks.values()) else 'FAIL:' + ','.join(k for k, v in checks.items() if not v)


# ── aggregation ───────────────────────────────────────────────
def collect_rnfl(recs, variant, band):
    """{param: (pred[], ref[])} in micron, mask value vs Cirrus printed value."""
    acc = {k: ([], []) for k in HOURS + QUADS + ['avg_rnfl']}
    for r in recs:
        m = (r.get('rnfl') or {}).get(variant)
        if not m:
            continue
        mb = m.get(f'{band:.2f}')
        if not mb:
            continue
        eye = r['eye']
        for h in HOURS:
            acc[h][0].append(mb[h])
            acc[h][1].append(r['ref']['hours'][h])
        for q in QUADS:
            acc[q][0].append(mb[quad_screen_key(q, eye)])
            acc[q][1].append(r['ref']['quads'][q])
        acc['avg_rnfl'][0].append(mb['avg'])
        acc['avg_rnfl'][1].append(r['ref']['avg_rnfl'])
    return acc


def collect_gca(recs, variant, convention):
    acc = {k: ([], []) for k in SECTORS + ['avg_gcl']}
    for r in recs:
        m = (r.get('gca') or {}).get(variant)
        if not m:
            continue
        eye = r['eye']
        for s in SECTORS:
            acc[s][0].append(m[sector_screen_key(s, eye, convention)])
            acc[s][1].append(r['ref']['sectors'][s])
        acc['avg_gcl'][0].append(m['avg'])
        acc['avg_gcl'][1].append(r['ref']['avg_gcl'])
    return acc


def gca_global_scale(acc):
    """One zero-intercept scale for the whole GCA table (the panel has no colorbar)."""
    p, r = [], []
    for k, (pv, rv) in acc.items():
        if k == 'avg_gcl':
            continue
        p += list(pv)
        r += list(rv)
    p = np.asarray(p, float)
    r = np.asarray(r, float)
    ok = np.isfinite(p) & np.isfinite(r)
    return float((p[ok] * r[ok]).sum() / (p[ok] ** 2).sum())


def table(acc, order):
    out = {}
    for k in order:
        s = stats(acc[k][0], acc[k][1])
        s['verdict'] = verdict(s)
        out[k] = s
    return out


def fmt(title, tbl, order, unit='um'):
    L = [title,
         f'{"param":10s} {"n":>4s} {"r":>6s} {"slope":>7s} {"icpt":>8s} {"MAE":>7s} '
         f'{"bias":>8s} {"LoA":>17s} {"refSD":>7s} {"MAE/SD":>7s}  verdict']
    for k in order:
        s = tbl[k]
        if 'pearson_r' not in s:
            L.append(f'{k:10s} {s["n"]:4d}   (insufficient n)')
            continue
        L.append(f'{k:10s} {s["n"]:4d} {s["pearson_r"]:6.3f} {s["slope"]:7.3f} '
                 f'{s["intercept"]:+8.2f} {s["mae"]:7.2f} {s["bias"]:+8.2f} '
                 f'[{s["loa_lower"]:+7.2f},{s["loa_upper"]:+7.2f}] '
                 f'{s["ref_sd"]:7.2f} {s["mae_over_ref_sd"]:7.3f}  {s["verdict"]}')
    return '\n'.join(L)


def hour_bias_table(tbl):
    """Signed per-clock-hour bias, decomposed the way the two failure modes separate.

    A wrong calculation-circle radius / band width shifts every hour the same way, so
    bias(h) ~ bias(h+6) and the antipodal sums are large. A wrong centre pushes one
    side of the ring outward and the opposite side inward, so bias(h) ~ -bias(h+6) and
    the antipodal sums cancel. U and A below are exactly those two components.
    """
    b = {h: tbl[h].get('bias', float('nan')) for h in HOURS}
    v = np.array([b[h] for h in HOURS])
    pair_sym = np.array([(b[f'h{k:02d}'] + b[f'h{k + 6:02d}']) / 2.0 for k in range(1, 7)])
    pair_anti = np.array([(b[f'h{k:02d}'] - b[f'h{k + 6:02d}']) / 2.0 for k in range(1, 7)])
    U = float(np.abs(pair_sym).mean())     # symmetric (radius/band) component
    A = float(np.abs(pair_anti).mean())    # antisymmetric (centring) component
    same = bool(np.all(v > 0) or np.all(v < 0))
    L = ['per-clock-hour signed bias (mask - Cirrus, um)',
         f'{"pair":10s} {"bias(h)":>9s} {"bias(h+6)":>10s} {"sym":>8s} {"anti":>8s}']
    for k in range(1, 7):
        h, o = f'h{k:02d}', f'h{k + 6:02d}'
        L.append(f'{h}/{o:6s} {b[h]:+9.2f} {b[o]:+10.2f} '
                 f'{pair_sym[k - 1]:+8.2f} {pair_anti[k - 1]:+8.2f}')
    L.append(f'mean bias {v.mean():+.2f}   all same sign: {same}   '
             f'symmetric |U|={U:.2f}   antisymmetric |A|={A:.2f}   A/U={A / U if U else float("nan"):.2f}')
    if A > 1.5 * U:
        interp = 'centring-dominated -> per-eye disc-centre correction needed'
    elif U > 1.5 * A:
        interp = 'radius/band-dominated -> calculation-circle geometry, config-fixable'
    else:
        interp = 'mixed: neither component dominates'
    L.append('interpretation: ' + interp)
    return '\n'.join(L), {'all_same_sign': same, 'mean_bias': float(v.mean()),
                          'symmetric_component': U, 'antisymmetric_component': A,
                          'interpretation': interp,
                          'bias': {h: float(b[h]) for h in HOURS}}


# ── mask PNGs ─────────────────────────────────────────────────
def dump_masks(recs_paths, lut_rgb, lut_val, outdir):
    os.makedirs(outdir, exist_ok=True)
    for pid, eye, rp, gp in recs_paths:
        if rp and os.path.exists(rp):
            crop = sm.crop_rnfl(Image.open(rp).convert('RGB'))
            arr = np.array(crop)
            H, W, _ = arr.shape
            ppm = W / CFG['rnfl']['map_extent_mm']
            disc = sm.detect_disc_center(arr)
            for v, ctr in (('a', (W / 2 - 0.5, H / 2 - 0.5)), ('b', disc)):
                if ctr is None:
                    continue
                m = rnfl_screen_masks((H, W), ctr, ppm, CFG['rnfl']['band_width_mm'])
                vis = arr.copy()
                for i, h in enumerate(HOURS):
                    vis[m[h]] = np.array([255, 0, 255]) if i % 2 else np.array([0, 255, 0])
                vis[m['h12']] = [255, 255, 255]
                cx, cy = int(round(ctr[0])), int(round(ctr[1]))
                vis[cy - 2:cy + 3, cx - 2:cx + 3] = [0, 0, 0]
                Image.fromarray(vis).save(os.path.join(outdir, f'rnfl_{pid}_{eye}_{v}.png'))
        if gp and os.path.exists(gp):
            crop = sm.crop_gca(Image.open(gp).convert('RGB'))
            arr = np.array(crop)
            H, W, _ = arr.shape
            ppm = W / CFG['gca']['map_extent_mm']
            sx = CFG['gca']['outer_semi_mm']['x'] * ppm
            sy = CFG['gca']['outer_semi_mm']['y'] * (H / CFG['gca']['map_extent_mm'])
            fit = sm.detect_gca_center(arr, CFG)
            for v, spec in (('a', ((W / 2 - 0.5, H / 2 - 0.5), (sx, sy))),
                            ('b', None if fit is None else ((fit[0], fit[1]), (fit[2], fit[3])))):
                if spec is None:
                    continue
                m = gca_screen_masks((H, W), *spec)
                vis = arr.copy()
                for i, c in enumerate(SCREEN_WEDGES):
                    vis[m[f'a{c:03d}']] = np.array([255, 0, 255]) if i % 2 else np.array([0, 255, 0])
                vis[m['a090']] = [255, 255, 255]
                cx, cy = int(round(spec[0][0])), int(round(spec[0][1]))
                vis[cy - 2:cy + 3, cx - 2:cx + 3] = [0, 0, 0]
                Image.fromarray(vis).save(os.path.join(outdir, f'gca_{pid}_{eye}_{v}.png'))


# ── main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--reuse', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'validation.json'))
    args = ap.parse_args()

    cohort = build_cohort()
    if args.limit:
        cohort = cohort[:args.limit]
    print(f'cohort eyes (deduped): {len(cohort)}')

    ref0 = next(c['rnfl_map'] for c in cohort if c['rnfl_map'] and os.path.exists(c['rnfl_map']))
    lut_rgb, lut_val = sm.extract_lut(ref0, CFG['rnfl']['colormap_max_um'])
    print(f'LUT entries: {len(lut_val)}  ({lut_val[0]:.0f} -> {lut_val[-1]:.0f} um)')

    bands = CFG['rnfl']['band_width_sweep_mm'] if args.sweep else [CFG['rnfl']['band_width_mm']]
    cache = os.path.join(HERE, '_measurements.npy')
    if args.reuse and os.path.exists(cache):
        recs = list(np.load(cache, allow_pickle=True))
        print(f'reused {len(recs)} cached measurements')
    else:
        recs = []
        for i, c in enumerate(cohort):
            rec = {'patient_id': c['patient_id'], 'eye': c['eye'],
                   'ref': {'hours': c['ref_hours'], 'quads': c['ref_quads'],
                           'avg_rnfl': c['ref_avg_rnfl'], 'sectors': c['ref_sectors'],
                           'avg_gcl': c['ref_avg_gcl']}}
            if c['rnfl_map'] and os.path.exists(c['rnfl_map']):
                rec['rnfl'] = measure_rnfl(c['rnfl_map'], lut_rgb, lut_val, bands)
            if c['gca_map'] and os.path.exists(c['gca_map']):
                rec['gca'] = measure_gca(c['gca_map'], lut_rgb, lut_val)
            recs.append(rec)
            if (i + 1) % 25 == 0:
                print(f'  {i + 1}/{len(cohort)}', flush=True)
        np.save(cache, np.array(recs, dtype=object), allow_pickle=True)

    report, out = [], {'config': CFG, 'n_eyes': len(recs), 'rnfl': {}, 'gca': {}}

    # RNFL --------------------------------------------------------------
    rnfl_order = HOURS + QUADS + ['avg_rnfl']
    best = None
    for band in bands:
        for variant in ('a', 'b'):
            acc = collect_rnfl(recs, variant, band)
            tbl = table(acc, rnfl_order)
            key = f'{variant}_band{band:.2f}'
            ht, hsum = hour_bias_table(tbl)
            out['rnfl'][key] = {'table': tbl, 'hour_bias': hsum}
            npass = sum(1 for k in rnfl_order if tbl[k]['verdict'] == 'PASS')
            label = ('(a) image centre' if variant == 'a' else '(b) detected disc centre')
            report.append(fmt(f'\n=== RNFL  variant {label}  band {band:.2f} mm  '
                              f'-- {npass}/{len(rnfl_order)} PASS ===', tbl, rnfl_order))
            report.append(ht)
            if best is None or npass > best[0]:
                best = (npass, key)
    out['rnfl_best'] = best[1]

    # GCA ---------------------------------------------------------------
    gca_order = SECTORS + ['avg_gcl']
    gbest = None
    for variant in ('a', 'b'):
        for conv in CFG['gca']['label_convention_options']:
            acc = collect_gca(recs, variant, conv)
            if not acc['avg_gcl'][0]:
                continue
            k = gca_global_scale(acc)
            acc = {kk: ([v * k for v in pv], rv) for kk, (pv, rv) in acc.items()}
            tbl = table(acc, gca_order)
            key = f'{variant}_{conv}'
            out['gca'][key] = {'table': tbl, 'global_scale_um_per_unit': k}
            npass = sum(1 for kk in gca_order if tbl[kk]['verdict'] == 'PASS')
            label = ('(a) image centre' if variant == 'a' else '(b) detected fovea centre')
            report.append(fmt(
                f'\n=== GCA  variant {label}  labels {conv}  -- {npass}/{len(gca_order)} PASS ===\n'
                f'    NOTE: the GCA panel has no colorbar, so values are LUT fractions rescaled\n'
                f'    by ONE global zero-intercept factor k={k:.1f} um fitted across every eye x\n'
                f'    sector. The pooled slope is therefore ~1 by construction and is NOT\n'
                f'    evidence; per-sector slope/intercept and MAE/SD stay free and diagnostic.',
                tbl, gca_order))
            if gbest is None or npass > gbest[0]:
                gbest = (npass, key)
    out['gca_best'] = gbest[1] if gbest else None

    # offsets -----------------------------------------------------------
    do = np.array([r['rnfl']['disc_offset_mm'] for r in recs
                   if r.get('rnfl') and r['rnfl'].get('disc_offset_mm')], float)
    fo = np.array([r['gca']['fovea_offset_mm'] for r in recs
                   if r.get('gca') and r['gca'].get('fovea_offset_mm')], float)
    off = {'disc_offset_mm': {'n': len(do), 'mean': do.mean(0).tolist(),
                              'sd': do.std(0).tolist(),
                              'abs_median': np.median(np.abs(do), 0).tolist()} if len(do) else None,
           'fovea_offset_mm': {'n': len(fo), 'mean': fo.mean(0).tolist(),
                               'sd': fo.std(0).tolist(),
                               'abs_median': np.median(np.abs(fo), 0).tolist()} if len(fo) else None}
    out['centre_offsets'] = off
    report.append('\n=== detected centre offsets from image centre (mm) ===')
    for nm, d in off.items():
        if d:
            report.append(f'{nm:18s} n={d["n"]:4d} mean=({d["mean"][0]:+.3f},{d["mean"][1]:+.3f}) '
                          f'sd=({d["sd"][0]:.3f},{d["sd"][1]:.3f}) '
                          f'|median|=({d["abs_median"][0]:.3f},{d["abs_median"][1]:.3f})')

    dump_masks([(c['patient_id'], c['eye'], c['rnfl_map'], c['gca_map']) for c in cohort[:4]],
               lut_rgb, lut_val, os.path.join(HERE, 'masks'))

    txt = '\n'.join(report)
    print(txt)
    open(os.path.join(HERE, 'validation_report.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=1, allow_nan=True)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
