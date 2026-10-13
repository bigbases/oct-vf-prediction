#!/usr/bin/env python3
"""Two-stage refit (annulus_fit.refit_annulus_v2) of the 16 flagged GCA maps, with starts
confined to where successful fits put the fovea. Draws the drawn ring next to the refit
(green, inner and outer) at 2x. Writes cache/refit16_v2.json and samples/zoomv2_<tile>.png."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
import annulus_fit as AF
R1 = json.load(open(S.CACHE / 'refit16.json'))
REGION, prev = R1['region'], R1['maps']
WANT = {1, 4, 7, 9, 10, 12, 13, 14, 15, 16}
rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
out, seen = {}, set()
for _, kind, ck, crop in S.thickness_jobs(rows, index):
    if kind != 'gca' or ck not in prev or ck in seen or prev[ck]['tile'] not in WANT:
        continue
    seen.add(ck)
    arr = np.asarray(crop)
    starts = [tuple(x) for x in prev[ck]['starts']]
    f = AF.refit_annulus_v2(arr, S.CFG, starts)
    tile = prev[ck]['tile']
    H, W, _ = arr.shape
    out[ck] = {'tile': tile, 'refit_v2': f}
    Z = 2
    base = crop.convert('RGB').resize((W * Z, H * Z), Image.NEAREST)
    fit = base.copy()
    if f:
        cx, cy, ox, oy, si = f[0] * Z, f[1] * Z, f[2] * Z, f[3] * Z, f[4]
        ppx = ppy = max(W, H) / S.CFG['gca']['map_extent_mm']
        ix, iy = si * S.CFG['gca']['inner_semi_mm']['x'] * ppx * Z, si * S.CFG['gca']['inner_semi_mm']['y'] * ppy * Z
        dr = ImageDraw.Draw(fit)
        dr.ellipse([cx - ox, cy - oy, cx + ox, cy + oy], outline=(0, 255, 0), width=1)
        dr.ellipse([cx - ix, cy - iy, cx + ix, cy + iy], outline=(0, 255, 0), width=1)
    t = Image.new('RGB', (2 * W * Z + 10, H * Z + 18), (255, 255, 255))
    t.paste(base, (0, 18)); t.paste(fit, (W * Z + 10, 18))
    lab = f"#{tile:02d}: drawn ring | two-stage refit (green)" + (f"  contrast inner {f[5]:.1f} outer {f[6]:.1f}" if f else '  no refit')
    ImageDraw.Draw(t).text((4, 2), lab, fill=(0, 0, 0))
    t.save(S.SAMPLES / f'zoomv3_{tile:02d}.png')
json.dump({'region': REGION, 'maps': out}, open(S.CACHE / 'refit16_v3.json', 'w'), indent=1)
print('done', len(out))
