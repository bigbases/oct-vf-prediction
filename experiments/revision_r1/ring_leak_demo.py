#!/usr/bin/env python3
"""Show and measure the GCA ring leak in the sector-averaged map (read-only).

1. zoom: original | sector-averaged (ring kept as original pixels) | ring painted one colour
2. measure: on 40 GCA maps, does the colour of each ring pixel track the thickness of the
   tissue right next to it? If it does, the kept ring carries thickness along the ellipse.
Writes samples/zoom_ring_<sample>.png. Prints no patient key.
"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
import phi_safe
phi_safe.install()

rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
lut_rgb, lut_val = S.load_lut(rows, index)


def ring_and_tissue(crop):
    arr = np.asarray(crop.convert('RGB'))
    vals, ramp = S.sm.decode(arr, lut_rgb, lut_val, S.MAX_DIST)
    c = S.sm.detect_gca_center(arr, S.CFG)
    ov = S.overlay_masks(arr, 'gca', ramp, c)
    non = np.zeros(ramp.shape, bool)
    for m in ov.values():
        non |= m
    return arr, vals, ov['contour'], ~non, c


# 1. zoom on one sample
sample = sys.argv[1] if len(sys.argv) > 1 else 'sample_02'
priv = Path.home() / 'hvf_private' / 'revision_r1_sample_map.csv'
key = next(tuple(r[1:]) for r in csv.reader(open(priv)) if r[0] == sample)
crop = S.IP._crop_gca_thickness(Image.open(phi_safe.lookup(index, [key], 'image index')[0][f'gca_{key[1].lower()}_thickness']).convert('RGB'))
sec = np.asarray(S.sector_average(crop, 'gca', lut_rgb, lut_val)[0])
arr, vals, ring, tissue, c = ring_and_tissue(crop)
fixed = sec.copy()
fixed[ring] = (98, 85, 170)
cx, cy, sx, sy = c
x0, x1 = int(cx - 0.45 * sx), int(cx + 0.45 * sx)
y0, y1 = int(cy - 0.45 * sy), int(cy + 0.45 * sy)
Z = 6
tiles = [Image.fromarray(np.ascontiguousarray(a[y0:y1, x0:x1])).resize(((x1 - x0) * Z, (y1 - y0) * Z), Image.NEAREST)
         for a in (arr, sec, fixed)]
W, H = tiles[0].size
canvas = Image.new('RGB', (3 * W + 40, H + 30), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
for i, (t, lab) in enumerate(zip(tiles, ['original (A)', 'sector-averaged now (B): ring = original pixels',
                                         'proposal: ring painted one colour'])):
    canvas.paste(t, (10 + i * (W + 10), 25))
    dr.text((10 + i * (W + 10), 6), lab, fill=(0, 0, 0))
out = HERE / 'samples' / f'zoom_ring_{sample}.png'
canvas.save(out)
print('zoom:', out.relative_to(HERE.parents[1]))
rc = arr[ring].astype(float)
print(f'{sample}: ring pixels {ring.sum()}, colour spread (RGB sd) {rc.std(0).round(1).tolist()}, '
      f'example range R {rc[:,0].min():.0f}-{rc[:,0].max():.0f}, B {rc[:,2].min():.0f}-{rc[:,2].max():.0f}')

# 2. measure on 40 GCA maps: ring colour vs thickness of the tissue next to it
rs, per_map = [], []
n = 0
for _, kind, ck, cr in S.thickness_jobs(rows, index):
    if kind != 'gca':
        continue
    arr, vals, ring, tissue, c = ring_and_tissue(cr)
    if ring.sum() < 50:
        continue
    t = np.where(tissue, vals, 0.0)
    w = tissue.astype(float)
    k = np.ones((9, 9))
    near = ndimage.convolve(t, k, mode='constant') / np.maximum(ndimage.convolve(w, k, mode='constant'), 1)
    warm = arr[..., 0].astype(float) - arr[..., 2].astype(float)    # R - B: brown/orange high, purple low
    x, y = warm[ring], near[ring]
    ok = np.isfinite(y) & (ndimage.convolve(w, k, mode='constant')[ring] > 10)
    if ok.sum() > 30:
        per_map.append(np.corrcoef(x[ok], y[ok])[0, 1])
    n += 1
    if n >= 40:
        break
pm = np.array(per_map)
print(f'40 GCA maps: within-map correlation of ring colour (R-B) with the tissue thickness next to it: '
      f'median r {np.median(pm):+.2f} (IQR {np.percentile(pm,25):+.2f} to {np.percentile(pm,75):+.2f}), '
      f'r > 0.5 in {int((pm > 0.5).sum())}/{len(pm)} maps')
