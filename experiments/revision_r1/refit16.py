#!/usr/bin/env python3
"""Refit the 16 flagged GCA maps and draw old (white) and refitted (green) ellipses.
Writes samples/qc_gca_refit16.png and cache/refit16.json (content key -> both fits)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
import annulus_fit as AF

adj = json.load(open(S.CACHE / 'fit_adjudication.json'))['maps']
# start-search region: where the successful fits put the fovea (0.5-99.5 percentile of the
# fit centres of the unflagged maps, as fractions of the crop) widened by 0.03
fc = json.load(open(S.CACHE / 'fit_centres.json'))
ok = np.array([[v['fx'], v['fy']] for k, v in fc.items() if v and k not in adj])
REGION = (float(np.percentile(ok[:, 0], 0.5)) - 0.03, float(np.percentile(ok[:, 0], 99.5)) + 0.03,
          float(np.percentile(ok[:, 1], 0.5)) - 0.03, float(np.percentile(ok[:, 1], 99.5)) + 0.03)
print('start region (fx0, fx1, fy0, fy1):', [round(x, 3) for x in REGION])
rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
out, tiles = {}, []


def ell(dr, c, col, w=2):
    cx, cy, sx, sy = c[:4]
    for f in (1.0, 0.25):
        dr.ellipse([cx - f * sx, cy - f * sy, cx + f * sx, cy + f * sy], outline=col, width=w)


for _, kind, ck, crop in S.thickness_jobs(rows, index):
    if kind != 'gca' or ck not in adj or ck in out:
        continue
    arr = np.asarray(crop)
    H, W, _ = arr.shape
    ppx, ppy = W / S.CFG['gca']['map_extent_mm'], H / S.CFG['gca']['map_extent_mm']
    size = (S.CFG['gca']['outer_semi_mm']['x'] * ppx, S.CFG['gca']['outer_semi_mm']['y'] * ppy)
    old = AF.fit_annulus(arr, S.CFG)
    lum, valid = AF._lum_and_valid(arr)
    old_c = AF.ring_contrast(lum, valid, *old[:4]) if old else None
    starts = [AF.fovea_start(arr, REGION)]
    ir = AF.inner_ring_start(arr, S.CFG, REGION)
    if ir:
        starts.append(ir)
    new = AF.refit_annulus(arr, S.CFG, size, starts)
    out[ck] = {'old': list(old) if old else None, 'old_contrast': old_c, 'refit': list(new),
               'starts': starts, 'old_support': adj[ck]['support']}
    im = crop.convert('RGB').copy()
    dr = ImageDraw.Draw(im)
    if old:
        ell(dr, old, (255, 255, 255))
    ell(dr, new, (0, 255, 0))
    for (x, y) in starts:
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 0, 255), width=2)
    im = im.resize((260, 260))
    ImageDraw.Draw(im).text((4, 4), f"sup {adj[ck]['support']:.2f}  c {old_c if old_c is None else round(old_c,1)} -> {new[4]:.1f}",
                            fill=(255, 255, 255))
    tiles.append((adj[ck]['support'], ck, im))
tiles.sort(key=lambda t: t[0])
g = Image.new('RGB', (4 * 264, 4 * 264), (255, 255, 255))
for i, (_, ck, im) in enumerate(tiles):
    g.paste(im, ((i % 4) * 264, (i // 4) * 264))
    ImageDraw.Draw(g).text(((i % 4) * 264 + 4, (i // 4) * 264 + 246), f'#{i + 1:02d}', fill=(255, 255, 0))
    out[ck]['tile'] = i + 1
g.save(S.SAMPLES / 'qc_gca_refit16.png')
json.dump({'region': REGION, 'maps': out}, open(S.CACHE / 'refit16.json', 'w'), indent=1)
print('done', len(tiles))
