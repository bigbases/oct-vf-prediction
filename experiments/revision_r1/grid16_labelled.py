#!/usr/bin/env python3
"""The 16 flagged GCA maps with their adjudication: first fit (white), two-stage refit
(green = accepted, orange = rejected), label a / b / normal. samples/qc_gca_16_labelled.png"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sector_maps as S
adj = json.load(open(S.CACHE / 'fit_adjudication.json'))['maps']
r1 = json.load(open(S.CACHE / 'refit16.json'))['maps']
r3 = json.load(open(S.CACHE / 'refit16_v3.json'))['maps']
rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
index = S.IP.build_image_index()
tiles = {}
for _, kind, ck, crop in S.thickness_jobs(rows, index):
    if kind != 'gca' or ck not in adj or ck in tiles:
        continue
    W, H = crop.size
    Z = 300 / max(W, H)
    im = crop.convert('RGB').resize((int(W * Z), int(H * Z)))
    dr = ImageDraw.Draw(im)
    a = adj[ck]
    def ell(c6, col):
        cx, cy, ox, oy, ix, iy = [v * Z for v in c6]
        for ax, ay in ((ox, oy), (ix, iy)):
            dr.ellipse([cx - ax, cy - ay, cx + ax, cy + ay], outline=col, width=2)
    old = r1[ck]['old']
    if old:
        ell(S.gca_ellipses(old[:4]), (255, 255, 255))
    if ck in r3 and r3[ck]['refit_v2'] and a['verdict'] != 'normal':
        f = r3[ck]['refit_v2']
        ppm = max(W, H) / S.CFG['gca']['map_extent_mm']
        c6 = (f[0], f[1], f[2], f[3], f[4] * S.CFG['gca']['inner_semi_mm']['x'] * ppm, f[4] * S.CFG['gca']['inner_semi_mm']['y'] * ppm)
        ell(c6, (0, 255, 0) if a['verdict'] == 'refit' else (255, 140, 0))
    lab = {'a': '(a) bad scan', 'b': '(b) bad fit', 'normal': 'normal'}[a['class']]
    lab += {'refit': ': refit accepted', 'refit_failed': ': refit rejected -> map mean', 'bad_scan': ' -> map mean', 'normal': ''}[a['verdict']]
    canvas = Image.new('RGB', (300, 300 + 30), (255, 255, 255))
    canvas.paste(im, (0, 0))
    d2 = ImageDraw.Draw(canvas)
    d2.text((4, 302), f"#{a['tile']:02d} sup {a['support']:.2f}  {lab}", fill=(0, 0, 0))
    d2.text((4, 316), a['reason'][:58], fill=(80, 80, 80))
    tiles[ck] = (a['tile'], canvas)
g = Image.new('RGB', (4 * 304, 4 * 334), (255, 255, 255))
for k, (t, c) in enumerate(sorted(tiles.values(), key=lambda x: x[0])):
    g.paste(c, ((k % 4) * 304, (k // 4) * 334))
g.save(S.SAMPLES / 'qc_gca_16_labelled.png')
print('saved', len(tiles))
