#!/usr/bin/env python3
"""R1-1 launcher: the submitted CNN training with the thickness maps replaced by their
sector-averaged versions (sector_maps.py). train.py is imported, not copied; every
training setting comes from the command line, as for the submitted runs.

  --map_mode identity   the cache is looked up for every crop (a missing key raises) but the
                        original crop is returned. Guard probe: must reproduce the submitted
                        history exactly, showing the substitution plumbing changes nothing.
  --map_mode sector     the sector-averaged map replaces the crop, after cropping and before
                        the transform. Nothing else changes.

Outputs go only to the absolute --out_dir given.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from PIL import Image                     # noqa: E402

import image_preprocessing as IP          # noqa: E402
import train as T                         # noqa: E402
from cache_key import content_key         # noqa: E402

assert T.ROOT == REPO, f'ROOT mismatch: {T.ROOT}'
CACHE = HERE / 'cache'


def _pop_flag(name: str) -> str:
    i = sys.argv.index(name)
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


MODE = _pop_flag('--map_mode')
assert MODE in ('identity', 'sector'), MODE
INDEX = json.load(open(CACHE / 'index.json'))['maps']
_SECTOR = {}


def _substitute(crop: Image.Image, kind: str) -> Image.Image:
    k = content_key(crop)
    assert k in INDEX and INDEX[k]['kind'] == kind, 'crop not in the sector-map cache'
    if MODE == 'identity':
        return crop
    if k not in _SECTOR:
        _SECTOR[k] = Image.open(CACHE / f'{k}.png').convert('RGB')
        assert _SECTOR[k].size == crop.size
    return _SECTOR[k].copy()


_orig_gca, _orig_rnfl = IP._crop_gca_thickness, IP._crop_rnfl_thickness
IP._crop_gca_thickness = lambda img: _substitute(_orig_gca(img), 'gca')
IP._crop_rnfl_thickness = lambda img, top_margin=16: _substitute(_orig_rnfl(img, top_margin), 'rnfl')

if __name__ == '__main__':
    out_dir = Path(sys.argv[sys.argv.index('--out_dir') + 1])
    assert out_dir.is_absolute(), 'give --out_dir as an absolute path'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'map_mode.json').write_text(json.dumps({'map_mode': MODE, 'cache_maps': len(INDEX)}) + '\n')
    print(f'map_mode = {MODE}; cache maps {len(INDEX)}', flush=True)
    T.main()
