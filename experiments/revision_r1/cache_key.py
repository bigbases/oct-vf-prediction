"""Content key of a cropped thickness map: links a crop to its sector-averaged version
without writing any patient key. Kept free of scipy so the training process (which
imports torch first) can use it; see sector_maps.py for why the order matters here."""
import hashlib

import numpy as np


def content_key(img) -> str:
    a = np.ascontiguousarray(np.asarray(img.convert('RGB')))
    return hashlib.sha1(a.tobytes() + repr(a.shape).encode()).hexdigest()
