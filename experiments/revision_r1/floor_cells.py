"""Floor cells: field locations recorded as "<0 dB" (-1 in the source report) that the
label file now carries as 0 dB.

Revision R1-4. The current label file no longer marks these cells; the pre-conversion
backup does. Rows are matched exactly as in scripts/neg1_mask_sensitivity.py
(the analysis behind neg1_mask_sensitivity.json): (patient, eye, test date) exact
match for 274 rows, and (patient, eye) unique candidate for the 6 rows whose test date
was corrected after the backup was taken. The same two integrity checks are asserted:
no non-floor value differs between the files, and every floor cell is 0 in the current
file.

Returns a dict keyed by (patient_id, eye, vf_date) of the current file -> bool[52]
in the label order of image_preprocessing.PT. Nothing here prints a key.

floor_cells() 는 키를 출력하지 않는다. 키 자체가 식별자다.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parents[2]))
PT52 = [f'p{i:02d}' for i in range(1, 55) if f'p{i:02d}' not in ('p26', 'p35')]
CURRENT = REPO / 'ml_final_90d_excl_empty_flip.csv'
BACKUP = REPO / 'runs' / 'backup_vf_neg1_fix_20260530_193547' / 'ml_final_90d_excl_empty_flip.csv'

# Expected counts, from neg1_mask_sensitivity.json (frozen) and its log.
N_ROWS = 280
N_FLOOR = 1792


def _read(p: Path):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _vec(r):
    return np.array([float(r[p]) for p in PT52], dtype=np.float64)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def floor_cells(current: Path = CURRENT, backup: Path = BACKUP):
    old, new = _read(backup), _read(current)
    oi = {}
    for i, r in enumerate(old):
        oi.setdefault((r['patient_id'], r['eye'], r['vf_date']), []).append(i)
    pairs, used_o, used_n = [], set(), set()
    for j, r in enumerate(new):
        k = (r['patient_id'], r['eye'], r['vf_date'])
        if oi.get(k):
            i = oi[k].pop(0)
            pairs.append((i, j)); used_o.add(i); used_n.add(j)
    n_exact = len(pairs)
    rest = [i for i in range(len(old)) if i not in used_o]
    for j in [j for j in range(len(new)) if j not in used_n]:
        cand = [i for i in rest
                if (old[i]['patient_id'], old[i]['eye']) == (new[j]['patient_id'], new[j]['eye'])]
        assert len(cand) == 1, 'row matching is not one-to-one'
        rest.remove(cand[0]); pairs.append((cand[0], j))
    assert len(pairs) == len(new) == N_ROWS, (len(pairs), len(new))

    out, bad_a, bad_b, n_floor = {}, 0, 0, 0
    for i, j in pairs:
        o, n = _vec(old[i]), _vec(new[j])
        m = o == -1.0
        bad_a += int(np.sum((~m) & (o != n)))
        bad_b += int(np.sum(n[m] != 0.0))
        n_floor += int(m.sum())
        key = (new[j]['patient_id'], new[j]['eye'], new[j]['vf_date'])
        assert key not in out, 'duplicate key in current file'
        out[key] = m
    assert bad_a == 0 and bad_b == 0, (bad_a, bad_b)
    assert n_floor == N_FLOOR, n_floor
    info = {'rows': len(pairs), 'rows_exact': n_exact, 'rows_unique_candidate': len(pairs) - n_exact,
            'floor_cells': n_floor, 'cells': len(pairs) * len(PT52),
            'floor_pct': round(100 * n_floor / (len(pairs) * len(PT52)), 2),
            'current_sha256': sha256(current), 'backup_sha256': sha256(backup)}
    return out, info


if __name__ == '__main__':
    _, info = floor_cells()
    print(info)
