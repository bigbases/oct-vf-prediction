#!/usr/bin/env python3
"""Guard before the R1-3 / R1-4 launches: a short fold-0 run of the current code must
reproduce the submitted run's first epochs exactly.

usage: probe_compare.py <probe results.json> <submitted results.json>
Compares every history field of fold 0, epoch by epoch, for as many epochs as the probe
ran. Exact float equality; any difference fails. Also reports seconds per epoch from the
probe's log if given as a third argument (epoch lines are not time-stamped, so this is
the wall time of the probe divided by its epochs and includes start-up).
"""
import json
import sys

probe = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
ph = probe['results'][0]['history']
rh = ref['results'][0]['history']
assert probe['results'][0]['fold'] == 0 and ref['results'][0]['fold'] == 0
fields = ['train_loss', 'train_mae', 'val_mse', 'val_rmse', 'val_mae']
bad = []
for e, (a, b) in enumerate(zip(ph, rh), start=1):
    for f in fields:
        if a[f] != b[f]:
            bad.append((e, f, a[f], b[f]))
print(f'epochs compared: {len(ph)}; mismatches: {len(bad)}')
for e, f, a, b in bad[:10]:
    print(f'  epoch {e} {f}: probe {a!r} vs submitted {b!r}')
sys.exit(1 if bad or not ph else 0)
