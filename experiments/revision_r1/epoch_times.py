#!/usr/bin/env python3
"""Seconds per epoch from a run_job.sh log (unix-time prefix on each line).
usage: epoch_times.py <log> ...   prints start-up, per-epoch mean (excluding epoch 1) and epochs done."""
import re
import sys

for p in sys.argv[1:]:
    L = open(p, encoding='utf-8', errors='replace').read().splitlines()
    if not L:
        print(p.rsplit('/', 1)[-1], 'empty'); continue
    t = lambda l: float(l.split()[0])
    t0 = t(L[0])
    fold_starts = [(t(l), l) for l in L if '시작 ===' in l]
    ep = [(t(l), int(re.search(r'\] ep +(\d+)/', l).group(1)), re.search(r'\[fold (\d)\]', l).group(1))
          for l in L if re.search(r'\] ep +\d+/', l)]
    if not ep:
        print(p.rsplit('/', 1)[-1], f'no epoch yet; {L[-1][:12]}'); continue
    gaps = [b[0] - a[0] for a, b in zip(ep, ep[1:]) if b[2] == a[2]]
    first = ep[0][0] - (fold_starts[0][0] if fold_starts else t0)
    mean = sum(gaps) / len(gaps) if gaps else float('nan')
    print(f'{p.rsplit("/", 1)[-1]}: startup {fold_starts[0][0] - t0 if fold_starts else float("nan"):.0f}s, '
          f'epoch1 {first:.1f}s, later epochs {mean:.2f}s (n={len(gaps)}), done {len(ep)} epochs, '
          f'current fold {ep[-1][2]} ep {ep[-1][1]}')
