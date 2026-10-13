#!/usr/bin/env python3
"""R1-4 launcher: the submitted CNN training with one change, the treatment of floor cells.

train.py is imported, not copied. Every training setting comes from the command line,
exactly as for the submitted runs. The only addition is --floor_rule:

  none      floor cells are 0 dB targets, as submitted. Used only by the guard probe:
            it must reproduce the submitted history bit for bit, which shows that the
            wrapper itself changes nothing.
  exclude   floor cells get m_j = 0. Because the validation loader uses the same
            dataset class, the validation MAE that selects the checkpoint also skips
            them. Loss and metric code are the submitted functions.
  censored  floor cells stay in the mask but are penalised only when the prediction is
            above 0 dB: (max(y_hat, 0))^2 in the loss, max(y_hat, 0) in the validation
            MAE. Non-floor cells keep the submitted squared / absolute error. The
            normaliser is the number of cells in the mask, as submitted.

Floor cells come from floor_cells.py. A key that is missing raises; nothing is guessed.
The dumped prediction files keep the batch mask, so for `exclude` their mask is the
reduced one. The analysis re-derives masks from the submitted files and does not read
the mask stored here.

Outputs go only to the absolute --out_dir given. Submitted runs/ and the manuscript are
not touched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np                        # noqa: E402
import torch                              # noqa: E402

import image_preprocessing as IP          # noqa: E402
import train as T                         # noqa: E402
from floor_cells import floor_cells, PT52  # noqa: E402

assert T.ROOT == REPO, f'ROOT mismatch: {T.ROOT}'
assert list(IP.PT) == PT52, 'label order differs from floor_cells.PT52'


def _pop_flag(name: str) -> str:
    argv = sys.argv
    i = argv.index(name)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


RULE = _pop_flag('--floor_rule')
assert RULE in ('none', 'exclude', 'censored'), RULE
FLOOR, FLOOR_INFO = floor_cells()

_orig_getitem = IP.VFDataset.__getitem__


def _getitem(self, idx):
    item = _orig_getitem(self, idx)
    key = (item['patient_id'], item['eye'], item['vf_date'])
    fl = torch.from_numpy(FLOOR[key].copy())
    # floor cells must be 0 dB targets inside the submitted mask
    assert bool(torch.all(item['mask'][fl])) and bool(torch.all(item['labels'][fl] == 0.0))
    if RULE == 'exclude':
        item['mask'] = item['mask'] & ~fl
    item['floor'] = fl
    return item


IP.VFDataset.__getitem__ = _getitem


def censored_sq(pred, target, mask, floor):
    m = mask.float()
    f = (mask & floor).float()
    e2 = (pred - target) ** 2 * (m - f) + torch.clamp(pred, min=0.0) ** 2 * f
    return e2.sum() / m.sum().clamp(min=1.0)


def censored_abs(pred, target, mask, floor):
    m = mask.float()
    f = (mask & floor).float()
    e = torch.abs(pred - target) * (m - f) + torch.clamp(pred, min=0.0) * f
    return e.sum() / m.sum().clamp(min=1.0)


# Same control flow as T.train_one_epoch / T.evaluate_metrics; only the two loss calls differ.
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        floor = batch['floor'].to(device)

        optimizer.zero_grad()
        pred = model(images, tabular)
        loss = censored_sq(pred, labels, mask, floor)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n_valid = mask.sum().item()
        loss_sum += loss.item() * n_valid
        mae_sum += censored_abs(pred.detach(), labels, mask, floor).item() * n_valid
        n += n_valid
    return loss_sum / max(n, 1), mae_sum / max(n, 1)


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device) if model.use_tabular else None
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        floor = batch['floor'].to(device)
        pred = model(images, tabular)
        n_valid = mask.sum().item()
        mse_sum += censored_sq(pred, labels, mask, floor).item() * n_valid
        mae_sum += censored_abs(pred, labels, mask, floor).item() * n_valid
        n += n_valid
    denom = max(n, 1)
    return mse_sum / denom, mae_sum / denom


if RULE == 'censored':
    T.train_one_epoch = train_one_epoch
    T.evaluate_metrics = evaluate_metrics

if __name__ == '__main__':
    out_dir = Path(sys.argv[sys.argv.index('--out_dir') + 1])
    assert out_dir.is_absolute(), 'give --out_dir as an absolute path'
    out_dir.mkdir(parents=True, exist_ok=True)
    side = out_dir / 'floor_rule.json'
    assert not side.exists(), f'{side} exists; use a new --out_dir'
    side.write_text(json.dumps({'floor_rule': RULE, 'floor_cells': FLOOR_INFO}, indent=2) + '\n')
    print(f'floor_rule = {RULE}; floor cells {FLOOR_INFO["floor_cells"]} '
          f'({FLOOR_INFO["floor_pct"]}%)', flush=True)
    T.main()
