#!/usr/bin/env python3
"""R1-1 analysis (plan: docs/revision_r1/prereg_R1-1.md, statistics of prereg_R1-3_R1-4.md §0).

Arms. CNN: submitted run per backbone (original map) and runs/r11_sector_<backbone>
(sector-averaged map). XGBoost: submitted printed26 (pass B runs/oof) and runs/r11_xgb/<set>
for map23 (primary), map26, printed26 (refit; must equal the submitted one). Eyes, folds and
the location mask are those of the submitted analysis (fusion_eval_common.load_all: 240
out-of-fold eyes, 37 held-out). Held-out predictions are the mean of the five fold models.
Fusion of a single backbone: w fitted on all out-of-fold folds (grid 0-1 step 0.01, pooled
RMSE), Table 3 convention. Arms whose predictions do not exist yet are skipped.

Comparisons, per-eye RMSE and MAE, paired Wilcoxon and patient-clustered bootstrap 95%
interval of the mean difference (a - b; negative = a better):
  cnn_sector vs cnn_original        primary for R1-1: IR-v2, out-of-fold, per-eye RMSE
  cnn_sector vs xgb_map23           same information, different model family
  xgb_map23 / xgb_map26 vs xgb_printed26
  fusion_sector vs xgb_printed26, fusion_sector vs fusion_original
each on all eyes and, as the sensitivity analysis, without the eyes whose GCA map is a
map-mean map (cache index gca_status == 'map_mean').

Guards (submitted values, pooled RMSE): out-of-fold summary 8.6603, IR-v2 8.5397; held-out
summary 8.9193, IR-v2 8.7323, IR-v2 fusion 8.4105.

  python r11_analysis.py [backbone ...]     # default inception_resnet_v2
Output: r11_analysis.json (no patient keys).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))
sys.path.insert(0, str(HERE))

import sector_maps as S                               # noqa: E402  (scipy before torch)
import fusion_eval_common as FE                       # noqa: E402
from oof_common import load_oof_npz, row_key          # noqa: E402
from cache_key import content_key                     # noqa: E402
import phi_safe                                       # noqa: E402
import calibration_multiplicity as CM                 # noqa: E402
from PIL import Image                                 # noqa: E402

BACKBONES = {'inception_resnet_v2': 'IR-v2', 'inception_v3': 'Inception-v3', 'vgg16': 'VGG16',
             'xception': 'Xception', 'densenet121': 'DenseNet121'}
XGB_SETS = ('map23', 'map26', 'printed26')
GUARD = {('OOF', 'xgb_printed26'): 8.6603, ('OOF', 'cnn_original'): 8.5397,
         ('TEST', 'xgb_printed26'): 8.9193, ('TEST', 'cnn_original'): 8.7323,
         ('TEST', 'fusion_original'): 8.4105}


def preds(path_of, keys_by_fold, split):
    """Predictions aligned to the submitted keys, or None if any file is missing."""
    files = [path_of(k) for k in range(5)]
    if not all(f.exists() for f in files):
        return None
    out = []
    for k, f in enumerate(files):
        z = load_oof_npz(f)
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        out.append(z['pred'][phi_safe.lookup(idx, keys_by_fold(k), f.name)])
    return np.concatenate(out) if split == 'OOF' else np.mean(out, 0)


def map_mean_keys():
    """Row keys whose GCA map has no usable annulus (map-mean map in the sector condition)."""
    status = json.load(open(S.CACHE / 'index.json'))['maps']
    rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
    index = S.IP.build_image_index()
    out = set()
    for r in rows:
        p = index.get(row_key(r), {}).get(f"gca_{r['eye'].lower()}_thickness")
        if p and status[content_key(S.IP._crop_gca_thickness(Image.open(p).convert('RGB')))].get('gca_status') == 'map_mean':
            out.add(row_key(r))
    return out


def pooled(p, l, m):
    d = (p - l)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def main(backbones):
    oof, tst, test_lab, test_mask = FE.load_all()
    mm = map_mean_keys()
    out = {'note': 'R1-1 analysis; delta = a - b, negative = a better. No patient keys.',
           'backbones': backbones, 'n_map_mean_eyes_total': len(mm), 'guards': [], 'splits': {}}
    W = {}
    data = {}
    for split in ('OOF', 'TEST'):
        if split == 'OOF':
            kf = lambda k: oof[k]['keys']
            keys = [kk for k in range(5) for kk in oof[k]['keys']]
            lab = np.concatenate([oof[k]['lab'] for k in range(5)])
            mask = np.concatenate([oof[k]['mask'] for k in range(5)]).astype(bool)
            s = 'val'
        else:
            kf = lambda k: tst[0]['keys']
            keys = list(tst[0]['keys'])
            lab, mask, s = test_lab, test_mask.astype(bool), 'test'
        B = {'xgb_printed26': preds(lambda k: FE.xgb_path(k, 'val' if split == 'OOF' else 'test'), kf, split)}
        for fs in XGB_SETS:
            p = preds(lambda k: HERE / 'runs/r11_xgb' / fs / f'xgb_90d_fold{k}_{s}.npz', kf, split)
            if p is not None:
                B[f'xgb_{fs}_r11'] = p
        for bb in backbones:
            B[f'{bb}:cnn_original'] = preds(lambda k: PASS_B / FE.BB[BACKBONES[bb]] / f'{s}_preds_fold{k}.npz', kf, split)
            p = preds(lambda k: HERE / f'runs/r11_sector_{bb}' / f'{s}_preds_fold{k}.npz', kf, split)
            if p is not None:
                B[f'{bb}:cnn_sector'] = p
        data[split] = dict(keys=keys, lab=lab, mask=mask, B=B,
                           pid=np.array([str(kk[0]) for kk in keys], dtype=object),
                           keep=np.array([tuple(kk) not in mm for kk in keys]))
    # fusion weights: all out-of-fold folds, pooled RMSE, grid 0-1 step 0.01
    d = data['OOF']
    for bb in backbones:
        for arm in ('original', 'sector'):
            c = d['B'].get(f'{bb}:cnn_{arm}')
            if c is not None:
                folds = [{'xgb': d['B']['xgb_printed26'], 'cnns': {'c': c}, 'lab': d['lab'], 'mask': d['mask']}]
                W[f'{bb}:{arm}'] = FE.fit_w(folds, 'c')
    for split, d in data.items():
        for bb in backbones:
            for arm in ('original', 'sector'):
                if f'{bb}:{arm}' in W:
                    w = W[f'{bb}:{arm}']
                    d['B'][f'{bb}:fusion_{arm}'] = w * d['B']['xgb_printed26'] + (1 - w) * d['B'][f'{bb}:cnn_{arm}']
    out['fusion_w'] = W

    # guards
    for (split, name), want in GUARD.items():
        key = name if name.startswith('xgb') else f'inception_resnet_v2:{name}'
        p = data[split]['B'].get(key)
        if p is None:
            continue
        got = pooled(p, data[split]['lab'], data[split]['mask'])[0]
        ok = abs(got - want) <= 5e-4
        out['guards'].append({'split': split, 'branch': key, 'got': round(got, 4), 'want': want, 'ok': ok})
        print(f"guard {split} {key}: {got:.4f} vs {want} {'ok' if ok else 'FAIL'}")
    for split, d in data.items():
        if 'xgb_printed26_r11' in d['B']:
            dev = float(np.nanmax(np.abs(d['B']['xgb_printed26_r11'] - d['B']['xgb_printed26'])))
            ok = dev <= 1e-5
            out['guards'].append({'split': split, 'branch': 'xgb_printed26_r11 = submitted', 'max_abs_diff': dev, 'ok': ok})
            print(f"guard {split} printed26 refit max |diff| {dev:.2e} {'ok' if ok else 'FAIL'}")
    if not all(g['ok'] for g in out['guards']):
        raise SystemExit('guard failed; nothing computed')

    cache = {}

    def Wboot(n):
        if n not in cache:
            cache[n] = CM.boot_weights(n, np.random.default_rng(CM.SEED))
        return cache[n]

    for split, d in data.items():
        lab, m, pid, keep, Bs = d['lab'], d['mask'], d['pid'], d['keep'], d['B']
        rec = {'n_eyes': int(m.shape[0]), 'n_eyes_map_mean': int((~keep).sum()), 'branches': {}, 'comparisons': []}
        for name, p in Bs.items():
            r, a = pooled(p, lab, m)
            rec['branches'][name] = {'pooled_rmse': r, 'pooled_mae': a,
                                     'eye_mean_rmse': float(FE.eye_metric(p, lab, m, 'rmse').mean()),
                                     'eye_mean_mae': float(FE.eye_metric(p, lab, m, 'mae').mean())}
        pairs = []
        for bb in backbones:
            pairs += [(f'{bb}:cnn_sector', f'{bb}:cnn_original', bb == 'inception_resnet_v2'),
                      (f'{bb}:cnn_sector', 'xgb_map23_r11', False),
                      (f'{bb}:fusion_sector', 'xgb_printed26', False),
                      (f'{bb}:fusion_sector', f'{bb}:fusion_original', False)]
        pairs += [('xgb_map23_r11', 'xgb_printed26', False), ('xgb_map26_r11', 'xgb_printed26', False)]
        for a, b, primary in pairs:
            if a not in Bs or b not in Bs:
                rec['comparisons'].append({'a': a, 'b': b, 'status': 'arm not available yet'})
                continue
            for subset, sel in (('all eyes', np.ones_like(keep)), ('without map-mean eyes', keep)):
                for kind in ('rmse', 'mae'):
                    ea = FE.eye_metric(Bs[a][sel], lab[sel], m[sel], kind)
                    eb = FE.eye_metric(Bs[b][sel], lab[sel], m[sel], kind)
                    r = CM.paired(ea, eb, pid[sel], Wboot)
                    r.update(a=a, b=b, metric=f'per-eye {kind}', subset=subset,
                             primary=bool(primary and kind == 'rmse' and split == 'OOF' and subset == 'all eyes'))
                    rec['comparisons'].append(r)
        out['splits'][split] = rec
    (HERE / 'r11_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for split, rec in out['splits'].items():
        for c in rec['comparisons']:
            if 'status' in c:
                print(f"{split} {c['a']} vs {c['b']}: {c['status']}")
            elif c['subset'] == 'all eyes':
                print(f"{split} {c['metric']:13s} {c['a']} - {c['b']}: {c['mean_delta']:+.3f} "
                      f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] p={c['p_wilcoxon']:.3g}{'  PRIMARY' if c['primary'] else ''}")


if __name__ == '__main__':
    phi_safe.install()
    main(sys.argv[1:] or ['inception_resnet_v2'])
