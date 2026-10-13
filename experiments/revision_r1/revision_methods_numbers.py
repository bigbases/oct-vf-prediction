#!/usr/bin/env python3
"""Numbers the revised Methods prints for R1-1, R1-3 and R1-4, gathered from the code
constants and the artefacts that define them, so that each printed value has a frozen
source. Nothing is fitted here.

  r11_regions   sector-averaged maps: geometry, ramp, non-thickness classes, band colours,
                build diagnostics, and a recount of the frame band on every cached map
  r11_qc        GCA annulus fit and adjudication counts; MD of the map-mean eyes
  r11_xgb       XGBoost feature sets and their missing values
  r13           mirrored-frame runs: settings that differ from the submitted run, epochs,
                and the three-epoch reproduction probes (incl. VGG16 run-to-run differences)
  r14           floor cells and the two training rules

Output: revision_methods_numbers.json (no patient keys).
"""
from __future__ import annotations

import collections
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

import sector_maps as S                   # noqa: E402  (scipy before torch)
import r11_xgb as RX                      # noqa: E402
import fusion_eval_common as FE           # noqa: E402
from oof_common import row_key            # noqa: E402
from cache_key import content_key         # noqa: E402
from floor_cells import floor_cells       # noqa: E402
from PIL import Image                     # noqa: E402

OUT = HERE / 'revision_methods_numbers.json'
HIST = ['train_loss', 'train_mae', 'val_mse', 'val_rmse', 'val_mae']


def band_recount(index):
    """Frame band of every cached sector-averaged map: pixels and distinct colours."""
    px = collections.Counter()
    one = collections.Counter()
    for k, d in index.items():
        a = np.asarray(Image.open(S.CACHE / f'{k}.png').convert('RGB'))
        H, W, _ = a.shape
        b = int(np.ceil(S.BORDER_FRAC[d['kind']] * W))
        m = np.ones((H, W), bool)
        m[b:H - b, b:W - b] = False
        u = np.unique(a[m].reshape(-1, 3), axis=0)
        px[d['kind']] += int(m.sum())
        one[d['kind']] += int(len(u) == 1 and tuple(u[0]) == S.BORDER_RGB[d['kind']])
    return {k: {'maps_with_one_band_colour': one[k], 'band_px': px[k]} for k in ('gca', 'rnfl')}


def md_stats(keys, md):
    v = np.array([md.get(k, np.nan) for k in keys], float)
    ok = np.isfinite(v)
    return {'n': int(len(keys)), 'n_with_md': int(ok.sum()),
            'median': float(np.median(v[ok])) if ok.any() else None,
            'min': float(v[ok].min()) if ok.any() else None, 'max': float(v[ok].max()) if ok.any() else None}


def probe_mismatches(a, b):
    ha, hb = [json.load(open(p))['results'][0]['history'] for p in (a, b)]
    return {'epochs': len(ha), 'values': len(ha) * len(HIST),
            'mismatches': sum(x[f] != y[f] for x, y in zip(ha, hb) for f in HIST)}


def main():
    doc = json.load(open(S.CACHE / 'index.json'))
    idx, summ = doc['maps'], doc['summary']
    gca, rnfl = S.CFG['gca'], S.CFG['rnfl']
    lut_top = 350.0
    out = {'note': 'Numbers for the revised Methods (R1-1, R1-3, R1-4); constants read from the code '
                   'that ran, counts from the artefacts. No patient keys.'}

    out['r11_regions'] = {
        'rnfl': {'n_wedges': 12, 'wedge_deg': 30, 'centre': 'optic disc (centroid of the grey disc overlay)',
                 'centre_detection_failures': summ['centre_fallback']['rnfl'],
                 'border_band_frac_of_width': [9, 271], 'border_rgb': list(S.BORDER_RGB['rnfl'])},
        'gca': {'n_regions': 8, 'n_annulus_sectors': gca['n_sectors'], 'sector_width_deg': gca['sector_width_deg'],
                'inner_axes_mm': [2 * gca['inner_semi_mm']['x'], 2 * gca['inner_semi_mm']['y']],
                'outer_axes_mm': [2 * gca['outer_semi_mm']['x'], 2 * gca['outer_semi_mm']['y']],
                'map_extent_mm': gca['map_extent_mm'], 'ring_band_halfwidth_px': S.RING_BAND_PX,
                'contour_band_px': S.CONTOUR_BAND_PX, 'ring_rgb': list(S.RING_RGB),
                'border_band_frac_of_width': [4, 343], 'border_rgb': list(S.BORDER_RGB['gca'])},
        'ramp': {'entries': summ['lut_entries'], 'step_um': summ['lut_step_um'], 'colorbar_um': [lut_top, lut_top / 2, 0.0],
                 'max_rgb_distance': S.MAX_DIST, 'panels_checked': summ['lut_stability_max_rgb_diff']['n_panels'],
                 'max_rgb_diff_between_panels': summ['lut_stability_max_rgb_diff']['max']},
        'non_thickness': {'grey_saturation_max': S.GREY_SAT, 'grey_max_between': [S.BLACK_MAX, S.GREY_MAX],
                          'black_max': S.BLACK_MAX, 'fringe_px': S.PURPLE_FRINGE_PX},
        'min_region_px': S.MIN_WEDGE_PX,
        'build': {'maps': summ['maps'], 'by_kind': summ['by_kind'],
                  'thickness_fraction_median': {k: summ['tissue_frac'][k]['median'] for k in ('gca', 'rnfl')},
                  'regions_filled_from_map_mean': summ['wedges_filled_from_map_mean'],
                  'roundtrip_max_err_um': summ['roundtrip_max_err']},
        'band_recount': band_recount(idx),
    }

    adj = json.load(open(S.CACHE / 'fit_adjudication.json'))['maps']
    cls = collections.Counter(f"{e['class']}/{e['verdict']}" for e in adj.values())
    rows = list(csv.DictReader(open(S.CSV, encoding='utf-8-sig')))
    index = S.IP.build_image_index()
    status, klass = {}, {}
    for r in rows:
        p = index.get(row_key(r), {}).get(f"gca_{r['eye'].lower()}_thickness")
        if p:
            ck = content_key(S.IP._crop_gca_thickness(Image.open(p).convert('RGB')))
            status[row_key(r)] = idx[ck].get('gca_status')
            klass[row_key(r)] = adj.get(ck, {}).get('class')
    md = {}
    for r in csv.DictReader(open(REPO / 'cohort_md.csv', encoding='utf-8-sig')):
        try:
            md[(r['patient_id'], r['eye'], r['vf_date'])] = float(r['MD'])
        except (ValueError, KeyError):
            pass
    oof, tst, _, _ = FE.load_all()
    ev_oof = {tuple(k) for f in oof.values() for k in f['keys']}
    ev_test = {tuple(k) for k in tst[0]['keys']}
    evaluated = ev_oof | ev_test
    mm = sorted(k for k, s in status.items() if s == 'map_mean')
    ca = sorted(k for k, c in klass.items() if c == 'a')
    out['r11_qc'] = {
        'fit_support_min': S.FIT_SUPPORT_MIN, 'gca_maps': summ['by_kind']['gca'],
        'flagged': sum(cls.values()), 'adjudication': dict(cls),
        'gca_status': summ['gca_status'],
        'map_mean_eyes': {**md_stats(mm, md), 'oof': sum(k in ev_oof for k in mm), 'test': sum(k in ev_test for k in mm)},
        'class_a_eyes': {**md_stats(ca, md), 'oof': sum(k in ev_oof for k in ca), 'test': sum(k in ev_test for k in ca)},
        'cohort_evaluated_eyes': md_stats(sorted(evaluated), md),
    }

    z = np.load(RX.OUT / 'features.npz', allow_pickle=True)
    keys = [tuple(k) for k in z['keys']]
    F23, F26 = z['map23'], z['map26']
    ev = np.array([k in evaluated for k in keys])
    n26 = np.isnan(F26).any(1)
    pv = np.array([float(r[f"{r['eye'].lower()}_vert_cd"]) if r.get(f"{r['eye'].lower()}_vert_cd", '') else np.nan
                   for r in rows])
    ok = np.isfinite(pv) & np.isfinite(F23[:, 22])
    out['r11_xgb'] = {
        'sets': {'map23': {'n_features': int(F23.shape[1]), 'region_means': 20, 'disc_measures': 3},
                 'map26': {'n_features': int(F26.shape[1]), 'cirrus_geometry': 25, 'vertical_cd': 1},
                 'printed26': {'n_features': 26}},
        'rnfl_circle_diameter_mm': rnfl['calc_circle_diameter_mm'], 'rnfl_band_width_mm': rnfl['band_width_mm'],
        'gca_min_spokes': 360, 'cup_grey_min': RX.CUP_GREY_MIN,
        'rows': len(keys), 'evaluated_rows': int(ev.sum()),
        'evaluated_rows_with_missing_map23': int((np.isnan(F23).any(1) & ev).sum()),
        'evaluated_rows_with_missing_map26': int((n26 & ev).sum()),
        'missing_map26_values_per_such_row': sorted(int(x) for x in np.isnan(F26[n26 & ev]).sum(1)),
        'r_map_vcd_vs_printed': {'r': float(np.corrcoef(F23[ok, 22], pv[ok])[0, 1]), 'n': int(ok.sum())},
    }

    def cfg(p):
        d = json.load(open(p))
        return d.get('args') or d.get('config'), [len(r['history']) for r in d['results']]
    runs = {'IR-v2': ('phasec_b0_inception_resnet_v2_5fold', REPO / 'runs/phasec_b0_inception_resnet_v2_5fold_osflip'),
            'Inception-v3': ('phasec_b0_clip_mse_5fold', HERE / 'runs/r13_osflip_inception_v3'),
            'Xception': ('phasec_b0_xception_5fold', HERE / 'runs/r13_osflip_xception'),
            'DenseNet121': ('phasec_b0_densenet121_5fold', HERE / 'runs/r13_osflip_densenet121'),
            'VGG16': ('phasec_b0_vgg16_5fold', HERE / 'runs/r13_osflip_vgg16')}
    flips = {}
    for bb, (ref, d) in runs.items():
        if not (d / 'results.json').exists() or len(json.load(open(d / 'results.json'))['results']) < 5:
            flips[bb] = {'status': 'run not complete'}
            continue
        ca, _ = cfg(REPO / 'runs' / ref / 'results.json')
        cb, ep = cfg(d / 'results.json')
        diff = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb))
                if ca.get(k) != cb.get(k) and k not in ('out_dir', 'allow_existing_out_dir')}
        # an argument the submitted code did not have yet is recorded as absent (None), not changed
        flips[bb] = {'args_changed': {k: v for k, v in diff.items() if v[0] is not None},
                     'args_absent_in_submitted_run': {k: v[1] for k, v in diff.items() if v[0] is None},
                     'flip_os_images': cb.get('flip_os_images'), 'epochs_per_fold': ep, 'epochs_total': sum(ep)}
    P = HERE / 'probes'
    sub = {'IR-v2': 'phasec_b0_inception_resnet_v2_5fold', 'Inception-v3': 'phasec_b0_clip_mse_5fold',
           'Xception': 'phasec_b0_xception_5fold', 'DenseNet121': 'phasec_b0_densenet121_5fold',
           'VGG16': 'phasec_b0_vgg16_5fold'}
    pr = {'IR-v2': 'probe_r14none_irv2', 'Inception-v3': 'probe_inception_v3', 'Xception': 'probe_xception',
          'DenseNet121': 'probe_densenet121', 'VGG16': 'probe_vgg16'}
    probes = {}
    for bb, name in pr.items():
        rec = {'vs_submitted': probe_mismatches(P / name / 'results.json', REPO / 'runs' / sub[bb] / 'results.json')}
        rep = P / name.replace('probe_', 'probe2_') / 'results.json'
        if rep.exists():
            rec['repeat_vs_first'] = probe_mismatches(rep, P / name / 'results.json')
        probes[bb] = rec
    out['r13'] = {'mirrored_runs': flips, 'probes_fold0': probes,
                  'probe_setting': {'fold': 0, 'epochs': 3, 'history_fields': HIST},
                  'unmirrored_reruns_current_code': ['VGG16', 'Xception', 'DenseNet121']}

    _, info = floor_cells()
    neg1 = json.load(open(PASS_B / 'runs/neg1_mask_sensitivity.json'))['splits']
    out['r14'] = {'floor_cells': info['floor_cells'], 'cells': info['cells'], 'floor_pct': info['floor_pct'],
                  'rows': info['rows'], 'rows_matched_exactly': info['rows_exact'],
                  'rows_matched_unique_candidate': info['rows_unique_candidate'],
                  'evaluation_floor_cells': {s: {'cells_dropped': neg1[s]['cells_dropped'], 'dropped_pct': neg1[s]['dropped_pct'],
                                                 'n_eyes': neg1[s]['n_eyes']} for s in ('oof', 'test')},
                  'rules': {'exclude': 'floor cells masked out of the training loss and of the validation MAE that selects the checkpoint',
                            'censored': 'floor cells kept; penalised only above 0 dB: max(pred, 0)^2 in the loss, max(pred, 0) in the validation MAE',
                            'xgb_exclude': 'per location, training rows whose cell is a floor cell are left out'}}
    OUT.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
