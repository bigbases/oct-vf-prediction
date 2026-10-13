#!/usr/bin/env python3
"""Values the submitted manuscript prints whose evidence was right but not frozen, and the
replacement values for the mismatches listed in docs/revision_r1_manuscript_changes.md.
Pass-B predictions and artefacts, nothing refitted.

  cohort_md            mean deviation over the 277 evaluated pairs (271 with MD) and the
                       four severity strata - methods.tex:19-23, related_work.tex:58/68/71
  sensitivity_bins     share of field points per measured-sensitivity bin, 277 pairs, and
                       the 240 out-of-fold / 37 held-out pairs - related_work.tex:59
  axis_counterfactual  pooled minus per-eye-mean RMSE if our per-eye spread were that of the
                       published studies: sqrt(mu^2 + sd^2) - mu, out-of-fold, per model -
                       related_work.tex:51 (guarded against experiments/aggregation_axis/axis_gap.json)
  bland_altman         regression-based limits of agreement, out-of-fold IR-v2 fusion, from
                       the figure script's own functions - results.tex:641-643
  derived              discussion.tex:178 (1.3 dB), :186 (35%), related_work.tex:61 (2.4)
  training_settings    CNN and XGBoost settings as run - methods.tex:138-158

Output: manuscript_support_numbers.json (no patient keys).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))

import fusion_eval_common as FE           # noqa: E402
import export_xgb_oof as X                # noqa: E402

OUT = HERE / 'manuscript_support_numbers.json'
W_FIXED = 0.47


def bland_altman_module():
    """make_fig_bland_altman_single without its plotting imports (this env has no
    matplotlib); only eye_means() and fit() are used, unchanged."""
    mpl = types.ModuleType('matplotlib')
    mpl.pyplot = types.ModuleType('matplotlib.pyplot')
    style = types.ModuleType('fig_style')
    style.figure = style.zero_line = style.save_fig = None
    style.GREYS = ['#000000'] * 5
    style.OKABE_ITO = ['#000000'] * 8
    saved = {k: sys.modules.get(k) for k in ('matplotlib', 'matplotlib.pyplot', 'fig_style')}
    sys.modules.update({'matplotlib': mpl, 'matplotlib.pyplot': mpl.pyplot, 'fig_style': style})
    try:
        import make_fig_bland_altman_single as BA
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return BA


def stats_md(v):
    v = np.asarray(v, float)
    strata = {'normal (MD > -3)': v > -3, 'early (-6 < MD <= -3)': (v > -6) & (v <= -3),
              'moderate (-12 < MD <= -6)': (v > -12) & (v <= -6), 'advanced (MD <= -12)': v <= -12}
    return {'n': int(v.size), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'sd_pop': float(v.std()),
            'median': float(np.median(v)), 'min': float(v.min()), 'max': float(v.max()),
            'strata': {k: {'n': int(m.sum()), 'pct': float(100 * m.mean())} for k, m in strata.items()}}


def main():
    oof, tst, test_lab, test_mask = FE.load_all()
    oof_keys = [tuple(k) for f in oof.values() for k in f['keys']]
    test_keys = [tuple(k) for k in tst[0]['keys']]
    out = {'note': 'Pass-B values for revised-manuscript sentences (see the module docstring). No patient keys.'}

    # ---- cohort MD over the evaluated pairs
    md = {}
    for r in csv.DictReader(open(REPO / 'cohort_md.csv', encoding='utf-8-sig')):
        try:
            md[(r['patient_id'], r['eye'], r['vf_date'])] = float(r['MD'])
        except (ValueError, KeyError):
            pass
    ev = oof_keys + test_keys
    vals = [md[k] for k in ev if k in md]
    out['cohort_md'] = {'pairs_evaluated': len(ev), 'pairs_with_md': len(vals), **stats_md(vals),
                        # the stratum boundaries this code applies (stats_md); -6 / -12 follow Hodapp
                        'strata_boundaries_db': {'normal_above': -3, 'early_down_to': -6, 'moderate_down_to': -12}}
    printed = {'mean': '-7.39', 'sd': '8.70', 'median': '-3.97', 'min': '-32.16', 'max': '2.34'}
    for k, p in printed.items():
        if f"{out['cohort_md'][k]:.2f}" != p:
            raise SystemExit(f'cohort MD {k} does not reproduce the printed {p}')

    # ---- share of points per measured-sensitivity bin
    edges = [0.0, 10.0, 20.0, 30.0, math.inf]
    names = ['[0,10)', '[10,20)', '[20,30)', '[30,inf)']
    lab_oof = np.concatenate([oof[k]['lab'] for k in range(5)])
    m_oof = np.concatenate([oof[k]['mask'] for k in range(5)]).astype(bool)
    sets = {'277 pairs': (np.concatenate([lab_oof[m_oof], test_lab[test_mask.astype(bool)]])),
            '240 out-of-fold': lab_oof[m_oof], '37 held-out': test_lab[test_mask.astype(bool)]}
    sb = {}
    for name, v in sets.items():
        sb[name] = {'points': int(v.size), 'bins': {b: {'n': int(((v >= lo) & (v < hi)).sum()),
                                                        'pct': float(100 * ((v >= lo) & (v < hi)).mean())}
                                                    for b, lo, hi in zip(names, edges[:-1], edges[1:])}}
    frozen_n = json.load(open(REPO / 'experiments/revision_r1/calibration_multiplicity.json'))[
        'calibration']['OOF']['Summary']['measured [0,10)']['n_cells']
    if sb['240 out-of-fold']['bins']['[0,10)']['n'] != frozen_n:
        raise SystemExit('out-of-fold [0,10) count differs from calibration_multiplicity.json')
    out['sensitivity_bins'] = sb

    # ---- axis counterfactual
    ref = json.load(open(REPO / 'experiments/aggregation_axis/axis_gap.json'))['passes']['B']
    branches = {'summary_xgb': lambda f: f['xgb'], 'image_cnn': lambda f: f['cnns']['IR-v2'],
                'fusion': lambda f: W_FIXED * f['xgb'] + (1 - W_FIXED) * f['cnns']['IR-v2']}
    ac = {'published_sd': {'park2020': 2.09, 'shin2021_sd_oct': 2.68}, 'models': {}}
    for name, get in branches.items():
        eye = np.concatenate([FE.eye_metric(get(oof[k]), oof[k]['lab'], oof[k]['mask'].astype(bool), 'rmse')
                              for k in range(5)])
        mu = float(eye.mean())
        if abs(mu - ref[name]['eye_rmse_mean']) > 1e-6:
            raise SystemExit(f'{name}: per-eye mean RMSE differs from axis_gap.json')
        rec = {'eye_rmse_mean': mu}
        for tag, sd in ac['published_sd'].items():
            g = math.sqrt(mu ** 2 + sd ** 2) - mu
            if abs(g - ref[name]['counterfactual_gap'][f'{sd:.2f}']) > 1e-6:
                raise SystemExit(f'{name}: counterfactual gap at {sd} differs from axis_gap.json')
            rec[f'gap_at_{tag}'] = g
        ac['models'][name] = rec
    allg = [v for m in ac['models'].values() for k, v in m.items() if k.startswith('gap_at_')]
    fus = [v for k, v in ac['models']['fusion'].items() if k.startswith('gap_at_')]
    ac['range_three_models'] = {'min': min(allg), 'max': max(allg)}
    ac['range_fusion'] = {'min': min(fus), 'max': max(fus)}
    out['axis_counterfactual'] = ac

    # ---- Bland-Altman limits (figure script's functions)
    BA = bland_altman_module()
    meas, pred, joined, w = BA.eye_means()
    s = BA.fit(meas, pred - meas)
    lim = {}
    for gx in BA.CAPTION_X:
        b = s['a'] + s['b'] * gx
        h = 2.46 * (s['qa'] + s['qb'] * gx)
        lim[f'x{gx}'] = {'measured_db': gx, 'bias': b, 'upper': b + h, 'lower': b - h}
    out['bland_altman'] = {'n_eyes': s['n'], 'w_xgb': w, 'intercept': s['a'], 'slope': s['b'],
                           'eyes_outside_limits': s['out'], 'limits': lim}

    # ---- derived quantities
    step = json.load(open(PASS_B / 'runs/stepsize_sensitivity.json'))
    sev = json.load(open(PASS_B / 'runs/severity_region.json'))['severity']
    hasan_floored = 2.49     # Hasan et al. 2025, floored axis; verify/literature_values.yaml (checked)
    raw_mae = step['clamp_label']['raw']['fusion_w047']['mae']
    floored_mae = step['clamp_retrain']['fusion_w047']['mae']
    out['derived'] = {
        'remaining_vs_hasan_db': {'value': floored_mae - hasan_floored, 'ours_floored_mae': floored_mae,
                                  'hasan_floored_mae': hasan_floored,
                                  'inputs': 'stepsize_sensitivity.json clamp_retrain.fusion_w047.mae minus Hasan 2.49'},
        'target_axis_change_pct': {'value': 100 * (1 - floored_mae / raw_mae), 'raw_mae': raw_mae, 'floored_mae': floored_mae,
                                   'inputs': 'stepsize_sensitivity.json 1 - clamp_retrain / clamp_label.raw (fusion_w047 MAE)'},
        'severity_rmse_ratio': {'value': sev['advanced']['fusion']['rmse'] / sev['normal']['fusion']['rmse'],
                                'advanced': sev['advanced']['fusion']['rmse'], 'normal': sev['normal']['fusion']['rmse'],
                                'inputs': 'severity_region.json severity.advanced / severity.normal fusion RMSE'},
    }

    # ---- training settings as run
    args = json.load(open(PASS_B / 'runs/phasec_b0_inception_resnet_v2_5fold/results.json'))['args']
    train_src = (REPO / 'train.py').read_text()
    ip_src = (REPO / 'image_preprocessing.py').read_text()
    clip = re.search(r'clip_grad_norm_\(.*?max_norm=([0-9.]+)', train_src)
    jit = re.search(r'ColorJitter\(brightness=([0-9.]+),\s*contrast=([0-9.]+)\)', ip_src)
    kw = X._xgb_kwargs('cpu')
    # the manuscript prints lr and weight decay as powers of ten ($10^{-4}$); the gate reads the
    # exponent as the printed number, so the exponents are stored too
    out['training_settings'] = {
        'cnn': {'optimizer': args['optimizer'], 'lr': args['lr'], 'weight_decay': args['weight_decay'],
                'momentum': args['momentum'], 'batch_size': args['batch_size'], 'epochs_max': args['epochs'],
                'lr_exponent': round(math.log10(args['lr'])), 'weight_decay_exponent': round(math.log10(args['weight_decay'])),
                'patience': args['patience'], 'grad_clip_max_norm': float(clip.group(1)),
                'jitter_brightness': float(jit.group(1)), 'jitter_contrast': float(jit.group(2))},
        'xgb': {k: kw[k] for k in ('n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree',
                                   'reg_alpha', 'reg_lambda')},
    }
    # ---- target-axis variants (results.tex:240-244): clamp placement (both / label) x threshold
    # (raw, 10, 14, 15, 19) + retrained, per split; raw is the same unclamped axis under both
    # placements and is counted under each, as in the manuscript's 11 per split
    tav = {}
    for split, fn in (('oof', 'stepsize_sensitivity.json'), ('test', 'stepsize_sensitivity_test.json')):
        d = json.load(open(PASS_B / 'runs' / fn))
        ps = [v['paired']['fusion_vs_xgb']['wilcoxon_p'] for ax in ('clamp_both', 'clamp_label')
              for v in d[ax].values() if isinstance(v, dict) and 'paired' in v]
        ps.append(d['clamp_retrain']['paired']['fusion_vs_xgb']['wilcoxon_p'])
        tav[split] = {'variants': len(ps), 'p_min': min(ps), 'p_max': max(ps),
                      'fusion_beats_summary_p_below_0_05': sum(x < 0.05 for x in ps)}
    tav['total_variants'] = tav['oof']['variants'] + tav['test']['variants']
    out['target_axis_variants'] = tav
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:5000])


if __name__ == '__main__':
    import phi_safe
    phi_safe.install()
    main()
