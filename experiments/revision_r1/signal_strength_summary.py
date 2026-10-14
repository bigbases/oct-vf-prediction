#!/usr/bin/env python3
"""OCT signal strength over the 277 evaluated pairs - methods.tex Cohort paragraph
("Signal strength is available for 276 of the 277 optic disc scans ...", manuscript change M15).

Input: the stage-10 output signal_strength.csv of the sister extraction pipeline. One row per
cohort row of that pipeline: patient, OCT date and the four values read from the OU report
headers (ss_gca_od/os = macular cube, ss_rnfl_od/os = optic disc cube). It holds patient numbers
and dates, so it stays outside the repository (HVF_SS_CSV, default
~/hvf_private/oct_ss_results/signal_strength.csv) and its sha256 must equal the stage-10 record.

Join: each evaluated pair (240 out-of-fold + 37 held-out, pass B) takes its OCT date from
ml_final_90d_excl_empty_flip.csv and then the report row of that (patient, OCT date); the eye
picks the column. Rows repeated for the same (patient, OCT date) must agree.

The stage-10 reader assigns values by position: the first "X/10" it reads goes to OD, the second
to OS. When it reads only one, that value lands in the OD column whichever eye it came from.
Three counting rules:

  own_eye                the pair's own column; blank = unavailable; 0 counted as a value
  own_eye_zero_excluded  as own_eye, and 0 (outside the 1-10 range) also unavailable
  single_value_fill      as own_eye_zero_excluded, except that a blank own column takes the
                         report's single read value (the OD column). Its eye is not known. This
                         is the rule that reproduces the submitted sentence (guarded below).

Output: signal_strength_summary.json (counts, medians, value histograms; no patient keys, no dates).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('HVF_ROOT', HERE.parents[1]))
PASS_B = Path(os.environ.get('HVF_PASS_B', REPO / 'experiments/laterality_qfix/step4_work/B'))
sys.path.insert(0, str(PASS_B / 'scripts'))
sys.path.insert(0, str(HERE))

import fusion_eval_common as FE           # noqa: E402
import phi_safe                           # noqa: E402

SS_CSV = Path(os.environ.get('HVF_SS_CSV', Path.home() / 'hvf_private/oct_ss_results/signal_strength.csv'))
# stage10_ss_sha256.json "value" in the sister pipeline's results/
STAGE10_SHA256 = '4157f4ca5b5d4180f0a6a120a8ef72f59c82af7b26dbdb45e00ceb103cdb61d2'
COLUMNS = ['patient_id', 'oct_date', 'ss_gca_od', 'ss_gca_os', 'ss_rnfl_od', 'ss_rnfl_os']
CUBES = {'optic_disc': 'ss_rnfl_', 'macular': 'ss_gca_'}
THRESHOLD = 6           # "would have removed": scans below this value
OUT = HERE / 'signal_strength_summary.json'

# the submitted sentence, reproduced by single_value_fill
PRINTED = {('optic_disc', 'available'): 276, ('optic_disc', 'pairs'): 277, ('macular', 'available'): 277,
           ('optic_disc', 'median'): 7, ('macular', 'median'): 8,
           ('optic_disc', 'below_6_pct'): '26.1', ('macular', 'below_6_pct'): '22.4'}


def read_value(text):
    text = text.strip()
    return None if text == '' else int(text)


def summarise(values, n_pairs):
    avail = [v for v in values if v is not None]
    below = sum(v < THRESHOLD for v in avail)
    return {'pairs': n_pairs, 'available': len(avail), 'unavailable': n_pairs - len(avail),
            'median': statistics.median(avail), 'min': min(avail), 'max': max(avail),
            'below_6': {'n': below, 'pct': 100 * below / len(avail)},
            'histogram': {str(v): n for v, n in sorted(Counter(avail).items())}}


def main():
    raw = SS_CSV.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != STAGE10_SHA256:
        raise SystemExit('signal_strength.csv sha256 differs from the stage-10 record')
    rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if list(rows[0].keys()) != COLUMNS:
        raise SystemExit('signal_strength.csv columns differ from the stage-10 layout')

    sessions = defaultdict(list)
    for r in rows:
        sessions[(r['patient_id'], r['oct_date'])].append(tuple(r[c] for c in COLUMNS[2:]))
    repeated = sum(len(v) > 1 for v in sessions.values())
    if any(len(set(v)) > 1 for v in sessions.values()):
        raise SystemExit('rows repeated for one (patient, OCT date) disagree')
    report = {s: dict(zip(COLUMNS[2:], v[0])) for s, v in sessions.items()}

    oof, tst, _, _ = FE.load_all()
    oof_keys = [tuple(map(str, k)) for f in oof.values() for k in f['keys']]
    test_keys = [tuple(map(str, k)) for k in tst[0]['keys']]
    ev = oof_keys + test_keys
    if (len(oof_keys), len(test_keys), len(set(ev))) != (240, 37, 277):
        raise SystemExit('evaluated pairs are not the 240 + 37 of pass B')

    ml = {(r['patient_id'], r['eye'], r['vf_date']): r['oct_date']
          for r in csv.DictReader(open(REPO / 'ml_final_90d_excl_empty_flip.csv', encoding='utf-8-sig'))}
    oct_dates = phi_safe.lookup(ml, ev, 'ml_final_90d_excl_empty_flip.csv')
    sess_keys = [(k[0], d) for k, d in zip(ev, oct_dates)]
    reps = phi_safe.lookup(report, sess_keys, 'signal_strength.csv sessions')
    ev_sessions = {(k[0], k[1], d) for k, d in zip(ev, oct_dates)}

    out = {'note': 'OCT signal strength (Cirrus, 0-10) over the 277 evaluated pairs; counting rules in the '
                   'producer docstring. No patient keys, no dates.',
           'input': {'sha256': sha, 'rows': len(rows), 'report_sessions': len(sessions),
                     'sessions_with_repeated_rows': repeated, 'repeated_rows_consistent': True},
           'pairs_evaluated': {'total': len(ev), 'out_of_fold': len(oof_keys), 'held_out': len(test_keys)},
           'threshold': THRESHOLD, 'rules': {}, 'single_value_cases': {}}

    other = {'OD': 'os', 'OS': 'od'}
    for rule in ('own_eye', 'own_eye_zero_excluded', 'single_value_fill'):
        out['rules'][rule] = {}
        for cube, pref in CUBES.items():
            vals = []
            for k, rep in zip(ev, reps):
                v = read_value(rep[pref + k[1].lower()])
                if v is None and rule == 'single_value_fill':
                    v = read_value(rep[pref + other[k[1]]])
                if v == 0 and rule != 'own_eye':
                    v = None
                vals.append(v)
            out['rules'][rule][cube] = summarise(vals, len(ev))

    # pairs whose own column is blank: by eye, whether the report has the other column, and
    # whether the other eye of the same session is itself an evaluated pair
    for cube, pref in CUBES.items():
        blank = [(k, d, rep) for k, d, rep in zip(ev, oct_dates, reps) if read_value(rep[pref + k[1].lower()]) is None]
        fill = [read_value(rep[pref + other[k[1]]]) for k, d, rep in blank]
        out['single_value_cases'][cube] = {
            'own_column_blank': len(blank),
            'by_eye': dict(sorted(Counter(k[1] for k, d, rep in blank).items())),
            'other_column_present': sum(v is not None for v in fill),
            'other_eye_of_session_evaluated': sum((k[0], other[k[1]].upper(), d) in ev_sessions for k, d, rep in blank),
            'filled_value_below_6': sum(v is not None and v < THRESHOLD for v in fill),
            'filled_value_zero': sum(v == 0 for v in fill)}

    sv = out['rules']['single_value_fill']
    for (cube, field), printed in PRINTED.items():
        got = f"{sv[cube]['below_6']['pct']:.1f}" if field == 'below_6_pct' else sv[cube][field]
        if got != printed:
            raise SystemExit(f'single_value_fill no longer reproduces the submitted {cube} {field}')

    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {OUT.name}: guards passed (sha256, layout, repeated rows, 240 + 37 pairs, submitted values)')


if __name__ == '__main__':
    phi_safe.install()
    main()
