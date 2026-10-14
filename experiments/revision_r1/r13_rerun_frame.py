#!/usr/bin/env python3
"""The submitted analyses with the matched current-code reruns in place of the submitted runs
that the current code does not reproduce (VGG16, Xception, DenseNet121). IR-v2 and
Inception-v3 keep the submitted runs, which their reruns match bit for bit.

Not in the plan: added 2026-10-14 at the authors' request, after the mirrored-frame results of
r13_analysis.py were seen. It answers whether the main conclusions (fusion vs summary per
backbone, the ensemble ceiling, and the sensitivity-bin trend) depend on the submitted
predictions of the three non-reproducible backbones.

Same computation as r13_analysis.endpoints; only the backbone -> prediction directory map
changes. While some reruns are still in the queue, the per-backbone rows of the finished ones
are computed with the other backbones left at their submitted runs (per-backbone rows do not
depend on the other backbones); the ensemble ceiling and trend wait for all three.

Output: r13_rerun_frame.json (no patient keys).
"""
from __future__ import annotations

import json

import r13_analysis as R

NON_REPRODUCIBLE = ('VGG16', 'Xception', 'DenseNet121')


def main():
    sub, *_ = R.endpoints(R.SUBMITTED)
    g = R.guards(sub)
    done = [nm for nm in NON_REPRODUCIBLE if R.complete(R.UNMIRRORED_REF[nm])]
    bbmap = dict(R.SUBMITTED)
    bbmap.update({nm: R.UNMIRRORED_REF[nm] for nm in done})
    doc = {'note': 'Submitted analyses with the matched reruns for the backbones whose submitted runs '
                   'are not reproducible. Added after the R1-3 results (not pre-specified).',
           'guards': g, 'reruns_used': done,
           'reruns_missing': [nm for nm in NON_REPRODUCIBLE if nm not in done],
           'submitted': sub}
    if done:
        rr, *_ = R.endpoints(bbmap)
        keep = set(done) | {'IR-v2', 'Inception-v3'}
        if len(done) == len(NON_REPRODUCIBLE):
            doc['rerun'] = rr
        else:
            # ceiling, ensemble trend and pooled RMSE mix submitted and rerun predictions: dropped
            doc['rerun'] = {'n_eyes_oof': rr['n_eyes_oof'], 'n_eyes_test': rr['n_eyes_test'],
                            'fusion_vs_summary': [r for r in rr['fusion_vs_summary'] if r['backbone'] in done],
                            'trend': [t for t in rr['trend'] if t['backbone'] in done],
                            'partial': True}
        doc['rerun_backbones'] = sorted(keep) if not doc['rerun'].get('partial') else done
    R.FE.BB.clear(); R.FE.BB.update(R.SUBMITTED)
    (R.HERE / 'r13_rerun_frame.json').write_text(json.dumps(doc, indent=1, default=float) + '\n')
    print('reruns used:', done or 'none', '| missing:', doc['reruns_missing'] or 'none')


if __name__ == '__main__':
    import phi_safe
    phi_safe.install()
    main()
