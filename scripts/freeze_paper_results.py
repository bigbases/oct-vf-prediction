"""Copy the JSON files the manuscript quotes into paper/results_frozen/.

The producers write into runs/ and experiments/, which this repository does not
publish (they also hold per-eye intermediates). This script takes the small
number of summary JSONs that tables and figures read, strips anything that
identifies an eye or a machine, and writes them to paper/results_frozen/ so a
reader can check a printed number without the withheld data.

Two transforms are applied, and only these two:

  * absolute paths are rewritten to repository-relative ones, and any home
    directory becomes /home/<user>;
  * in case_profile.json the pseudonym, the laterality and the examination date
    are removed. Each alone is harmless; together they identify a study eye.
    The metrics and the cohort position, which is what the manuscript cites,
    are kept unchanged.

Numbers are never rewritten.

Usage:  python scripts/freeze_paper_results.py --source <path to the working tree>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'results_frozen'

# The manuscript reports the pass-B numbers: the laterality correction in
# make_rnfl_flip.py (QUAD_SWAP) was applied upstream and every analysis re-run
# into the pass-B tree. The top-level runs/ directory of the working tree still
# holds the pass-A values (for example XGB out-of-fold RMSE 8.65 rather than
# 8.66), so the summary JSONs are taken from the pass-B tree, not from runs/.
# The experiments/ scripts already read pass B directly and have no pass-B copy.
PASS_B = 'experiments/laterality_qfix/step4_work/B'

# frozen filename -> (source-relative path, pass)
SOURCES = {
    'trivial_baselines.json':         (f'{PASS_B}/runs/trivial_baselines.json', 'B'),
    'final_model_comparison.json':    (f'{PASS_B}/runs/final_model_comparison.json', 'B'),
    'backbone_matrix_table.json':     (f'{PASS_B}/runs/backbone_matrix_table.json', 'B'),
    'aggregation_comparability.json': (f'{PASS_B}/runs/aggregation_comparability.json', 'B'),
    'ensemble_decisive.json':         (f'{PASS_B}/runs/ensemble_decisive.json', 'B'),
    'severity_region.json':           (f'{PASS_B}/runs/severity_region.json', 'B'),
    'skeleton_numbers.json':          (f'{PASS_B}/runs/skeleton_numbers.json', 'B'),
    'fusion_noninferiority.json':     (f'{PASS_B}/runs/fusion_noninferiority.json', 'B'),
    'fusion_consistency_matrix.json': (f'{PASS_B}/runs/fusion_consistency_matrix.json', 'B'),
    'patient_cluster_stats.json':     (f'{PASS_B}/runs/patient_cluster_stats.json', 'B'),
    'robustness_gains.json':          (f'{PASS_B}/runs/robustness_gains.json', 'B'),
    'multiseed_fusion.json':          (f'{PASS_B}/runs/multiseed_fusion.json', 'B'),
    'heldout_mae_reversal.json':      (f'{PASS_B}/runs/heldout_mae_reversal.json', 'B'),
    'fusion_sensitivity_trend.json':   (f'{PASS_B}/runs/fusion_sensitivity_trend.json', 'B'),
    'reliability_sensitivity.json':    (f'{PASS_B}/runs/reliability_sensitivity.json', 'B'),
    'stepsize_sensitivity.json':       (f'{PASS_B}/runs/stepsize_sensitivity.json', 'B'),
    'forest_rows.json':          ('experiments/forest_robustness/forest_rows.json', 'B'),
    'bias_by_bin.json':          ('experiments/bias_structure/bias_by_bin.json', 'B'),
    'bias_by_bin_refit.json':    ('experiments/bias_structure/bias_by_bin_refit.json', 'B'),
    'case_profile.json':         ('experiments/case_study/case_profile.json', 'A+B'),
    'resnet50_late_fusion.json': ('experiments/reviewer_round3/resnet50_late_fusion.json', 'B'),

    # Added 2026-09-17. These six back sentences that entered the manuscript
    # after this list was first written - the summary-branch ensemble, the
    # centred ensemble, the image-vs-summary head-to-head, the precision /
    # minimum-detectable-effect table, the error-structure decomposition and
    # the vertical cup-to-disc ablation. The experiments had all been run; the
    # results were simply never frozen, so check_numbers.py reported 42 numbers
    # as unsupported. They close 33 of them. All six declare pass B.
    #
    # sixth_backbone.json earns its place even though it lowers the MISS count
    # by nothing: its comparisons_basisB[1] (p 0.538, 120 of 240 eyes) is the
    # actual source of "not significant (p = 0.54, fusion better in 120 of 240
    # eyes)" in the ceiling subsection. Without it that sentence was passing
    # against the 0.54 in the weight column of the table just above it - a
    # coincidence, not evidence.
    'sixth_backbone.json':       ('experiments/reviewer_round3/sixth_backbone.json', 'B'),
    'symmetric_summary_ensemble.json': ('experiments/reviewer_round3/symmetric_summary_ensemble.json', 'B'),
    'centred_ensemble.json':     ('experiments/reviewer_round3/centred_ensemble.json', 'B'),
    'direct_test.json':          ('experiments/image_vs_summary/direct_test.json', 'B'),
    'fusion_precision_mde.json': (f'{PASS_B}/runs/fusion_precision_mde.json', 'B'),
    'fusion_error_structure.json': (f'{PASS_B}/runs/fusion_error_structure.json', 'B'),
    'ablation_vert_cd.json':     (f'{PASS_B}/runs/ablation_vert_cd.json', 'B'),

    # 반대편 눈 ablation (§4.3 "ten fellow-eye values ... 26 to 46").
    # 이 파일은 2026-09-17 **게이트 버그를 고치고 나서야** 필요해졌다. extract()
    # 가 LaTeX 이스케이프 `\%` 를 주석 시작으로 읽어 그 뒤를 통째로 버리던 탓에,
    # "(95\% CI $-0.089$ to $+0.238$" 의 CI 상한이 **한 번도 검사된 적이 없었다**.
    # 실험은 사전등록(docs/fellow_eye_ablation_prereg.md)까지 돼 있었고 값은
    # primary.{delta,ci_lo,ci_hi,p_wilcoxon} 에 그대로 있다. 패스 B.
    'fellow_eye_ablation.json':  (f'{PASS_B}/runs/fellow_eye_ablation.json', 'B'),

    # 아래 둘은 2026-09-17 **우연 통과 훑기**로 필요해졌다. 두 실험 모두 돌아
    # 있었고 값도 정확한데 동결만 안 돼 있었다 — 그래서 원고 수치가 근거 풀의
    # 무관한 값에 우연히 맞아 [OK] 로 떠 있었다. MISS 50 때와 같은 구조다.
    #
    # neg1_mask_sensitivity: §4.6 "p moves from 0.026 to 1.9e-4 on the held-out
    #   set". 진짜 근거는 splits.test.paired_excl.fusion_vs_xgb.p = 1.886e-4.
    #   여태는 reliability_sensitivity.json 의 **vgg16 OOF fusion_vs_cnn**
    #   1.852e-4 에 맞아 통과했다 — 백본도 split 도 비교쌍도 전부 다른 값이다.
    # osflip_compare: §4.4 좌우 미러링 절제 네 수치(+0.147/p=0.79/-0.261/p=0.19).
    #   p 둘은 paired_wilcoxon_flip_vs_noflip 에 그대로 있고, delta 둘은
    #   flip - noflip 차라 DERIVED 로 등록한다.
    'neg1_mask_sensitivity.json': (f'{PASS_B}/runs/neg1_mask_sensitivity.json', 'B'),
    'osflip_compare.json':       (f'{PASS_B}/runs/osflip_compare.json', 'B'),
}

# keys dropped from case_profile.json: pseudonym + eye + date is re-identifying
CASE_DROP = {'pseudonym', 'eye', 'vf_date'}

_ABS = re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+(?:/hvf_project)?')


def scrub_text(s: str) -> str:
    return _ABS.sub('/home/<user>', s)


def scrub(node):
    if isinstance(node, dict):
        return {k: scrub(v) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str):
        return scrub_text(node)
    return node


def scrub_case(doc: dict) -> dict:
    doc = scrub(doc)
    case = doc.get('case')
    if isinstance(case, dict):
        doc['case'] = {k: v for k, v in case.items() if k not in CASE_DROP}
    prov = doc.get('provenance')
    if isinstance(prov, dict):
        doc['provenance'] = {
            'note': 'The eye is supplied on the command line to '
                    'scripts/make_case_heatmap.py; no identifier is stored here '
                    'or in the code. The selection rule was not recorded at the '
                    'time, and the eye sits in the favourable tail of the '
                    'cohort - see distribution_position below.',
        }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    help='working tree that holds runs/ and experiments/')
    ap.add_argument('--check', action='store_true',
                    help='compare against what is already frozen, write nothing')
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    missing, changed = [], []

    for name, (rel, _pass) in SOURCES.items():
        p = src / rel
        if not p.exists():
            missing.append(rel)
            continue
        doc = json.loads(p.read_text(encoding='utf-8'))
        doc = scrub_case(doc) if name == 'case_profile.json' else scrub(doc)
        text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + '\n'
        dst = OUT / name
        if args.check:
            if not dst.exists() or dst.read_text(encoding='utf-8') != text:
                changed.append(name)
        else:
            dst.write_text(text, encoding='utf-8')

    for rel in missing:
        print(f'missing: {rel}')
    if args.check:
        for name in changed:
            print(f'drifted: {name}')
        print(f'{len(SOURCES) - len(missing) - len(changed)} / {len(SOURCES)} match')
        return 1 if (missing or changed) else 0
    print(f'froze {len(SOURCES) - len(missing)} / {len(SOURCES)} into {OUT}')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
