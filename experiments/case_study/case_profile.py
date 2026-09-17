#!/usr/bin/env python3
"""Profile of the eye used as the representative case in Figure 6.

It produces numbers only, no figure. What it measures is where that one eye
stands in the cohort: its per-eye error for each branch, its percentile in the
out-of-fold distribution of the fusion-minus-summary difference, its residual by
sensitivity bin, and its severity stratum. The eye sits in the favourable tail,
and no selection rule was recorded at the time; the output says so.

The eye is identified only by the pseudonym P-19. The record that binds that
pseudonym to a study eye is an internal record that is not part of this release, and
is not distributed with this repository.

대표 사례(fig_case_representative)로 쓴 안구의 프로파일 — 값만 낸다.

원고 Figure 6 (그림 번호 재배정 전에는 3, 그 뒤 4) 의 heat map 한 장이 어떤
안구인지는 공개 대상이 아닌 내부 등록부가 정본이다 (가명 P-19).
그림과 캡션 수치는 원고 §4.6 (Representative case).
`scripts/make_case_heatmap.py` 는 안구를 위치인자로 받으므로 선택 규칙이
코드에 없다. 이 스크립트는 그 안구가 코호트 안에서 어디에 서 있는지를 잰다.

  (2) 분포상 위치  per-eye RMSE/MAE 차이(fusion-summary, fusion-image)의
                   out-of-fold 240안 분포 백분위
  (3) 구간별 잔차  §4.4 와 같은 네 감도 구간의 branch별 평균 잔차 (양수=과대예측)
  (4) 중증도       cohort_md.csv 의 MD 와 층

기준 트리는 **패스 B** (`step4_work/B`). 안구 동일성 확인만 정본(패스 A)에서
원고 §4.6 의 캡션 수치와 대조한다. 둘 다 읽기 전용이다.
w 는 그림과 같은 고정 0.47 을 쓴다 (`make_case_heatmap.W`). 코호트 대조값은
`experiments/bias_structure/bias_by_bin.json` (fold별 nested w) 과 함께 낸다 —
xgb/cnn 잔차는 w 와 무관하므로 그 두 줄은 가드로 쓴다.

PHI: 실제 환자 ID 는 출력에 쓰지 않는다. `~/hvf_private/pseudonym_map.csv` 로
프로세스 안에서만 조회하고, 산출물에는 가명 P-19 만 남는다.

출력: experiments/case_study/case_profile.json / .md
env: aaa (matplotlib 계열 — ENVIRONMENT.md 참조)
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import percentileofscore

REPO = Path(__file__).resolve().parents[2]
TREES = {'A': REPO / 'experiments/laterality_qfix/step4_work/A',
         'B': REPO / 'experiments/laterality_qfix/step4_work/B'}
CANON = REPO
PSEUDO = 'P-19'
EYE = 'OS'
PSEUDO_MAP = Path.home() / 'hvf_private' / 'pseudonym_map.csv'
BIAS_JSON = REPO / 'experiments/bias_structure/bias_by_bin.json'
OUT = REPO / 'experiments/case_study'

# 원고 §4.6 캡션 수치(패스 A 기준). 안구 동일성 확인용.
CAPTION_A = {'summary': (8.380, 7.047), 'image': (8.152, 5.177),
             'fusion': (6.553, 4.560)}
TOL = 0.005
BRANCH = ('summary', 'image', 'fusion')   # XGB / CNN / late fusion


def real_pid() -> str:
    if not PSEUDO_MAP.exists():
        raise SystemExit(f'매핑표 없음: {PSEUDO_MAP}')
    with PSEUDO_MAP.open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if str(r.get('pseudonym', '')).strip() == PSEUDO:
                return str(r['real_patient_id']).strip()
    raise SystemExit(f'{PSEUDO} 없음')


def load_tree(tree: Path):
    """섀도 트리의 scripts/ 를 import 한다. ROOT 가 __file__ 로 정해지므로
    경로를 갈아끼우면 그 트리의 산출물을 읽는다 (bias_by_bin.py 와 같은 방식)."""
    for m in ('make_case_heatmap', 'oof_common', 'fig_style',
              'fusion_sensitivity_trend', 'fusion_eval_common',
              'build_severity_region'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tree / 'scripts'))
    try:
        mch = importlib.import_module('make_case_heatmap')
        fst = importlib.import_module('fusion_sensitivity_trend')
        assert mch.ROOT == tree, f'ROOT 불일치: {mch.ROOT} != {tree}'
        return mch, fst
    finally:
        sys.path.pop(0)


def load_severity(tree: Path):
    for m in ('build_severity_region', 'oof_common', 'fig_style'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(tree / 'scripts'))
    try:
        bsr = importlib.import_module('build_severity_region')
        assert bsr.ROOT == tree
        return bsr
    finally:
        sys.path.pop(0)


def all_eyes(mch):
    """5 fold 의 XGB val 과 CNN val 을 키로 맞춰 240안을 쌓는다.
    make_case_heatmap.find_eye 가 한 안에 하는 짝짓기를 전체로 넓힌 것이다."""
    keys, xgb, cnn, lab, mask, fold = [], [], [], [], [], []
    for k in range(5):
        xz = mch.load_oof_npz(mch.XGB[k])
        cz = mch.load_oof_npz(mch.CNN / f'val_preds_fold{k}.npz')
        cidx = {kk: j for j, kk in enumerate(cz['keys'])}
        for i, key in enumerate(xz['keys']):
            if key not in cidx:
                continue
            j = cidx[key]
            keys.append(tuple(str(v) for v in key))
            xgb.append(xz['pred'][i]); cnn.append(cz['pred'][j])
            lab.append(xz['labels'][i])
            mask.append(xz['mask'][i] & cz['mask'][j])
            fold.append(k)
    d = dict(keys=keys, fold=np.array(fold),
             summary=np.array(xgb), image=np.array(cnn),
             lab=np.array(lab), mask=np.array(mask).astype(bool))
    d['fusion'] = mch.W * d['summary'] + (1 - mch.W) * d['image']
    return d


def eye_metric(pred, lab, mask, kind):
    """안별 RMSE 또는 MAE. mask 밖 지점은 뺀다."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        e = pred[i][m] - lab[i][m]
        out[i] = np.sqrt(np.mean(e ** 2)) if kind == 'rmse' else np.mean(np.abs(e))
    return out


def bin_residuals(fst, pred, lab, mask):
    """구간별 (평균 잔차, 지점 수). 잔차 = 예측 - 실측, 양수가 과대예측."""
    m = mask.astype(bool)
    bi = fst.bin_index(lab[m])
    r = (pred - lab)[m]
    means = np.array([r[bi == b].mean() if (bi == b).any() else np.nan
                      for b in range(4)])
    n = np.array([int((bi == b).sum()) for b in range(4)])
    return means, n


def main() -> int:
    pid = real_pid()
    checks: dict[str, dict] = {}

    def chk(name, got, want, tol):
        ok = bool(np.isfinite(got) and np.isfinite(want) and abs(got - want) <= tol)
        checks[name] = {'got': round(float(got), 6), 'want': round(float(want), 6),
                        'tol': tol, 'ok': ok}
        return ok

    # ── 안구 동일성: 정본(패스 A) 에서 캡션 수치를 재현하는가 ──────────────
    mch_a, _ = load_tree(CANON)
    ea = mch_a.find_eye(pid, EYE)
    passA = {}
    for b, key in zip(BRANCH, ('xgb', 'cnn', 'fus')):
        r = mch_a.rmse(ea[key], ea['lab'], ea['mask'])
        a = mch_a.mae(ea[key], ea['lab'], ea['mask'])
        passA[b] = {'rmse': round(float(r), 4), 'mae': round(float(a), 4)}
        chk(f'passA_{b}_rmse', r, CAPTION_A[b][0], TOL)
        chk(f'passA_{b}_mae', a, CAPTION_A[b][1], TOL)
    vf_date = str(ea['key'][2])
    passA['fold'] = int(ea['fold'])
    passA['n_points'] = int(ea['mask'].sum())
    chk('passA_n_points', ea['mask'].sum(), 52, 0)

    # ── 패스 B: 240안 전체 ────────────────────────────────────────────────
    mch, fst = load_tree(TREES['B'])
    D = all_eyes(mch)
    n_eye = len(D['keys'])
    chk('n_eyes_oof', n_eye, 240, 0)
    idx = [i for i, k in enumerate(D['keys']) if k[0] == pid and k[1] == EYE]
    if len(idx) != 1:
        raise SystemExit(f'패스 B 에서 안구 {len(idx)}건 (1이어야 함)')
    i0 = idx[0]
    chk('vf_date_match_A_B', float(D['keys'][i0][2] == vf_date), 1.0, 0)

    # (2) 분포상 위치 ------------------------------------------------------
    dist = {}
    case_metric = {}
    for kind in ('rmse', 'mae'):
        per = {b: eye_metric(D[b], D['lab'], D['mask'], kind) for b in BRANCH}
        case_metric[kind] = {b: round(float(per[b][i0]), 4) for b in BRANCH}
        for ref in ('summary', 'image'):
            d = per['fusion'] - per[ref]
            v = d[i0]
            rank = int((d < v).sum()) + 1          # 오름차순 순위, 1 = 가장 유리
            dist[f'{kind}_fusion_minus_{ref}'] = {
                'case_delta': round(float(v), 4),
                'percentile_rank': round(float(percentileofscore(d, v, 'rank')), 2),
                'percentile_strict_below': round(float(percentileofscore(d, v, 'strict')), 2),
                'rank_ascending': rank, 'n': int(d.size),
                'cohort_mean': round(float(d.mean()), 4),
                'cohort_median': round(float(np.median(d)), 4),
                'cohort_min': round(float(d.min()), 4),
                'cohort_max': round(float(d.max()), 4),
                'n_negative': int((d < 0).sum()),
            }

    # (3) 구간별 잔차 ------------------------------------------------------
    one = {b: D[b][i0:i0 + 1] for b in BRANCH}
    case_bins, cohort_bins = {}, {}
    for b in BRANCH:
        m, n = bin_residuals(fst, one[b], D['lab'][i0:i0 + 1], D['mask'][i0:i0 + 1])
        case_bins[b] = {'residual_mean_by_bin': [None if np.isnan(x) else round(float(x), 4) for x in m],
                        'n_points_by_bin': n.tolist(),
                        'residual_mean_global': round(float(
                            (one[b][0] - D['lab'][i0])[D['mask'][i0]].mean()), 4)}
        cm, cn = bin_residuals(fst, D[b], D['lab'], D['mask'])
        cohort_bins[b] = {'residual_mean_by_bin': [round(float(x), 4) for x in cm],
                          'n_points_by_bin': cn.tolist(),
                          'residual_mean_global': round(float(
                              (D[b] - D['lab'])[D['mask']].mean()), 4)}
    case_bins['n_points_by_bin'] = case_bins['summary']['n_points_by_bin']

    # 코호트 xgb/cnn 잔차는 w 와 무관하므로 bias_by_bin.json 패스 B 와 같아야 한다.
    ref = json.loads(BIAS_JSON.read_text(encoding='utf-8'))['passes']['B']['IR-v2']['OOF']
    for b, key in (('summary', 'xgb'), ('image', 'cnn')):
        for j in range(4):
            chk(f'cohort_{b}_bin{j}', cohort_bins[b]['residual_mean_by_bin'][j],
                ref[f'{key}_residual_mean_by_bin'][j], 1e-3)
    for j in range(4):
        chk(f'cohort_n_bin{j}', cohort_bins['summary']['n_points_by_bin'][j],
            ref['n_points_by_bin'][j], 0)

    # (4) 중증도 ----------------------------------------------------------
    bsr = load_severity(TREES['B'])
    md = bsr.MD.get(bsr.nkey(pid, EYE, vf_date))
    stratum = None
    if md is not None and np.isfinite(md):
        stratum = next((nm for nm, fn in bsr.STRATA if fn(md)), None)
    md_all = np.array([v for v in (bsr.MD.get(bsr.nkey(*k)) for k in D['keys'])
                       if v is not None and np.isfinite(v)])
    sev = {'md': None if md is None else round(float(md), 2),
           'stratum': stratum,
           'md_source': 'cohort_md.csv (step4_work/B)',
           'cohort_md_n': int(md_all.size),
           'md_percentile_rank': None if md is None else
               round(float(percentileofscore(md_all, md, 'rank')), 2),
           'cohort_md_median': round(float(np.median(md_all)), 2),
           'strata_counts': {nm: int(sum(1 for v in md_all if fn(v)))
                             for nm, fn in bsr.STRATA}}

    payload = {
        'note': 'fig_case_representative 대표안 프로파일. 값만 산출, 그림 없음.',
        'case': {'pseudonym': PSEUDO, 'eye': EYE, 'vf_date': vf_date,
                 'fold': int(D['fold'][i0]),
                 'n_valid_points': int(D['mask'][i0].sum()),
                 'label_range_dB': [round(float(D['lab'][i0][D['mask'][i0]].min()), 1),
                                    round(float(D['lab'][i0][D['mask'][i0]].max()), 1)]},
        'provenance': {
            'how_the_eye_is_supplied': 'scripts/make_case_heatmap.py 의 위치인자 '
                                       '(ap.add_argument("pid"), ap.add_argument("eye")). '
                                       '선택 규칙이 코드에 없다.',
            'canonical_record': 'internal figure registry, not released (P-19 / fold 2)',
            'how_that_record_was_made': '원고 §4.6 캡션에 인쇄된 안 단위 '
                                        '지표를 240안 전체에 tol 0.005 로 역추적해 사후 복원. '
                                        '§5 는 이 복원이 "우연 덕" 이라고 적는다.',
            'runtime_binding': 'experiments/laterality_qfix/step4_runner.py 의 '
                               'CASE_EYE / case_pid(pseudonym="P-19")',
            'selection_rationale_recorded': False,
            'searched': ['docs/', 'scripts/', 'experiments/',
                         'scripts/make_case_heatmap.py 주석'],
        },
        'basis': {'tree_primary': str(TREES['B']), 'tree_identity_check': str(CANON),
                  'w_fixed': mch.W, 'backbone': 'IR-v2 (Inception-ResNet-v2)',
                  'branch_map': {'summary': 'XGB', 'image': 'CNN',
                                 'fusion': 'w*XGB + (1-w)*CNN'},
                  'residual_sign': 'prediction - measurement, 양수 = 과대예측',
                  'bins': fst.BIN_LABELS, 'bin_edges_dB': [0, 10, 20, 30, 'inf']},
        'passA_identity_check': passA,
        'passB_case_metrics': case_metric,
        'distribution_position': dist,
        'bins_case': case_bins,
        'bins_cohort_w047': cohort_bins,
        'bins_cohort_nested_w_reference': {
            'source': 'experiments/bias_structure/bias_by_bin.json::passes.B.IR-v2.OOF',
            'n_points_by_bin': ref['n_points_by_bin'],
            'summary_residual_mean_by_bin': [round(float(x), 4) for x in ref['xgb_residual_mean_by_bin']],
            'image_residual_mean_by_bin': [round(float(x), 4) for x in ref['cnn_residual_mean_by_bin']],
            'summary_residual_mean_global': round(float(ref['xgb_residual_mean_global']), 4),
            'image_residual_mean_global': round(float(ref['cnn_residual_mean_global']), 4),
        },
        'severity': sev,
        'checks': checks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'case_profile.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    bad = [k for k, v in checks.items() if not v['ok']]
    print(f'가드 {len(checks) - len(bad)}/{len(checks)} 통과' + (f' — 실패 {bad}' if bad else ''))
    for k in ('rmse', 'mae'):
        for r in ('summary', 'image'):
            e = dist[f'{k}_fusion_minus_{r}']
            print(f'  {k} fusion-{r:8s} Δ={e["case_delta"]:+7.3f}  '
                  f'{e["percentile_rank"]:6.2f} 백분위  {e["rank_ascending"]:3d}/{e["n"]}')
    print(f'  MD {sev["md"]} ({sev["stratum"]})')
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
