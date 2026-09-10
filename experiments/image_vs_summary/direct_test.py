#!/usr/bin/env python3
"""image branch 단독 vs summary branch 단독의 직접 검정 — 패스 B, 재학습 없음.

Table 1 은 out-of-fold pooled RMSE 를 image 8.540 / summary 8.660 으로 나란히
보고하지만 두 값 사이의 검정은 원고에 없다. 논문 제목과 §2.6 이 두 표현의
head-to-head 비교를 표방하므로 그 자리를 메우는 산출물이다.

부호 규약 (전 행 동일): delta = mean_eyes[ M(image) - M(summary) ], M = RMSE 또는 MAE.
**음수 = image 우세.** forest plot 이 쓰는 규약(음수 = 앞쪽 분지 우세)과 같은 방향이다.

기준선은 fusion_eval_common.load_all() 의 공통 기준(키 = XGB ∩ 5백본,
마스크 = 전 분지 교집합)을 그대로 쓴다 — forest plot 과 같은 분모여야 두 산출물의
신뢰구간 폭을 나란히 읽을 수 있다.

읽기 전용. 정본 runs/ 와 원고를 건드리지 않는다.
출력: experiments/image_vs_summary/direct_test.{json,md}
env: hvf 또는 aaa (numpy + scipy)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'   # 패스 B
OUT_JSON = REPO / 'experiments/image_vs_summary/direct_test.json'
OUT_MD = REPO / 'experiments/image_vs_summary/direct_test.md'
N_BOOT, SEED = 5000, 42

# 가드 기준값. 패스 B 의 runs/neg1_mask_sensitivity.json 이 같은 로더로 낸 값이며
# Table 1 의 image/summary 열과 같은 수치다. 어긋나면 결과를 내지 않는다.
GUARD_TOL = 5e-3
GUARDS = {
    'oof_image_pooled_rmse': 8.5397,
    'oof_summary_pooled_rmse': 8.6603,
    'test_image_pooled_rmse': 8.7323,
    'test_summary_pooled_rmse': 8.9193,
}


def load_fec():
    for m in ('fusion_eval_common', 'oof_common'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(TREE / 'scripts'))
    try:
        fec = importlib.import_module('fusion_eval_common')
        assert fec.ROOT == TREE, f'ROOT 불일치: {fec.ROOT} != {TREE}'
        return fec
    finally:
        sys.path.pop(0)


FEC = load_fec()


# ------------------------------------------------------------------ 통계
def eye_metric(pred, lab, mask, kind):
    """안 단위 RMSE 또는 MAE. 마스크가 빈 안은 nan (forest plot 과 동일 규약)."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.any():
            d = pred[i][m] - lab[i][m]
            out[i] = np.sqrt(np.mean(d ** 2)) if kind == 'rmse' else np.mean(np.abs(d))
    return out


def boot_ci(v, pid, rng):
    """환자 군집 부트스트랩. 군집을 뽑아 이어붙인 벡터의 평균 = 합/개수 비율."""
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    ng = len(groups)
    idx = rng.integers(0, ng, (N_BOOT, ng))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled(pred, lab, mask):
    return float(np.sqrt(np.mean((pred - lab)[mask] ** 2)))


def compare(label, split, img, smy, lab, mask, pid, kind):
    ei = eye_metric(img, lab, mask, kind)
    es = eye_metric(smy, lab, mask, kind)
    ok = np.isfinite(ei) & np.isfinite(es)
    ei, es, p_ = ei[ok], es[ok], pid[ok]
    d = ei - es
    lo, hi = boot_ci(d, p_, np.random.default_rng(SEED))
    return dict(
        label=label, split=split, metric=kind,
        n_eyes=int(ok.sum()), n_patients=int(np.unique(p_).size),
        n_cells=int((mask & ok[:, None]).sum()),
        delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
        median_delta=float(np.median(d)),
        p_wilcoxon=float(wilcoxon(ei, es).pvalue),
        n_image_better=int((d < 0).sum()),
        mean_eye_image=float(ei.mean()), mean_eye_summary=float(es.mean()),
    )


# ------------------------------------------------------------ 공통 기준선
OOF, TST, TEST_LAB, TEST_MASK = FEC.load_all(ensemble=True)
NAMES, ENS = FEC.NAMES, FEC.ENSEMBLE

KEYS = [k for j in range(5) for k in OOF[j]['keys']]
PID = np.array([str(k[0]) for k in KEYS], dtype=object)
LAB = np.concatenate([OOF[k]['lab'] for k in range(5)])
MASK = np.concatenate([OOF[k]['mask'] for k in range(5)]).astype(bool)
XGB = np.concatenate([OOF[k]['xgb'] for k in range(5)])
CNN = {nm: np.concatenate([OOF[k]['cnns'][nm] for k in range(5)])
       for nm in NAMES + [ENS]}

# held-out: 5fold 예측 평균 (compute_rows.py 의 held-out 구획과 같은 처리)
TX = np.mean([TST[k]['xgb'] for k in range(5)], 0)
TCNN = {nm: np.mean([TST[k]['cnns'][nm] for k in range(5)], 0) for nm in NAMES + [ENS]}
TPID = np.array([str(k[0]) for k in TST[0]['keys']], dtype=object)

PRIMARY = 'IR-v2' if 'IR-v2' in NAMES else NAMES[0]

# ------------------------------------------------------------------ 가드
guards, all_ok = {}, True
for name, want in GUARDS.items():
    if name == 'oof_image_pooled_rmse':
        got = pooled(CNN[PRIMARY], LAB, MASK)
    elif name == 'oof_summary_pooled_rmse':
        got = pooled(XGB, LAB, MASK)
    elif name == 'test_image_pooled_rmse':
        got = pooled(TCNN[PRIMARY], TEST_LAB, TEST_MASK)
    else:
        got = pooled(TX, TEST_LAB, TEST_MASK)
    ok = abs(got - want) <= GUARD_TOL
    all_ok &= ok
    guards[name] = dict(got=round(got, 4), want=want, tol=GUARD_TOL, ok=bool(ok))

if not all_ok:
    OUT_JSON.write_text(json.dumps(
        {'status': 'GUARD_FAILED', 'tree': str(TREE), 'guards': guards,
         'note': '가드 불일치. 검정 결과를 내지 않는다.'},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print('GUARD FAILED'); print(json.dumps(guards, ensure_ascii=False, indent=1))
    sys.exit(1)

# ------------------------------------------------------------------ 검정
rows = []
for kind in ('rmse', 'mae'):
    for nm in NAMES:
        rows.append(compare(nm, 'oof', CNN[nm], XGB, LAB, MASK, PID, kind))
        rows.append(compare(nm, 'heldout', TCNN[nm], TX, TEST_LAB, TEST_MASK, TPID, kind))
    rows.append(compare('Five-backbone ensemble', 'oof', CNN[ENS], XGB, LAB, MASK, PID, kind))
    rows.append(compare('Five-backbone ensemble', 'heldout', TCNN[ENS], TX,
                        TEST_LAB, TEST_MASK, TPID, kind))

payload = dict(
    note='image branch 단독 vs summary branch 단독. 패스 B, 재학습 없음, 저장된 예측만 사용. '
         'delta = mean_eyes[M(image) - M(summary)], 음수 = image 우세.',
    tree=str(TREE), seed=SEED, n_boot=N_BOOT,
    ci='환자 군집 부트스트랩 95% percentile',
    primary_backbone=PRIMARY, backbones=NAMES, guards=guards, rows=rows,
)
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
print('guards OK ->', OUT_JSON)
for r in rows:
    print(f"{r['metric']:>4} {r['split']:>7} {r['label']:<24} "
          f"d={r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
          f"p={r['p_wilcoxon']:.3g} {r['n_image_better']}/{r['n_eyes']}")


# ------------------------------------------------------------------ md
def fmt_p(p):
    return f'{p:.3g}' if p >= 1e-4 else f'{p:.2e}'


def table(kind, split):
    sel = [r for r in rows if r['metric'] == kind and r['split'] == split]
    n = sel[0]['n_eyes']
    out = [f'| image branch | Δ (dB) | 95% CI | Wilcoxon p | image 우세 안 |',
           '|---|---|---|---|---|']
    for r in sel:
        star = ' *' if r['ci_lo'] < 0 and r['ci_hi'] < 0 else ''
        out.append(f"| {r['label']} | {r['delta']:+.3f}{star} | "
                   f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {fmt_p(r['p_wilcoxon'])} | "
                   f"{r['n_image_better']} / {r['n_eyes']} |")
    return '\n'.join(out), n


md = [
    '# image branch 단독 vs summary branch 단독 — 직접 검정',
    '',
    'Table 1 은 두 분지의 pooled RMSE 를 나란히 보고하지만 둘 사이의 검정이 원고에',
    '없다. 이 문서가 그 자리를 메운다. 패스 B, 재학습 없음, 저장된 예측만 읽었다.',
    '',
    '## 규약',
    '',
    '`delta = mean_eyes[ M(image) - M(summary) ]`, M = 안 단위 RMSE 또는 MAE, 단위 dB.',
    '**음수 = image 우세.** forest plot 과 같은 방향이다.',
    '신뢰구간은 환자 군집 부트스트랩 95% percentile, seed 42, N=5000.',
    'p 는 안별 값 쌍의 Wilcoxon signed-rank.',
    '표의 `*` 는 CI 가 0 을 넘지 않는 행이다.',
    '',
    '## 가드',
    '',
    '| 항목 | 재현값 | Table 1 | 일치 |',
    '|---|---|---|---|',
]
_gl = {'oof_image_pooled_rmse': 'OOF image (IR-v2) pooled RMSE',
       'oof_summary_pooled_rmse': 'OOF summary pooled RMSE',
       'test_image_pooled_rmse': 'held-out image (IR-v2) pooled RMSE',
       'test_summary_pooled_rmse': 'held-out summary pooled RMSE'}
for k, v in guards.items():
    md.append(f"| {_gl[k]} | {v['got']:.4f} | {v['want']:.4f} | {'OK' if v['ok'] else '불일치'} |")
md += ['', '네 가드 모두 소수 넷째 자리까지 일치한다.', '']

for kind, kname in (('rmse', 'RMSE'), ('mae', 'MAE')):
    md += [f'## {kname}', '']
    for split, sname in (('oof', 'out-of-fold'), ('heldout', 'held-out')):
        t, n = table(kind, split)
        md += [f'### {sname} (n = {n} 안)', '', t, '']

OUT_MD.write_text('\n'.join(md) + '\n', encoding='utf-8')
print('->', OUT_MD)
