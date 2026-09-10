#!/usr/bin/env python3
"""Figure 6 (forest plot) 의 행 값을 산출한다 — 패스 B, 재학습 없음.

각 행은 하나의 강건성 조건에서 "fusion 이 summary branch(XGB) 를 이기는가" 를
안 단위 RMSE 차이 하나로 요약한다.

부호 규약 (전 행 동일): delta = mean_eyes[ RMSE(fusion) - RMSE(summary) ], dB.
음수 = fusion 우세. 원고 본문은 이득을 양수로 쓰는 곳과 차이를 음수로 쓰는 곳이
섞여 있으므로 여기서는 음수 규약 하나만 쓴다.

기준선 통일: 모든 행의 안 집합·평가 마스크는 fusion_eval_common.load_all() 이
만드는 공통 기준(키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합)에서 출발하고,
조건별 제한(셀 제외 / 안 제외 / 다른 XGB)만 그 위에 얹는다. 조건마다 다른
로더를 쓰면 분모가 달라져 신뢰구간 폭이 조건이 아니라 정렬 차이를 반영한다.

읽기 전용. 정본 runs/ 와 원고를 건드리지 않는다.
출력: experiments/forest_robustness/forest_rows.json
env: hvf 또는 aaa (numpy + scipy)
"""
from __future__ import annotations

import csv
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'   # 패스 B
OUT = REPO / 'experiments/forest_robustness/forest_rows.json'
N_BOOT, SEED = 5000, 42
W_FIXED = 0.47          # §4.2 의 seed / floor / reliability arm 이 쓴 고정 w


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
from oof_common import load_oof_npz  # noqa: E402  (TREE/scripts 와 동일 파일)


# ------------------------------------------------------------------ 통계
def eye_rmse(pred, lab, mask):
    """안 단위 RMSE. 마스크가 빈 안은 nan."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.any():
            out[i] = np.sqrt(np.mean((pred[i][m] - lab[i][m]) ** 2))
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


def row(label, group, fus, xgb, lab, mask, pid, *, keep=None, w=None,
        source='new', note=''):
    """delta = fusion - summary 의 안 단위 평균, 환자 군집 CI, Wilcoxon."""
    if keep is None:
        keep = np.ones(pred_n := fus.shape[0], bool)
    ef = eye_rmse(fus, lab, mask)
    ex = eye_rmse(xgb, lab, mask)
    ok = keep & np.isfinite(ef) & np.isfinite(ex)
    ef, ex, p_ = ef[ok], ex[ok], pid[ok]
    d = ef - ex
    lo, hi = boot_ci(d, p_, np.random.default_rng(SEED))
    m = mask & ok[:, None]
    return dict(
        label=label, group=group, n_eyes=int(ok.sum()), n_patients=int(np.unique(p_).size),
        n_cells=int(m.sum()), w=w, source=source, note=note,
        delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
        median_delta=float(np.median(d)),
        p_wilcoxon=float(wilcoxon(ef, ex).pvalue),
        n_fusion_better=int((d < 0).sum()),
        pooled_rmse_fusion=float(np.sqrt(np.mean((fus - lab)[m] ** 2))),
        pooled_rmse_summary=float(np.sqrt(np.mean((xgb - lab)[m] ** 2))),
        mean_eye_rmse_fusion=float(ef.mean()), mean_eye_rmse_summary=float(ex.mean()),
    )


# ------------------------------------------------------------ 공통 기준선
OOF, TST, TEST_LAB, TEST_MASK = FEC.load_all(ensemble=True)
NAMES = FEC.NAMES
ENS = FEC.ENSEMBLE

KEYS = [k for j in range(5) for k in OOF[j]['keys']]
PID = np.array([str(k[0]) for k in KEYS], dtype=object)
LAB = np.concatenate([OOF[k]['lab'] for k in range(5)])
MASK = np.concatenate([OOF[k]['mask'] for k in range(5)]).astype(bool)
XGB = np.concatenate([OOF[k]['xgb'] for k in range(5)])
CNN = {nm: np.concatenate([OOF[k]['cnns'][nm] for k in range(5)])
       for nm in NAMES + [ENS]}
BOUNDS, _n = [], 0
for k in range(5):
    BOUNDS.append((_n, _n + OOF[k]['xgb'].shape[0])); _n += OOF[k]['xgb'].shape[0]

W_GLOBAL = {nm: FEC.fit_w([OOF[k] for k in range(5)], nm) for nm in NAMES + [ENS]}
W_NESTED = {nm: [FEC.fit_w([OOF[i] for i in range(5) if i != k], nm) for k in range(5)]
            for nm in NAMES + [ENS]}


def fuse_global(nm, w=None):
    w = W_GLOBAL[nm] if w is None else w
    return w * XGB + (1 - w) * CNN[nm]


def fuse_nested(nm):
    out = np.empty_like(XGB)
    for w, (a, b) in zip(W_NESTED[nm], BOUNDS):
        out[a:b] = w * XGB[a:b] + (1 - w) * CNN[nm][a:b]
    return out


# ------------------------------------------------- 조건별 안/셀 제한 재현
def neg1_drop_mask():
    """runs/backup_vf_neg1_fix_.../CSV 의 원본 -1 셀. neg1_mask_sensitivity.py 와 동일."""
    PT52 = [f'p{i:02d}' for i in range(1, 55) if f'p{i:02d}' not in ('p26', 'p35')]

    def read(p):
        with open(p, encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))

    old = read(TREE / 'runs/backup_vf_neg1_fix_20260530_193547/ml_final_90d_excl_empty_flip.csv')
    new = read(TREE / 'ml_final_90d_excl_empty_flip.csv')
    vec = lambda r: np.array([float(r[p]) for p in PT52], float)

    oi = {}
    for i, r in enumerate(old):
        oi.setdefault((r['patient_id'], r['eye'], r['vf_date']), []).append(i)
    pairs, used_o, used_n = [], set(), set()
    for j, r in enumerate(new):
        k = (r['patient_id'], r['eye'], r['vf_date'])
        if oi.get(k):
            i = oi[k].pop(0); pairs.append((i, j)); used_o.add(i); used_n.add(j)
    ro = [i for i in range(len(old)) if i not in used_o]
    for j in [j for j in range(len(new)) if j not in used_n]:
        cand = [i for i in ro if (old[i]['patient_id'], old[i]['eye'])
                == (new[j]['patient_id'], new[j]['eye'])]
        assert len(cand) == 1, (j, cand)
        i = cand[0]; ro.remove(i); pairs.append((i, j))
    assert len(pairs) == len(new) == 280

    neg1, bad_a, bad_b = {}, 0, 0
    for i, j in pairs:
        o, n = vec(old[i]), vec(new[j])
        m = (o == -1.0)
        neg1[j] = m
        bad_a += int(np.sum((~m) & (o != n)))
        bad_b += int(np.sum(n[m] != 0.0))
    assert bad_a == 0 and bad_b == 0, (bad_a, bad_b)

    nk = {}
    for j, r in enumerate(new):
        nk.setdefault((r['patient_id'], r['eye'], r['vf_date']), []).append(j)
    drop = np.zeros_like(MASK)
    for i, k in enumerate(KEYS):
        js = nk[(str(k[0]), str(k[1]), str(k[2]))]
        assert len(js) == 1
        drop[i] = neg1[js[0]]
    return drop


def reliability_sets():
    """cohort_md.csv 에서 FL / FL|FP 배제 키. reliability_sensitivity.py 와 동일."""
    norm = lambda p, e, d: (re.sub(r'\D', '', str(p)), str(e).strip().upper(),
                            re.sub(r'\D', '', str(d)))
    bad_A, bad_B = set(), set()
    with open(TREE / 'cohort_md.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            k = norm(r['patient_id'], r['eye'], r['vf_date'])
            m = re.match(r'\s*(\d+)\s*/\s*(\d+)', str(r.get('fix_loss', '')))
            flr = int(m.group(1)) / int(m.group(2)) if m and int(m.group(2)) > 0 else None
            try:
                fp = float(r.get('false_pos', '') or 'nan')
            except ValueError:
                fp = float('nan')
            if str(r.get('low_reliability', '')).strip().upper() == 'Y':
                bad_A.add(k)
            if (flr is not None and flr >= 0.20) or (np.isfinite(fp) and fp >= 15.0):
                bad_B.add(k)
    kn = [norm(*k) for k in KEYS]
    return (np.array([k not in bad_A for k in kn]),
            np.array([k not in bad_B for k in kn]))


def load_alt_cnn(dirname):
    """다른 seed 의 IR-v2 OOF 예측을 공통 키 순서로 정렬."""
    parts = []
    for k in range(5):
        z = load_oof_npz(TREE / dirname / f'val_preds_fold{k}.npz')
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        parts.append(z['pred'][[idx[kk] for kk in OOF[k]['keys']]])
    return np.concatenate(parts)


def load_alt_xgb(tag, base=None):
    """다른 feature set 의 XGB OOF 예측 + 마스크를 공통 키 순서로 정렬."""
    base = TREE / 'runs/oof' if base is None else base
    ps, ms = [], []
    for k in range(5):
        z = load_oof_npz(base / f'xgb_{tag}_fold{k}_val.npz')
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        ii = [idx[kk] for kk in OOF[k]['keys']]
        ps.append(z['pred'][ii]); ms.append(z['mask'][ii])
    return np.concatenate(ps), np.concatenate(ms).astype(bool)


def fit_w_arrays(x, c, lab, mask):
    xm, cm, lm = x[mask], c[mask], lab[mask]
    errs = [np.mean((w * xm + (1 - w) * cm - lm) ** 2) for w in np.linspace(0, 1, 101)]
    return float(np.linspace(0, 1, 101)[int(np.argmin(errs))])


# ------------------------------------------------------------------- 행
rows, checks = [], {}

# --- primary + 백본 4종 (global w = tab:backbones 규약) --------------------
DISPLAY = {'IR-v2': 'Primary: Inception-ResNet-v2',
           'Inception-v3': 'Inception-v3', 'VGG16': 'VGG16',
           'Xception': 'Xception', 'DenseNet121': 'DenseNet121'}
for nm in ['IR-v2', 'Inception-v3', 'VGG16', 'Xception', 'DenseNet121']:
    rows.append(row(DISPLAY[nm], 'primary' if nm == 'IR-v2' else 'backbone',
                    fuse_global(nm), XGB, LAB, MASK, PID,
                    w=W_GLOBAL[nm], note='global w (whole-OOF fit)'))

# --- 앙상블: 두 규약 모두 산출, 그림에는 nested (tab:ceiling 과 같은 규약) ---
ens_nested = row('Five-backbone ensemble', 'ensemble', fuse_nested(ENS), XGB,
                 LAB, MASK, PID, w=None, note='nested w per fold (tab:ceiling)')
ens_nested['w_per_fold'] = [round(w, 3) for w in W_NESTED[ENS]]
ens_global = row('Five-backbone ensemble (global w)', 'ensemble_alt',
                 fuse_global(ENS), XGB, LAB, MASK, PID,
                 w=W_GLOBAL[ENS], note='global w — 대조용, 그림에 넣지 않는다')
rows.append(ens_nested)

# --- seeds (w 는 multiseed_fusion.py 와 같은 0.47 고정) --------------------
for s, d in ((43, 'runs/repro_final_ir_v2_5fold_s43'),
             (44, 'runs/repro_final_ir_v2_5fold_s44')):
    c = load_alt_cnn(d)
    rows.append(row(f'Seed {s}', 'seed', W_FIXED * XGB + (1 - W_FIXED) * c,
                    XGB, LAB, MASK, PID, w=W_FIXED, note='w fixed at 0.47'))

# --- no vertical C/D (25 파라미터) ----------------------------------------
# XGB(25) 예측은 refit_xgb25.py 가 재적합해 저장했다(같은 재적합으로 26feat
# 기준선이 정본 8.6603/8.0688 을 재현하는 것까지 확인). feature set 이 바뀐
# arm 이므로 fellow-eye 행과 같이 w 를 이 arm 에서 재적합한다(w=0.46).
XGB25_DIR = REPO / 'experiments/forest_robustness/xgb25'
x25, m25 = load_alt_xgb('25feat_no_vertcd', base=XGB25_DIR)
mask25 = MASK & m25
w25 = fit_w_arrays(x25, CNN['IR-v2'], LAB, mask25)
rows.append(row('No vertical C/D (25 params)', 'ablation',
                w25 * x25 + (1 - w25) * CNN['IR-v2'], x25, LAB, mask25, PID,
                w=w25, note='summary = XGB(25), fusion 도 XGB(25) 기반. '
                            'w 는 이 arm 에서 재적합 (w=0.47 이면 pooled 8.0832)'))

# --- floor removed (원본 -1 셀 제외) --------------------------------------
drop = neg1_drop_mask()
mask_excl = MASK & ~drop
checks['neg1_cells_dropped'] = int((MASK & drop).sum())
rows.append(row('Floor labels removed', 'mask',
                W_FIXED * XGB + (1 - W_FIXED) * CNN['IR-v2'], XGB, LAB,
                mask_excl, PID, w=W_FIXED, note='원본 -1 셀을 평가 마스크에서 제외'))

# --- reliability 필터 ------------------------------------------------------
keepA, keepB = reliability_sets()
for lab_, keep in (('FL < 20%', keepA), ('FL < 20% and FP < 15%', keepB)):
    rows.append(row(lab_, 'filter', W_FIXED * XGB + (1 - W_FIXED) * CNN['IR-v2'],
                    XGB, LAB, MASK, PID, keep=keep, w=W_FIXED,
                    note='안 단위 제외, 마스크는 그대로'))

# --- fellow-eye arm (46 파라미터) -----------------------------------------
x46, m46 = load_alt_xgb('90d_fellow')
mask46 = MASK & m46
w46 = fit_w_arrays(x46, CNN['IR-v2'], LAB, mask46)
rows.append(row('Fellow-eye features (46 params)', 'ablation',
                w46 * x46 + (1 - w46) * CNN['IR-v2'], x46, LAB, mask46, PID,
                w=w46, note='summary = XGB(46), fusion 도 XGB(46) 기반. w 는 이 arm 에서 재적합'))

# --- held-out (별도 구획) --------------------------------------------------
TX = np.mean([TST[k]['xgb'] for k in range(5)], 0)
TPID = np.array([str(k[0]) for k in TST[0]['keys']], dtype=object)
for nm, lab_ in (('IR-v2', 'Held-out: primary'), (ENS, 'Held-out: ensemble')):
    tc = np.mean([TST[k]['cnns'][nm] for k in range(5)], 0)
    w = W_GLOBAL[nm]
    rows.append(row(lab_, 'heldout', w * TX + (1 - w) * tc, TX, TEST_LAB,
                    TEST_MASK, TPID, w=w, note='w = 전체 OOF 적합, test 예측은 5fold 평균'))


# ------------------------------------------------------------------ 가드
def chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    checks[name] = dict(got=float(got), want=float(want), tol=tol, ok=bool(ok))
    return ok


by = {r['label']: r for r in rows}
gB = json.loads((TREE / 'runs/robustness_gains.json').read_text(encoding='utf-8'))
mB = json.loads((TREE / 'runs/multiseed_fusion.json').read_text(encoding='utf-8'))
rB = json.loads((TREE / 'runs/reliability_sensitivity.json').read_text(encoding='utf-8'))
nB = json.loads((TREE / 'runs/neg1_mask_sensitivity.json').read_text(encoding='utf-8'))
fB = json.loads((TREE / 'runs/fellow_eye_ablation.json').read_text(encoding='utf-8'))

chk('primary_pooled_fusion_vs_robustness_gains',
    by['Primary: Inception-ResNet-v2']['pooled_rmse_fusion'],
    gB['fusion_machinery']['global_w_dev']['rmse'], 5e-3)
chk('ensemble_nested_advantage_vs_robustness_gains',
    -ens_nested['delta'], gB['per_eye_advantage']['vs_xgb']['mean_advantage_db'], 5e-3)
chk('ensemble_nested_n_better_vs_robustness_gains',
    ens_nested['n_fusion_better'], gB['per_eye_advantage']['vs_xgb']['n_eyes_fusion_better'], 0)
chk('seed43_pooled_fusion_vs_multiseed', by['Seed 43']['pooled_rmse_fusion'],
    mB['seeds']['43']['oof']['fusion_rmse'], 5e-3)
chk('seed44_pooled_fusion_vs_multiseed', by['Seed 44']['pooled_rmse_fusion'],
    mB['seeds']['44']['oof']['fusion_rmse'], 5e-3)
_ir = rB['backbones']['inception_resnet_v2']['OOF']
chk('FL_n_eyes_vs_reliability', by['FL < 20%']['n_eyes'], _ir['crA_FL']['n_eyes'], 0)
chk('FLFP_n_eyes_vs_reliability', by['FL < 20% and FP < 15%']['n_eyes'],
    _ir['crB_FL_or_FP']['n_eyes'], 0)
chk('FL_pooled_fusion_vs_reliability', by['FL < 20%']['pooled_rmse_fusion'],
    _ir['crA_FL']['rmse']['fusion'], 5e-3)
chk('floor_cells_dropped_vs_neg1', checks['neg1_cells_dropped'],
    nB['splits']['oof']['cells_dropped'], 0)
chk('floor_pooled_fusion_vs_neg1', by['Floor labels removed']['pooled_rmse_fusion'],
    nB['splits']['oof']['fusion']['rmse_excl'], 5e-3)
chk('fellow_mean_eye_summary_vs_fellow_json',
    by['Fellow-eye features (46 params)']['mean_eye_rmse_summary'],
    fB['primary']['mean_a'], 5e-3)
aB = json.loads((TREE / 'runs/ablation_vert_cd.json').read_text(encoding='utf-8'))
_a25 = aB['25_feat_no_vertcd']['oof']
chk('xgb25_pooled_summary_vs_ablation_json',
    by['No vertical C/D (25 params)']['pooled_rmse_summary'], _a25['xgb_rmse'], 5e-4)
chk('xgb25_pooled_fusion_vs_ablation_json',
    by['No vertical C/D (25 params)']['pooled_rmse_fusion'],
    _a25['w_refit']['fusion_rmse'], 5e-4)
chk('xgb25_w_vs_ablation_json', by['No vertical C/D (25 params)']['w'],
    _a25['w_refit']['w_xgb'], 1e-9)
chk('xgb25_n_better_vs_ablation_json',
    by['No vertical C/D (25 params)']['n_fusion_better'],
    int(_a25['w_refit']['fus_vs_xgb_win'].split('/')[0]), 0)
chk('heldout_primary_pooled_fusion_vs_multiseed',
    by['Held-out: primary']['pooled_rmse_fusion'],
    mB['seeds']['42']['test']['fusion_rmse'], 5e-3)

payload = dict(
    note=('Figure 6 forest plot 행 값. 패스 B. 재학습 없음, 저장된 예측만 사용. '
          'delta = mean_eyes[RMSE(fusion) - RMSE(summary)] dB, 음수 = fusion 우세.'),
    tree=str(TREE), seed=SEED, n_boot=N_BOOT,
    ci='환자 군집 부트스트랩 95% percentile',
    w_global=W_GLOBAL, w_nested=W_NESTED,
    rows=rows, ensemble_global_w_alt=ens_global, blocked=[], checks=checks)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

print(f"{'행':<34}{'n':>5}{'delta':>9}{'CI lo':>9}{'CI hi':>9}{'p':>11}{'w':>7}")
print('-' * 84)
for r in rows:
    print(f"{r['label']:<34}{r['n_eyes']:>5}{r['delta']:>+9.3f}{r['ci_lo']:>+9.3f}"
          f"{r['ci_hi']:>+9.3f}{r['p_wilcoxon']:>11.2e}"
          f"{('-' if r['w'] is None else f'{r[chr(119)]:.2f}'):>7}")
print('-' * 84)
print(f"{ens_global['label']:<34}{ens_global['n_eyes']:>5}{ens_global['delta']:>+9.3f}"
      f"{ens_global['ci_lo']:>+9.3f}{ens_global['ci_hi']:>+9.3f}"
      f"{ens_global['p_wilcoxon']:>11.2e}{ens_global['w']:>7.2f}")
print('\n가드:')
for k, v in checks.items():
    if isinstance(v, dict):
        print(f"  {'OK ' if v['ok'] else '!! '}{k:<48} {v['got']:.4f} vs {v['want']:.4f}")
    else:
        print(f"     {k:<48} {v}")
print(f'\n저장: {OUT}')
