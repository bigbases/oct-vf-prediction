#!/usr/bin/env python3
"""(3) 6번째 CNN 백본 — 5->6 앙상블 이득 대 XGB 추가 이득.

§5 는 "gradient boosting 을 더해도 앙상블이 나아지지 않았으니 fusion 은 다양성이
아니다"라고 논증하지만, 5->6 의 앙상블 이득 포화를 통제하지 않는다. 같은 설정으로
학습한 6번째 CNN 을 더했을 때의 이득이 유일하게 공정한 기준선이다.

기존 5백본과 동일 설정으로 학습한 ResNet50 (동일 폴드/seed 42/동일 head/증강 없음/
동일 optimizer) 을 읽어 6백본 앙상블을 만든다. 학습은 train_r3.py 가 했고 산출물은
experiments/reviewer_round3/runs_r3/ 에 있다 — 정본 runs/ 밖이다.

기준선 두 개를 모두 낸다:
  basis A = XGB ∩ 기존 5백본  (원고와 같은 분모. 가드 8.115 는 여기서 확인한다)
  basis B = XGB ∩ 6백본       (ResNet50 을 넣으면 키/마스크 교집합이 줄 수 있으므로,
                               5백본과 6백본을 like-for-like 로 비교하려면 여기여야 한다)

부호 규약: delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.
읽기 전용(예측 npz 만 읽는다). 원고와 정본 runs/ 를 건드리지 않는다.
출력: experiments/reviewer_round3/sixth_backbone.{json,md}
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'
R50 = REPO / 'experiments/reviewer_round3/runs_r3/phasec_b0_resnet50_5fold'
OUT_JSON = REPO / 'experiments/reviewer_round3/sixth_backbone.json'
N_BOOT, SEED = 5000, 42
GUARD_TOL = 5e-3
GUARDS = {'oof_ensemble5_pooled_rmse_basisA': 8.115,
          'oof_ensemble5_fusion_pooled_rmse_basisA': 7.963,
          'oof_summary_pooled_rmse_basisA': 8.6603}


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
sys.path.insert(0, str(TREE / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

NAMES5 = list(FEC.NAMES)
NEW = 'ResNet50'
NAMES6 = NAMES5 + [NEW]


# --------------------------------------------------------- 폴드 로딩 (6백본판)
def load_fold6(k: int, split: str) -> dict:
    """FEC.load_fold 와 같은 절차에 ResNet50 을 한 항목 더 넣은 것."""
    xz = load_oof_npz(FEC.xgb_path(k, split))
    fn = f'{split}_preds_fold{k}.npz'
    czs = {nm: load_oof_npz(TREE / d / fn) for nm, d in FEC.BB.items()}
    czs[NEW] = load_oof_npz(R50 / fn)
    keys = set(xz['keys'])
    for cz in czs.values():
        keys &= set(cz['keys'])
    keys = sorted(keys)

    def pick(z):
        idx = {kk: i for i, kk in enumerate(z['keys'])}
        ii = [idx[kk] for kk in keys]
        return z['pred'][ii], z['labels'][ii], z['mask'][ii]

    xp, lab, xm = pick(xz)
    cnns, cmask = {}, None
    for nm, cz in czs.items():
        cp, _, cm = pick(cz)
        cnns[nm] = cp
        cmask = cm if cmask is None else (cmask & cm)
    return dict(keys=keys, xgb=xp, cnns=cnns, lab=lab, mask=(xm & cmask).astype(bool))


def add_ens(d, names, tag):
    d['cnns'][tag] = np.mean(np.stack([d['cnns'][n] for n in names]), 0)
    return d


E5, E6 = 'ENS5', 'ENS6'
A = {k: add_ens(FEC.load_fold(k, 'val'), NAMES5, E5) for k in range(5)}
B = {k: add_ens(add_ens(load_fold6(k, 'val'), NAMES5, E5), NAMES6, E6) for k in range(5)}


# ------------------------------------------------------------------ 통계 도구
def cat(folds, f):
    return np.concatenate([f(folds[k]) for k in range(5)])


def pooled(pred, lab, mask):
    return float(np.sqrt(np.mean((pred - lab)[mask] ** 2)))


def pooled_mae(pred, lab, mask):
    return float(np.mean(np.abs(pred - lab)[mask]))


def boot_ci(v, pid, rng):
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    sums = np.array([v[g].sum() for g in groups], float)
    cnts = np.array([g.size for g in groups], float)
    idx = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def nested_fusion(folds, cnn_tag):
    """폴드 k 의 w 는 나머지 4폴드에서 적합. FEC.oof_raw 와 같은 규약."""
    out, ws = [], []
    for k in range(5):
        tr = [folds[i] for i in range(5) if i != k]
        w = FEC.fit_w(tr, cnn_tag)
        ws.append(w)
        f = folds[k]
        out.append(w * f['xgb'] + (1 - w) * f['cnns'][cnn_tag])
    return np.concatenate(out), ws


def block(label, pred, lab, mask):
    ei = FEC.eye_metric(pred, lab, mask, 'rmse')
    em = FEC.eye_metric(pred, lab, mask, 'mae')
    return dict(label=label, pooled_rmse=pooled(pred, lab, mask),
                pooled_mae=pooled_mae(pred, lab, mask),
                eye_rmse_mean=float(ei.mean()), eye_rmse_sd_pop=float(ei.std(ddof=0)),
                eye_mae_mean=float(em.mean()), eye_mae_sd_pop=float(em.std(ddof=0)),
                n_eyes=int(pred.shape[0]), n_cells=int(mask.sum()))


def compare(a_lab, b_lab, a, b, lab, mask, pid, kind='rmse'):
    ea = FEC.eye_metric(a, lab, mask, kind)
    eb = FEC.eye_metric(b, lab, mask, kind)
    ok = np.isfinite(ea) & np.isfinite(eb)
    d = ea[ok] - eb[ok]
    lo, hi = boot_ci(d, pid[ok], np.random.default_rng(SEED))
    return dict(a=a_lab, b=b_lab, metric=kind, n_eyes=int(ok.sum()),
                delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
                median_delta=float(np.median(d)),
                p_wilcoxon=float(wilcoxon(ea[ok], eb[ok]).pvalue),
                n_a_better=int((d < 0).sum()))


# --------------------------------------------------------------------- basis A
LA = cat(A, lambda f: f['lab'])
MA = cat(A, lambda f: f['mask']).astype(bool)
E5A = cat(A, lambda f: f['cnns'][E5])
FUSA, WA = nested_fusion(A, E5)
XA = cat(A, lambda f: f['xgb'])

guards = {}
for name, want in GUARDS.items():
    got = {'oof_ensemble5_pooled_rmse_basisA': pooled(E5A, LA, MA),
           'oof_ensemble5_fusion_pooled_rmse_basisA': pooled(FUSA, LA, MA),
           'oof_summary_pooled_rmse_basisA': pooled(XA, LA, MA)}[name]
    ok = abs(got - want) <= GUARD_TOL
    guards[name] = dict(got=round(got, 4), want=want, tol=GUARD_TOL, ok=bool(ok))

if not all(v['ok'] for v in guards.values()):
    OUT_JSON.write_text(json.dumps(
        {'status': 'GUARD_FAILED', 'guards': guards,
         'note': '가드 불일치. 결과를 내지 않는다.'}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('GUARD FAILED'); print(json.dumps(guards, ensure_ascii=False, indent=1))
    sys.exit(1)
print('가드 통과.')

# --------------------------------------------------------------------- basis B
LB = cat(B, lambda f: f['lab'])
MB = cat(B, lambda f: f['mask']).astype(bool)
PIDB = np.array([str(kk[0]) for k in range(5) for kk in B[k]['keys']], dtype=object)
E5B = cat(B, lambda f: f['cnns'][E5])
E6B = cat(B, lambda f: f['cnns'][E6])
R50B = cat(B, lambda f: f['cnns'][NEW])
XB = cat(B, lambda f: f['xgb'])
FUS5B, W5B = nested_fusion(B, E5)
FUS6B, W6B = nested_fusion(B, E6)

keys_a = [kk for k in range(5) for kk in A[k]['keys']]
keys_b = [kk for k in range(5) for kk in B[k]['keys']]
basis_shift = dict(
    basis_a_eyes=len(keys_a), basis_b_eyes=len(keys_b),
    basis_a_cells=int(MA.sum()), basis_b_cells=int(MB.sum()),
    eyes_dropped=len(set(map(tuple, keys_a)) - set(map(tuple, keys_b))),
    note='basis B 는 ResNet50 의 키/마스크까지 교집합에 넣은 것이다.')

rows_b = [block(f'{NEW} 단독', R50B, LB, MB),
          block('5백본 앙상블', E5B, LB, MB),
          block('6백본 앙상블 (+ResNet50)', E6B, LB, MB),
          block('5백본 앙상블 + XGB fusion', FUS5B, LB, MB),
          block('6백본 앙상블 + XGB fusion', FUS6B, LB, MB),
          block('summary XGB 단독', XB, LB, MB)]
rows_b += [block(nm, cat(B, lambda f, n=nm: f['cnns'][n]), LB, MB) for nm in NAMES5]

cmps = []
for kind in ('rmse', 'mae'):
    cmps.append(compare('6백본 앙상블', '5백본 앙상블', E6B, E5B, LB, MB, PIDB, kind))
    cmps.append(compare('5백본+XGB fusion', '5백본 앙상블', FUS5B, E5B, LB, MB, PIDB, kind))
    cmps.append(compare('6백본+XGB fusion', '6백본 앙상블', FUS6B, E6B, LB, MB, PIDB, kind))
    cmps.append(compare('6백본 앙상블', '5백본+XGB fusion', E6B, FUS5B, LB, MB, PIDB, kind))

g5 = pooled(E5B, LB, MB)
gains = dict(
    basis='B',
    ens5_pooled_rmse=g5,
    gain_add_6th_cnn=pooled(E6B, LB, MB) - g5,
    gain_add_xgb=pooled(FUS5B, LB, MB) - g5,
    gain_add_6th_then_xgb=pooled(FUS6B, LB, MB) - g5,
    note='음수 = 개선. 두 이득을 같은 basis B, 같은 출발점(5백본 앙상블)에서 잰다.')

payload = dict(
    note='6번째 CNN(ResNet50) 추가 이득 대 XGB 추가 이득. 패스 B, out-of-fold. '
         'delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.',
    tree=str(TREE), resnet50_run=str(R50), seed=SEED, n_boot=N_BOOT,
    backbones5=NAMES5, backbones6=NAMES6,
    guards=guards, basis_shift=basis_shift,
    nested_w_ens5_basisB=W5B, nested_w_ens6_basisB=W6B,
    rows_basisB=rows_b, comparisons_basisB=cmps, gains=gains,
    resnet50_training=dict(
        launcher='experiments/reviewer_round3/train_r3.py',
        note='train.py 를 수정하지 않고 build_image_backbone 만 런타임에 확장했다. '
             '설정은 정본 5백본과 동일(300 epoch, RMSprop, lr 1e-4, wd 1e-5, '
             'patience 100, early_stop mae, seed 42, 증강 없음, use_deviation False).'),
)
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
print('->', OUT_JSON)
for r in rows_b:
    print(f"  {r['label']:<28} pooled RMSE {r['pooled_rmse']:.4f}  MAE {r['pooled_mae']:.4f}  "
          f"per-eye {r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f}")
print()
for c in cmps:
    print(f"  {c['metric']:>4} {c['a']:<18} vs {c['b']:<20} d={c['delta']:+.4f} "
          f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p_wilcoxon']:.3g} "
          f"{c['n_a_better']}/{c['n_eyes']}")
print()
print(f"  5백본 {gains['ens5_pooled_rmse']:.4f} | +6번째 CNN {gains['gain_add_6th_cnn']:+.4f} | "
      f"+XGB {gains['gain_add_xgb']:+.4f}")
