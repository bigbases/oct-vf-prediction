#!/usr/bin/env python3
"""fusion 비열등성(non-inferiority) 검정.

일관성 매트릭스에서 fusion 이 영상단독(CNN)에게 '유의하게 진' 칸은 0/20 이었다.
그러나 "지지 않았다"는 ns 만으로는 논증이 약하다(검정력 부족과 구분 안 됨).
여기서는 안 단위 paired bootstrap 으로 delta = fusion - CNN 의 95% CI 상한을 구해
"열등폭이 임상적으로 무시할 수준(margin) 이내"임을 양적으로 보인다.

margin 근거: VF 재검사 변동(test-retest) 수준. 보수적으로 0.5 dB 를 1차 margin 으로
쓰고, 0.25 dB 도 함께 보고한다.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

XGB = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGT = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
BB = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'Inception-v3': 'runs/phasec_b0_clip_mse_5fold',
    'VGG16': 'runs/phasec_b0_vgg16_5fold',
    'Xception': 'runs/phasec_b0_xception_5fold',
    'DenseNet121': 'runs/phasec_b0_densenet121_5fold',
}
NAMES = list(BB)
MARGINS = (0.25, 0.5)
NBOOT = 10000
SEED = 42


def load_fold(k, split):
    xz = load_oof_npz((XGB if split == 'val' else XGT)[k])
    fn = f'{split}_preds_fold{k}.npz'
    czs = {nm: load_oof_npz(ROOT / d / fn) for nm, d in BB.items()}
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
    return dict(keys=keys, xgb=xp, cnns=cnns, lab=lab, mask=(xm & cmask).astype(bool),
                pid=np.array([kk[0] for kk in keys], dtype=object))


OOF = {k: load_fold(k, 'val') for k in range(5)}
TST = {k: load_fold(k, 'test') for k in range(5)}
TEST_LAB = TST[0]['lab']
TEST_MASK = np.ones_like(TST[0]['mask']).astype(bool)
for k in range(5):
    TEST_MASK &= TST[k]['mask'].astype(bool)


def eye_metric(pred, lab, mask, metric):
    out = []
    for i in range(pred.shape[0]):
        d = (pred[i] - lab[i])[mask[i]]
        out.append(np.sqrt(np.mean(d ** 2)) if metric == 'rmse' else np.mean(np.abs(d)))
    return np.array(out)


def fit_w(folds, cnn_name):
    x = np.concatenate([f['xgb'][f['mask']] for f in folds])
    c = np.concatenate([f['cnns'][cnn_name][f['mask']] for f in folds])
    l = np.concatenate([f['lab'][f['mask']] for f in folds])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * x + (1 - w) * c - l) ** 2))
        if r < best[0]:
            best = (r, float(w))
    return best[1]


def cluster_boot_ci(delta, pid, nboot=NBOOT, seed=SEED):
    """환자 단위 클러스터 부트스트랩 (양안 상관 반영). delta = fus - other (음수=fusion 우세)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(pid)
    idx_by_pat = {p: np.flatnonzero(pid == p) for p in pats}
    stats = np.empty(nboot)
    for b in range(nboot):
        pick = rng.choice(pats, size=len(pats), replace=True)
        ii = np.concatenate([idx_by_pat[p] for p in pick])
        stats[b] = delta[ii].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


rows = []
for cnn_name in NAMES:
    # OOF (nested w)
    fus, cnn, labs, masks, pids = [], [], [], [], []
    for k in range(5):
        tr = [OOF[i] for i in range(5) if i != k]
        w = fit_w(tr, cnn_name)
        f = OOF[k]
        fus.append(w * f['xgb'] + (1 - w) * f['cnns'][cnn_name])
        cnn.append(f['cnns'][cnn_name])
        labs.append(f['lab']); masks.append(f['mask']); pids.append(f['pid'])
    pid_oof = np.concatenate(pids)
    for metric in ('rmse', 'mae'):
        fe = np.concatenate([eye_metric(fus[k], labs[k], masks[k], metric) for k in range(5)])
        ce = np.concatenate([eye_metric(cnn[k], labs[k], masks[k], metric) for k in range(5)])
        d, lo, hi = cluster_boot_ci(fe - ce, pid_oof)
        rows.append(dict(backbone=cnn_name, split='OOF', metric=metric, comparator='CNN',
                         delta=d, ci_lo=lo, ci_hi=hi,
                         noninf={f'{m}': bool(hi < m) for m in MARGINS}))
    # TEST
    w_all = fit_w([OOF[k] for k in range(5)], cnn_name)
    tx = np.mean([TST[k]['xgb'] for k in range(5)], 0)
    tc = np.mean([TST[k]['cnns'][cnn_name] for k in range(5)], 0)
    tf = w_all * tx + (1 - w_all) * tc
    pid_test = TST[0]['pid']
    for metric in ('rmse', 'mae'):
        fe = eye_metric(tf, TEST_LAB, TEST_MASK, metric)
        ce = eye_metric(tc, TEST_LAB, TEST_MASK, metric)
        d, lo, hi = cluster_boot_ci(fe - ce, pid_test)
        rows.append(dict(backbone=cnn_name, split='TEST', metric=metric, comparator='CNN',
                         delta=d, ci_lo=lo, ci_hi=hi,
                         noninf={f'{m}': bool(hi < m) for m in MARGINS}))

print("fusion - 영상단독(CNN) 차이 (dB, 음수 = fusion 우세), 환자 클러스터 부트스트랩 95% CI")
print(f"{'backbone':<14}{'set':<6}{'metric':<6}{'delta':>8}{'95% CI':>22}   비열등 (m=0.25 / 0.50)")
print('-' * 92)
for r in rows:
    ni = f"{'예' if r['noninf']['0.25'] else '아니오':<6}/ {'예' if r['noninf']['0.5'] else '아니오'}"
    print(f"{r['backbone']:<14}{r['split']:<6}{r['metric'].upper():<6}{r['delta']:>+8.3f}"
          f"   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   {ni}")
print('-' * 92)
for m in MARGINS:
    n = sum(r['noninf'][str(m)] for r in rows)
    print(f"  margin {m} dB 비열등 성립: {n}/{len(rows)} 칸")
print(f"  CI 상한이 0 미만(=우월): {sum(r['ci_hi'] < 0 for r in rows)}/{len(rows)} 칸")
print(f"  CI 하한이 0 초과(=열등): {sum(r['ci_lo'] > 0 for r in rows)}/{len(rows)} 칸")

out = ROOT / 'runs/fusion_noninferiority.json'
out.write_text(json.dumps(dict(
    note='delta = eye-level(fusion) - eye-level(CNN). 음수=fusion 우세. 환자 클러스터 부트스트랩 10000회, seed 42. w=RMSE 기준, OOF는 nested.',
    margins=list(MARGINS), n_boot=NBOOT, rows=rows), ensure_ascii=False, indent=2))
print(f"\n저장: {out}")
