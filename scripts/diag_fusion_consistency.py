#!/usr/bin/env python3
"""진단: fusion이 두 단일모달을 '일관되게' 이기지 못하는 지점이 어디이고,
그 원인이 방법론적 비대칭(=고칠 수 있음)인지 실제 정보 한계(=못 고침)인지 분리한다.

읽기 전용(npz만 사용, 재학습 없음). nested-w 로 낙관 제거.

축1) w 목적함수: RMSE로 튜닝한 w를 MAE로 평가 -> MAE 패배가 목적함수 불일치 때문인지.
축2) 브랜치 용량 대칭: 5백본 앙상블 CNN vs 단일 XGB (비대칭) -> XGB도 seed 앙상블로 맞추면?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

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
    mask = (xm & cmask).astype(bool)
    ens = np.mean([cnns[nm] for nm in NAMES], axis=0)
    return dict(keys=keys, xgb=xp, cnns=cnns, ens=ens, ir=cnns['IR-v2'],
                lab=lab, mask=mask)


OOF = {k: load_fold(k, 'val') for k in range(5)}
TST = {k: load_fold(k, 'test') for k in range(5)}


def pooled(preds, labs, masks):
    d = np.concatenate([(p - l)[m.astype(bool)] for p, l, m in zip(preds, labs, masks)])
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def fit_w(fl, xkey, ckey, obj):
    x = np.concatenate([f[xkey][f['mask']] for f in fl])
    c = np.concatenate([f[ckey][f['mask']] for f in fl])
    l = np.concatenate([f['lab'][f['mask']] for f in fl])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = w * x + (1 - w) * c - l
        v = np.sqrt(np.mean(r ** 2)) if obj == 'rmse' else np.mean(np.abs(r))
        if v < best[0]:
            best = (v, float(w))
    return best[1]


def eye_metric(pred, lab, mask, metric):
    """안(eye) 단위 RMSE/MAE 벡터."""
    out = []
    for i in range(pred.shape[0]):
        m = mask[i]
        d = (pred[i] - lab[i])[m]
        out.append(np.sqrt(np.mean(d ** 2)) if metric == 'rmse' else np.mean(np.abs(d)))
    return np.array(out)


def nested_fusion(xkey, ckey, obj):
    """held fold k 를 나머지 4fold에서 적합한 w로 융합."""
    preds, labs, masks, ws = [], [], [], []
    for k in range(5):
        tr = [OOF[i] for i in range(5) if i != k]
        w = fit_w(tr, xkey, ckey, obj)
        ws.append(w)
        f = OOF[k]
        preds.append(w * f[xkey] + (1 - w) * f[ckey])
        labs.append(f['lab'])
        masks.append(f['mask'])
    return preds, labs, masks, ws


def test_cols(name):
    return np.mean([TST[k][name] for k in range(5)], 0)


TEST_LAB = TST[0]['lab']
TEST_MASK = np.ones_like(TST[0]['mask']).astype(bool)
for k in range(5):
    TEST_MASK &= TST[k]['mask'].astype(bool)


def paired_p(a, b):
    """안 단위 paired Wilcoxon (a=fusion, b=comparator). a<b 이면 fusion 우세."""
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float('nan')


def report(tag, xkey, ckey, obj):
    preds, labs, masks, ws = nested_fusion(xkey, ckey, obj)
    fus_oof = pooled(preds, labs, masks)
    x_oof = pooled([OOF[k][xkey] for k in range(5)], [OOF[k]['lab'] for k in range(5)],
                   [OOF[k]['mask'] for k in range(5)])
    c_oof = pooled([OOF[k][ckey] for k in range(5)], [OOF[k]['lab'] for k in range(5)],
                   [OOF[k]['mask'] for k in range(5)])
    # 안 단위 검정 (OOF)
    rows = {}
    for mi, metric in enumerate(('rmse', 'mae')):
        fe = np.concatenate([eye_metric(preds[k], labs[k], masks[k], metric) for k in range(5)])
        xe = np.concatenate([eye_metric(OOF[k][xkey], OOF[k]['lab'], OOF[k]['mask'], metric) for k in range(5)])
        ce = np.concatenate([eye_metric(OOF[k][ckey], OOF[k]['lab'], OOF[k]['mask'], metric) for k in range(5)])
        rows[metric] = (paired_p(fe, xe), paired_p(fe, ce), (fe < xe).sum(), (fe < ce).sum(), len(fe))
    # test
    w_all = fit_w([OOF[k] for k in range(5)], xkey, ckey, obj)
    tp = w_all * test_cols(xkey) + (1 - w_all) * test_cols(ckey)
    tx, tc = test_cols(xkey), test_cols(ckey)
    def tm(p, metric):
        d = (p - TEST_LAB)[TEST_MASK]
        return float(np.sqrt(np.mean(d ** 2))) if metric == 'rmse' else float(np.mean(np.abs(d)))
    trows = {}
    for metric in ('rmse', 'mae'):
        fe = eye_metric(tp, TEST_LAB, TEST_MASK, metric)
        xe = eye_metric(tx, TEST_LAB, TEST_MASK, metric)
        ce = eye_metric(tc, TEST_LAB, TEST_MASK, metric)
        trows[metric] = (paired_p(fe, xe), paired_p(fe, ce), (fe < xe).sum(), (fe < ce).sum(), len(fe))

    print(f"\n### {tag}   (w 목적함수 = {obj.upper()}, nested w={np.round(ws,2).tolist()}, test w={w_all:.2f})")
    print(f"  OOF  pooled  XGB {x_oof[0]:.3f}/{x_oof[1]:.3f} | CNN {c_oof[0]:.3f}/{c_oof[1]:.3f} | FUS {fus_oof[0]:.3f}/{fus_oof[1]:.3f}   (RMSE/MAE)")
    print(f"  TEST pooled  XGB {tm(tx,'rmse'):.3f}/{tm(tx,'mae'):.3f} | CNN {tm(tc,'rmse'):.3f}/{tm(tc,'mae'):.3f} | FUS {tm(tp,'rmse'):.3f}/{tm(tp,'mae'):.3f}")
    for setname, rr in (('OOF', rows), ('TEST', trows)):
        for metric in ('rmse', 'mae'):
            pxg, pcn, wx, wc, n = rr[metric]
            print(f"   {setname:<4} {metric.upper():<4} fus>XGB p={pxg:.2e} ({wx}/{n})   fus>CNN p={pcn:.2e} ({wc}/{n})")


print("=" * 96)
print("축1) w 목적함수 불일치 진단 — 같은 예측, w만 RMSE/MAE 기준으로 각각 적합")
print("=" * 96)
for obj in ('rmse', 'mae'):
    report('단일 IR-v2 fusion', 'xgb', 'ir', obj)
print()
print("=" * 96)
print("축2) 브랜치 용량 비대칭 — 5백본 앙상블 CNN vs 단일 XGB")
print("=" * 96)
for obj in ('rmse', 'mae'):
    report('앙상블 CNN fusion', 'xgb', 'ens', obj)
