#!/usr/bin/env python3
"""fusion 일관성 매트릭스: 5백본 x {OOF,TEST} x {RMSE,MAE} = 20칸에서
fusion 이 각 단일모달 대비 (a) 유의하게 좋음 / (b) 비유의 / (c) 유의하게 나쁨 을 센다.

목적: "fusion 이 일관되게 좋다"를 만들어내는 대신, 실제로 성립하는 형태
      = "정량 대비 지배적 + 영상 대비 어디서도 유의하게 나쁘지 않다(비열등)"
      를 20칸 전수로 검증한다.

읽기 전용. w 는 nested(OOF) / 전체 OOF 적합(test) — 낙관 제거.
"""
from __future__ import annotations
import json
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
ALPHA = 0.05


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
    return dict(keys=keys, xgb=xp, cnns=cnns, lab=lab, mask=(xm & cmask).astype(bool))


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


def fit_w(folds, cnn_name, obj):
    x = np.concatenate([f['xgb'][f['mask']] for f in folds])
    c = np.concatenate([f['cnns'][cnn_name][f['mask']] for f in folds])
    l = np.concatenate([f['lab'][f['mask']] for f in folds])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = w * x + (1 - w) * c - l
        v = np.sqrt(np.mean(r ** 2)) if obj == 'rmse' else np.mean(np.abs(r))
        if v < best[0]:
            best = (v, float(w))
    return best[1]


def verdict(fus_eye, other_eye):
    """fus 가 작을수록 좋음. 양측 Wilcoxon + 방향."""
    p = float(wilcoxon(fus_eye, other_eye).pvalue)
    better = float(np.mean(fus_eye) < np.mean(other_eye))
    if p < ALPHA:
        return ('WIN' if better else 'LOSS'), p
    return 'ns', p


results = []
for cnn_name in NAMES:
    for obj in ('rmse',):  # w 는 정본대로 RMSE 기준으로 적합
        # ---- OOF (nested w) ----
        fus_p, xgb_p, cnn_p, labs, masks = [], [], [], [], []
        for k in range(5):
            tr = [OOF[i] for i in range(5) if i != k]
            w = fit_w(tr, cnn_name, obj)
            f = OOF[k]
            fus_p.append(w * f['xgb'] + (1 - w) * f['cnns'][cnn_name])
            xgb_p.append(f['xgb'])
            cnn_p.append(f['cnns'][cnn_name])
            labs.append(f['lab'])
            masks.append(f['mask'])
        for metric in ('rmse', 'mae'):
            fe = np.concatenate([eye_metric(fus_p[k], labs[k], masks[k], metric) for k in range(5)])
            xe = np.concatenate([eye_metric(xgb_p[k], labs[k], masks[k], metric) for k in range(5)])
            ce = np.concatenate([eye_metric(cnn_p[k], labs[k], masks[k], metric) for k in range(5)])
            vx, px = verdict(fe, xe)
            vc, pc = verdict(fe, ce)
            results.append(dict(backbone=cnn_name, split='OOF', metric=metric, n=len(fe),
                                fus=float(np.mean(fe)), xgb=float(np.mean(xe)), cnn=float(np.mean(ce)),
                                vs_xgb=vx, p_xgb=px, vs_cnn=vc, p_cnn=pc))
        # ---- TEST (w = 전체 OOF 적합) ----
        w_all = fit_w([OOF[k] for k in range(5)], cnn_name, obj)
        tx = np.mean([TST[k]['xgb'] for k in range(5)], 0)
        tc = np.mean([TST[k]['cnns'][cnn_name] for k in range(5)], 0)
        tf = w_all * tx + (1 - w_all) * tc
        for metric in ('rmse', 'mae'):
            fe = eye_metric(tf, TEST_LAB, TEST_MASK, metric)
            xe = eye_metric(tx, TEST_LAB, TEST_MASK, metric)
            ce = eye_metric(tc, TEST_LAB, TEST_MASK, metric)
            vx, px = verdict(fe, xe)
            vc, pc = verdict(fe, ce)
            results.append(dict(backbone=cnn_name, split='TEST', metric=metric, n=len(fe),
                                fus=float(np.mean(fe)), xgb=float(np.mean(xe)), cnn=float(np.mean(ce)),
                                vs_xgb=vx, p_xgb=px, vs_cnn=vc, p_cnn=pc))

print(f"{'backbone':<14}{'set':<6}{'metric':<6}{'fus':>6}{'xgb':>6}{'cnn':>6}  {'vs XGB':<22}{'vs CNN'}")
print('-' * 96)
for r in results:
    print(f"{r['backbone']:<14}{r['split']:<6}{r['metric'].upper():<6}"
          f"{r['fus']:>6.2f}{r['xgb']:>6.2f}{r['cnn']:>6.2f}  "
          f"{r['vs_xgb']:<5}(p={r['p_xgb']:.1e})     {r['vs_cnn']:<5}(p={r['p_cnn']:.1e})")

print('-' * 96)


def tally(key):
    from collections import Counter
    return dict(Counter(r[key] for r in results))


print("20칸(5백본 x {OOF,TEST} x {RMSE,MAE}) 판정 집계 [안 단위 paired Wilcoxon, alpha=.05]")
print("  fusion vs 정량단독(XGB):", tally('vs_xgb'))
print("  fusion vs 영상단독(CNN):", tally('vs_cnn'))
print()
print("  point-estimate 방향(유의성 무관):")
print("    fusion < XGB :", sum(r['fus'] < r['xgb'] for r in results), "/", len(results))
print("    fusion < CNN :", sum(r['fus'] < r['cnn'] for r in results), "/", len(results))

out = ROOT / 'runs/fusion_consistency_matrix.json'
out.write_text(json.dumps(dict(
    note='5백본 x {OOF,TEST} x {RMSE,MAE} = 20칸. w=RMSE기준, OOF는 nested, test는 전체 OOF 적합. 안 단위 paired Wilcoxon 양측.',
    alpha=ALPHA, rows=results,
    tally_vs_xgb=tally('vs_xgb'), tally_vs_cnn=tally('vs_cnn'),
), ensure_ascii=False, indent=2))
print(f"\n저장: {out}")
