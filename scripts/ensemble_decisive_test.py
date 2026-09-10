#!/usr/bin/env python3
"""Table 5: the decisive test - does adding the summary branch improve the
five-backbone ensemble? That is, does the quantitative branch still carry
information once the image model is strong.

결정적 검정: (5백본 앙상블 CNN) + XGB fusion 이 앙상블 CNN 단독을 이기는가?
= "강한 image 모델에도 정량지표가 정보를 더하는가" (정량→영상 방향의 진짜 테스트).
robustness_gains의 7.964 / p=0.46 을 신선하게 재확인. 재학습 없음(저장 npz만).
출력: runs/ensemble_decisive.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

BACKBONES = ['phasec_b0_inception_resnet_v2_5fold', 'phasec_b0_vgg16_5fold',
             'phasec_b0_xception_5fold', 'phasec_b0_densenet121_5fold',
             'phasec_b0_clip_mse_5fold']
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}

def norm_keys(d):
    return [tuple(str(x) for x in kk) for kk in d['keys']]
def idxmap(d):
    return {k: i for i, k in enumerate(norm_keys(d))}
def pick(d, im, common, field):
    return np.stack([d[field][im[c]] for c in common])

def gather(bbs, xz, common):
    imx = idxmap(xz)
    bb_pred = []
    mk = pick(xz, imx, common, 'mask')
    for b in bbs:
        im = idxmap(b)
        bb_pred.append(pick(b, im, common, 'pred'))
        mk = mk & pick(b, im, common, 'mask')
    ens = np.mean(np.stack(bb_pred), 0)
    xp = pick(xz, imx, common, 'pred')
    lb = pick(xz, imx, common, 'labels')
    return xp, ens, lb, mk

def common_keys(dicts):
    sets = [set(norm_keys(d)) for d in dicts]
    return sorted(set.intersection(*sets))

def load_oof():
    XS, ES, LS, MS = [], [], [], []
    for k in range(5):
        bbs = [load_oof_npz(ROOT / f'runs/{d}/val_preds_fold{k}.npz') for d in BACKBONES]
        xz = load_oof_npz(XGB_VAL[k])
        common = common_keys(bbs + [xz])
        xp, ens, lb, mk = gather(bbs, xz, common)
        XS.append(xp); ES.append(ens); LS.append(lb); MS.append(mk)
    return (np.concatenate(XS), np.concatenate(ES), np.concatenate(LS), np.concatenate(MS))

def load_test():
    # 각 백본: 5 fold test 평균 → bb_test ; 앙상블 = bb_test 평균. XGB = fold 평균.
    bb_tests, xgb_folds, keyref = [], [], None
    # common keys = 교집합(모든 백본 fold0 + xgb fold0) — test는 fold별 동일 eyes
    d0 = [load_oof_npz(ROOT / f'runs/{d}/test_preds_fold0.npz') for d in BACKBONES]
    x0 = load_oof_npz(XGB_TEST[0])
    common = common_keys(d0 + [x0])
    for d in BACKBONES:
        folds = [load_oof_npz(ROOT / f'runs/{d}/test_preds_fold{k}.npz') for k in range(5)]
        stk = np.mean(np.stack([pick(f, idxmap(f), common, 'pred') for f in folds]), 0)
        bb_tests.append(stk)
    ens = np.mean(np.stack(bb_tests), 0)
    xf = [load_oof_npz(XGB_TEST[k]) for k in range(5)]
    xp = np.mean(np.stack([pick(f, idxmap(f), common, 'pred') for f in xf]), 0)
    lb = pick(x0, idxmap(x0), common, 'labels')
    mk = pick(x0, idxmap(x0), common, 'mask')
    for d in d0:
        mk = mk & pick(d, idxmap(d), common, 'mask')
    return xp, ens, lb, mk

def pooled(pred, labels, mask):
    m = mask.astype(bool); d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))
def per_eye(pred, labels, mask):
    return np.array([np.sqrt(np.mean((pred[i][mask[i].astype(bool)] - labels[i][mask[i].astype(bool)]) ** 2))
                     if mask[i].any() else np.nan for i in range(pred.shape[0])])
def best_w(xp, cp, lb, mk):
    m = mk.astype(bool); yx, yc, y = xp[m], cp[m], lb[m]
    bw, br = 0.0, 1e9
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * yx + (1 - w) * yc - y) ** 2))
        if r < br: br, bw = r, float(w)
    return bw
def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    try: _, wp = stats.wilcoxon(a, b)
    except ValueError: wp = float('nan')
    return float(wp), int((a < b).sum()), len(a)

def block(xp, ens, lb, mk, name):
    er, em = pooled(ens, lb, mk); xr, xm = pooled(xp, lb, mk)
    out = {'ensemble_cnn_rmse': er, 'ensemble_cnn_mae': em, 'xgb_rmse': xr, 'xgb_mae': xm}
    print(f'\n=== {name} ===')
    print(f'  앙상블-CNN {er:.3f}/{em:.3f} | XGB {xr:.3f}/{xm:.3f}')
    for wl, w in [('w0.47', 0.47), ('w_refit', best_w(xp, ens, lb, mk))]:
        F = w * xp + (1 - w) * ens
        fr, fm = pooled(F, lb, mk)
        ef, ec, ex = per_eye(F, lb, mk), per_eye(ens, lb, mk), per_eye(xp, lb, mk)
        pc, wc, n = paired(ef, ec); px, wx, _ = paired(ef, ex)
        out[wl] = {'w_xgb': round(w, 3), 'fusion_rmse': fr, 'fusion_mae': fm,
                   'fus_vs_ensCNN_p': pc, 'fus_vs_ensCNN_win': f'{wc}/{n}',
                   'fus_vs_XGB_p': px, 'fus_vs_XGB_win': f'{wx}/{n}'}
        print(f'  [{wl} w={w:.2f}] fusion {fr:.3f}/{fm:.3f} '
              f'| fus>앙상블CNN Δ p={pc:.4g} ({wc}/{n}) '
              f'| fus>XGB p={px:.4g} ({wx}/{n})')
    return out

def main():
    res = {'note': '앙상블=5백본 CNN 평균. 결정적: fus>앙상블CNN(=정량이 강한 영상에 더하나)'}
    xp, ens, lb, mk = load_oof()
    res['OOF'] = block(xp, ens, lb, mk, f'OOF (n_eyes={xp.shape[0]})')
    xpt, enst, lbt, mkt = load_test()
    res['TEST'] = block(xpt, enst, lbt, mkt, f'TEST (n_eyes={xpt.shape[0]})')
    (ROOT / 'runs/ensemble_decisive.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print('\n저장: runs/ensemble_decisive.json')

if __name__ == '__main__':
    main()
