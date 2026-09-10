#!/usr/bin/env python3
"""정직한 성능·견고성 탐색.

모든 튜닝 방법은 nested CV(다른 fold에서 파라미터 적합 → held fold 예측)로 평가한다.
=> "OOF에서 튜닝하고 OOF에서 보고"하는 낙관을 제거한 honest OOF 수치.

비교 config:
  0) XGB 단독            (baseline, 튜닝 없음)
  1) IR-v2 CNN 단독       (baseline)
  2) 5백본 평균 CNN 단독  (baseline)
  3) fusion(IR-v2), global-w  nested
  4) fusion(5백본앙상블), global-w  nested
  5) fusion(IR-v2), per-point w  nested
  6) fusion(5백본앙상블), per-point w  nested
  7) stacking NNLS [XGB, IR-v2]  per-point  nested
  8) stacking NNLS [XGB, 5CNN]   per-point  nested

test(37)도 함께: 파라미터는 전체 OOF로 적합 후 test 평가(방향 확인용).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz, align_oof  # noqa: E402

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
    """held fold k의 XGB와 5백본 CNN을 공통 키로 정렬. returns dict."""
    xz = load_oof_npz((XGB if split == 'val' else XGT)[k])
    fn = f'{split}_preds_fold{k}.npz'
    czs = {nm: load_oof_npz(ROOT / d / fn) for nm, d in BB.items()}
    # 공통 키 = XGB ∩ 모든 CNN
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
    return dict(keys=keys, xgb=xp, cnns=cnns, ens=ens, ir=cnns['IR-v2'], lab=lab, mask=mask)


OOF = {k: load_fold(k, 'val') for k in range(5)}
TST = {k: load_fold(k, 'test') for k in range(5)}


def pooled(preds, labs, masks):
    d = np.concatenate([(p - l)[m.astype(bool)] for p, l, m in zip(preds, labs, masks)])
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def gw(fl, xkey, ckey):
    """global scalar w on pooled points (min RMSE)."""
    x = np.concatenate([f[xkey][f['mask'].astype(bool)] for f in fl])
    c = np.concatenate([f[ckey][f['mask'].astype(bool)] for f in fl])
    l = np.concatenate([f['lab'][f['mask'].astype(bool)] for f in fl])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * x + (1 - w) * c - l) ** 2))
        if r < best[0]:
            best = (r, w)
    return best[1]


def ppw(fl, xkey, ckey, shrink=0.0, w_global=0.5):
    """per-point w (52,), optional shrink toward global."""
    P = fl[0][xkey].shape[1]
    w = np.full(P, 0.5)
    Xall = np.concatenate([f[xkey] for f in fl]); Call = np.concatenate([f[ckey] for f in fl])
    Lall = np.concatenate([f['lab'] for f in fl]); Mall = np.concatenate([f['mask'] for f in fl]).astype(bool)
    for p in range(P):
        m = Mall[:, p]
        if m.sum() < 5:
            w[p] = w_global; continue
        xp, cp, lp = Xall[m, p], Call[m, p], Lall[m, p]
        best = (1e9, 0.5)
        for ww in np.linspace(0, 1, 101):
            r = np.mean((ww * xp + (1 - ww) * cp - lp) ** 2)
            if r < best[0]:
                best = (r, ww)
        w[p] = (1 - shrink) * best[1] + shrink * w_global
    return w


def stack_nnls(fl, cols_key):
    """per-point NNLS weights over columns [xgb, *cnns]. returns (P, ncol)."""
    cols = cols_key
    P = fl[0]['xgb'].shape[1]
    ncol = len(cols)
    W = np.zeros((P, ncol))
    def getcol(f, name):
        return f['xgb'] if name == 'xgb' else (f['ens'] if name == 'ens' else f['cnns'][name])
    stacks = {name: np.concatenate([getcol(f, name) for f in fl]) for name in cols}
    Lall = np.concatenate([f['lab'] for f in fl]); Mall = np.concatenate([f['mask'] for f in fl]).astype(bool)
    for p in range(P):
        m = Mall[:, p]
        if m.sum() < ncol + 2:
            W[p] = np.array([1.0] + [0.0] * (ncol - 1)); continue
        A = np.stack([stacks[name][m, p] for name in cols], axis=1)
        b = Lall[m, p]
        coef, _ = nnls(A, b)
        s = coef.sum()
        W[p] = coef / s if s > 1e-8 else np.array([1.0] + [0.0] * (ncol - 1))
    return W


def apply_w(f, xkey, ckey, w):
    return w * f[xkey] + (1 - w) * f[ckey]


def apply_stack(f, cols, W):
    def getcol(name):
        return f['xgb'] if name == 'xgb' else (f['ens'] if name == 'ens' else f['cnns'][name])
    out = np.zeros_like(f['xgb'])
    for j, name in enumerate(cols):
        out += W[:, j][None, :] * getcol(name)
    return out


def nested_oof(kind, **kw):
    """held fold k 예측을 다른 4 fold에서 적합한 파라미터로 생성 → pooled."""
    preds, labs, masks = [], [], []
    for k in range(5):
        tr = [OOF[i] for i in range(5) if i != k]
        f = OOF[k]
        if kind == 'gw':
            w = gw(tr, kw['xkey'], kw['ckey']); p = apply_w(f, kw['xkey'], kw['ckey'], w)
        elif kind == 'ppw':
            wg = gw(tr, kw['xkey'], kw['ckey'])
            w = ppw(tr, kw['xkey'], kw['ckey'], kw.get('shrink', 0.0), wg); p = apply_w(f, kw['xkey'], kw['ckey'], w)
        elif kind == 'stack':
            W = stack_nnls(tr, kw['cols']); p = apply_stack(f, kw['cols'], W)
        preds.append(p); labs.append(f['lab']); masks.append(f['mask'])
    return pooled(preds, labs, masks)


def test_eval(kind, **kw):
    """전체 OOF로 파라미터 적합 → test(각 fold model, 5-fold 예측 평균) 평가."""
    tr = [OOF[i] for i in range(5)]
    # test는 fold별 예측을 평균(앙상블) — dossier §4.2 test 정의와 동일
    x4 = np.mean([TST[k]['xgb'] for k in range(5)], 0)
    lab = TST[0]['lab']; mask = np.ones_like(TST[0]['mask']).astype(bool)
    for k in range(5):
        mask &= TST[k]['mask'].astype(bool)
    def _pick(z, name):
        if name == 'xgb':
            return z['xgb']
        if name == 'ens':
            return z['ens']
        if name == 'ir':
            return z['ir']
        return z['cnns'][name]

    def col_test(name):
        return np.mean([_pick(TST[k], name) for k in range(5)], 0)
    if kind == 'gw':
        w = gw(tr, kw['xkey'], kw['ckey']); p = w * col_test(kw['xkey']) + (1 - w) * col_test(kw['ckey'])
    elif kind == 'ppw':
        wg = gw(tr, kw['xkey'], kw['ckey']); w = ppw(tr, kw['xkey'], kw['ckey'], kw.get('shrink', 0.0), wg)
        p = w * col_test(kw['xkey']) + (1 - w) * col_test(kw['ckey'])
    elif kind == 'stack':
        W = stack_nnls(tr, kw['cols']); p = np.zeros_like(col_test('xgb'))
        for j, name in enumerate(kw['cols']):
            p += W[:, j][None, :] * col_test(name)
    elif kind == 'solo':
        p = col_test(kw['col'])
    d = (p - lab)[mask]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def solo_oof(col):
    return pooled([OOF[k][col] for k in range(5)], [OOF[k]['lab'] for k in range(5)], [OOF[k]['mask'] for k in range(5)])


print(f"{'config':42s} {'OOF(honest) RMSE/MAE':>22s}   {'TEST RMSE/MAE':>16s}")
print('-' * 86)
rows = [
    ('0 XGB 단독',                    solo_oof('xgb'),                      test_eval('solo', col='xgb')),
    ('1 IR-v2 CNN 단독',              solo_oof('ir'),                       test_eval('solo', col='ir')),
    ('2 5백본 평균 CNN 단독',          solo_oof('ens'),                      test_eval('solo', col='ens')),
    ('3 fusion IR-v2, global-w',      nested_oof('gw', xkey='xgb', ckey='ir'),  test_eval('gw', xkey='xgb', ckey='ir')),
    ('4 fusion 5백본앙상블, global-w', nested_oof('gw', xkey='xgb', ckey='ens'), test_eval('gw', xkey='xgb', ckey='ens')),
    ('5 fusion IR-v2, per-point w',   nested_oof('ppw', xkey='xgb', ckey='ir'), test_eval('ppw', xkey='xgb', ckey='ir')),
    ('6 fusion 앙상블, per-point w',   nested_oof('ppw', xkey='xgb', ckey='ens'),test_eval('ppw', xkey='xgb', ckey='ens')),
    ('7 stack NNLS [XGB,IR-v2] pp',   nested_oof('stack', cols=['xgb', 'IR-v2']), test_eval('stack', cols=['xgb', 'IR-v2'])),
    ('8 stack NNLS [XGB,5CNN] pp',    nested_oof('stack', cols=['xgb'] + NAMES),  test_eval('stack', cols=['xgb'] + NAMES)),
]
for name, oof, tst in rows:
    print(f"{name:42s}   {oof[0]:5.3f}/{oof[1]:5.3f}          {tst[0]:5.2f}/{tst[1]:5.2f}")
print('-' * 86)
print("n_oof_eyes =", sum(int(OOF[k]['mask'].shape[0]) for k in range(5)),
      " n_test_eyes =", TST[0]['mask'].shape[0])
