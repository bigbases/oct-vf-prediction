#!/usr/bin/env python3
"""재학습 없는 성능 여지 탐색: 이미 학습된 5백본 CNN 앙상블 + fusion.

- 단일 백본 image-only / fusion(OOF-tuned w) 대비
- 5백본 평균 앙상블 image-only / fusion
OOF(240) + test(37). pooled RMSE/MAE.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402

XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
BB = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'Inception-v3': 'runs/phasec_b0_clip_mse_5fold',
    'VGG16': 'runs/phasec_b0_vgg16_5fold',
    'Xception': 'runs/phasec_b0_xception_5fold',
    'DenseNet121': 'runs/phasec_b0_densenet121_5fold',
}


def gather(xgb_paths, cnn_dir, split):
    xs, cs, ls, ms = [], [], [], []
    for k in range(5):
        xz = load_oof_npz(xgb_paths[k])
        fn = f'{"val" if split == "oof" else "test"}_preds_fold{k}.npz'
        cz = load_oof_npz(ROOT / cnn_dir / fn)
        ax, bx, _ = align_oof(xz, cz)
        xs.append(ax['pred']); cs.append(bx['pred'])
        ls.append(ax['labels']); ms.append(ax['mask'] & bx['mask'])
    if split == 'oof':
        return (np.concatenate(xs), np.concatenate(cs), np.concatenate(ls), np.concatenate(ms))
    X = np.mean(np.stack(xs), 0); C = np.mean(np.stack(cs), 0); L = ls[0]; M = ms[0]
    for i in range(1, 5):
        M = M & ms[i]
    return X, C, L, M


def pooled(p, l, m):
    m = m.astype(bool); d = (p - l)[m]
    return round(float(np.sqrt(np.mean(d ** 2))), 2), round(float(np.mean(np.abs(d))), 2)


def opt_w(x, c, l, m):
    m = m.astype(bool); ya, yb, y = x[m], c[m], l[m]; best = (1e9, .5)
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (r, w)
    return round(float(best[1]), 2)


def main():
    # 단일 백본
    print(f'{"backbone":14s} {"img OOF":>9s} {"fus OOF":>9s} {"img TST":>9s} {"fus TST":>9s}', flush=True)
    print('-' * 56, flush=True)
    cnn_oof, cnn_test, Lo, Mo, Lt, Mt, Xo, Xt = {}, {}, None, None, None, None, None, None
    for name, d in BB.items():
        Xo, Co, Lo, Mo = gather(XGB_VAL, d, 'oof')
        Xt, Ct, Lt, Mt = gather(XGB_TEST, d, 'test')
        cnn_oof[name] = Co; cnn_test[name] = Ct
        w = opt_w(Xo, Co, Lo, Mo)
        rio = pooled(Co, Lo, Mo)[0]; rfo = pooled(w * Xo + (1 - w) * Co, Lo, Mo)[0]
        rit = pooled(Ct, Lt, Mt)[0]; rft = pooled(w * Xt + (1 - w) * Ct, Lt, Mt)[0]
        print(f'{name:14s} {rio:9.2f} {rfo:9.2f} {rit:9.2f} {rft:9.2f}', flush=True)

    # 앙상블 (5백본 평균) — OOF/test 키가 백본 간 동일하다고 가정(같은 fold/split)
    Ceo = np.mean(np.stack(list(cnn_oof.values())), 0)
    Cet = np.mean(np.stack(list(cnn_test.values())), 0)
    w = opt_w(Xo, Ceo, Lo, Mo)
    print('-' * 56, flush=True)
    io = pooled(Ceo, Lo, Mo); fo = pooled(w * Xo + (1 - w) * Ceo, Lo, Mo)
    it = pooled(Cet, Lt, Mt); ft = pooled(w * Xt + (1 - w) * Cet, Lt, Mt)
    print(f'{"ENSEMBLE(5)":14s} {io[0]:9.2f} {fo[0]:9.2f} {it[0]:9.2f} {ft[0]:9.2f}   (w={w})', flush=True)
    print(f'\n  ENSEMBLE image-only  OOF {io[0]}/{io[1]}  test {it[0]}/{it[1]}', flush=True)
    print(f'  ENSEMBLE late fusion OOF {fo[0]}/{fo[1]}  test {ft[0]}/{ft[1]}', flush=True)
    print(f'  (참고 단일 IR-v2 fusion: OOF 8.06/5.89  test 8.40/6.29)', flush=True)


if __name__ == '__main__':
    main()
