#!/usr/bin/env python3
"""공헌 프레이밍 근거: 백본별 late-fusion이 XGB·image-only CNN을 이기는지 + paired 유의성.

image-only 모드(우리의 fusion 레시피)만. OOF(n=240, 검정력 있음)와 test(n=37) 각각:
  - pooled RMSE: XGB / CNN / fusion(w는 OOF에서 튜닝)
  - eye 단위 paired: fusion vs CNN, fusion vs XGB (Wilcoxon + paired t)
출력: runs/fusion_contribution_per_backbone.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402

XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
IMG_DIR = {
    'inception_v3': 'runs/phasec_b0_clip_mse_5fold',
    'inception_resnet_v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'vgg16': 'runs/phasec_b0_vgg16_5fold',
    'xception': 'runs/phasec_b0_xception_5fold',
    'densenet121': 'runs/phasec_b0_densenet121_5fold',
}


def npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {'keys': list(zip(z['patient_id'], z['eye'], z['vf_date'])),
            'pred': z['pred'], 'labels': z['labels'], 'mask': z['mask'], 'meta': {}}


def pooled(pred, labels, mask):
    m = mask.astype(bool)
    d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2)))


def per_eye(pred, labels, mask):
    out = []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        if m.sum() == 0:
            continue
        d = pred[i][m] - labels[i][m]
        out.append(np.sqrt(np.mean(d ** 2)))
    return np.array(out)


def opt_w(px, pc, labels, mask, grid=101):
    m = mask.astype(bool)
    ya, yb, y = px[m], pc[m], labels[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, grid):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (float(r), float(w))
    return best[1]


def paired(fus, solo):
    try:
        _, wp = stats.wilcoxon(fus, solo)
    except ValueError:
        wp = float('nan')
    _, tp = stats.ttest_rel(fus, solo)
    return {'fusion_mean_eye_rmse': round(float(fus.mean()), 3),
            'solo_mean_eye_rmse': round(float(solo.mean()), 3),
            'delta': round(float((fus - solo).mean()), 3),
            'fusion_wins': f'{int(np.sum(fus < solo))}/{len(fus)}',
            'wilcoxon_p': round(float(wp), 4), 'paired_t_p': round(float(tp), 4)}


def collect(xgb_paths, cnn_dir, split):
    xs, cs, ls, ms = [], [], [], []
    fold_iter = range(5)
    per_fold = []
    for k in fold_iter:
        xz = load_oof_npz(xgb_paths[k])
        fn = f'{"val" if split=="oof" else "test"}_preds_fold{k}.npz'
        cz = npz_dict(ROOT / cnn_dir / fn)
        ax, bx, _ = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        xs.append(ax['pred']); cs.append(bx['pred']); ls.append(ax['labels']); ms.append(m)
        per_fold.append((ax['pred'], bx['pred'], ax['labels'], m))
    return xs, cs, ls, ms, per_fold


def main():
    result = {}
    for bb, cnn_dir in IMG_DIR.items():
        # OOF: concat (each eye once)
        xs, cs, ls, ms, _ = collect(XGB_VAL, cnn_dir, 'oof')
        XV, CV, LV, MV = (np.concatenate(a) for a in (xs, cs, ls, ms))
        w = opt_w(XV, CV, LV, MV)
        FV = w * XV + (1 - w) * CV
        oof = {
            'w_xgb': round(w, 2),
            'pooled_rmse': {'xgb': round(pooled(XV, LV, MV), 3),
                            'cnn': round(pooled(CV, LV, MV), 3),
                            'fusion': round(pooled(FV, LV, MV), 3)},
            'fusion_vs_cnn': paired(per_eye(FV, LV, MV), per_eye(CV, LV, MV)),
            'fusion_vs_xgb': paired(per_eye(FV, LV, MV), per_eye(XV, LV, MV)),
        }
        # TEST: fold 예측 평균(앙상블) → eye 1회
        xs, cs, ls, ms, pf = collect(XGB_TEST, cnn_dir, 'test')
        XT = np.mean(np.stack(xs), 0); CT = np.mean(np.stack(cs), 0)
        LT = ls[0]; MT = ms[0]
        for i in range(1, 5):  # 안전: 마스크 교집합
            MT = MT & ms[i]
        FT = w * XT + (1 - w) * CT
        test = {
            'pooled_rmse': {'xgb': round(pooled(XT, LT, MT), 3),
                            'cnn': round(pooled(CT, LT, MT), 3),
                            'fusion': round(pooled(FT, LT, MT), 3)},
            'fusion_vs_cnn': paired(per_eye(FT, LT, MT), per_eye(CT, LT, MT)),
            'fusion_vs_xgb': paired(per_eye(FT, LT, MT), per_eye(XT, LT, MT)),
        }
        result[bb] = {'oof': oof, 'test': test}
        print(f'\n=== {bb} (w_xgb={w:.2f}) ===', flush=True)
        print(f'  OOF  pooled  XGB {oof["pooled_rmse"]["xgb"]} | CNN {oof["pooled_rmse"]["cnn"]} | FUS {oof["pooled_rmse"]["fusion"]}', flush=True)
        print(f'       fus<CNN? {oof["fusion_vs_cnn"]["fusion_mean_eye_rmse"]}<{oof["fusion_vs_cnn"]["solo_mean_eye_rmse"]} p={oof["fusion_vs_cnn"]["wilcoxon_p"]} | fus<XGB? p={oof["fusion_vs_xgb"]["wilcoxon_p"]}', flush=True)
        print(f'  TEST pooled  XGB {test["pooled_rmse"]["xgb"]} | CNN {test["pooled_rmse"]["cnn"]} | FUS {test["pooled_rmse"]["fusion"]}', flush=True)
        print(f'       fus vs CNN p={test["fusion_vs_cnn"]["wilcoxon_p"]} | fus vs XGB p={test["fusion_vs_xgb"]["wilcoxon_p"]}', flush=True)

    out = ROOT / 'runs/fusion_contribution_per_backbone.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {out}', flush=True)


if __name__ == '__main__':
    main()
