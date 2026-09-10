#!/usr/bin/env python3
"""(B) 2·3항: 집계 축 괴리의 정량화와 MAE 축 동일성 검증.

1항(선행연구 축 조사)은 이 스크립트의 범위가 아니다 — 사용자가 직접 채운다.

안구당 지점 수가 52 로 고정이면 pooled RMSE = sqrt(mu^2 + sigma^2) 가 항등식이다
(mu = 안구별 RMSE 평균, sigma = 그 모집단 SD). 그래서 축 괴리는 오직 sigma 가
만든다. 여기서는 우리 코호트의 괴리를 내고, sigma 가 선행 보고값이었다면
괴리가 얼마였을지를 같은 mu 위에서 계산한다.

읽기 전용. 저장된 예측만 쓰고 재학습하지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402

CNN_DIR = 'runs/phasec_b0_inception_resnet_v2_5fold'   # 주 모델의 image branch
W_PRIMARY = 0.47
PRIOR_SD = [2.09, 2.68]      # 02_related_work.tex:79 / 05_discussion.tex:169


def npz_cnn(path):
    z = np.load(path, allow_pickle=True)
    return {'keys': list(zip(z['patient_id'], z['eye'], z['vf_date'])),
            'pred': z['pred'], 'labels': z['labels'], 'mask': z['mask'], 'meta': {}}


def collect(tree):
    xs, cs, ls, ms = [], [], [], []
    for k in range(5):
        xz = load_oof_npz(tree / f'runs/oof/xgb_90d_fold{k}_val.npz')
        cz = npz_cnn(tree / CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, _ = align_oof(xz, cz)
        xs.append(ax['pred']); cs.append(bx['pred'])
        ls.append(ax['labels']); ms.append(ax['mask'] & bx['mask'])
    return (np.concatenate(xs), np.concatenate(cs),
            np.concatenate(ls), np.concatenate(ms))


def axis_stats(pred, labels, mask):
    m = mask.astype(bool)
    d = pred - labels
    per_n, per_rmse, per_mae = [], [], []
    for i in range(pred.shape[0]):
        mi = m[i]
        if mi.sum() == 0:
            continue
        di = d[i][mi]
        per_n.append(int(mi.sum()))
        per_rmse.append(float(np.sqrt(np.mean(di ** 2))))
        per_mae.append(float(np.mean(np.abs(di))))
    per_rmse = np.array(per_rmse); per_mae = np.array(per_mae)
    pooled_rmse = float(np.sqrt(np.mean(d[m] ** 2)))
    pooled_mae = float(np.mean(np.abs(d[m])))
    mu = float(per_rmse.mean())
    sd_pop = float(per_rmse.std(ddof=0))
    return {
        'n_eyes': len(per_rmse),
        'points_per_eye_unique': sorted(set(per_n)),
        'pooled_rmse': pooled_rmse,
        'eye_rmse_mean': mu,
        'eye_rmse_sd_pop': sd_pop,
        'eye_rmse_sd_sample': float(per_rmse.std(ddof=1)),
        'identity_sqrt_mu2_plus_sd2': float(np.sqrt(mu ** 2 + sd_pop ** 2)),
        'identity_abs_err': abs(pooled_rmse - float(np.sqrt(mu ** 2 + sd_pop ** 2))),
        'gap_pooled_minus_eyemean': pooled_rmse - mu,
        'pooled_mae': pooled_mae,
        'eye_mae_mean': float(per_mae.mean()),
        'mae_abs_err': abs(pooled_mae - float(per_mae.mean())),
        # 같은 mu 위에서 sigma 만 선행 보고값으로 바꿔 본다. 코호트 이질성이
        # 축 괴리를 얼마나 키우는지가 요점이다.
        'counterfactual_gap': {
            str(s): float(np.sqrt(mu ** 2 + s ** 2) - mu) for s in PRIOR_SD
        },
    }


def main():
    out = {'note': 'OOF only. mask = XGB & CNN 교집합. '
                   'image/fusion 은 주 모델(IR-v2, w_xgb=0.47).',
           'prior_sd_reference': PRIOR_SD, 'passes': {}}
    for pk, tree in (('A', ROOT / 'experiments/laterality_qfix/step4_work/A'),
                     ('B', ROOT / 'experiments/laterality_qfix/step4_work/B')):
        X, C, L, M = collect(tree)
        F = W_PRIMARY * X + (1 - W_PRIMARY) * C
        out['passes'][pk] = {
            'summary_xgb': axis_stats(X, L, M),
            'image_cnn': axis_stats(C, L, M),
            'fusion': axis_stats(F, L, M),
        }
        print(f'=== 패스 {pk} ===')
        for name, s in out['passes'][pk].items():
            print(f'  {name:<12} pooled={s["pooled_rmse"]:.4f} mu={s["eye_rmse_mean"]:.4f} '
                  f'sd={s["eye_rmse_sd_pop"]:.4f} 괴리={s["gap_pooled_minus_eyemean"]:.4f} '
                  f'| 항등오차={s["identity_abs_err"]:.2e} '
                  f'| MAE pooled={s["pooled_mae"]:.6f} eye={s["eye_mae_mean"]:.6f} '
                  f'차={s["mae_abs_err"]:.2e}')
            cf = s['counterfactual_gap']
            print(f'{"":<16}반사실 괴리: SD=2.09 -> {cf["2.09"]:.4f} | SD=2.68 -> {cf["2.68"]:.4f}')
    p = ROOT / 'experiments/aggregation_axis/axis_gap.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {p}')


if __name__ == '__main__':
    main()
