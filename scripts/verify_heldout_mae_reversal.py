#!/usr/bin/env python
"""held-out에서 MAE만 역전되는 이유를 잔차 분위수로 확인한다.

원고 §Discussion "What the held-out set does not support" 문단의 근거.
5개 백본 전부에서 fusion이 하위 분위수에서 나빠지고 상위에서 좋아지며,
교차점이 p84-p86에 모인다는 것을 보인다.

    conda activate hvf
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    python scripts/verify_heldout_mae_reversal.py
"""
import json

import numpy as np

# w는 각 백본의 OOF 선택값(runs/skeleton_numbers.json과 동일).
# inception_v3의 예측은 clip_mse 디렉터리에 있다(디렉터리명이 백본명이 아님).
BACKBONES = {
    'inception_resnet_v2': ('runs/phasec_b0_inception_resnet_v2_5fold', 0.47),
    'inception_v3':        ('runs/phasec_b0_clip_mse_5fold',            0.60),
    'vgg16':               ('runs/phasec_b0_vgg16_5fold',               0.63),
    'xception':            ('runs/phasec_b0_xception_5fold',            0.43),
    'densenet121':         ('runs/phasec_b0_densenet121_5fold',         0.50),
}


def key(d):
    return [f'{p}|{e}|{v}' for p, e, v in zip(d['patient_id'], d['eye'], d['vf_date'])]


def load(path):
    return np.load(path, allow_pickle=True)


xgb_folds = [load(f'runs/oof/xgb_90d_fold{i}_test.npz') for i in range(5)]
kx = key(xgb_folds[0])

out = {}
print(f"{'backbone':22s} {'RMSE cnn->fus':>16s} {'MAE cnn->fus':>16s} {'cross':>6s}"
      f" {'p50Δ':>7s} {'p95Δ':>7s}")

for name, (cnn_dir, w) in BACKBONES.items():
    cnn_folds = [load(f'{cnn_dir}/test_preds_fold{i}.npz') for i in range(5)]
    kc = key(cnn_folds[0])

    # 두 branch의 held-out 안 집합이 1안 다르므로 교집합에서만 평가한다.
    common = [k for k in kc if k in set(kx)]
    ic = [kc.index(k) for k in common]
    ix = [kx.index(k) for k in common]

    cnn = np.mean([c['pred'] for c in cnn_folds], 0)[ic]
    xgb = np.mean([x['pred'] for x in xgb_folds], 0)[ix]
    y = cnn_folds[0]['labels'][ic]
    m = cnn_folds[0]['mask'][ic] & xgb_folds[0]['mask'][ix]
    fus = w * xgb + (1 - w) * cnn

    rmse = lambda p: float(np.sqrt(((p - y)[m] ** 2).mean()))
    mae = lambda p: float(np.abs((p - y)[m]).mean())

    ec = np.abs(cnn - y)[m]
    ef = np.abs(fus - y)[m]
    q = lambda a, p: float(np.percentile(a, p))
    cross = next((p for p in range(50, 100) if q(ef, p) < q(ec, p)), None)

    out[name] = {
        'n_eyes': len(common), 'w_xgb': w,
        'cnn': {'rmse': round(rmse(cnn), 3), 'mae': round(mae(cnn), 3)},
        'fusion': {'rmse': round(rmse(fus), 3), 'mae': round(mae(fus), 3)},
        'crossing_percentile': cross,
        'quantile_delta': {f'p{p}': round(q(ef, p) - q(ec, p), 3)
                           for p in (10, 25, 50, 75, 90, 95, 99)},
    }
    print(f'{name:22s} {rmse(cnn):7.3f}->{rmse(fus):<8.3f} {mae(cnn):7.3f}->{mae(fus):<8.3f}'
          f' {("p%d" % cross) if cross else "none":>6s}'
          f' {q(ef, 50) - q(ec, 50):+7.3f} {q(ef, 95) - q(ec, 95):+7.3f}')

crossings = [v['crossing_percentile'] for v in out.values()]
print(f'\n교차점 범위: p{min(crossings)}-p{max(crossings)} (5/5 백본)')
print('MAE 역전:', sum(v['fusion']['mae'] > v['cnn']['mae'] for v in out.values()), '/ 5')
print('RMSE 유지:', sum(v['fusion']['rmse'] < v['cnn']['rmse'] for v in out.values()), '/ 5')

with open('runs/heldout_mae_reversal.json', 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print('→ runs/heldout_mae_reversal.json')
