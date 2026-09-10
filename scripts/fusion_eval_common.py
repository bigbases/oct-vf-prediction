"""Shared loader for the fusion evaluations. It reproduces the procedure of
fusion_consistency_matrix.py so that new analysis scripts cannot drift from it,
and self-checks its eye-level deltas against runs/fusion_noninferiority.json.
Note the run-directory naming trap documented below: runs/phasec_b0_clip_mse_5fold
holds an Inception-v3 run, not a CLIP one.

fusion 평가 공용 로더 — fusion_consistency_matrix.py 와 동일한 절차를 재사용한다.

기존 스크립트(fusion_consistency_matrix.py, fusion_noninferiority.py)는 재현성을 위해
건드리지 않는다. 신규 분석 스크립트만 이 모듈을 쓴다.

정합성 요구: 이 모듈이 만드는 안 단위 delta 는 runs/fusion_noninferiority.json 의
delta 와 일치해야 한다. 신규 스크립트는 그 대조를 self-check 로 수행한다.

주의 — 런 디렉터리 이름 오독 금지:
  runs/phasec_b0_clip_mse_5fold 는 **Inception-v3** 런이다. 근거는 체크포인트
  가중치다: best_fold0.pt 의 image_branch 최상위 모듈이 Conv2d_1a_3x3 및
  Mixed_5b~Mixed_7c (572 tensors, 24.6M params) 로 torchvision Inception-v3 의
  구조이며 CLIP 흔적(visual/transformer/attn/ln_/token)은 하나도 없다.
  대조: IR-v2 는 1312 tensors/56.6M, VGG16 은 38 tensors/139.1M 이다.
  'clip_mse' 라는 이름의 유래는 기록이 없다. 저장된 args 에 clip·loss 관련 키가
  없고 early_stop_metric 은 'mae' 이므로, 이름의 뜻을 문서에 단정하지 말 것.
  확정된 것은 백본이 Inception-v3 라는 사실 하나뿐이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

XGB_TAG = '90d'


def xgb_path(k: int, split: str, tag: str = XGB_TAG) -> Path:
    return ROOT / f'runs/oof/xgb_{tag}_fold{k}_{"val" if split == "val" else "test"}.npz'

BB = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'Inception-v3': 'runs/phasec_b0_clip_mse_5fold',
    'VGG16': 'runs/phasec_b0_vgg16_5fold',
    'Xception': 'runs/phasec_b0_xception_5fold',
    'DenseNet121': 'runs/phasec_b0_densenet121_5fold',
}
NAMES = list(BB)


def load_fold(k: int, split: str, xgb_tag: str = XGB_TAG) -> dict:
    xz = load_oof_npz(xgb_path(k, split, xgb_tag))
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


ENSEMBLE = 'ENSEMBLE'


def add_ensemble(d: dict) -> dict:
    """5백본 평균을 의사 백본으로 추가한다. 논문 §fusion ceiling 의 앙상블과 동일."""
    d['cnns'][ENSEMBLE] = np.mean(np.stack([d['cnns'][n] for n in NAMES]), 0)
    return d


def load_all(xgb_tag: str = XGB_TAG, ensemble: bool = False):
    """(OOF, TST, TEST_LAB, TEST_MASK) — consistency matrix 와 동일한 구성."""
    oof = {k: load_fold(k, 'val', xgb_tag) for k in range(5)}
    tst = {k: load_fold(k, 'test', xgb_tag) for k in range(5)}
    if ensemble:
        for k in range(5):
            add_ensemble(oof[k])
            add_ensemble(tst[k])
    test_lab = tst[0]['lab']
    test_mask = np.ones_like(tst[0]['mask']).astype(bool)
    for k in range(5):
        test_mask &= tst[k]['mask'].astype(bool)
    return oof, tst, test_lab, test_mask


def eye_metric(pred, lab, mask, metric: str) -> np.ndarray:
    out = []
    for i in range(pred.shape[0]):
        d = (pred[i] - lab[i])[mask[i]]
        out.append(np.sqrt(np.mean(d ** 2)) if metric == 'rmse' else np.mean(np.abs(d)))
    return np.array(out)


def fit_w(folds, cnn_name: str, obj: str = 'rmse') -> float:
    x = np.concatenate([f['xgb'][f['mask']] for f in folds])
    c = np.concatenate([f['cnns'][cnn_name][f['mask']] for f in folds])
    lab = np.concatenate([f['lab'][f['mask']] for f in folds])
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, 101):
        r = w * x + (1 - w) * c - lab
        v = np.sqrt(np.mean(r ** 2)) if obj == 'rmse' else np.mean(np.abs(r))
        if v < best[0]:
            best = (v, float(w))
    return best[1]


def oof_arrays(oof, cnn_name: str, metric: str):
    """nested w 로 만든 OOF 안 단위 (fusion, xgb, cnn, patient_id) 배열."""
    fus, xgb, cnn, pid = [], [], [], []
    for k in range(5):
        tr = [oof[i] for i in range(5) if i != k]
        w = fit_w(tr, cnn_name)
        f = oof[k]
        fp = w * f['xgb'] + (1 - w) * f['cnns'][cnn_name]
        fus.append(eye_metric(fp, f['lab'], f['mask'], metric))
        xgb.append(eye_metric(f['xgb'], f['lab'], f['mask'], metric))
        cnn.append(eye_metric(f['cnns'][cnn_name], f['lab'], f['mask'], metric))
        pid += [str(kk[0]) for kk in f['keys']]
    return (np.concatenate(fus), np.concatenate(xgb), np.concatenate(cnn),
            np.array(pid, dtype=object))


def test_arrays(oof, tst, test_lab, test_mask, cnn_name: str, metric: str):
    """w = 전체 OOF 적합, test 예측은 5fold 평균."""
    w = fit_w([oof[k] for k in range(5)], cnn_name)
    tx = np.mean([tst[k]['xgb'] for k in range(5)], 0)
    tc = np.mean([tst[k]['cnns'][cnn_name] for k in range(5)], 0)
    tf = w * tx + (1 - w) * tc
    pid = np.array([str(kk[0]) for kk in tst[0]['keys']], dtype=object)
    return (eye_metric(tf, test_lab, test_mask, metric),
            eye_metric(tx, test_lab, test_mask, metric),
            eye_metric(tc, test_lab, test_mask, metric),
            pid)


def test_raw(oof, tst, cnn_name: str):
    """test 의 안 x 지점 예측 원본 (구간별 층화용). (fusion, xgb, cnn, w)."""
    w = fit_w([oof[k] for k in range(5)], cnn_name)
    tx = np.mean([tst[k]['xgb'] for k in range(5)], 0)
    tc = np.mean([tst[k]['cnns'][cnn_name] for k in range(5)], 0)
    return w * tx + (1 - w) * tc, tx, tc, w


def oof_raw(oof, cnn_name: str):
    """OOF 의 안 x 지점 예측 원본을 fold 순서대로 이어붙인다 (구간별 층화용)."""
    fus, xgb, cnn, lab, mask, pid = [], [], [], [], [], []
    for k in range(5):
        tr = [oof[i] for i in range(5) if i != k]
        w = fit_w(tr, cnn_name)
        f = oof[k]
        fus.append(w * f['xgb'] + (1 - w) * f['cnns'][cnn_name])
        xgb.append(f['xgb'])
        cnn.append(f['cnns'][cnn_name])
        lab.append(f['lab'])
        mask.append(f['mask'])
        pid += [str(kk[0]) for kk in f['keys']]
    return (np.concatenate(fus), np.concatenate(xgb), np.concatenate(cnn),
            np.concatenate(lab), np.concatenate(mask),
            np.array(pid, dtype=object))
