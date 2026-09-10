#!/usr/bin/env python3
"""§4 Robustness 의 산문 수치를 산출물로 승격한다.

배경 (2026-08-26). 아래 네 값은 원고에만 있고 `runs/` 어디에도 없었다.
근거가 사람이 쓴 중간 문서(`robustness_gains_*.md`, `professor_memo_*.md`)뿐이라,
그 경로에서 이중 반올림·모델 혼동 오류가 세 건 나왔다. 이 스크립트는 원고가
산문 대신 JSON 을 근거로 갖게 만든다.

담는 값:
  fusion_machinery — fusion 천장의 대안 가중치 (04_results.tex:158-166)
    8.064  global-w, 평가 fold 포함 적합 (development estimate)
    8.097  global-w, nested (평가 fold 제외 적합)
    8.065  w 를 0.5 로 고정
    8.117  field 위치별 w (per-point), nested
    8.116  per-point NNLS 스택 [XGB, IR-v2], nested
  per_eye_advantage — 5백본 앙상블 fusion vs XGB 단독 (04_results.tex:265-269)
    0.861  안 단위 RMSE 평균차, nested-w
    p = 6.8e-13, 240 안 중 171 안에서 fusion 우세

기존 스크립트는 건드리지 않는다. `explore_fusion_gains.py` 는 표를 print 만 하고
w=0.5 config 도 Wilcoxon 도 없어서 어차피 새 코드가 필요했다.
정렬·마스크 규약은 `explore_fusion_gains.py` 와 동일하다:
공통 키 = XGB ∩ 5백본 전부, 마스크 = XGB 마스크 & 5백본 마스크 교집합 (240안 12,480점).

실행: python scripts/export_robustness_gains.py  ->  runs/robustness_gains.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

BB = {
    'IR-v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    # 주의: 아래 디렉터리 이름은 clip_mse 지만 실제 체크포인트는 Inception-v3 다.
    'Inception-v3': 'runs/phasec_b0_clip_mse_5fold',
    'VGG16': 'runs/phasec_b0_vgg16_5fold',
    'Xception': 'runs/phasec_b0_xception_5fold',
    'DenseNet121': 'runs/phasec_b0_densenet121_5fold',
}
NAMES = list(BB)
W_GRID = np.linspace(0, 1, 101)


def load_fold(k: int) -> dict:
    xz = load_oof_npz(ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz')
    czs = {nm: load_oof_npz(ROOT / d / f'val_preds_fold{k}.npz') for nm, d in BB.items()}
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
    return dict(
        xgb=xp, cnns=cnns, ir=cnns['IR-v2'],
        ens=np.mean([cnns[nm] for nm in NAMES], axis=0),
        lab=lab, mask=(xm & cmask).astype(bool),
        pid=np.array([str(kk[0]) for kk in keys], dtype=object),
    )


OOF = {k: load_fold(k) for k in range(5)}


def pooled_rmse(preds) -> float:
    d = np.concatenate([(p - OOF[k]['lab'])[OOF[k]['mask']] for k, p in preds.items()])
    return float(np.sqrt(np.mean(d ** 2)))


def fit_gw(folds, ckey: str) -> float:
    x = np.concatenate([f['xgb'][f['mask']] for f in folds])
    c = np.concatenate([f[ckey][f['mask']] for f in folds])
    l = np.concatenate([f['lab'][f['mask']] for f in folds])
    err = [np.mean((w * x + (1 - w) * c - l) ** 2) for w in W_GRID]
    return float(W_GRID[int(np.argmin(err))])


def fit_ppw(folds, ckey: str, w_global: float) -> np.ndarray:
    X = np.concatenate([f['xgb'] for f in folds])
    C = np.concatenate([f[ckey] for f in folds])
    L = np.concatenate([f['lab'] for f in folds])
    M = np.concatenate([f['mask'] for f in folds])
    w = np.full(X.shape[1], w_global)
    for p in range(X.shape[1]):
        m = M[:, p]
        if m.sum() < 5:
            continue
        xp, cp, lp = X[m, p], C[m, p], L[m, p]
        err = [np.mean((ww * xp + (1 - ww) * cp - lp) ** 2) for ww in W_GRID]
        w[p] = W_GRID[int(np.argmin(err))]
    return w


def fit_nnls(folds, cols) -> np.ndarray:
    def col(f, nm):
        return f['xgb'] if nm == 'xgb' else (f['ens'] if nm == 'ens' else f['cnns'][nm])
    S = {nm: np.concatenate([col(f, nm) for f in folds]) for nm in cols}
    L = np.concatenate([f['lab'] for f in folds])
    M = np.concatenate([f['mask'] for f in folds])
    P, ncol = L.shape[1], len(cols)
    W = np.zeros((P, ncol))
    fallback = np.array([1.0] + [0.0] * (ncol - 1))
    for p in range(P):
        m = M[:, p]
        if m.sum() < ncol + 2:
            W[p] = fallback
            continue
        A = np.stack([S[nm][m, p] for nm in cols], axis=1)
        coef, _ = nnls(A, L[m, p])
        s = coef.sum()
        W[p] = coef / s if s > 1e-8 else fallback
    return W


def apply_nnls(f, cols, W) -> np.ndarray:
    def col(nm):
        return f['xgb'] if nm == 'xgb' else (f['ens'] if nm == 'ens' else f['cnns'][nm])
    out = np.zeros_like(f['xgb'])
    for j, nm in enumerate(cols):
        out += W[:, j][None, :] * col(nm)
    return out


def eye_rmse(pred, f) -> np.ndarray:
    """안 단위 RMSE. 마스크 지점이 없는 안은 nan."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = f['mask'][i]
        if m.any():
            out[i] = np.sqrt(np.mean((pred[i][m] - f['lab'][i][m]) ** 2))
    return out


def main() -> None:
    others = {k: [OOF[i] for i in range(5) if i != k] for k in range(5)}

    # --- fusion machinery: 5개 대안 ---------------------------------------
    gw_all = fit_gw([OOF[k] for k in range(5)], 'ir')
    machinery = {}

    machinery['global_w_dev'] = dict(
        rmse=pooled_rmse({k: gw_all * OOF[k]['xgb'] + (1 - gw_all) * OOF[k]['ir'] for k in range(5)}),
        w=gw_all,
        note='w 를 전체 OOF 로 적합하고 같은 OOF 에서 평가 — development estimate.',
    )

    w_nested = {k: fit_gw(others[k], 'ir') for k in range(5)}
    machinery['global_w_nested'] = dict(
        rmse=pooled_rmse({k: w_nested[k] * OOF[k]['xgb'] + (1 - w_nested[k]) * OOF[k]['ir'] for k in range(5)}),
        w_per_fold=[w_nested[k] for k in range(5)],
        note='평가 fold 를 뺀 4 fold 에서 w 적합 — 선택 낙관 제거.',
    )

    machinery['fixed_w_0.5'] = dict(
        rmse=pooled_rmse({k: 0.5 * OOF[k]['xgb'] + 0.5 * OOF[k]['ir'] for k in range(5)}),
        w=0.5,
        note='적합 없이 두 분지를 동일 가중으로 평균.',
    )

    pp = {k: fit_ppw(others[k], 'ir', w_nested[k]) for k in range(5)}
    machinery['per_point_w_nested'] = dict(
        rmse=pooled_rmse({k: pp[k] * OOF[k]['xgb'] + (1 - pp[k]) * OOF[k]['ir'] for k in range(5)}),
        note='52 개 field 위치마다 w 를 따로 적합, nested.',
    )

    cols = ['xgb', 'IR-v2']
    Wn = {k: fit_nnls(others[k], cols) for k in range(5)}
    machinery['nnls_stack_nested'] = dict(
        rmse=pooled_rmse({k: apply_nnls(OOF[k], cols, Wn[k]) for k in range(5)}),
        columns=cols,
        note='위치별 비음수 최소제곱 스택(합 1 로 정규화), nested.',
    )

    # --- per-eye advantage: 앙상블 fusion vs XGB 단독 ----------------------
    we = {k: fit_gw(others[k], 'ens') for k in range(5)}
    fus_e, xgb_e, cnn_e = [], [], []
    for k in range(5):
        f = OOF[k]
        p = we[k] * f['xgb'] + (1 - we[k]) * f['ens']
        fus_e.append(eye_rmse(p, f))
        xgb_e.append(eye_rmse(f['xgb'], f))
        cnn_e.append(eye_rmse(f['ens'], f))
    fus_e = np.concatenate(fus_e); xgb_e = np.concatenate(xgb_e); cnn_e = np.concatenate(cnn_e)
    ok = np.isfinite(fus_e) & np.isfinite(xgb_e) & np.isfinite(cnn_e)
    fus_e, xgb_e, cnn_e = fus_e[ok], xgb_e[ok], cnn_e[ok]

    def cmp(a, b, label):
        d = a - b
        return dict(
            comparator=label,
            mean_delta_rmse_db=float(np.mean(d)),
            mean_advantage_db=float(-np.mean(d)),
            median_delta_rmse_db=float(np.median(d)),
            n_eyes=int(d.size),
            n_eyes_fusion_better=int((d < 0).sum()),
            p_wilcoxon=float(wilcoxon(a, b).pvalue),
        )

    advantage = dict(
        model='fusion(XGB + 5백본 평균 CNN), nested global-w',
        w_per_fold=[we[k] for k in range(5)],
        vs_xgb=cmp(fus_e, xgb_e, 'XGB 단독'),
        vs_cnn_ensemble=cmp(fus_e, cnn_e, '5백본 평균 CNN'),
        note='안 단위 RMSE 의 대응표본 Wilcoxon. 음수 delta = fusion 우세.',
    )

    n_pts = int(sum(OOF[k]['mask'].sum() for k in range(5)))
    n_eyes = int(sum(OOF[k]['mask'].shape[0] for k in range(5)))
    payload = dict(
        note=('§4 Robustness 산문 수치의 산출물 근거. 정렬 규약은 explore_fusion_gains.py 와 동일 '
              '(공통 키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합).'),
        split='OOF (5-fold)',
        backbone_single='IR-v2',
        n_eyes=n_eyes,
        n_points=n_pts,
        w_grid_step=float(W_GRID[1] - W_GRID[0]),
        fusion_machinery=machinery,
        per_eye_advantage=advantage,
    )

    out = ROOT / 'runs/robustness_gains.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f'OOF {n_eyes} 안 / {n_pts} 점\n')
    print(f"{'config':<26}{'RMSE':>9}   원고")
    print('-' * 52)
    for key, want in (('global_w_dev', '8.064'), ('global_w_nested', '8.097'),
                      ('fixed_w_0.5', '8.065'), ('per_point_w_nested', '8.117'),
                      ('nnls_stack_nested', '8.116')):
        r = machinery[key]['rmse']
        mark = 'OK' if f'{r:.3f}' == want else f'!! {want}'
        print(f'{key:<26}{r:>9.4f}   {mark}')
    print('-' * 52)
    a = advantage['vs_xgb']
    print(f"per-eye advantage vs XGB : {a['mean_advantage_db']:.4f} dB, "
          f"{a['n_eyes_fusion_better']} of {a['n_eyes']} 안, p = {a['p_wilcoxon']:.3g}")
    print(f"           원고 기대치   : 0.861 dB, 171 of 240 안, p = 6.8e-13")
    b = advantage['vs_cnn_ensemble']
    print(f"per-eye vs CNN ensemble  : {-b['mean_delta_rmse_db']:.4f} dB, "
          f"{b['n_eyes_fusion_better']} of {b['n_eyes']} 안, p = {b['p_wilcoxon']:.3g}")
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
