#!/usr/bin/env python3
"""Robustness of the significance tests to the correlation between the two eyes of
one patient. No retraining.

Two variants remove that correlation entirely: (A) one value per patient by
averaging their eyes, (B) a cluster bootstrap that resamples patients rather
than eyes.

환자 클러스터(양안 상관) 강건성 검정 — 재학습 없음.

리뷰어 우려: 안 단위(eye) 검정은 한 환자의 양안을 독립으로 취급 → pseudo-replication으로
p값이 과신될 수 있음. 여기서 양안 상관을 완전히 제거한 두 가지를 계산:
  (A) 환자 평균 검정: 환자당 눈들의 52-pt RMSE를 평균 → 환자 1명 = 1값 → paired Wilcoxon/t
  (B) 환자 클러스터 부트스트랩: 눈이 아니라 '환자'를 재표본 → pooled RMSE 95% CI
OOF(240안/125명)와 holdout test(37안/20명) 각각. 안 단위 결과와 나란히 비교.
출력: runs/patient_cluster_stats.json
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

CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
W = 0.47
RNG = np.random.default_rng(42)
N_BOOT = 5000


def per_eye_rmse(pred, labels, mask):
    out = []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        if m.sum() == 0:
            out.append(np.nan)
            continue
        d = pred[i][m] - labels[i][m]
        out.append(float(np.sqrt(np.mean(d ** 2))))
    return np.array(out)


def pooled_rmse(pred, labels, mask):
    m = mask.astype(bool)
    d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2)))


def load_oof():
    xs, cs, ls, ms, keys = [], [], [], [], []
    for k in range(5):
        xz = load_oof_npz(XGB_VAL[k])
        cz = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        xs.append(ax['pred']); cs.append(bx['pred']); ls.append(ax['labels']); ms.append(m); keys += common
    X = np.concatenate(xs); C = np.concatenate(cs); L = np.concatenate(ls); M = np.concatenate(ms)
    pid = np.array([k[0] for k in keys], dtype=object)
    return X, C, L, M, pid


def load_test():
    xk, ck, keys0, LB, MK = [], [], None, None, None
    for k in range(5):
        xz = load_oof_npz(XGB_TEST[k])
        cz = load_oof_npz(CNN_DIR / f'test_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        if keys0 is None:
            keys0 = common
            LB = ax['labels']; MK = ax['mask'] & bx['mask']
        xk.append(ax['pred']); ck.append(bx['pred'])
    X = np.mean(np.stack(xk), axis=0)
    C = np.mean(np.stack(ck), axis=0)
    pid = np.array([k[0] for k in keys0], dtype=object)
    return X, C, LB, MK, pid


def paired(a, b):
    """a=fusion, b=solo. 음수 diff = fusion 우세."""
    a = np.asarray(a); b = np.asarray(b)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    try:
        _, wp = stats.wilcoxon(a, b)
    except ValueError:
        wp = float('nan')
    _, tp = stats.ttest_rel(a, b)
    return {
        'n': int(len(a)),
        'mean_fusion': float(a.mean()),
        'mean_solo': float(b.mean()),
        'mean_diff': float((a - b).mean()),
        'fusion_better': f'{int(np.sum(a < b))}/{len(a)}',
        'wilcoxon_p': float(wp),
        'paired_t_p': float(tp),
    }


def patient_means(metric, pid):
    upid = sorted(set(pid))
    return upid, np.array([np.nanmean(metric[pid == p]) for p in upid])


def cluster_bootstrap(pred, labels, mask, pid):
    upid = np.array(sorted(set(pid)), dtype=object)
    idxmap = {p: np.where(pid == p)[0] for p in upid}
    vals = []
    for _ in range(N_BOOT):
        samp = RNG.integers(0, len(upid), len(upid))
        rows = np.concatenate([idxmap[upid[s]] for s in samp])
        vals.append(pooled_rmse(pred[rows], labels[rows], mask[rows]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def eye_bootstrap(pred, labels, mask):
    n = pred.shape[0]
    vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        vals.append(pooled_rmse(pred[idx], labels[idx], mask[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyze(X, C, L, M, pid, name):
    F = W * X + (1 - W) * C
    ef, ex, ec = per_eye_rmse(F, L, M), per_eye_rmse(X, L, M), per_eye_rmse(C, L, M)
    n_eyes = int(np.sum(~np.isnan(ef)))
    n_pat = len(set(pid))
    print('=' * 74)
    print(f'{name}: {n_eyes} eyes / {n_pat} patients')
    print('=' * 74)

    block = {'n_eyes': n_eyes, 'n_patients': n_pat}

    # --- eye-level (기존, 참조) ---
    block['eye_level'] = {
        'fusion_vs_XGB': paired(ef, ex),
        'fusion_vs_CNN': paired(ef, ec),
    }
    # --- patient-level (양안 상관 제거) ---
    _, pf = patient_means(ef, pid)
    _, pfx = patient_means(ex, pid)
    _, pfc = patient_means(ec, pid)
    block['patient_level'] = {
        'fusion_vs_XGB': paired(pf, pfx),
        'fusion_vs_CNN': paired(pf, pfc),
    }
    for lvl in ('eye_level', 'patient_level'):
        print(f'\n[{lvl}]')
        for comp in ('fusion_vs_XGB', 'fusion_vs_CNN'):
            r = block[lvl][comp]
            print(f'  {comp:16s} n={r["n"]:>3} | fusion {r["mean_fusion"]:.3f} vs solo {r["mean_solo"]:.3f} '
                  f'| Δ={r["mean_diff"]:+.3f} | {r["fusion_better"]:>7} | Wilcoxon p={r["wilcoxon_p"]:.4g} '
                  f'| t p={r["paired_t_p"]:.4g}')

    # --- bootstrap CI: eye vs patient cluster ---
    e_lo, e_hi = eye_bootstrap(F, L, M)
    c_lo, c_hi = cluster_bootstrap(F, L, M, pid)
    block['fusion_pooled_rmse'] = pooled_rmse(F, L, M)
    block['bootstrap'] = {
        'eye_resample_ci95': [e_lo, e_hi],
        'patient_cluster_ci95': [c_lo, c_hi],
    }
    print(f'\n  fusion pooled RMSE={block["fusion_pooled_rmse"]:.3f}')
    print(f'    eye-resample   95%CI [{e_lo:.3f}, {e_hi:.3f}]')
    print(f'    patient-cluster 95%CI [{c_lo:.3f}, {c_hi:.3f}]  (양안 상관 반영 → 보통 더 넓음)')
    return block


def main():
    result = {'w_xgb': W, 'n_boot': N_BOOT,
              'note': '환자 평균 검정(양안 상관 제거) + 환자 클러스터 부트스트랩. GEE 대체(statsmodels 부재).'}
    Xo, Co, Lo, Mo, pido = load_oof()
    result['OOF'] = analyze(Xo, Co, Lo, Mo, pido, 'OOF (5-fold val)')
    Xt, Ct, Lt, Mt, pidt = load_test()
    result['TEST'] = analyze(Xt, Ct, Lt, Mt, pidt, 'TEST (holdout, fold-ensemble)')
    out = ROOT / 'runs/patient_cluster_stats.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
