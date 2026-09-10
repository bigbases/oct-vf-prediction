#!/usr/bin/env python3
"""Late fusion 통계 검증 — paired test + bootstrap CI (재학습 없음).

분석 단위 = eye (환자눈). 각 eye의 52-point RMSE를 계산한 뒤:
  1) paired 검정: fusion vs CNN, fusion vs XGB (Wilcoxon + paired t)
  2) bootstrap: eye 재표본 → pooled RMSE 95% CI
OOF(val 240 eyes)와 holdout test(37 eyes) 각각.
출력: runs/fusion_stats.json
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
    """eye별 52-point masked RMSE. (n_eyes,) 반환 (유효 point 있는 eye만)."""
    out = []
    keep = []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        if m.sum() == 0:
            keep.append(False)
            continue
        d = pred[i][m] - labels[i][m]
        out.append(np.sqrt(np.mean(d ** 2)))
        keep.append(True)
    return np.array(out), np.array(keep)


def pooled_rmse(pred, labels, mask):
    m = mask.astype(bool)
    d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2)))


def bootstrap_ci(pred, labels, mask, n=N_BOOT):
    """eye 단위 부트스트랩 → pooled RMSE 95% CI."""
    n_eyes = pred.shape[0]
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, n_eyes, n_eyes)
        vals.append(pooled_rmse(pred[idx], labels[idx], mask[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())


def paired_block(px, pc, labels, mask, label):
    pf = W * px + (1 - W) * pc
    e_f, keep = per_eye_rmse(pf, labels, mask)
    e_x, _ = per_eye_rmse(px, labels, mask)
    e_c, _ = per_eye_rmse(pc, labels, mask)
    n_eyes = len(e_f)

    def _pair(a, b, name):
        # a=fusion, b=solo; 음수 diff = fusion이 더 좋음
        diff = a - b
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        t_stat, t_p = stats.ttest_rel(a, b)
        wins = int(np.sum(a < b))
        return {
            'comparison': name,
            'n_eyes': n_eyes,
            'mean_eye_rmse_fusion': float(a.mean()),
            'mean_eye_rmse_solo': float(b.mean()),
            'mean_diff (fusion-solo)': float(diff.mean()),
            'median_diff': float(np.median(diff)),
            'fusion_better_eyes': f'{wins}/{n_eyes}',
            'wilcoxon_p': float(w_p),
            'paired_t_p': float(t_p),
        }

    print(f'\n[{label}] n_eyes={n_eyes}', flush=True)
    res = []
    for solo, name in [(e_c, 'fusion vs CNN'), (e_x, 'fusion vs XGB')]:
        r = _pair(e_f, solo, name)
        res.append(r)
        print(f'  {name}: fusion {r["mean_eye_rmse_fusion"]:.3f} vs solo {r["mean_eye_rmse_solo"]:.3f} '
              f'| Δ={r["mean_diff (fusion-solo)"]:.3f} | 우세 {r["fusion_better_eyes"]} '
              f'| Wilcoxon p={r["wilcoxon_p"]:.4f} | t p={r["paired_t_p"]:.4f}', flush=True)
    return res


def load_split(paths_xgb, cnn_name):
    xk, ck, lk, mk = [], [], [], []
    for k in range(5):
        xz = load_oof_npz(paths_xgb[k])
        cz = load_oof_npz(CNN_DIR / f'{cnn_name}_fold{k}.npz')
        ax, bx, _ = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        if cnn_name == 'test_preds':
            # 각 fold test는 동일 eyes → concat하면 중복. 대신 per-fold 유지 후 stack 평가는
            # paired에는 부적절하므로 test는 fold별 평균 예측 사용.
            xk.append(ax['pred']); ck.append(bx['pred']); lk.append(ax['labels']); mk.append(m)
        else:
            xk.append(ax['pred']); ck.append(bx['pred']); lk.append(ax['labels']); mk.append(m)
    return xk, ck, lk, mk


def main():
    result = {'w_xgb': W, 'unit': 'eye (52-point RMSE)', 'n_boot': N_BOOT}

    # ---- OOF (240 eyes, fold concat: 각 eye 1회) ----
    xk, ck, lk, mk = [], [], [], []
    for k in range(5):
        xz = load_oof_npz(XGB_VAL[k])
        cz = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, _ = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        xk.append(ax['pred']); ck.append(bx['pred']); lk.append(ax['labels']); mk.append(m)
    XP, CP, LB, MK = (np.concatenate(a) for a in (xk, ck, lk, mk))
    print('=' * 70 + '\nOOF (5-fold val, eye 단위)\n' + '=' * 70, flush=True)
    result['oof_paired'] = paired_block(XP, CP, LB, MK, 'OOF')
    pf = W * XP + (1 - W) * CP
    lo, hi, mean = bootstrap_ci(pf, LB, MK)
    result['oof_fusion_bootstrap'] = {'point_rmse': pooled_rmse(pf, LB, MK), 'ci95': [lo, hi], 'boot_mean': mean}
    # solo CI
    lox, hix, _ = bootstrap_ci(XP, LB, MK)
    loc, hic, _ = bootstrap_ci(CP, LB, MK)
    result['oof_solo_bootstrap'] = {
        'xgb': {'point_rmse': pooled_rmse(XP, LB, MK), 'ci95': [lox, hix]},
        'cnn': {'point_rmse': pooled_rmse(CP, LB, MK), 'ci95': [loc, hic]},
    }
    print(f'  bootstrap fusion RMSE={result["oof_fusion_bootstrap"]["point_rmse"]:.3f} '
          f'95%CI [{lo:.3f}, {hi:.3f}]', flush=True)
    print(f'  bootstrap XGB  95%CI [{lox:.3f}, {hix:.3f}] | CNN 95%CI [{loc:.3f}, {hic:.3f}]', flush=True)

    # ---- TEST (37 eyes) : fold별 예측 평균(앙상블)으로 eye 1회 평가 ----
    print('\n' + '=' * 70 + '\nTEST (holdout 37 eyes, fold 예측 평균)\n' + '=' * 70, flush=True)
    xk2, ck2, lk2, mk2, keys0 = [], [], [], [], None
    for k in range(5):
        xz = load_oof_npz(XGB_TEST[k])
        cz = load_oof_npz(CNN_DIR / f'test_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        if keys0 is None:
            keys0 = common
            LBt = ax['labels']; MKt = ax['mask'] & bx['mask']
        xk2.append(ax['pred']); ck2.append(bx['pred'])
    XPt = np.mean(np.stack(xk2), axis=0)
    CPt = np.mean(np.stack(ck2), axis=0)
    result['test_paired'] = paired_block(XPt, CPt, LBt, MKt, 'TEST (ensemble)')
    pft = W * XPt + (1 - W) * CPt
    lo, hi, mean = bootstrap_ci(pft, LBt, MKt)
    result['test_fusion_bootstrap'] = {'point_rmse': pooled_rmse(pft, LBt, MKt), 'ci95': [lo, hi], 'boot_mean': mean}
    lox, hix, _ = bootstrap_ci(XPt, LBt, MKt)
    loc, hic, _ = bootstrap_ci(CPt, LBt, MKt)
    result['test_solo_bootstrap'] = {
        'xgb': {'point_rmse': pooled_rmse(XPt, LBt, MKt), 'ci95': [lox, hix]},
        'cnn': {'point_rmse': pooled_rmse(CPt, LBt, MKt), 'ci95': [loc, hic]},
    }
    print(f'  bootstrap fusion(ensemble) RMSE={result["test_fusion_bootstrap"]["point_rmse"]:.3f} '
          f'95%CI [{lo:.3f}, {hi:.3f}]', flush=True)
    print(f'  bootstrap XGB 95%CI [{lox:.3f}, {hix:.3f}] | CNN 95%CI [{loc:.3f}, {hic:.3f}]', flush=True)

    out = ROOT / 'runs/fusion_stats.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {out}', flush=True)


if __name__ == '__main__':
    main()
