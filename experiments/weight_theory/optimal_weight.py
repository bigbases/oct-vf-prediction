#!/usr/bin/env python3
"""(A) 두 branch 오차 상관과 분산 최적 가중치.

원고 §4.1 의 "the weight placed on the summary branch rises monotonically"
는 지금 관찰로만 적혀 있다. 이 스크립트는 그 관찰을 검정으로 승격하기 위한
재료를 낸다 — 오차 상관 rho 와, rho 로부터 예측되는 가중치 w*.

읽기 전용이다. 재학습하지 않고, 어떤 정본도 쓰지 않는다.
입력은 step4_work/{A,B}/runs/ 의 저장된 예측뿐이고,
마스크·정렬·w 탐색은 scripts/fusion_contribution_per_backbone.py 와 같다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa: E402

SEED = 42
N_BOOT = 10000

# fusion_contribution_per_backbone.py 의 IMG_DIR 그대로. image-only 가 fusion 레시피다.
IMG_DIR = {
    'inception_v3':        'runs/phasec_b0_clip_mse_5fold',
    'inception_resnet_v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'vgg16':               'runs/phasec_b0_vgg16_5fold',
    'xception':            'runs/phasec_b0_xception_5fold',
    'densenet121':         'runs/phasec_b0_densenet121_5fold',
}
DISPLAY = {
    'inception_v3': 'Inception-v3', 'inception_resnet_v2': 'IR-v2',
    'vgg16': 'VGG16', 'xception': 'Xception', 'densenet121': 'DenseNet121',
}
# 원고 표의 관측 w (패스 A). 재계산값과 대조하기 위해 박아 둔다.
W_PAPER = {'xception': 0.43, 'inception_resnet_v2': 0.47, 'densenet121': 0.50,
           'inception_v3': 0.60, 'vgg16': 0.63}
# image-only CNN 의 OOF pooled RMSE. 원고가 단조성을 주장할 때 쓰는 정렬축이다.
# 두 패스에서 image branch 는 동일하므로(라테랄리티는 표 특징만 건드린다) 상수다.
IMG_RMSE = {'xception': 8.352, 'inception_resnet_v2': 8.540, 'densenet121': 8.669,
            'inception_v3': 9.110, 'vgg16': 9.165}


def npz_cnn(path):
    z = np.load(path, allow_pickle=True)
    return {'keys': list(zip(z['patient_id'], z['eye'], z['vf_date'])),
            'pred': z['pred'], 'labels': z['labels'], 'mask': z['mask'], 'meta': {}}


def opt_w(px, pc, labels, mask, grid=101):
    """producer 와 동일한 101격자 탐색 (pooled RMSE 최소화)."""
    m = mask.astype(bool)
    ya, yb, y = px[m], pc[m], labels[m]
    best = (1e9, 0.5)
    for w in np.linspace(0, 1, grid):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (float(r), float(w))
    return best[1]


def collect_oof(tree, cnn_dir):
    """5 fold 를 concat. 각 안구가 정확히 한 번 들어간다."""
    xs, cs, ls, ms, ks = [], [], [], [], []
    for k in range(5):
        xz = load_oof_npz(tree / f'runs/oof/xgb_90d_fold{k}_val.npz')
        cz = npz_cnn(tree / cnn_dir / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        xs.append(ax['pred']); cs.append(bx['pred']); ls.append(ax['labels'])
        ms.append(ax['mask'] & bx['mask']); ks.extend(common)
    return (np.concatenate(xs), np.concatenate(cs), np.concatenate(ls),
            np.concatenate(ms), ks)


def w_star(s1, s2, rho):
    """예측기1(XGB)에 걸리는 분산 최적 가중치."""
    den = s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2
    return float((s2 ** 2 - rho * s1 * s2) / den)


def cluster_stats(x, y, pid):
    """환자별 충분통계. 부트스트랩을 행렬곱으로 돌리기 위한 준비."""
    pats = sorted(set(pid))
    idx = {p: i for i, p in enumerate(pats)}
    S = np.zeros((len(pats), 6))
    for xi, yi, p in zip(x, y, pid):
        r = S[idx[p]]
        r[0] += 1; r[1] += xi; r[2] += yi
        r[3] += xi * xi; r[4] += yi * yi; r[5] += xi * yi
    return S


def rho_from_stats(T):
    """(..,6) 합계 -> Pearson r. 표본 분산(ddof=0)으로 계산 — r 은 무관하다."""
    n, sx, sy, sxx, syy, sxy = (T[..., i] for i in range(6))
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return cov / np.sqrt(vx * vy)


def boot_rho_ci(x, y, pid, seed=SEED, n_boot=N_BOOT):
    """환자 클러스터 부트스트랩. 환자를 복원추출하고 그 안의 관측은 통째로 따라온다."""
    S = cluster_stats(x, y, pid)
    P = S.shape[0]
    rng = np.random.default_rng(seed)
    cnt = np.zeros((n_boot, P))
    draws = rng.integers(0, P, size=(n_boot, P))
    for b in range(n_boot):
        np.add.at(cnt[b], draws[b], 1)
    T = cnt @ S                       # (n_boot, 6)
    r = rho_from_stats(T)
    r = r[np.isfinite(r)]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), len(r), P


def per_eye_rmse(pred, labels, mask):
    out, keep = [], []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        if m.sum() == 0:
            continue
        d = pred[i][m] - labels[i][m]
        out.append(float(np.sqrt(np.mean(d ** 2)))); keep.append(i)
    return np.array(out), keep


def analyse(tree, label):
    res = {}
    for bb, cnn_dir in IMG_DIR.items():
        X, C, L, M, keys = collect_oof(tree, cnn_dir)
        w_obs = opt_w(X, C, L, M)

        entry = {'w_observed_recomputed': round(w_obs, 4),
                 'w_paper_passA': W_PAPER[bb], 'axes': {}}

        # ---- (a) 풀링 축: 마스크된 모든 지점 ----
        m = M.astype(bool)
        ex = (X - L)[m]
        ec = (C - L)[m]
        pid_pt = np.array([keys[i][0] for i in range(len(keys))], dtype=object)
        pid_pt = np.repeat(pid_pt, m.sum(axis=1))     # 지점마다 환자 꼬리표
        entry['axes']['pooled'] = axis_block(ex, ec, pid_pt, w_obs, exact_mse=True)

        # ---- (b) 안구별 RMSE 축 ----
        rx, keep = per_eye_rmse(X, L, M)
        rc, _ = per_eye_rmse(C, L, M)
        pid_eye = np.array([keys[i][0] for i in keep], dtype=object)
        entry['axes']['per_eye'] = axis_block(rx, rc, pid_eye, w_obs, exact_mse=False)

        entry['img_pooled_rmse'] = IMG_RMSE[bb]
        entry['n_eyes'] = len(rx)
        entry['n_points'] = int(m.sum())
        res[bb] = entry
        print(f'  [{label}] {DISPLAY[bb]:<13} w_obs={w_obs:.2f} '
              f'rho_pooled={entry["axes"]["pooled"]["rho"]:.4f} '
              f'w*={entry["axes"]["pooled"]["w_star_rho_obs"]:.4f}', flush=True)
    return res


def axis_block(a, b, pid, w_obs, exact_mse):
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    rho = float(np.corrcoef(a, b)[0, 1])
    lo, hi, nb, P = boot_rho_ci(a, b, pid)
    out = {
        'sd_xgb': s1, 'sd_cnn': s2, 'mean_xgb': m1, 'mean_cnn': m2,
        'rho': rho, 'rho_ci95': [lo, hi], 'n_boot_finite': nb, 'n_patients': P,
        'w_star_rho_obs': w_star(s1, s2, rho),
        'w_star_rho_zero': w_star(s1, s2, 0.0),
        'w_observed': w_obs,
    }
    out['abs_diff_obs_vs_wstar'] = abs(w_obs - out['w_star_rho_obs'])
    if exact_mse:
        # 관측 w 는 분산이 아니라 pooled MSE 를 최소화한다. 편향이 0이 아니면
        # 위 공식과 어긋나므로, 편향까지 포함한 정확해도 같이 낸다(중심적률 -> 원적률).
        q1, q2, q12 = float(np.mean(a ** 2)), float(np.mean(b ** 2)), float(np.mean(a * b))
        out['w_star_exact_mse'] = float((q2 - q12) / (q1 + q2 - 2 * q12))
        out['abs_diff_obs_vs_exact'] = abs(w_obs - out['w_star_exact_mse'])
    return out


def verdicts(res):
    """4번 판정. 값을 맞추려는 조정은 하지 않는다 — 어긋나면 어긋난 채로 낸다."""
    order = sorted(IMG_RMSE, key=IMG_RMSE.get)          # image RMSE 오름차순
    v = {'sort_axis': 'image-only OOF pooled RMSE 오름차순',
         'order': [DISPLAY[b] for b in order]}
    for ax in ('pooled', 'per_eye'):
        A = v[ax] = {}
        wobs = [res_ax(res, b, ax, 'w_observed') for b in order]
        wsr  = [res_ax(res, b, ax, 'w_star_rho_obs') for b in order]
        wsz  = [res_ax(res, b, ax, 'w_star_rho_zero') for b in order]
        A['w_observed_in_order'] = wobs
        A['w_star_rho_obs_in_order'] = wsr
        A['w_star_rho_zero_in_order'] = wsz
        A['monotonic_w_observed'] = all(x < y for x, y in zip(wobs, wobs[1:]))
        A['monotonic_w_star_rho_obs'] = all(x < y for x, y in zip(wsr, wsr[1:]))
        A['monotonic_w_star_rho_zero'] = all(x < y for x, y in zip(wsz, wsz[1:]))
        d = [abs(a - b) for a, b in zip(wobs, wsr)]
        A['abs_diff_per_backbone'] = dict(zip([DISPLAY[b] for b in order], d))
        A['abs_diff_mean'] = sum(d) / len(d)
        A['abs_diff_max'] = max(d)
        A['range_w_observed'] = max(wobs) - min(wobs)
        A['range_w_star_rho_obs'] = max(wsr) - min(wsr)
        A['range_w_star_rho_zero'] = max(wsz) - min(wsz)
        A['observed_range_wider_than_rho_zero'] = (
            A['range_w_observed'] > A['range_w_star_rho_zero'])
        # 관측 w 는 pooled MSE 최소해다. 분산 공식이 어긋나는 원인이 상관인지
        # 편향인지 가르려면 편향까지 넣은 정확해로도 같은 판정을 해 봐야 한다.
        if ax == 'pooled':
            we = [res_ax(res, b, ax, 'w_star_exact_mse') for b in order]
            A['w_star_exact_mse_in_order'] = we
            A['monotonic_w_star_exact_mse'] = all(x < y for x, y in zip(we, we[1:]))
            A['range_w_star_exact_mse'] = max(we) - min(we)
            de = [abs(a - b) for a, b in zip(wobs, we)]
            A['abs_diff_exact_mean'] = sum(de) / len(de)
            A['abs_diff_exact_max'] = max(de)
        # 안구별 RMSE 축의 w* 는 볼록결합 밖으로 나갈 수 있다. 그 축의 sd 는
        # 오차의 산포가 아니라 양수 통계량의 산포라, 공식의 전제가 성립하지 않는다.
        A['w_star_outside_unit_interval'] = [
            DISPLAY[b] for b in order
            if not (0.0 <= res_ax(res, b, ax, 'w_star_rho_obs') <= 1.0)]
    return v


def res_ax(res, bb, ax, key):
    return res[bb]['axes'][ax][key]


def main():
    trees = {'A': ROOT / 'experiments/laterality_qfix/step4_work/A',
             'B': ROOT / 'experiments/laterality_qfix/step4_work/B'}
    out = {'seed': SEED, 'n_boot': N_BOOT,
           'note': 'OOF only. mask = XGB mask & CNN mask, '
                   'fusion_contribution_per_backbone.py 와 동일.',
           'passes': {}}
    for k, t in trees.items():
        print(f'=== 패스 {k} ===', flush=True)
        r = analyse(t, k)
        out['passes'][k] = {'backbones': r, 'verdicts': verdicts(r)}
    p = ROOT / 'experiments/weight_theory/optimal_weight.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n저장: {p}', flush=True)


if __name__ == '__main__':
    main()
