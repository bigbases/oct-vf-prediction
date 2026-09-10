#!/usr/bin/env python3
"""반대편 눈 피처 ablation 평가 — 사전등록 docs/fellow_eye_ablation_prereg.md 대로만 본다.

주 종말점: XGB(46) 대 XGB(26). OOF 안 단위 RMSE 의 대응 Wilcoxon.
부차: 융합(46) 대 융합(26), 융합(46) 대 영상 단독.

**금지(사전등록 §4.1): 이 결과를 C5(융합 > 영상 단독) 주장에 쓰지 않는다.**
융합(46) 대 영상 단독은 정보 비대칭 비교이므로 진단용으로만 출력한다.

읽기 전용. 출력: runs/fellow_eye_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_eval_common import NAMES, load_all, oof_arrays, test_arrays  # noqa: E402

BASE_TAG, FELLOW_TAG = '90d', '90d_fellow'
N_BOOT, SEED = 10000, 42


def boot_ci(d, pid, rng):
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    ng = len(groups)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        means[b] = np.mean(d[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a, b, pid, label):
    """a - b. 음수 = a 가 우세."""
    d = a - b
    p = float(wilcoxon(a, b).pvalue)
    lo, hi = boot_ci(d, pid, np.random.default_rng(SEED))
    return dict(what=label, n=int(d.size), mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                delta=float(np.mean(d)), ci_lo=lo, ci_hi=hi, p_wilcoxon=p,
                a_better=bool(np.mean(d) < 0), significant=bool(p < 0.05))


def main() -> None:
    oofb, tstb, labb, mskb = load_all(BASE_TAG)
    ooff, tstf, labf, mskf = load_all(FELLOW_TAG)

    out = {'note': ('반대편 눈 ablation. 사전등록 docs/fellow_eye_ablation_prereg.md. '
                    '주 종말점 = XGB(46) vs XGB(26) OOF RMSE. '
                    '융합(46) vs 영상단독은 정보 비대칭이므로 C5 주장에 쓰지 않는다.'),
           'primary': None, 'secondary': [], 'diagnostic': []}

    # ---- 주 종말점: XGB(46) vs XGB(26). CNN 과 무관하므로 대표 백본 하나로 뽑는다 ----
    nm0 = NAMES[0]
    for metric in ('rmse', 'mae'):
        _, x26, _, pid = oof_arrays(oofb, nm0, metric)
        _, x46, _, _ = oof_arrays(ooff, nm0, metric)
        r = compare(x46, x26, pid, f'XGB(46) vs XGB(26) | OOF {metric.upper()}')
        if metric == 'rmse':
            out['primary'] = r
        else:
            out['secondary'].append(r)

    for metric in ('rmse', 'mae'):
        _, x26, _, pidt = test_arrays(oofb, tstb, labb, mskb, nm0, metric)
        _, x46, _, _ = test_arrays(ooff, tstf, labf, mskf, nm0, metric)
        out['secondary'].append(
            compare(x46, x26, pidt, f'XGB(46) vs XGB(26) | TEST {metric.upper()}'))

    # ---- 부차: 융합(46) vs 융합(26) ----
    for nm in NAMES:
        for metric in ('rmse', 'mae'):
            f26, _, c26, pid = oof_arrays(oofb, nm, metric)
            f46, _, _, _ = oof_arrays(ooff, nm, metric)
            out['secondary'].append(
                compare(f46, f26, pid, f'fusion(46) vs fusion(26) | {nm} OOF {metric.upper()}'))
            # ---- 진단용(주장 금지): 융합(46) vs 영상 단독 ----
            out['diagnostic'].append(
                compare(f46, c26, pid, f'fusion(46) vs CNN | {nm} OOF {metric.upper()}'))

    # ================= 출력 =================
    def show(r):
        arrow = '우세' if r['a_better'] else '열세'
        sig = '유의' if r['significant'] else 'ns  '
        print(f"  {r['what']:<48} {r['mean_a']:>6.3f} vs {r['mean_b']:>6.3f} | "
              f"delta {r['delta']:>+6.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] | "
              f"p={r['p_wilcoxon']:.3g} {sig} ({arrow})")

    print('주 종말점 (사전등록) — 반대편 눈이 정량 브랜치의 천장을 올리는가')
    show(out['primary'])
    print('\n부차')
    for r in out['secondary']:
        show(r)
    print('\n진단용 — 정보 비대칭 비교. 사전등록 §4.1 에 따라 C5 주장에 쓰지 않는다.')
    for r in out['diagnostic']:
        show(r)

    pr = out['primary']
    print('\n판정')
    if pr['significant'] and pr['a_better']:
        v = ('(A) 또는 (B): XGB(46) > XGB(26). 반대편 눈이 정량 브랜치를 강화했다. '
             '이득의 원천이 스캔 외 정보로 바뀌었음을 명시해야 한다.')
    elif pr['significant'] and not pr['a_better']:
        v = '(D): XGB(46) < XGB(26). 피처 20개 추가가 과적합으로 작용했다.'
    else:
        v = ('(C): XGB(46) ≈ XGB(26). 반대편 눈 요약값은 대상안 시야 예측에 정보를 '
             '더하지 않는다. 정량 브랜치의 천장이 피처 부족이 아니라 표현의 한계에서 온다.')
    print(f'  {v}')
    out['verdict'] = v

    p = ROOT / 'runs/fellow_eye_ablation.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'\n저장: {p}')


if __name__ == '__main__':
    main()
