#!/usr/bin/env python3
"""no vertical C/D (25 파라미터) XGB 를 재적합하고 안 단위 예측을 저장한다.

`scripts/ablate_vert_cd.py` 는 같은 재적합을 하고도 pooled RMSE 와 Wilcoxon p 만
JSON 에 남긴다 — 안 단위 예측이 없어 Figure 6 의 이 행만 비어 있었다. 이 스크립트는
그 스크립트의 함수를 그대로 import 해서(설정을 베끼지 않는다) 재적합하고, 예측을
npz 로 저장한다. 저장 위치는 `experiments/forest_robustness/xgb25/` 이며 정본
`runs/` 와 패스 B 트리 어디에도 쓰지 않는다.

가드: 재적합 결과가 기존 `runs/ablation_vert_cd.json` 의 pooled 값(25feat XGB
8.7182, fusion w_refit 8.0828, 26feat 8.6603/8.0688)을 재현하지 못하면 저장하지
않고 멈춘다.

실행: python experiments/forest_robustness/refit_xgb25.py
env: hvf (xgboost 필요)
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / 'experiments/laterality_qfix/step4_work/B'      # 패스 B
OUTDIR = REPO / 'experiments/forest_robustness/xgb25'
TOL = 5e-4                                                     # dB


def load_mod():
    for m in ('ablate_vert_cd', 'oof_common'):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(TREE / 'scripts'))
    try:
        m = importlib.import_module('ablate_vert_cd')
        assert m.ROOT == TREE, f'ROOT 불일치: {m.ROOT} != {TREE}'
        return m
    finally:
        sys.path.pop(0)


A = load_mod()
sys.path.insert(0, str(REPO / 'scripts'))
from oof_common import save_oof_npz  # noqa: E402


def main() -> int:
    rows = list(csv.DictReader(open(A.CSV, encoding='utf-8-sig')))
    ref = json.loads((TREE / 'runs/ablation_vert_cd.json').read_text(encoding='utf-8'))
    saved, ok = {}, True

    for drop, tag, rkey in [(False, '26feat_baseline', '26_feat_baseline'),
                            (True, '25feat_no_vertcd', '25_feat_no_vertcd')]:
        nf = len(A.feats('OD', drop))
        print(f'\n=== {tag} ({nf} feat) 재적합 ===', flush=True)
        val, tst = A.xgb_oof(rows, drop)
        XP, CP, LB, MK = A.build_oof(val, 'val_preds')
        xr, _ = A.pooled(XP, LB, MK)
        w_ref = A.best_w(XP, CP, LB, MK)
        f047, _ = A.pooled(0.47 * XP + 0.53 * CP, LB, MK)
        fref, _ = A.pooled(w_ref * XP + (1 - w_ref) * CP, LB, MK)

        r = ref[rkey]['oof']
        for name, got, want in [('xgb_rmse', xr, r['xgb_rmse']),
                                ('fusion w0.47', f047, r['w0.47']['fusion_rmse']),
                                ('fusion w_refit', fref, r['w_refit']['fusion_rmse']),
                                ('w_refit', w_ref, r['w_refit']['w_xgb'])]:
            good = abs(got - want) <= TOL
            ok &= good
            print(f"  {'OK ' if good else '!! '}{name:<15} {got:.4f} vs {want:.4f}")
        saved[tag] = (val, tst, nf)

    if not ok:
        print('\n재현 실패. npz 를 저장하지 않는다. 그림도 고치지 않는다.')
        return 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (val, tst, nf) in saved.items():
        for split, folds in (('val', val), ('test', tst)):
            for k, f in enumerate(folds):
                meta = {'source': 'experiments/forest_robustness/refit_xgb25.py',
                        'basis': 'pass B', 'n_features': nf, 'tag': tag,
                        'random_state': 42, 'fold': k, 'split': split}
                save_oof_npz(OUTDIR / f'xgb_{tag}_fold{k}_{split}.npz',
                             f['keys'], f['pred'], f['labels'],
                             np.asarray(f['mask']), meta)
    print(f'\n가드 8/8 통과. 저장: {OUTDIR}/ ({len(list(OUTDIR.glob("*.npz")))} 파일)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
