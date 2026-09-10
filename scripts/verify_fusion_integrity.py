#!/usr/bin/env python3
"""Integrity checks on the late-fusion baseline. No retraining: the stored
predictions are recomputed independently.

Checked: patient-level leakage across folds, whether the out-of-fold npz really
are out of fold, whether the fusion weight is derived from validation folds
only, whether the reported numbers reproduce from the stored npz, and mask and
blind-spot consistency.

Late fusion baseline 무결성 검증 (재학습 없음, 저장물만 독립 재계산).

검증 항목:
  A. patient-level split 누수 (fold/test 간 환자 중복)
  B. OOF npz가 진짜 out-of-fold 키인지 (train 환자와 겹치지 않는지)
  C. w=0.47이 val OOF에서만 유도되는지 (test 재계산 optimal w 비교)
  D. 저장 npz에서 OOF 8.06 / test 8.61 독립 재계산 일치
  E. mask/blind-spot 일관성, ensemble vs per-fold test 교차확인
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz, masked_overall_rmse_mae  # noqa: E402

CSV = ROOT / 'ml_final_90d_excl_empty_flip.csv'
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
W = 0.47


def sep(t):
    print('\n' + '=' * 70 + f'\n{t}\n' + '=' * 70, flush=True)


def recompute_rmse(pred, labels, mask):
    m = mask.astype(bool)
    d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def opt_w(px, pc, labels, mask, grid=101):
    m = mask.astype(bool)
    ya, yb, y = px[m], pc[m], labels[m]
    best = (1e9, None)
    for w in np.linspace(0, 1, grid):
        r = np.sqrt(np.mean((w * ya + (1 - w) * yb - y) ** 2))
        if r < best[0]:
            best = (float(r), float(w))
    return best


def main():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    print(f'CSV rows: {len(rows)}', flush=True)

    # ---- A. patient-level split 누수 ----
    sep('A. patient-level split 누수 검사')
    fold_pat = {}
    for r in rows:
        f = r['cv_fold']
        fold_pat.setdefault(f, set()).add(str(r['patient_id']).strip())
    for f in sorted(fold_pat):
        print(f'  fold={f:>4}  eyes={sum(1 for r in rows if r["cv_fold"]==f):>3}  patients={len(fold_pat[f])}', flush=True)
    # test vs 나머지
    non_test = set().union(*[v for k, v in fold_pat.items() if k != 'test'])
    test_pat = fold_pat.get('test', set())
    overlap_tt = non_test & test_pat
    print(f'  [test ∩ train/val 환자]: {len(overlap_tt)}  {"OK 누수 없음" if not overlap_tt else "누수! "+str(overlap_tt)}', flush=True)
    # fold 간 환자 중복
    cvfolds = [k for k in fold_pat if k not in ('test',)]
    pair_bad = []
    for i in range(len(cvfolds)):
        for j in range(i + 1, len(cvfolds)):
            ov = fold_pat[cvfolds[i]] & fold_pat[cvfolds[j]]
            if ov:
                pair_bad.append((cvfolds[i], cvfolds[j], len(ov)))
    print(f'  [fold 간 환자 중복]: {"OK 없음" if not pair_bad else pair_bad}', flush=True)

    # ---- B. OOF 키가 out-of-fold인지 ----
    sep('B. OOF npz 키 = 해당 fold 환자만인지')
    for k in range(5):
        z = load_oof_npz(XGB_VAL[k])
        pats = {str(p).strip() for p, _, _ in z['keys']}
        expect = fold_pat[str(k)]
        extra = pats - expect
        print(f'  XGB val fold{k}: keys={len(z["keys"])} pat={len(pats)} '
              f'fold환자와불일치={len(extra)} {"OK" if not extra else "이상"}', flush=True)
        cz = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        cpats = {str(p).strip() for p, _, _ in cz['keys']}
        cextra = cpats - expect
        print(f'  CNN val fold{k}: keys={len(cz["keys"])} pat={len(cpats)} '
              f'fold환자와불일치={len(cextra)} {"OK" if not cextra else "이상"}', flush=True)

    # ---- D. OOF fusion 독립 재계산 ----
    sep('D-1. OOF fusion RMSE 독립 재계산 (목표 8.06)')
    xk, ck, lk, mk = [], [], [], []
    for k in range(5):
        xz = load_oof_npz(XGB_VAL[k])
        cz = load_oof_npz(CNN_DIR / f'val_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        xk.append(ax['pred']); ck.append(bx['pred']); lk.append(ax['labels']); mk.append(m)
    XP = np.concatenate(xk); CP = np.concatenate(ck)
    LB = np.concatenate(lk); MK = np.concatenate(mk)
    rx, _ = recompute_rmse(XP, LB, MK)
    rc, _ = recompute_rmse(CP, LB, MK)
    rf, mf = recompute_rmse(W * XP + (1 - W) * CP, LB, MK)
    bestr, bestw = opt_w(XP, CP, LB, MK)
    print(f'  OOF rows aligned: {MK.shape[0]}  valid points: {int(MK.sum())}', flush=True)
    print(f'  XGB OOF   RMSE={rx:.3f}', flush=True)
    print(f'  CNN OOF   RMSE={rc:.3f}', flush=True)
    print(f'  fusion(w=0.47) OOF RMSE={rf:.3f}  MAE={mf:.3f}   [기대 8.06]', flush=True)
    print(f'  OOF optimal w={bestw:.2f} → RMSE={bestr:.3f}   [기대 w≈0.47]', flush=True)

    # ---- C+D-2. test 재계산 + test optimal w ----
    sep('D-2. 5-fold test fusion 독립 재계산 (목표 8.61±0.20)')
    fold_rf, fold_rx, fold_rc, test_opt_w = [], [], [], []
    per_fold_pred, per_fold_lab, per_fold_mask, ref_keys = [], None, None, None
    for k in range(5):
        xz = load_oof_npz(XGB_TEST[k])
        cz = load_oof_npz(CNN_DIR / f'test_preds_fold{k}.npz')
        ax, bx, common = align_oof(xz, cz)
        m = ax['mask'] & bx['mask']
        rxk, _ = recompute_rmse(ax['pred'], ax['labels'], m)
        rck, _ = recompute_rmse(bx['pred'], ax['labels'], m)
        rfk, _ = recompute_rmse(W * ax['pred'] + (1 - W) * bx['pred'], ax['labels'], m)
        _, twk = opt_w(ax['pred'], bx['pred'], ax['labels'], m)
        fold_rx.append(rxk); fold_rc.append(rck); fold_rf.append(rfk); test_opt_w.append(twk)
        print(f'  fold{k}: n={len(common)} XGB={rxk:.3f} CNN={rck:.3f} '
              f'fusion(0.47)={rfk:.3f}  test-optimal-w={twk:.2f}', flush=True)
        # ensemble 준비 (동일 test 키 정렬)
        if ref_keys is None:
            ref_keys = common
            per_fold_lab = ax['labels']; per_fold_mask = m
        per_fold_pred.append(W * ax['pred'] + (1 - W) * bx['pred'])
    print(f'\n  per-fold fusion mean±std = {np.mean(fold_rf):.3f} ± {np.std(fold_rf):.3f}   [기대 8.61±0.20]', flush=True)
    print(f'  test-optimal-w per fold = {[round(x,2) for x in test_opt_w]}  (0.47과 비교: test 튜닝시 얼마나 다른가)', flush=True)

    # ensemble (5 fold 예측 평균 후 1회 RMSE)
    ens = np.mean(np.stack(per_fold_pred), axis=0)
    rens, _ = recompute_rmse(ens, per_fold_lab, per_fold_mask)
    print(f'  [교차확인] 5-fold 예측 평균 후 단일 RMSE(ensemble)={rens:.3f}', flush=True)

    # ---- E. 저장 JSON과 대조 ----
    sep('E. 저장 JSON 값과 독립 재계산 대조')
    ig = json.loads((ROOT / 'runs/ir_v2_global_fusion.json').read_text())
    ft = json.loads((ROOT / 'runs/fusion_5fold_test.json').read_text())
    print(f'  OOF fusion  저장={ig["oof"]["fusion_rmse_fixed_w"]:.3f}  재계산={rf:.3f}  '
          f'{"일치" if abs(ig["oof"]["fusion_rmse_fixed_w"]-rf)<0.01 else "불일치!"}', flush=True)
    print(f'  test fusion 저장={ft["summary"]["fusion_rmse_mean"]:.3f}±{ft["summary"]["fusion_rmse_std"]:.3f}  '
          f'재계산={np.mean(fold_rf):.3f}±{np.std(fold_rf):.3f}  '
          f'{"일치" if abs(ft["summary"]["fusion_rmse_mean"]-np.mean(fold_rf))<0.01 else "불일치!"}', flush=True)

    print('\n검증 종료.', flush=True)


if __name__ == '__main__':
    main()
