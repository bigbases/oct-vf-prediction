#!/usr/bin/env python3
"""신뢰도(reliability) 필터 민감도 분석 — 재학습 없음.

기존 OOF(val 5-fold pooled)·test 예측을 그대로 쓰되,
신뢰안(reliable eyes)만 남겼을 때 헤드라인(fusion>XGB, fusion>CNN)이
유지되는지 확인. 5 백본 전부.

reliability 기준(코호트 cohort_md.csv의 실측 지표):
  - FL  = fixation loss ratio (fix_loss "a/b" -> a/b)
  - FP  = false_pos (SITA %; 원판 수치)
  - low_reliability=='Y' (기존 플래그 ~= FL>=20%)
세 가지 배제 시나리오:
  full   : 배제 없음(현재 논문 상태)
  crA    : low_reliability=='Y' 배제 (FL>=20% 근사, 기존 플래그)
  crB    : FL>=20% OR FP>=15% 배제 (표준 녹내장 기준)
출력: runs/reliability_sensitivity.json
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import align_oof, load_oof_npz  # noqa

W = 0.47
BACKBONES = {
    'inception_resnet_v2': 'runs/phasec_b0_inception_resnet_v2_5fold',
    'inception_v3(clip_mse)': 'runs/phasec_b0_clip_mse_5fold',
    'vgg16': 'runs/phasec_b0_vgg16_5fold',
    'xception': 'runs/phasec_b0_xception_5fold',
    'densenet121': 'runs/phasec_b0_densenet121_5fold',
}
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}


def norm_key(pid, eye, date):
    pid = re.sub(r'\D', '', str(pid))
    eye = str(eye).strip().upper()
    date = re.sub(r'\D', '', str(date))
    return (pid, eye, date)


def load_reliability():
    import csv
    bad_A, bad_B, seen = set(), set(), set()
    fl_missing = 0
    with open(ROOT / 'cohort_md.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            k = norm_key(r['patient_id'], r['eye'], r['vf_date'])
            seen.add(k)
            # FL ratio
            fl = r.get('fix_loss', '')
            flr = None
            m = re.match(r'\s*(\d+)\s*/\s*(\d+)', str(fl))
            if m and int(m.group(2)) > 0:
                flr = int(m.group(1)) / int(m.group(2))
            # FP
            try:
                fp = float(r.get('false_pos', '') or 'nan')
            except ValueError:
                fp = float('nan')
            lowrel = str(r.get('low_reliability', '')).strip().upper() == 'Y'
            if lowrel:
                bad_A.add(k)
            fl_bad = (flr is not None and flr >= 0.20)
            fp_bad = (np.isfinite(fp) and fp >= 15.0)
            if fl_bad or fp_bad:
                bad_B.add(k)
    return bad_A, bad_B, seen


def per_eye_rmse(pred, labels, mask):
    out, keep = [], []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        if m.sum() == 0:
            keep.append(False); continue
        d = pred[i][m] - labels[i][m]
        out.append(np.sqrt(np.mean(d**2))); keep.append(True)
    return np.array(out), np.array(keep, bool)


def pooled(pred, labels, mask):
    m = mask.astype(bool); d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))


def wilcox(a, b):
    """a=fusion, b=solo per-eye rmse. 음수 median diff = fusion 우세."""
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = float('nan')
    return float(p), int(np.sum(a < b)), len(a)


def gather(backbone_dir, split):
    """예측을 (keys, xgb, cnn, labels, mask)로 반환.
    val: 5-fold의 서로소 val을 concat(=OOF 240).
    test: 고정 test set을 5-fold 예측 평균(fold-ensemble)."""
    if split == 'val':
        K, PX, PC, Y, M = [], [], [], [], []
        for k in range(5):
            cnn = load_oof_npz(ROOT / backbone_dir / f'val_preds_fold{k}.npz')
            xgb = load_oof_npz(XGB_VAL[k])
            a, b, common = align_oof(xgb, cnn)
            K += list(common); PX.append(a['pred']); PC.append(b['pred'])
            Y.append(a['labels']); M.append(a['mask'])
        return (K, np.concatenate(PX), np.concatenate(PC),
                np.concatenate(Y), np.concatenate(M))
    # test: 눈별로 5 fold 예측 평균
    acc = {}  # key -> dict(px=[], pc=[], y, m)
    for k in range(5):
        cnn = load_oof_npz(ROOT / backbone_dir / f'test_preds_fold{k}.npz')
        xgb = load_oof_npz(XGB_TEST[k])
        a, b, common = align_oof(xgb, cnn)
        for i, key in enumerate(common):
            e = acc.setdefault(key, {'px': [], 'pc': [], 'y': a['labels'][i], 'm': a['mask'][i]})
            e['px'].append(a['pred'][i]); e['pc'].append(b['pred'][i])
    K = sorted(acc)
    PX = np.stack([np.mean(acc[k]['px'], axis=0) for k in K])
    PC = np.stack([np.mean(acc[k]['pc'], axis=0) for k in K])
    Y = np.stack([acc[k]['y'] for k in K])
    M = np.stack([acc[k]['m'] for k in K])
    return (K, PX, PC, Y, M)


def analyze_split(backbone_dir, split, bad_A, bad_B):
    K, PX, PC, Y, M = gather(backbone_dir, split)
    PF = W * PX + (1 - W) * PC
    keys_norm = [norm_key(*k) for k in K]
    res = {}
    for name, bad in [('full', set()), ('crA_FL', bad_A), ('crB_FL_or_FP', bad_B)]:
        sel = np.array([kn not in bad for kn in keys_norm], bool)
        n = int(sel.sum())
        rmse_x, mae_x = pooled(PX[sel], Y[sel], M[sel])
        rmse_c, mae_c = pooled(PC[sel], Y[sel], M[sel])
        rmse_f, mae_f = pooled(PF[sel], Y[sel], M[sel])
        ef, keep = per_eye_rmse(PF[sel], Y[sel], M[sel])
        ex, _ = per_eye_rmse(PX[sel], Y[sel], M[sel])
        ec, _ = per_eye_rmse(PC[sel], Y[sel], M[sel])
        p_fx, win_fx, ne = wilcox(ef, ex)
        p_fc, win_fc, _ = wilcox(ef, ec)
        res[name] = {
            'n_eyes': n, 'n_excluded': int((~sel).sum()),
            'rmse': {'xgb': round(rmse_x, 3), 'cnn': round(rmse_c, 3), 'fusion': round(rmse_f, 3)},
            'mae': {'xgb': round(mae_x, 3), 'cnn': round(mae_c, 3), 'fusion': round(mae_f, 3)},
            'fusion_vs_xgb': {'wilcoxon_p': p_fx, 'wins': win_fx, 'n': ne},
            'fusion_vs_cnn': {'wilcoxon_p': p_fc, 'wins': win_fc, 'n': ne},
        }
    return res


def main():
    bad_A, bad_B, seen = load_reliability()
    out = {'W': W, 'criteria': {
        'crA_FL': 'low_reliability==Y (~FL>=20%)',
        'crB_FL_or_FP': 'FL>=20% OR FP>=15%'},
        'n_bad_A_total': len(bad_A), 'n_bad_B_total': len(bad_B),
        'backbones': {}}
    for bb, d in BACKBONES.items():
        out['backbones'][bb] = {
            'OOF': analyze_split(d, 'val', bad_A, bad_B),
            'test': analyze_split(d, 'test', bad_A, bad_B),
        }
    (ROOT / 'runs/reliability_sensitivity.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    # 요약 출력
    print(f"reliability bad keys: A(FL)={len(bad_A)}  B(FL|FP)={len(bad_B)}  (cohort_md {len(seen)} rows)\n")
    hdr = f"{'backbone':22s} {'set':4s} {'crit':13s} {'n':>4s} {'XGB':>6s} {'CNN':>6s} {'FUS':>6s}  {'p(f>XGB)':>10s} {'p(f>CNN)':>10s}"
    print(hdr); print('-'*len(hdr))
    for bb in BACKBONES:
        for split in ('OOF', 'test'):
            for cr in ('full', 'crA_FL', 'crB_FL_or_FP'):
                r = out['backbones'][bb][split][cr]
                print(f"{bb:22s} {split:4s} {cr:13s} {r['n_eyes']:4d} "
                      f"{r['rmse']['xgb']:6.2f} {r['rmse']['cnn']:6.2f} {r['rmse']['fusion']:6.2f}  "
                      f"{r['fusion_vs_xgb']['wilcoxon_p']:10.2e} {r['fusion_vs_cnn']['wilcoxon_p']:10.2e}")
        print()


if __name__ == '__main__':
    main()
