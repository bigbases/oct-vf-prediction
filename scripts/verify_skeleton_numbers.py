#!/usr/bin/env python3
"""Recompute every number quoted in the manuscript from the raw npz and csv
outputs, and write the confirmed values to runs/skeleton_numbers.json.

스켈레톤/원고에 인용된 모든 수치를 raw 산출물(npz, csv)에서 독립 재계산하여 검증한다.

출력: runs/skeleton_numbers.json  (문서에 인용할 확정 수치)
실행: python scripts/verify_skeleton_numbers.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "runs" / "oof"
BACKBONES = {
    "inception_v3": "Inception-v3",
    "inception_resnet_v2": "IR-v2",
    "vgg16": "VGG16",
    "xception": "Xception",
    "densenet121": "DenseNet121",
}
# Inception-v3 run은 backbone 인자 기본값(None)으로 돌아 디렉토리명이 clip_mse
DIR_NAME = {"inception_v3": "clip_mse"}
CNN_DIR = {b: ROOT / "runs" / f"phasec_b0_{DIR_NAME.get(b, b)}_5fold" for b in BACKBONES}
W = {"inception_v3": 0.60, "inception_resnet_v2": 0.47,
     "vgg16": 0.63, "xception": 0.43, "densenet121": 0.50}


def key_of(pid, eye, vfd):
    return (str(pid), str(eye), str(vfd))


def load_xgb(split):
    """val -> OOF(5 fold 이어붙임, 각 안 1회).
    test -> 5개 fold 모델 예측의 평균(프로토콜 §2.7: test는 fold 모델 앙상블)."""
    if split == "val":
        out = {}
        for k in range(5):
            z = np.load(OOF / f"xgb_90d_fold{k}_val.npz", allow_pickle=True)
            for i in range(len(z["pred"])):
                out[key_of(z["patient_id"][i], z["eye"][i], z["vf_date"][i])] = (
                    z["pred"][i], z["labels"][i], z["mask"][i])
        return out
    acc, lab, msk = {}, {}, {}
    for k in range(5):
        z = np.load(OOF / f"xgb_90d_fold{k}_test.npz", allow_pickle=True)
        for i in range(len(z["pred"])):
            kk = key_of(z["patient_id"][i], z["eye"][i], z["vf_date"][i])
            acc.setdefault(kk, []).append(z["pred"][i])
            lab[kk], msk[kk] = z["labels"][i], z["mask"][i]
    return {k: (np.mean(v, axis=0), lab[k], msk[k]) for k, v in acc.items()}


def load_cnn(backbone, split):
    """CNN val=OOF 이어붙임; test=5 fold 예측 평균(동일 37안)."""
    d = CNN_DIR[backbone]
    if split == "val":
        out = {}
        for k in range(5):
            z = np.load(d / f"val_preds_fold{k}.npz", allow_pickle=True)
            for i in range(len(z["pred"])):
                out[key_of(z["patient_id"][i], z["eye"][i], z["vf_date"][i])] = (
                    z["pred"][i], z["labels"][i], z["mask"][i])
        return out
    acc, lab, msk = {}, {}, {}
    for k in range(5):
        z = np.load(d / f"test_preds_fold{k}.npz", allow_pickle=True)
        for i in range(len(z["pred"])):
            kk = key_of(z["patient_id"][i], z["eye"][i], z["vf_date"][i])
            acc.setdefault(kk, []).append(z["pred"][i])
            lab[kk], msk[kk] = z["labels"][i], z["mask"][i]
    return {k: (np.mean(v, axis=0), lab[k], msk[k]) for k, v in acc.items()}


def align(a, b):
    ks = sorted(set(a) & set(b))
    P = np.array([a[k][0] for k in ks])
    Q = np.array([b[k][0] for k in ks])
    Y = np.array([a[k][1] for k in ks])
    M = np.array([a[k][2] for k in ks]).astype(bool)
    return ks, P, Q, Y, M


def pooled(pred, y, m):
    e = (pred - y)[m]
    return float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean())


def eye_rmse(pred, y, m):
    return np.array([float(np.sqrt((((pred[i] - y[i]) ** 2)[m[i]]).mean()))
                     for i in range(len(pred))])


def wil(a, b):
    """a<b 이면 음의 Δ. 양측 p."""
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


res = {"_note": "raw npz/csv에서 독립 재계산. 낮을수록 우수(RMSE/MAE, dB).",
       "backbones": {}, "checks": []}

# ---------------------------------------------------------------- 백본별 전 수치
for bb, label in BACKBONES.items():
    row = {"label": label, "w_xgb": W[bb]}
    for split in ("val", "test"):
        xg, cn = load_xgb(split), load_cnn(bb, split)
        ks, PX, PC, Y, M = align(xg, cn)
        w = W[bb]
        PF = w * PX + (1 - w) * PC
        rx, mx = pooled(PX, Y, M)
        rc, mc = pooled(PC, Y, M)
        rf, mf = pooled(PF, Y, M)
        ex, ec, ef = eye_rmse(PX, Y, M), eye_rmse(PC, Y, M), eye_rmse(PF, Y, M)
        row["oof" if split == "val" else "test"] = {
            "n_eyes": len(ks),
            "n_points": int(M.sum()),
            "xgb": {"rmse": round(rx, 3), "mae": round(mx, 3)},
            "cnn": {"rmse": round(rc, 3), "mae": round(mc, 3)},
            "fusion": {"rmse": round(rf, 3), "mae": round(mf, 3)},
            "delta_fus_minus_xgb_rmse": round(rf - rx, 3),
            "delta_fus_minus_cnn_rmse": round(rf - rc, 3),
            "p_fus_vs_xgb": float(f"{wil(ef, ex):.3g}"),
            "p_fus_vs_cnn": float(f"{wil(ef, ec):.3g}"),
            "win_fus_vs_xgb": f"{int((ef < ex).sum())}/{len(ks)}",
            "win_fus_vs_cnn": f"{int((ef < ec).sum())}/{len(ks)}",
        }
    res["backbones"][bb] = row

# ---------------------------------------------------------------- 앙상블(5백본 CNN 평균)
ens = {}
for split in ("val", "test"):
    xg = load_xgb(split)
    cnns = [load_cnn(b, split) for b in BACKBONES]
    ks = sorted(set(xg) & set.intersection(*[set(c) for c in cnns]))
    PX = np.array([xg[k][0] for k in ks])
    PC = np.mean([[c[k][0] for k in ks] for c in cnns], axis=0)
    Y = np.array([xg[k][1] for k in ks])
    M = np.array([xg[k][2] for k in ks]).astype(bool)
    PF = 0.47 * PX + 0.53 * PC
    rx, mx = pooled(PX, Y, M); rc, mc = pooled(PC, Y, M); rf, mf = pooled(PF, Y, M)
    ex, ec, ef = eye_rmse(PX, Y, M), eye_rmse(PC, Y, M), eye_rmse(PF, Y, M)
    ens["oof" if split == "val" else "test"] = {
        "n_eyes": len(ks),
        "ens_cnn": {"rmse": round(rc, 3), "mae": round(mc, 3)},
        "xgb": {"rmse": round(rx, 3), "mae": round(mx, 3)},
        "ens_fusion_w047": {"rmse": round(rf, 3), "mae": round(mf, 3)},
        "p_fus_vs_ensCNN": float(f"{wil(ef, ec):.3g}"),
        "win_fus_vs_ensCNN": f"{int((ef < ec).sum())}/{len(ks)}",
        "p_fus_vs_xgb": float(f"{wil(ef, ex):.3g}"),
        "win_fus_vs_xgb": f"{int((ef < ex).sum())}/{len(ks)}",
    }
res["ensemble_5backbone"] = ens

# ---------------------------------------------------------------- 환자 단위 (IR-v2)
pat = {}
for split in ("val", "test"):
    xg, cn = load_xgb(split), load_cnn("inception_resnet_v2", split)
    ks, PX, PC, Y, M = align(xg, cn)
    PF = 0.47 * PX + 0.53 * PC
    ex, ec, ef = eye_rmse(PX, Y, M), eye_rmse(PC, Y, M), eye_rmse(PF, Y, M)
    pids = np.array([k[0] for k in ks])
    up = sorted(set(pids))
    gx = np.array([ex[pids == p].mean() for p in up])
    gc = np.array([ec[pids == p].mean() for p in up])
    gf = np.array([ef[pids == p].mean() for p in up])
    pat["oof" if split == "val" else "test"] = {
        "n_eyes": len(ks), "n_patients": len(up),
        "p_fus_vs_xgb": float(f"{wil(gf, gx):.3g}"),
        "win_fus_vs_xgb": f"{int((gf < gx).sum())}/{len(up)}",
        "p_fus_vs_cnn": float(f"{wil(gf, gc):.3g}"),
        "win_fus_vs_cnn": f"{int((gf < gc).sum())}/{len(up)}",
        "mean_diff_fus_minus_xgb": round(float(gf.mean() - gx.mean()), 3),
    }
res["patient_level_ir_v2"] = pat

# ---------------------------------------------------------------- 코호트/라벨 (CSV)
PT = [f"p{i:02d}" for i in range(1, 55)]
USE = [p for p in PT if p not in ("p26", "p35")]


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


cur = read_csv(ROOT / "ml_final_90d_excl_empty_flip.csv")
old = read_csv(ROOT / "runs/backup_vf_neg1_fix_20260530_193547/ml_final_90d_excl_empty_flip.csv")


def cells(rows):
    return np.array([[float(r[p]) if str(r[p]).strip() != "" else np.nan
                      for p in USE] for r in rows])


C, O = cells(cur), cells(old)
res["labels"] = {
    "n_eyes": len(cur),
    "n_patients": len({r["patient_id"] for r in cur}),
    "n_cells_52pt": int(C.size),
    "n_zero_current": int((C == 0).sum()),
    "pct_zero_current": round((C == 0).sum() / C.size * 100, 2),
    "n_from_minus1": int((O == -1).sum()),
    "pct_from_minus1": round((O == -1).sum() / C.size * 100, 2),
    "n_true_zero_before_rule": int((O == 0).sum()),
    "pct_true_zero_before_rule": round((O == 0).sum() / C.size * 100, 2),
    "share_of_zeros_from_minus1_pct": round((O == -1).sum() / (C == 0).sum() * 100, 1),
    "mean_db": round(float(np.nanmean(C)), 2),
    "median_db": float(np.nanmedian(C)),
    "pct_ge30": round(float((C >= 30).sum() / C.size * 100), 1),
    "pct_lt10": round(float((C < 10).sum() / C.size * 100), 1),
}

# 코호트 중증도(EMR MD)
md_rows = read_csv(ROOT / "cohort_md.csv")
mdmap = {(r["patient_id"], r["eye"], r["vf_date"]): r["MD"] for r in md_rows}
mds = []
for r in cur:
    v = mdmap.get((r["patient_id"], r["eye"], r["vf_date"]), "")
    if str(v).strip() not in ("", "nan"):
        mds.append(float(v))
mds = np.array(mds)
res["cohort_severity_EMR_MD"] = {
    "n_with_md": int(len(mds)), "n_missing": len(cur) - len(mds),
    "mean": round(float(mds.mean()), 2), "sd": round(float(mds.std(ddof=1)), 2),
    "median": float(np.median(mds)), "min": float(mds.min()), "max": float(mds.max()),
    "normal_MD_gt_-3": [int((mds > -3).sum()), round(float((mds > -3).mean() * 100), 1)],
    "early_-6_to_-3": [int(((mds > -6) & (mds <= -3)).sum()),
                       round(float(((mds > -6) & (mds <= -3)).mean() * 100), 1)],
    "moderate_-12_to_-6": [int(((mds > -12) & (mds <= -6)).sum()),
                           round(float(((mds > -12) & (mds <= -6)).mean() * 100), 1)],
    "advanced_le_-12": [int((mds <= -12).sum()), round(float((mds <= -12).mean() * 100), 1)],
}

# gap days
gaps = np.array([float(r["gap_days"]) for r in cur])
res["oct_vf_gap_days"] = {
    "mean": round(float(gaps.mean()), 2), "sd": round(float(gaps.std(ddof=1)), 2),
    "median": float(np.median(gaps)), "max": float(gaps.max()),
    "pct_same_day": round(float((gaps == 0).mean() * 100), 1),
}

# split 구조
folds = {}
for r in cur:
    folds[r["cv_fold"]] = folds.get(r["cv_fold"], 0) + 1
res["split_structure"] = {
    "cv_fold_eye_counts": dict(sorted(folds.items())),
    "n_eyes_total": len(cur),
    "note": "cv_fold는 280안 전체에 환자 단위로 배정(split-first) 후 양 모달 존재 안만 필터(filter-second)",
}

print(json.dumps(res, ensure_ascii=False, indent=1))
(ROOT / "runs" / "skeleton_numbers.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nwrote runs/skeleton_numbers.json")
