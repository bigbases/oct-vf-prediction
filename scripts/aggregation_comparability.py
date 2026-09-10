#!/usr/bin/env python3
"""
선행연구와의 비교 가능성 확보: 동일 예측을 두 집계 방식으로 보고한다.

- pooled       : 전 지점을 모아 한 번에 RMSE/MAE (본 연구 기존 방식)
- per-eye mean : 안별 RMSE/MAE를 구한 뒤 안에 걸쳐 mean ± SD
                 (Park 2020 4.44±2.09, Shin 2021 5.29±2.68, Abdullahi 2026 3.32±2.35 방식)

Jensen 부등식으로 pooled >= per-eye mean 이 항상 성립하므로,
두 방식을 섞어 비교하면 본 연구가 부당하게 나빠 보인다.

출력: runs/aggregation_comparability.json
실행: python scripts/aggregation_comparability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "runs" / "oof"
BACKBONES = {
    "inception_v3": "Inception-v3",
    "inception_resnet_v2": "IR-v2",
    "vgg16": "VGG16",
    "xception": "Xception",
    "densenet121": "DenseNet121",
}
DIR_NAME = {"inception_v3": "clip_mse"}
W = {"inception_v3": 0.60, "inception_resnet_v2": 0.47,
     "vgg16": 0.63, "xception": 0.43, "densenet121": 0.50}


def key_of(p, e, v):
    return (str(p), str(e), str(v))


def _gather(paths):
    acc, lab, msk = {}, {}, {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for i in range(len(z["pred"])):
            k = key_of(z["patient_id"][i], z["eye"][i], z["vf_date"][i])
            acc.setdefault(k, []).append(z["pred"][i])
            lab[k], msk[k] = z["labels"][i], z["mask"][i]
    return {k: (np.mean(v, axis=0), lab[k], msk[k]) for k, v in acc.items()}


def load_xgb(split):
    if split == "val":
        return _gather([OOF / f"xgb_90d_fold{k}_val.npz" for k in range(5)])
    return _gather([OOF / f"xgb_90d_fold{k}_test.npz" for k in range(5)])


def load_cnn(bb, split):
    d = ROOT / "runs" / f"phasec_b0_{DIR_NAME.get(bb, bb)}_5fold"
    tag = "val" if split == "val" else "test"
    return _gather([d / f"{tag}_preds_fold{k}.npz" for k in range(5)])


def stats(pred, y, m):
    """pooled 와 per-eye mean±SD 를 함께 계산."""
    err = pred - y
    pooled_rmse = float(np.sqrt((err[m] ** 2).mean()))
    pooled_mae = float(np.abs(err[m]).mean())
    eye_rmse = np.array([float(np.sqrt((err[i][m[i]] ** 2).mean())) for i in range(len(err))])
    eye_mae = np.array([float(np.abs(err[i][m[i]]).mean()) for i in range(len(err))])
    n_pts = np.array([int(m[i].sum()) for i in range(len(err))])
    # 모든 안이 동일 지점수(52)를 가지면 pooled = sqrt(mean^2 + sd_pop^2) 가 항등식으로 성립.
    identity = float(np.sqrt(eye_rmse.mean() ** 2 + eye_rmse.std(ddof=0) ** 2))
    return {
        "pooled_rmse": round(pooled_rmse, 3),
        "pooled_mae": round(pooled_mae, 3),
        "eye_rmse_mean": round(float(eye_rmse.mean()), 3),
        "eye_rmse_sd": round(float(eye_rmse.std(ddof=1)), 3),
        "eye_rmse_sd_pop": round(float(eye_rmse.std(ddof=0)), 3),
        "eye_mae_mean": round(float(eye_mae.mean()), 3),
        "eye_mae_sd": round(float(eye_mae.std(ddof=1)), 3),
        "n_points_per_eye_unique": sorted(set(n_pts.tolist())),
        "identity_sqrt_mean2_plus_sdpop2": round(identity, 3),
        "identity_abs_err_vs_pooled": round(abs(identity - pooled_rmse), 6),
        "gap_pooled_minus_eyemean_rmse": round(pooled_rmse - float(eye_rmse.mean()), 3),
    }


res = {
    "_note": "동일 예측을 pooled / per-eye mean±SD 두 방식으로 집계. "
             "선행연구(Park·Shin·Abdullahi)는 per-eye mean±SD 방식이므로 그쪽 열로 비교할 것.",
    "_jensen": "pooled >= per-eye mean (Jensen). 섞어 비교하면 본 연구가 부당하게 나빠 보인다.",
    "models": {},
}

for split, tag in (("val", "OOF"), ("test", "TEST")):
    xg = load_xgb(split)
    cnns = {bb: load_cnn(bb, split) for bb in BACKBONES}
    # 모든 모델이 동일 안 집합에서 평가되도록 교집합으로 고정 (헤드라인 수치와 동일 기준)
    ks = sorted(set(xg).intersection(*[set(c) for c in cnns.values()]))
    PX = np.array([xg[k][0] for k in ks])
    Y = np.array([xg[k][1] for k in ks])
    M = np.array([xg[k][2] for k in ks]).astype(bool)

    per_bb = {}
    for bb, cn in cnns.items():
        PC = np.array([cn[k][0] for k in ks])
        w = W[bb]
        per_bb[bb] = {
            "label": BACKBONES[bb],
            "cnn": stats(PC, Y, M),
            "fusion": stats(w * PX + (1 - w) * PC, Y, M),
        }
    res["models"][tag] = {
        "n_eyes": len(ks),
        "xgb": stats(PX, Y, M),
        "backbones": per_bb,
    }

# ------------------------------------------------ 선행연구 대조표 (IR-v2 기준)
def ref_row(tag):
    d = res["models"][tag]
    ir = d["backbones"]["inception_resnet_v2"]
    return {
        "xgb": d["xgb"],
        "cnn_ir_v2": ir["cnn"],
        "fusion_ir_v2": ir["fusion"],
    }


LIT = [
    ("Park 2020",      "RMSE", "IR-v2 / SS-OCT / test 305안", 4.44, 2.09),
    ("Shin 2021",      "RMSE", "SD-OCT",                       5.29, 2.68),
    ("Shin 2021",      "RMSE", "SS-OCT",                       4.51, 2.54),
    ("Abdullahi 2026", "MAE",  "제안모델",                      3.32, 2.35),
    ("Abdullahi 2026", "MAE",  "재학습 IR-v2 baseline",         5.77, 2.01),
    ("Abdullahi 2026", "MAE",  "재학습 InceptionV3 baseline",   6.70, 2.15),
]
res["comparison_ready"] = {
    "OOF": ref_row("OOF"),
    "TEST": ref_row("TEST"),
    "literature_axis_conversion": [
        {
            "study": s, "metric": met, "setting": note,
            "reported_per_eye_mean_sd": f"{mu:.2f} ± {sd:.2f}",
            # 안별 RMSE가 mean±SD로 보고되면 pooled RMSE = sqrt(mean^2 + SD^2) (동일 지점수 가정).
            # MAE는 이 변환이 성립하지 않으므로 참고용 상한이 아니라 '비교 불가' 표시.
            "pooled_equivalent_rmse": (round((mu ** 2 + sd ** 2) ** 0.5, 2)
                                       if met == "RMSE" else None),
            "note": ("안별 RMSE → pooled 환산 가능" if met == "RMSE"
                     else "MAE는 pooled와 안별평균이 동일값이므로 환산 불필요 (본 연구 eye_mae_mean과 직접 비교)"),
        }
        for s, met, note, mu, sd in LIT
    ],
    "_caution": "위 문헌값은 사용자 2차보고 기반. 1차출처 재확인 전 원고 인용 금지.",
}


# ------------------------------------------------ 격차 분해: 집계방식이 얼마나 설명하는가
def decompose(study, mu, sd, ours):
    """문헌 안별평균 mu±sd 대비, 본 연구 격차를 축별로 분해."""
    lit_pooled = (mu ** 2 + sd ** 2) ** 0.5
    naive = ours["pooled_rmse"] - mu                    # 축이 섞인 잘못된 비교
    same_eye = ours["eye_rmse_mean"] - mu               # 둘 다 안별평균
    same_pooled = ours["pooled_rmse"] - lit_pooled      # 둘 다 pooled
    return {
        "study": study,
        "lit_eye_mean_sd": f"{mu:.2f} ± {sd:.2f}",
        "lit_pooled_equiv": round(lit_pooled, 2),
        "gap_naive_mixed_axis": round(naive, 2),
        "gap_same_axis_eye_mean": round(same_eye, 2),
        "gap_same_axis_pooled": round(same_pooled, 2),
        "explained_by_aggregation_dB": [round(naive - same_pooled, 2),
                                        round(naive - same_eye, 2)],
        "explained_by_aggregation_pct": [round(100 * (naive - same_pooled) / naive),
                                         round(100 * (naive - same_eye) / naive)],
        "our_eye_rmse_sd_vs_lit_sd": [ours["eye_rmse_sd"], sd],
    }


_oof_fus = res["models"]["OOF"]["backbones"]["inception_resnet_v2"]["fusion"]
res["gap_decomposition_vs_literature"] = {
    "_basis": "본 연구 OOF late fusion (IR-v2): pooled RMSE "
              f"{_oof_fus['pooled_rmse']}, 안별 RMSE {_oof_fus['eye_rmse_mean']} ± {_oof_fus['eye_rmse_sd']}",
    "_read": "explained_by_aggregation_* 는 [pooled축 기준, 안별평균축 기준] 두 값. "
             "집계방식이 설명하는 몫은 일부일 뿐 격차 대부분은 남는다 — 과대해석 금지.",
    "rows": [decompose("Park 2020 (SS-OCT, IR-v2)", 4.44, 2.09, _oof_fus),
             decompose("Shin 2021 (SD-OCT)", 5.29, 2.68, _oof_fus),
             decompose("Shin 2021 (SS-OCT)", 4.51, 2.54, _oof_fus)],
    "MAE_axis_note": {
        "fact": "모든 안의 유효지점이 52로 동일하므로 pooled MAE = 안별 MAE 평균 (정확히 일치).",
        "implication": "MAE로 보고된 선행연구(Abdullahi, Kihara PMAE)와의 비교에는 "
                       "집계방식 보정이 아예 필요 없다. 이 경로는 변명 없이 그대로 성립.",
        "our_oof_fusion_mae": _oof_fus["pooled_mae"],
        "our_oof_cnn_irv2_mae": res["models"]["OOF"]["backbones"]["inception_resnet_v2"]["cnn"]["pooled_mae"],
    },
}

print(json.dumps(res, ensure_ascii=False, indent=1))
(ROOT / "runs" / "aggregation_comparability.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nwrote runs/aggregation_comparability.json")
