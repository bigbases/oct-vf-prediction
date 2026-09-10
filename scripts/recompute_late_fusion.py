#!/usr/bin/env python3
"""새 CNN/XGB 5-fold NPZ에서 primary late-fusion 결과를 독립 재계산한다.

기존 논문 OOF 수치는 전체 OOF에서 weight를 정하고 같은 OOF에서 표시한
development estimate다. 이 수치와 함께, 각 validation fold를 제외한 나머지
네 fold에서 weight를 정하는 nested OOF sensitivity도 계산한다. Held-out
성능은 전체 OOF에서 정한 weight와 각 branch의 다섯 fold 평균 prediction을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from oof_common import load_oof_npz, save_oof_npz  # noqa: E402


def align_verified(xgb: dict, cnn: dict) -> dict:
    ix = {tuple(map(str, key)): i for i, key in enumerate(xgb["keys"])}
    ic = {tuple(map(str, key)): i for i, key in enumerate(cnn["keys"])}
    keys = sorted(set(ix) & set(ic))
    if not keys:
        raise ValueError("XGB/CNN 공통 행이 없다")
    xi = [ix[key] for key in keys]
    ci = [ic[key] for key in keys]
    labels_x = xgb["labels"][xi]
    labels_c = cnn["labels"][ci]
    mask_x = xgb["mask"][xi].astype(bool)
    mask_c = cnn["mask"][ci].astype(bool)
    if not np.array_equal(labels_x, labels_c, equal_nan=True):
        raise ValueError("정렬된 XGB/CNN label이 다르다")
    if not np.array_equal(mask_x, mask_c):
        raise ValueError("정렬된 XGB/CNN mask가 다르다")
    return {
        "keys": keys,
        "xgb": xgb["pred"][xi],
        "cnn": cnn["pred"][ci],
        "labels": labels_x,
        "mask": mask_x,
    }


def pooled_metrics(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    diff = (pred - labels)[mask]
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "n_points": int(diff.size),
    }


def eye_rmse(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.sqrt(np.mean((pred[i][mask[i]] - labels[i][mask[i]]) ** 2)))
            for i in range(len(pred))
        ]
    )


def patient_rmse(
    pred: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    keys: list[tuple[str, str, str]],
) -> tuple[np.ndarray, list[str]]:
    patient_ids = sorted({key[0] for key in keys})
    per_eye = eye_rmse(pred, labels, mask)
    out = []
    for patient_id in patient_ids:
        rows = np.array([key[0] == patient_id for key in keys])
        out.append(float(np.mean(per_eye[rows])))
    return np.asarray(out), patient_ids


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float | None:
    delta = a - b
    if len(delta) == 0 or np.allclose(delta, 0):
        return None
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


def fit_weight(folds: list[dict]) -> float:
    xgb = np.concatenate([fold["xgb"][fold["mask"]] for fold in folds])
    cnn = np.concatenate([fold["cnn"][fold["mask"]] for fold in folds])
    labels = np.concatenate([fold["labels"][fold["mask"]] for fold in folds])
    best_rmse = float("inf")
    best_weight = 0.0
    for weight in np.linspace(0.0, 1.0, 101):
        pred = weight * xgb + (1.0 - weight) * cnn
        rmse = float(np.sqrt(np.mean((pred - labels) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight


def load_fold(
    cnn_dir: Path,
    xgb_dir: Path,
    xgb_tag: str,
    fold: int,
    split: str,
) -> dict:
    xgb_path = xgb_dir / f"xgb_{xgb_tag}_fold{fold}_{split}.npz"
    cnn_path = cnn_dir / f"{split}_preds_fold{fold}.npz"
    return align_verified(load_oof_npz(xgb_path), load_oof_npz(cnn_path))


def stack_nested_oof(oof: dict[int, dict]) -> tuple[dict, list[float]]:
    keys: list[tuple[str, str, str]] = []
    fusion, xgb, cnn, labels, masks = [], [], [], [], []
    weights = []
    for fold in range(5):
        weight = fit_weight([oof[k] for k in range(5) if k != fold])
        data = oof[fold]
        weights.append(weight)
        keys.extend(data["keys"])
        fusion.append(weight * data["xgb"] + (1.0 - weight) * data["cnn"])
        xgb.append(data["xgb"])
        cnn.append(data["cnn"])
        labels.append(data["labels"])
        masks.append(data["mask"])
    if len(keys) != len(set(keys)):
        raise ValueError("OOF validation key가 fold 사이에서 중복된다")
    return {
        "keys": keys,
        "fusion": np.concatenate(fusion),
        "xgb": np.concatenate(xgb),
        "cnn": np.concatenate(cnn),
        "labels": np.concatenate(labels),
        "mask": np.concatenate(masks),
    }, weights


def global_oof_from_nested_base(nested_oof: dict, weight: float) -> dict:
    return {
        **nested_oof,
        "fusion": (
            weight * nested_oof["xgb"] + (1.0 - weight) * nested_oof["cnn"]
        ),
    }


def average_held_out(test: dict[int, dict], weight: float) -> dict:
    reference = test[0]
    keys = reference["keys"]
    for fold in range(1, 5):
        current = test[fold]
        if current["keys"] != keys:
            raise ValueError(f"held-out key 순서가 fold0과 fold{fold}에서 다르다")
        if not np.array_equal(current["labels"], reference["labels"], equal_nan=True):
            raise ValueError(f"held-out label이 fold0과 fold{fold}에서 다르다")
        if not np.array_equal(current["mask"], reference["mask"]):
            raise ValueError(f"held-out mask가 fold0과 fold{fold}에서 다르다")
    xgb = np.mean(np.stack([test[k]["xgb"] for k in range(5)]), axis=0)
    cnn = np.mean(np.stack([test[k]["cnn"] for k in range(5)]), axis=0)
    return {
        "keys": keys,
        "fusion": weight * xgb + (1.0 - weight) * cnn,
        "xgb": xgb,
        "cnn": cnn,
        "labels": reference["labels"],
        "mask": reference["mask"],
    }


def summarize(data: dict) -> dict:
    result = {
        name: pooled_metrics(data[name], data["labels"], data["mask"])
        for name in ("fusion", "xgb", "cnn")
    }
    eye = {
        name: eye_rmse(data[name], data["labels"], data["mask"])
        for name in ("fusion", "xgb", "cnn")
    }
    patient = {}
    patient_ids = None
    for name in ("fusion", "xgb", "cnn"):
        patient[name], ids = patient_rmse(
            data[name], data["labels"], data["mask"], data["keys"]
        )
        patient_ids = ids if patient_ids is None else patient_ids
        if ids != patient_ids:
            raise ValueError("patient-level aggregation 순서 불일치")
    result["n_eyes"] = len(data["keys"])
    result["n_patients"] = len(patient_ids or [])
    result["paired_tests"] = {
        "eye_wilcoxon_fusion_vs_xgb": paired_wilcoxon(eye["fusion"], eye["xgb"]),
        "eye_wilcoxon_fusion_vs_cnn": paired_wilcoxon(eye["fusion"], eye["cnn"]),
        "patient_wilcoxon_fusion_vs_xgb": paired_wilcoxon(
            patient["fusion"], patient["xgb"]
        ),
        "patient_wilcoxon_fusion_vs_cnn": paired_wilcoxon(
            patient["fusion"], patient["cnn"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-dir", required=True)
    parser.add_argument("--xgb-dir", required=True)
    parser.add_argument("--xgb-tag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cnn_dir = Path(args.cnn_dir)
    xgb_dir = Path(args.xgb_dir)
    out_dir = Path(args.out_dir)
    for path_name, value in (
        ("cnn-dir", cnn_dir),
        ("xgb-dir", xgb_dir),
        ("out-dir", out_dir),
    ):
        if not value.is_absolute():
            value = ROOT / value
        if path_name == "cnn-dir":
            cnn_dir = value
        elif path_name == "xgb-dir":
            xgb_dir = value
        else:
            out_dir = value

    out_json = out_dir / "late_fusion_recomputed.json"
    if out_json.exists() and not args.overwrite:
        raise SystemExit(f"{out_json}이 이미 존재한다. 새 경로 또는 --overwrite를 사용하라.")
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = {
        fold: load_fold(cnn_dir, xgb_dir, args.xgb_tag, fold, "val")
        for fold in range(5)
    }
    test = {
        fold: load_fold(cnn_dir, xgb_dir, args.xgb_tag, fold, "test")
        for fold in range(5)
    }
    nested_oof, nested_weights = stack_nested_oof(oof)
    global_weight = fit_weight([oof[fold] for fold in range(5)])
    apparent_oof = global_oof_from_nested_base(nested_oof, global_weight)
    held_out = average_held_out(test, global_weight)

    result = {
        "protocol": {
            "oof_apparent": "global weight fitted and displayed on the same OOF predictions",
            "oof_nested_sensitivity": (
                "each fold weight fitted on the other four folds"
            ),
            "held_out": "global OOF weight; branch predictions averaged across five folds",
            "weight_grid": "0.00 to 1.00 by 0.01; weight applies to XGB",
            "cnn_dir": str(cnn_dir),
            "xgb_dir": str(xgb_dir),
            "xgb_tag": args.xgb_tag,
        },
        "weights": {
            "nested_oof_xgb": nested_weights,
            "global_xgb": global_weight,
        },
        "oof_apparent": summarize(apparent_oof),
        "oof_nested_sensitivity": summarize(nested_oof),
        "held_out": summarize(held_out),
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "protocol": result["protocol"],
        "weights": result["weights"],
    }
    for split, data in (
        ("oof_apparent", apparent_oof),
        ("oof_nested", nested_oof),
        ("held_out", held_out),
    ):
        save_oof_npz(
            out_dir / f"late_fusion_{split}.npz",
            data["keys"],
            data["fusion"],
            data["labels"],
            data["mask"],
            {**meta, "split": split},
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
