#!/usr/bin/env python3
"""논문용 XGB/CNN 예측 NPZ의 키·라벨·마스크·형상 무결성을 전수검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = ROOT / "runs" / "oof"
CNN_DIRS = {
    "inception_v3": ROOT / "runs" / "phasec_b0_clip_mse_5fold",
    "inception_resnet_v2": ROOT / "runs" / "phasec_b0_inception_resnet_v2_5fold",
    "vgg16": ROOT / "runs" / "phasec_b0_vgg16_5fold",
    "xception": ROOT / "runs" / "phasec_b0_xception_5fold",
    "densenet121": ROOT / "runs" / "phasec_b0_densenet121_5fold",
}
N_POINTS = 52


def key_of(pid: object, eye: object, vf_date: object) -> tuple[str, str, str]:
    return str(pid), str(eye), str(vf_date)


def load_bundle(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    required = {"pred", "labels", "mask", "patient_id", "eye", "vf_date"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"missing arrays: {missing}")

    pred = np.asarray(z["pred"])
    labels = np.asarray(z["labels"])
    mask = np.asarray(z["mask"]).astype(bool)
    keys = [
        key_of(pid, eye, date)
        for pid, eye, date in zip(z["patient_id"], z["eye"], z["vf_date"])
    ]
    return {
        "pred": pred,
        "labels": labels,
        "mask": mask,
        "keys": keys,
    }


def validate_bundle(path: Path, bundle: dict, failures: list[str]) -> None:
    pred = bundle["pred"]
    labels = bundle["labels"]
    mask = bundle["mask"]
    keys = bundle["keys"]
    expected = (len(keys), N_POINTS)

    if pred.shape != expected:
        failures.append(f"{path}: pred shape {pred.shape}, expected {expected}")
    if labels.shape != expected:
        failures.append(f"{path}: labels shape {labels.shape}, expected {expected}")
    if mask.shape != expected:
        failures.append(f"{path}: mask shape {mask.shape}, expected {expected}")
    if len(set(keys)) != len(keys):
        failures.append(f"{path}: duplicate keys")
    if pred.shape == expected and not np.isfinite(pred).all():
        failures.append(f"{path}: non-finite predictions")
    if labels.shape == expected and not np.isfinite(labels[mask]).all():
        failures.append(f"{path}: non-finite valid labels")
    if mask.shape == expected and np.any(mask.sum(axis=1) == 0):
        failures.append(f"{path}: eye with zero valid labels")


def indexed(bundle: dict) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        key: (bundle["pred"][i], bundle["labels"][i], bundle["mask"][i])
        for i, key in enumerate(bundle["keys"])
    }


def compare_truth(
    left_name: str,
    left: dict,
    right_name: str,
    right: dict,
    failures: list[str],
) -> int:
    a = indexed(left)
    b = indexed(right)
    common = sorted(set(a) & set(b))
    label_bad = 0
    mask_bad = 0
    for key in common:
        _, ya, ma = a[key]
        _, yb, mb = b[key]
        label_bad += int(not np.array_equal(ya, yb, equal_nan=True))
        mask_bad += int(not np.array_equal(ma, mb))
    if label_bad:
        failures.append(f"{left_name} vs {right_name}: label mismatch on {label_bad} keys")
    if mask_bad:
        failures.append(f"{left_name} vs {right_name}: mask mismatch on {mask_bad} keys")
    return len(common)


def main() -> None:
    failures: list[str] = []
    summary: dict = {"folds": {}, "backbones": {}, "failures": failures}
    xgb: dict[tuple[str, int], dict] = {}

    for split in ("val", "test"):
        for fold in range(5):
            path = OOF_DIR / f"xgb_90d_fold{fold}_{split}.npz"
            bundle = load_bundle(path)
            validate_bundle(path, bundle, failures)
            xgb[(split, fold)] = bundle

    xgb_oof_keys = [
        key for fold in range(5) for key in xgb[("val", fold)]["keys"]
    ]
    if len(xgb_oof_keys) != len(set(xgb_oof_keys)):
        failures.append("XGB OOF keys overlap across validation folds")
    xgb_test_sets = [set(xgb[("test", fold)]["keys"]) for fold in range(5)]
    if any(keys != xgb_test_sets[0] for keys in xgb_test_sets[1:]):
        failures.append("XGB held-out key sets differ across folds")
    for fold in range(1, 5):
        compare_truth(
            "XGB test fold0",
            xgb[("test", 0)],
            f"XGB test fold{fold}",
            xgb[("test", fold)],
            failures,
        )

    summary["xgb"] = {
        "oof_rows": len(xgb_oof_keys),
        "oof_unique_rows": len(set(xgb_oof_keys)),
        "held_out_rows_per_fold": [len(xgb[("test", fold)]["keys"]) for fold in range(5)],
    }

    for backbone, directory in CNN_DIRS.items():
        cnn: dict[tuple[str, int], dict] = {}
        fold_summary = {}
        for split in ("val", "test"):
            for fold in range(5):
                path = directory / f"{split}_preds_fold{fold}.npz"
                bundle = load_bundle(path)
                validate_bundle(path, bundle, failures)
                cnn[(split, fold)] = bundle
                common = compare_truth(
                    f"XGB {split} fold{fold}",
                    xgb[(split, fold)],
                    f"{backbone} {split} fold{fold}",
                    bundle,
                    failures,
                )
                fold_summary[f"{split}_fold{fold}"] = {
                    "cnn_rows": len(bundle["keys"]),
                    "xgb_rows": len(xgb[(split, fold)]["keys"]),
                    "aligned_rows": common,
                }

        cnn_oof_keys = [
            key for fold in range(5) for key in cnn[("val", fold)]["keys"]
        ]
        if len(cnn_oof_keys) != len(set(cnn_oof_keys)):
            failures.append(f"{backbone}: OOF keys overlap across folds")

        test_sets = [set(cnn[("test", fold)]["keys"]) for fold in range(5)]
        if any(keys != test_sets[0] for keys in test_sets[1:]):
            failures.append(f"{backbone}: held-out key sets differ across folds")
        for fold in range(1, 5):
            compare_truth(
                f"{backbone} test fold0",
                cnn[("test", 0)],
                f"{backbone} test fold{fold}",
                cnn[("test", fold)],
                failures,
            )

        summary["backbones"][backbone] = {
            "oof_rows": len(cnn_oof_keys),
            "oof_unique_rows": len(set(cnn_oof_keys)),
            "held_out_rows_per_fold": [
                len(cnn[("test", fold)]["keys"]) for fold in range(5)
            ],
            "folds": fold_summary,
        }

    summary["status"] = "PASS" if not failures else "FAIL"
    out = ROOT / "runs" / "prediction_artifact_audit.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
