#!/usr/bin/env python3
"""Write 180d 5-seed table + interpretation for REPRO_TRACK."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs/repro_180d_5seed_summary.json"
OUT_MD = ROOT / "runs/repro_180d_5seed_report.md"

seeds = [42, 43, 44, 45, 46]
rows_img, rows_mm = [], []
for s in seeds:
    for tag, store in [("imgonly", rows_img), ("multimodal", rows_mm)]:
        p = ROOT / f"runs/repro_180d_{tag}_flip_papertrack_lr1e4_p100_s{s}/results.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        r = json.loads(p.read_text(encoding="utf-8"))
        store.append({
            "seed": s,
            "best_epoch": r["best_epoch"],
            "val_rmse": r["best_val_rmse"],
            "holdout_rmse": r["best_holdout_test_rmse"],
            "holdout_mae": r["best_holdout_test_mae"],
        })

summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.is_file() else {}

lines = [
    "# 180d Repro 5-seed 결과 (자동 생성)\n",
    "## per-seed holdout_test_rmse (dB)\n",
    "| seed | image-only | multimodal | mm − img |",
    "|---:|---:|---:|---:|",
]
for i, s in enumerate(seeds):
    a, b = rows_img[i]["holdout_rmse"], rows_mm[i]["holdout_rmse"]
    lines.append(f"| {s} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")

if "image-only" in summary and "multimodal" in summary:
    io, mo = summary["image-only"], summary["multimodal"]
    d = summary.get("multimodal_minus_image_only", {})
    lines += [
        "",
        "## 5-seed 요약 (holdout RMSE)",
        f"- image-only: **{io['holdout_test_rmse_mean']:.3f} ± {io['holdout_test_rmse_std']:.3f}**",
        f"- multimodal: **{mo['holdout_test_rmse_mean']:.3f} ± {mo['holdout_test_rmse_std']:.3f}**",
    ]
    if d:
        lines.append(
            f"- multimodal − image-only: **{d['holdout_test_rmse_diff_mean']:+.3f} ± {d['holdout_test_rmse_diff_std']:.3f}**"
        )

lines += [
    "",
    "## 해석 메모",
    "1. **절대 성능**: holdout RMSE가 seed마다 7~12 dB대로 크게 흔들림 → 단일 split 운 + 180d 코호트 난이도 영향 큼. 논문 4.79 dB와 직접 비교 불가.",
    "2. **tabular (multimodal)**: seed별로 우위가 뒤집힘(예: s43 img<mm, s45 mm<img). 5-seed 평균 차이로만 '안정적 우위' 판단.",
    "3. **90d 3-seed**: 임상 primary window; repro 동일 설정으로 후속 비교 예정.",
]

text = "\n".join(lines) + "\n"
OUT_MD.write_text(text, encoding="utf-8")
print(text)
print(f"저장: {OUT_MD}")
