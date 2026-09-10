#!/usr/bin/env python3
"""Aggregate repro 5-seed results (holdout_test_rmse primary)."""
import argparse
import json
import statistics
from pathlib import Path


def load_metrics(path: Path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    return {
        "seed": r["args"]["seed"],
        "best_epoch": r["best_epoch"],
        "val_rmse": r["best_val_rmse"],
        "holdout_test_rmse": r["best_holdout_test_rmse"],
        "holdout_test_mae": r["best_holdout_test_mae"],
    }


def mean_std(xs):
    if len(xs) < 2:
        m = xs[0] if xs else float("nan")
        return m, 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--prefix", default="runs/repro_180d")
    p.add_argument("--suffix", default="flip_papertrack_lr1e4_p100")
    p.add_argument("--out", default="runs/repro_180d_5seed_summary.json")
    args = p.parse_args()

    modes = {
        "image-only": "imgonly",
        "multimodal": "multimodal",
    }
    all_rows = {}
    for label, tag in modes.items():
        rows = []
        for seed in args.seeds:
            path = Path(f"{args.prefix}_{tag}_{args.suffix}_s{seed}/results.json")
            if not path.is_file():
                print(f"MISSING: {path}")
                continue
            rows.append(load_metrics(path))
        all_rows[label] = rows

    print("\n--- per-seed holdout_test_rmse ---")
    for label, rows in all_rows.items():
        print(f"\n{label}:")
        for row in rows:
            print(
                f"  seed {row['seed']}: holdout_rmse={row['holdout_test_rmse']:.4f} "
                f"mae={row['holdout_test_mae']:.3f} val_rmse={row['val_rmse']:.4f}"
            )

    summary = {}
    diffs = []
    common = []
    for label, rows in all_rows.items():
        if not rows:
            continue
        hr = [r["holdout_test_rmse"] for r in rows]
        hm, hs = mean_std(hr)
        summary[label] = {
            "n": len(rows),
            "holdout_test_rmse_mean": hm,
            "holdout_test_rmse_std": hs,
            "per_seed": rows,
        }

    if "image-only" in summary and "multimodal" in summary:
        img_by_seed = {r["seed"]: r for r in summary["image-only"]["per_seed"]}
        mm_by_seed = {r["seed"]: r for r in summary["multimodal"]["per_seed"]}
        common = sorted(set(img_by_seed) & set(mm_by_seed))
        diffs = [
            mm_by_seed[s]["holdout_test_rmse"] - img_by_seed[s]["holdout_test_rmse"]
            for s in common
        ]
    if diffs:
        dm, ds = mean_std(diffs)
        summary["multimodal_minus_image_only"] = {
            "seeds": common,
            "holdout_test_rmse_diff_mean": dm,
            "holdout_test_rmse_diff_std": ds,
            "per_seed_diff": dict(zip(common, diffs)),
        }
        print("\n--- 5-seed summary (holdout_test_rmse) ---")
        for label in ("image-only", "multimodal"):
            s = summary[label]
            print(f"{label}: {s['holdout_test_rmse_mean']:.4f} ± {s['holdout_test_rmse_std']:.4f} (n={s['n']})")
        d = summary["multimodal_minus_image_only"]
        print(f"multimodal − image-only: {d['holdout_test_rmse_diff_mean']:+.4f} ± {d['holdout_test_rmse_diff_std']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
