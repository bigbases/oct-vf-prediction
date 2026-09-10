"""
백본 비교 결과 집계.
runs/bbcmp_{window}_{backbone}_{tag}_s{seed}/results.json 들을 읽어
백본별 holdout test RMSE/MAE 평균±SD 표를 출력하고 JSON으로 저장.

사용법:
  python scripts/summarize_backbone_compare.py --window 90d --tag imgonly ...
  python scripts/summarize_backbone_compare.py --window 90d --tag multimodal ...
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(window, bk, tag, seed):
    p = ROOT / f"runs/bbcmp_{window}_{bk}_{tag}_s{seed}/results.json"
    if not p.is_file():
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="90d")
    ap.add_argument("--tag", default="imgonly",
                    help="run 디렉터리 접미사: imgonly | multimodal")
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summary = {}
    rows = []
    for bk in args.backbones:
        rmses, maes, best_eps, missing = [], [], [], []
        for s in args.seeds:
            r = load(args.window, bk, args.tag, s)
            if r is None:
                missing.append(s)
                continue
            rmses.append(r.get("best_holdout_test_rmse"))
            maes.append(r.get("best_holdout_test_mae"))
            best_eps.append(r.get("best_epoch"))
        rmses = [x for x in rmses if x is not None]
        maes = [x for x in maes if x is not None]
        if rmses:
            entry = {
                "n": len(rmses),
                "rmse_mean": statistics.mean(rmses),
                "rmse_std": statistics.pstdev(rmses) if len(rmses) > 1 else 0.0,
                "mae_mean": statistics.mean(maes),
                "mae_std": statistics.pstdev(maes) if len(maes) > 1 else 0.0,
                "best_epochs": best_eps,
                "missing_seeds": missing,
            }
        else:
            entry = {"n": 0, "missing_seeds": missing}
        summary[bk] = entry
        if rmses:
            rows.append((entry["rmse_mean"], bk, entry))

    rows.sort()  # RMSE 낮은 순
    print(f"\n=== {args.window} {args.tag} 백본 비교 (seeds={args.seeds}) ===")
    print(f"{'backbone':18s} {'n':>2s}  {'test RMSE (mean±SD)':>22s}  {'test MAE (mean±SD)':>22s}")
    print("-" * 70)
    for _, bk, e in rows:
        print(f"{bk:18s} {e['n']:>2d}  "
              f"{e['rmse_mean']:8.3f} ± {e['rmse_std']:6.3f}        "
              f"{e['mae_mean']:8.3f} ± {e['mae_std']:6.3f}")
    miss = {bk: e.get("missing_seeds") for bk, e in summary.items() if e.get("missing_seeds")}
    if miss:
        print("\n[경고] 누락된 seed:", miss)

    out = args.out or f"runs/bbcmp_{args.window}_{args.tag}_summary.json"
    json.dump(summary, open(ROOT / out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
