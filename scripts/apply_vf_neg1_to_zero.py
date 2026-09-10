#!/usr/bin/env python3
"""
VF 라벨: 맹점(p26,p35) 제외 -1 → 0
- sfa_review_master_fixed.csv 수정
- sfa_thresholds.csv 동기화
- ml_final_* (non-flip) VF 컬럼 갱신 후 *_flip 재생성 (cv_fold·RNFL 등 유지)
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PT = [f"p{i:02d}" for i in range(1, 55)]
BLIND = frozenset({"p26", "p35"})
FLIP_ROWS = [
    ("p01", "p04"),
    ("p05", "p10"),
    ("p11", "p18"),
    ("p19", "p27"),
    ("p28", "p36"),
    ("p37", "p44"),
    ("p45", "p50"),
    ("p51", "p54"),
]

NON_FLIP_ML = [
    "ml_final_90d.csv",
    "ml_final_90d_excl_empty.csv",
    "ml_final_180d.csv",
    "ml_final_180d_excl_empty.csv",
]

SFA_THRESH_FIELDS = ["patient_id", "eye", "vf_date"] + PT + ["n_detected", "n_missing"]


def is_neg_one(v: str) -> bool:
    s = str(v).strip()
    return s in ("-1", "-1.0")


def patch_fixed(path: Path) -> int:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    fields = rows[0].keys() if rows else []
    n = 0
    for r in rows:
        for p in PT:
            if p in BLIND:
                continue
            if is_neg_one(r.get(p, "")):
                r[p] = "0"
                n += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return n


def load_fixed_index(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {(r["patient_id"], r["eye"], r["vf_date"]): r for r in rows}


def write_sfa_thresholds(fixed_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(fixed_path, encoding="utf-8-sig")))
    out_rows = []
    for r in rows:
        o = {k: r.get(k, "") for k in SFA_THRESH_FIELDS}
        out_rows.append(o)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SFA_THRESH_FIELDS)
        w.writeheader()
        w.writerows(out_rows)


def flip_os_row(row: dict) -> dict:
    out = dict(row)
    for a, b in FLIP_ROWS:
        pts = [f"p{i:02d}" for i in range(int(a[1:]), int(b[1:]) + 1)]
        vals = [row[p] for p in pts]
        for p, v in zip(pts, reversed(vals)):
            out[p] = v
    out["os_s_sup_t"], out["os_s_sup_n"] = row.get("os_s_sup_n", ""), row.get("os_s_sup_t", "")
    out["os_s_inf_t"], out["os_s_inf_n"] = row.get("os_s_inf_n", ""), row.get("os_s_inf_t", "")
    return out


def update_ml_final(path: Path, fixed_idx: dict) -> tuple[int, int]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    fields = list(rows[0].keys()) if rows else []
    updated = missing = 0
    for r in rows:
        key = (r["patient_id"], r["eye"], r["vf_date"])
        fr = fixed_idx.get(key)
        if fr is None:
            missing += 1
            continue
        for p in PT:
            if p in fields:
                r[p] = fr.get(p, "")
        updated += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return updated, missing


def rebuild_flip(non_flip_path: Path, flip_path: Path) -> None:
    rows = list(csv.DictReader(open(non_flip_path, encoding="utf-8-sig")))
    fields = list(rows[0].keys()) if rows else []
    out = []
    for r in rows:
        if r.get("eye") == "OS":
            out.append(flip_os_row(r))
        else:
            out.append(dict(r))
    with open(flip_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


def count_neg(path: Path, blind_only: bool = False) -> Counter:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    c = Counter()
    for r in rows:
        for p in PT:
            if blind_only and p not in BLIND:
                continue
            if not blind_only and p in BLIND:
                continue
            if is_neg_one(r.get(p, "")):
                c["neg1"] += 1
    return c


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fixed = ROOT / "sfa_review_master_fixed.csv"
    backup_dir = ROOT / "runs" / f"backup_vf_neg1_fix_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for name in [fixed.name, "sfa_thresholds.csv", *NON_FLIP_ML, *[p.replace(".csv", "_flip.csv") for p in NON_FLIP_ML]]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)

    n_fixed = patch_fixed(fixed)
    write_sfa_thresholds(fixed, ROOT / "sfa_thresholds.csv")

    fixed_idx = load_fixed_index(fixed)
    for nf in NON_FLIP_ML:
        p = ROOT / nf
        upd, miss = update_ml_final(p, fixed_idx)
        flip_name = nf.replace(".csv", "_flip.csv")
        flip_path = ROOT / flip_name
        rebuild_flip(p, flip_path)
        n_flip = patch_fixed(flip_path)  # OS flip 후 맹점 -1이 다른 pXX로 이동할 수 있음
        print(
            f"{nf}: rows_updated={upd} missing_in_fixed={miss} -> {flip_name} "
            f"rebuilt (flip extra -1->0: {n_flip})"
        )

    # summary
    print(f"\nfixed: patched {n_fixed} cells (-1->0, excl p26/p35)")
    for label, path in [
        ("fixed non-blind -1", fixed),
        ("ml_final_90d_excl_empty_flip non-blind -1", ROOT / "ml_final_90d_excl_empty_flip.csv"),
    ]:
        c = count_neg(path, blind_only=False)
        print(f"  {label}: {c.get('neg1', 0)}")
    c_blind = count_neg(fixed, blind_only=True)
    print(f"  fixed blind p26/p35 -1 kept: {c_blind.get('neg1', 0)}")
    print(f"backup: {backup_dir}")


if __name__ == "__main__":
    main()
