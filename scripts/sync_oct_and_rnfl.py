#!/usr/bin/env python3
"""ml_final에 oct_values(OCT) 반영 + RNFL merge (rnfl_detail / rnfl_detail_flip)."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_vf_neg1_to_zero import NON_FLIP_ML, flip_os_row  # noqa: E402

OCT_COLS = [
    "avg_gcl_od", "avg_gcl_os", "min_gcl_od", "min_gcl_os",
    "od_s_sup", "od_s_sup_t", "od_s_inf_t", "od_s_inf", "od_s_inf_n", "od_s_sup_n",
    "os_s_sup", "os_s_sup_t", "os_s_inf_t", "os_s_inf", "os_s_inf_n", "os_s_sup_n",
    "od_avg_rnfl", "os_avg_rnfl", "od_vert_cd", "os_vert_cd",
]


def load_oct_index() -> dict:
    rows = list(csv.DictReader(open(ROOT / "oct_values.csv", encoding="utf-8-sig")))
    return {(r["patient_id"], r["eye"], r["oct_date"]): r for r in rows}


def sync_oct_to_ml_final(path: Path, oct_idx: dict) -> tuple[int, int]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    updated = missing = 0
    for r in rows:
        key = (r["patient_id"], r["eye"], r["oct_date"])
        oc = oct_idx.get(key)
        if oc is None:
            missing += 1
            continue
        for c in OCT_COLS:
            if c in fields and c in oc:
                r[c] = oc.get(c, "")
        updated += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return updated, missing


def refresh_flip_oct_swap(flip_path: Path, non_flip_path: Path) -> None:
    """OS 행만 OCT temporal/nasal swap (RNFL·VF는 flip 파일 유지)."""
    nf = {
        (r["patient_id"], r["eye"], r["vf_date"]): r
        for r in csv.DictReader(open(non_flip_path, encoding="utf-8-sig"))
    }
    rows = list(csv.DictReader(open(flip_path, encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    for i, r in enumerate(rows):
        if r.get("eye") != "OS":
            continue
        key = (r["patient_id"], r["eye"], r["vf_date"])
        base = nf.get(key)
        if base is None:
            continue
        swapped = flip_os_row(base)
        for c in OCT_COLS:
            if c in fields:
                r[c] = swapped.get(c, r.get(c, ""))
        rows[i] = r
    with open(flip_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "runs" / f"backup_oct_rnfl_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    targets = NON_FLIP_ML + [p.replace(".csv", "_flip.csv") for p in NON_FLIP_ML]
    for name in targets:
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, backup / name)

    oct_idx = load_oct_index()
    print("=== OCT sync (oct_values.csv, join=oct_date) ===")
    for nf in NON_FLIP_ML:
        p = ROOT / nf
        upd, miss = sync_oct_to_ml_final(p, oct_idx)
        print(f"  {nf}: updated={upd} missing_oct={miss}")

    print("\n=== RNFL merge ===")
    subprocess.run(
        [sys.executable, str(ROOT / "merge_rnfl_into_ml_final.py")],
        cwd=str(ROOT),
        check=True,
    )

    print("\n=== Flip OCT T/N swap (OS only, VF/RNFL 유지) ===")
    for nf in NON_FLIP_ML:
        flip = ROOT / nf.replace(".csv", "_flip.csv")
        refresh_flip_oct_swap(flip, ROOT / nf)
        print(f"  {flip.name}: OS OCT swap refreshed")

    # verify 90d flip rnfl_q_i
    rows = list(csv.DictReader(open(ROOT / "ml_final_90d_excl_empty_flip.csv", encoding="utf-8-sig")))
    qi = sum(1 for r in rows if str(r.get("rnfl_q_i", "")).strip() not in ("",))
    print(f"\n90d flip: rnfl_q_i filled {qi}/{len(rows)}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
