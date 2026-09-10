#!/usr/bin/env python3
"""RNFL non-flip 데이터값 자가검증 (읽기 전용, 이미지 불필요).

컨벤션: h01=1시 ... h12=12시.
내부 일관성 규칙:
  R1) mean(clock 12) ≈ avg_rnfl            (|diff|>10 → flag)
  R2) mean(quad 4)   ≈ avg_rnfl            (|diff|>10 → flag)
  R3) superior quad  ≈ mean(h11,12,1)  (|diff|>20, 90°=clock 3개)
  R4) inferior quad  ≈ mean(h5,6,7)    (|diff|>20)
  R5) 범위: quad/clock ∈ [3,230]
  R6) mean(clock 12) ≈ mean(quad 4)    (|diff|>12, OCR 교차검증)
어긋난 행 = 이미지 대조 후보. avg_rnfl 출처: oct_values.csv.
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = sys.argv[1] if len(sys.argv) > 1 else "clockhours.csv"

def f(v):
    v = str(v).strip()
    if v in ("", "NaN", "nan", "None"): return None
    try:
        x = float(v); return None if math.isnan(x) else x
    except ValueError: return None

def m(ks, r):
    vs = [f(r.get(f"rnfl_h{k:02d}", "")) for k in ks]
    vs = [x for x in vs if x is not None]
    return sum(vs)/len(vs) if vs else None

rows = list(csv.DictReader(open(ROOT/SRC, encoding="utf-8-sig")))
octv = {(r["patient_id"], r["eye"], r["oct_date"]): r
        for r in csv.DictReader(open(ROOT/"oct_values.csv", encoding="utf-8-sig"))}

QUAD = ["rnfl_q_s", "rnfl_q_t", "rnfl_q_i", "rnfl_q_n"]
CLK = [f"rnfl_h{i:02d}" for i in range(1, 13)]
flags = {"R1": [], "R2": [], "R3": [], "R4": [], "R5": [], "R6": []}

for r in rows:
    pid, eye, od = r["patient_id"], r["eye"], r["oct_date"]
    avg = None
    oc = octv.get((pid, eye, od))
    if oc:
        avg = f(oc.get(f"{eye.lower()}_avg_rnfl", ""))
    clk = [f(r.get(c, "")) for c in CLK]; clk_ok = [x for x in clk if x is not None]
    qv = [f(r.get(c, "")) for c in QUAD]; qv_ok = [x for x in qv if x is not None]
    tag = f"{pid} {eye} {od}"
    if avg and len(clk_ok) >= 8:
        d = abs(sum(clk_ok)/len(clk_ok) - avg)
        if d > 10: flags["R1"].append((tag, round(d, 1)))
    if avg and len(qv_ok) >= 3:
        d = abs(sum(qv_ok)/len(qv_ok) - avg)
        if d > 10: flags["R2"].append((tag, round(d, 1)))
    qs = f(r.get("rnfl_q_s", "")); ms = m((11, 12, 1), r)
    if qs and ms and abs(qs - ms) > 20: flags["R3"].append((tag, qs, round(ms, 1)))
    qi = f(r.get("rnfl_q_i", "")); mi = m((5, 6, 7), r)
    if qi and mi and abs(qi - mi) > 20: flags["R4"].append((tag, qi, round(mi, 1)))
    for c in QUAD + CLK:
        v = f(r.get(c, ""))
        if v is not None and not (3 <= v <= 230):
            flags["R5"].append((tag, c, v))
    if len(clk_ok) >= 8 and len(qv_ok) >= 3:
        d = abs(sum(clk_ok)/len(clk_ok) - sum(qv_ok)/len(qv_ok))
        if d > 12: flags["R6"].append((tag, round(d, 1)))

print(f"=== RNFL 자가검증: {SRC} ({len(rows)}행) ===\n")
names = {"R1": "clock평균≉avg_rnfl", "R2": "quad평균≉avg_rnfl",
         "R3": "superior quad≉mean(h11,12,1)", "R4": "inferior quad≉mean(h5,6,7)",
         "R5": "범위밖", "R6": "clock평균≉quad평균"}
for k in ["R1", "R2", "R3", "R4", "R5", "R6"]:
    print(f"[{k}] {names[k]}: {len(flags[k])}건")
    for e in flags[k][:12]:
        print("     ", e)

suspect = set()
for k in flags:
    for e in flags[k]:
        suspect.add(e[0])
print(f"\n=== 이미지 대조 후보(고유 행): {len(suspect)}개 / {len(rows)} ===")
for s in sorted(suspect)[:40]:
    print("   ", s)
