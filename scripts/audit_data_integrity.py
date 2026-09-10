#!/usr/bin/env python3
"""데이터 정합성 종합 감사 (읽기 전용).

검증:
  A) 구조: 헤더 vs 데이터 컬럼 수
  B) 타입/범위 sanity = "한 칸 밀림" 핵심 탐지
       vert_cd∈[0,1], avg_rnfl∈[25,160] 등. 비율↔두께가 바뀌면 범위 위반으로 잡힘.
  C) cross-validation: avg_gcl≈mean(sector), avg_rnfl≈mean(quad), min_gcl≤min(sector)
  D) 소스 정합성: ml_final(non-flip) OCT==oct_values, RNFL==rnfl_detail, VF==sfa_thresholds
  E) flip 정합성: OD행 불변, OS행 VF reverse + GCA swap 재계산 일치, RNFL 값 집합 보존
  F) 환자 단위 split 누수, 중복 행 키, 허용되지 않은 split 값
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.apply_vf_neg1_to_zero import flip_os_row, PT  # noqa: E402

GCA = ["avg_gcl_od", "avg_gcl_os", "min_gcl_od", "min_gcl_os",
       "od_s_sup", "od_s_sup_t", "od_s_inf_t", "od_s_inf", "od_s_inf_n", "od_s_sup_n",
       "os_s_sup", "os_s_sup_t", "os_s_inf_t", "os_s_inf", "os_s_inf_n", "os_s_sup_n"]
RNFL_AVG = ["od_avg_rnfl", "os_avg_rnfl"]
VERT = ["od_vert_cd", "os_vert_cd"]
QUAD = ["rnfl_q_s", "rnfl_q_t", "rnfl_q_i", "rnfl_q_n"]
CLK = [f"rnfl_h{i:02d}" for i in range(1, 13)]
OCT_ALL = GCA + RNFL_AVG + VERT
RNFL_ALL = QUAD + CLK

# 범위: 위반 시 밀림/오염 의심
RANGE = {
    **{c: (20, 130) for c in ["avg_gcl_od", "avg_gcl_os"]},
    **{c: (5, 120) for c in ["min_gcl_od", "min_gcl_os"]},
    **{c: (10, 150) for c in GCA if c.endswith(("_sup", "_sup_t", "_inf_t", "_inf", "_inf_n", "_sup_n"))},
    **{c: (25, 160) for c in RNFL_AVG},
    **{c: (0.0, 1.0) for c in VERT},
}


def fnum(s):
    s = str(s).strip()
    if s in ("", "NaN", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return "ERR"


def load(name):
    p = ROOT / name
    if not p.exists():
        return None
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


ML_NF = ["ml_final_90d.csv", "ml_final_90d_excl_empty.csv",
         "ml_final_180d.csv", "ml_final_180d_excl_empty.csv"]
ML_FL = ["ml_final_90d_flip.csv", "ml_final_90d_excl_empty_flip.csv",
         "ml_final_180d_flip.csv", "ml_final_180d_excl_empty_flip.csv"]
ALL = ML_NF + ML_FL + ["oct_values.csv", "rnfl_detail.csv", "rnfl_detail_180d.csv",
                       "rnfl_detail_flip.csv", "rnfl_detail_180d_flip.csv"]

hard_issues = 0
range_candidates = 0


def add_hard(n):
    global hard_issues
    hard_issues += n


def add_candidate(n):
    global range_candidates
    range_candidates += n


# ── A) 구조 ─────────────────────────────────────────────
banner("A) 구조 (헤더 vs 데이터 컬럼 수)")
for name in ALL:
    p = ROOT / name
    if not p.exists():
        continue
    with open(p, encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        bad = [i + 2 for i, row in enumerate(rdr) if len(row) != len(hdr)]
    add_hard(len(bad))
    print(f"  {name:36s} cols={len(hdr):3d} 행오류={len(bad)} {'OK' if not bad else bad[:5]}")


# ── B) 범위 sanity ──────────────────────────────────────
banner("B) 타입/범위 sanity = 밀림/오염 핵심 탐지")
for name in ML_NF + ["oct_values.csv"]:
    rows = load(name)
    if rows is None:
        continue
    errs, oob = [], []
    for r in rows:
        for c, (lo, hi) in RANGE.items():
            if c not in r:
                continue
            v = fnum(r[c])
            if v == "ERR":
                errs.append((r["patient_id"], r["eye"], c, r[c]))
            elif v is not None and not (lo <= v <= hi):
                oob.append((r["patient_id"], r["eye"], c, v))
    add_hard(len(errs))
    add_candidate(len(oob))
    print(f"  {name:36s} 형식깨짐={len(errs)} 범위위반={len(oob)}")
    for e in errs[:6]:
        print("       ERR ", e)
    for o in oob[:10]:
        print("       OOB ", o)


# ── C) cross-validation ─────────────────────────────────
banner("C) cross-validation (avg≈mean, min≤sector)")
SEC = {"od": ["od_s_sup", "od_s_sup_t", "od_s_inf_t", "od_s_inf", "od_s_inf_n", "od_s_sup_n"],
       "os": ["os_s_sup", "os_s_sup_t", "os_s_inf_t", "os_s_inf", "os_s_inf_n", "os_s_sup_n"]}


def mean(vs):
    vs = [x for x in vs if isinstance(x, float)]
    return sum(vs) / len(vs) if vs else None


rows = load("oct_values.csv")
cv_gcl = cv_min = 0
for r in rows:
    for eye in ("od", "os"):
        ag = fnum(r.get(f"avg_gcl_{eye}"))
        secs = [fnum(r.get(c)) for c in SEC[eye]]
        m = mean(secs)
        if isinstance(ag, float) and m is not None and abs(ag - m) > 8:
            cv_gcl += 1
        mg = fnum(r.get(f"min_gcl_{eye}"))
        sec_min = min([x for x in secs if isinstance(x, float)], default=None)
        if isinstance(mg, float) and sec_min is not None and mg > sec_min + 6:
            cv_min += 1
print(f"  oct_values: avg_gcl≉mean(sector) {cv_gcl}건, min_gcl>min(sector) {cv_min}건")
print("  (참고: OCR 단계 cv_flags로 이미 표시된 항목과 중복 가능 — 치명적 아님)")


# ── D) 소스 정합성 ───────────────────────────────────────
banner("D) 소스 정합성 (ml_final non-flip ↔ 원본)")
oct_idx = {(r["patient_id"], r["eye"], r["oct_date"]): r for r in load("oct_values.csv")}


def norm(s):
    s = str(s).strip()
    return "" if s in ("NaN", "nan", "None") else s


for name in ML_NF:
    rows = load(name)
    rdname = "rnfl_detail.csv" if "90d" in name else "rnfl_detail_180d.csv"
    rnfl_idx = {(r["patient_id"], r["eye"], r["oct_date"]): r for r in load(rdname)}
    d_oct = d_rnfl = nokey = 0
    for r in rows:
        k = (r["patient_id"], r["eye"], r["oct_date"])
        oc = oct_idx.get(k)
        if oc is None:
            nokey += 1
        else:
            for c in OCT_ALL:
                if c in r and norm(r[c]) != norm(oc.get(c, "")):
                    d_oct += 1
        rd = rnfl_idx.get(k)
        if rd:
            for c in RNFL_ALL:
                if c in r and c in rd and norm(r[c]) != norm(rd.get(c, "")):
                    d_rnfl += 1
    add_hard(d_oct + d_rnfl)
    print(f"  {name:36s} OCT불일치={d_oct} RNFL불일치={d_rnfl} oct키없음={nokey} (vs {rdname})")


# ── E) flip 정합성 ──────────────────────────────────────
banner("E) flip 정합성 (OD 불변 / OS reverse+swap / RNFL 값보존)")
for base, flip in zip(ML_NF, ML_FL):
    nf = {(r["patient_id"], r["eye"], r["vf_date"]): r for r in load(base)}
    fl = load(flip)
    od_changed = os_vf_bad = os_gca_bad = rnfl_set_bad = 0
    for r in fl:
        b = nf.get((r["patient_id"], r["eye"], r["vf_date"]))
        if not b:
            continue
        if r["eye"] == "OD":
            for c in PT + OCT_ALL + RNFL_ALL:
                if c in r and c in b and norm(r[c]) != norm(b[c]):
                    od_changed += 1
        else:
            exp = flip_os_row(b)
            for p in PT:
                rv, ev = norm(r.get(p, "")), norm(exp.get(p, ""))
                # 맹점(blind spot)의 -1 → 0 변환(apply_vf_neg1_to_zero)은 정상
                if rv != ev and not (ev == "-1" and rv == "0"):
                    os_vf_bad += 1
            for c in GCA:
                if c.startswith("os_") and norm(r.get(c, "")) != norm(exp.get(c, "")):
                    os_gca_bad += 1
            # RNFL 값 집합 보존 (거울반전이면 multiset 동일)
            sb = sorted(norm(b.get(c, "")) for c in RNFL_ALL)
            sf = sorted(norm(r.get(c, "")) for c in RNFL_ALL)
            if sb != sf:
                rnfl_set_bad += 1
    add_hard(od_changed + os_vf_bad + os_gca_bad + rnfl_set_bad)
    print(f"  {flip:36s} OD변형={od_changed} OS_VF오류={os_vf_bad} "
          f"OS_GCA swap오류={os_gca_bad} RNFL값집합깨짐={rnfl_set_bad}")


banner("F) 환자 단위 split·행 키 무결성")
for name in ML_FL:
    rows = load(name)
    key_counts = Counter(
        (r["patient_id"], r["eye"], r["vf_date"])
        for r in rows
    )
    duplicate_keys = sum(count - 1 for count in key_counts.values() if count > 1)
    patient_splits = {}
    invalid_split = 0
    for row in rows:
        split = row.get("cv_fold", "")
        if split not in {"0", "1", "2", "3", "4", "test"}:
            invalid_split += 1
        patient_splits.setdefault(row["patient_id"], set()).add(split)
    split_leakage = sum(len(splits) > 1 for splits in patient_splits.values())
    add_hard(duplicate_keys + invalid_split + split_leakage)
    print(
        f"  {name:36s} duplicate_key={duplicate_keys} "
        f"invalid_split={invalid_split} patient_split_leakage={split_leakage}"
    )


banner(f"종합: 확정 정합성 문제 = {hard_issues}건")
print(f"  범위 밖 원본 대조 후보 출현 = {range_candidates}건")
print("  동일 후보가 원본/90d/180d 파생본에 반복되므로 고유 오류 수가 아니다.")
print("  병적 극단값·segmentation artifact도 포함하므로 원본 대조 전 오류로 판정하지 않는다.")
print("  C(cross-validation)는 OCR 품질 참고용이라 위 합계에서 제외.")
