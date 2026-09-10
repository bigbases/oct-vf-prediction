"""
RNFL Quadrant + Clock-hour OCR (개선판)
- 기존 parse_quadrants_cv / parse_clockhours_cv 좌표·임계값 재조정
- 다단계 contrast / conf 완화 / 매칭 허용범위 확대
- 결과: rnfl_detail.csv 저장 (patient_id, eye, vf_date, oct_date, 16개 수치, cv_flags, notes)
- 결측은 빈 셀로 두고 사용자가 수동 보완
"""
import csv, os, re, math, sys, time
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from paths import ROOT
BASE = str(ROOT)
ML_FULL = os.path.join(BASE, 'ml_dataset.csv')
ML_90D  = os.path.join(BASE, 'ml_final_90d.csv')
ML_180D = os.path.join(BASE, 'ml_final_180d.csv')
OCT     = os.path.join(BASE, 'oct_values.csv')
OUT_90  = os.path.join(BASE, 'rnfl_detail.csv')
OUT_180 = os.path.join(BASE, 'rnfl_detail_180d.csv')


# ── OCR 후보 수집 (다단계 fallback) ──────────────────────────
_CFG = '--psm 11 --oem 1 -c tessedit_char_whitelist=0123456789'

def collect_candidates(pil_l, conf_thr=10):
    """주어진 grayscale 이미지에서 (val, cx, cy) 후보 리스트 반환"""
    data = pytesseract.image_to_data(pil_l, config=_CFG, output_type=pytesseract.Output.DICT)
    out = []
    for i, txt in enumerate(data['text']):
        m = re.search(r'\d+', txt.strip())
        if not m or data['conf'][i] < conf_thr:
            continue
        v = int(m.group())
        if v < 5 or v > 300:
            continue
        cx = data['left'][i] + data['width'][i] // 2
        cy = data['top'][i]  + data['height'][i] // 2
        out.append((v, cx, cy))
    return out


def multi_stage_candidates(img_rgb, scale=2):
    """다단계 contrast로 후보 수집 → 좌표 dedup (같은 위치는 1번만)"""
    big = img_rgb.resize((img_rgb.width*scale, img_rgb.height*scale), Image.LANCZOS)
    gray = big.convert('L')
    all_cands = []
    for contrast in (3.0, 2.0, 4.0, 1.5, 6.0):
        enh = ImageEnhance.Contrast(gray).enhance(contrast)
        all_cands.extend(collect_candidates(enh, conf_thr=10))
    # 반전 1회
    inv = ImageOps.invert(ImageEnhance.Contrast(gray).enhance(3.0))
    all_cands.extend(collect_candidates(inv, conf_thr=10))

    # 좌표 dedup (15px 이내 같은 값이면 중복)
    dedup = []
    for v, cx, cy in all_cands:
        is_dup = False
        for v2, cx2, cy2 in dedup:
            if v == v2 and abs(cx-cx2) < 15 and abs(cy-cy2) < 15:
                is_dup = True
                break
        if not is_dup:
            dedup.append((v, cx, cy))
    # scale 보정 → 원본 좌표계로
    return [(v, cx//scale, cy//scale) for v, cx, cy in dedup]


# ── 1. Quadrants (4개 × 2안 = 8개) ──────────────────────────
def parse_quadrants(path):
    keys = ['S', 'T', 'I', 'N']
    empty = {f'{e}_{k}': None for e in ('od', 'os') for k in keys}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    cands = multi_stage_candidates(img)

    # OD/OS 원의 중심 (대략 W*0.21 / W*0.79, H*0.5)
    # quadrants.png에서 수치 위치를 더 넓게 잡음
    targets = {
        'od_S': (W*0.21,  H*0.13),
        'od_T': (W*0.040, H*0.55),
        'od_I': (W*0.21,  H*0.92),
        'od_N': (W*0.378, H*0.55),
        'os_S': (W*0.79,  H*0.13),
        'os_N': (W*0.660, H*0.55),
        'os_I': (W*0.79,  H*0.92),
        'os_T': (W*0.985, H*0.55),
    }

    # 거리 매칭 (허용범위 W의 8% 확대 = ~50px)
    max_d = max(W, H) * 0.10
    result = {}
    used = set()
    # 가까운 순으로 그리디 할당
    pairs = []
    for key, (tx, ty) in targets.items():
        for idx, (v, cx, cy) in enumerate(cands):
            d = math.sqrt((cx-tx)**2 + (cy-ty)**2)
            if d < max_d:
                pairs.append((d, key, idx, v))
    pairs.sort()
    for d, key, idx, v in pairs:
        if key in result or idx in used:
            continue
        result[key] = v
        used.add(idx)
    for key in targets:
        result.setdefault(key, None)
    return result


# ── 2. Clock-hours (12개 × 2안 = 24개) ──────────────────────
def parse_clockhours(path):
    empty = {f'{e}_h{n:02d}': None for e in ('od', 'os') for n in range(1, 13)}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    cands = multi_stage_candidates(img)

    cx_od = W * 0.206
    cx_os = W * 0.817
    cy_c  = H * 0.522
    r_min = min(W, H) * 0.10
    r_max = min(W, H) * 0.55

    cands_od, cands_os = [], []
    for v, cx, cy in cands:
        d_od = math.sqrt((cx-cx_od)**2 + (cy-cy_c)**2)
        d_os = math.sqrt((cx-cx_os)**2 + (cy-cy_c)**2)
        if d_od < d_os:
            if r_min <= d_od <= r_max:
                ang = math.degrees(math.atan2(-(cy-cy_c), cx-cx_od))
                cands_od.append((v, ang, d_od))
        else:
            if r_min <= d_os <= r_max:
                ang = math.degrees(math.atan2(-(cy-cy_c), cx-cx_os))
                cands_os.append((v, ang, d_os))

    def assign(cands_eye):
        target = {f'h{n:02d}': 90 - n*30 for n in range(1, 13)}
        # 12시(h12)는 90도, 3시(h03)는 0도, 6시(h06)는 -90도, 9시(h09)는 180/-180도
        # 위 target은 h01=60, h02=30, h03=0, h04=-30, h05=-60, h06=-90, h07=-120(=240), h08=-150, h09=180, h10=150, h11=120, h12=90
        max_d = 25  # 각도 허용범위 (30도 슬라이스 ÷ 2 + 여유)
        pairs = []
        for key, tgt in target.items():
            for idx, (v, ang, _) in enumerate(cands_eye):
                diff = abs((ang - tgt + 180) % 360 - 180)
                if diff < max_d:
                    pairs.append((diff, key, idx, v))
        pairs.sort()
        res = {}; used = set()
        for diff, key, idx, v in pairs:
            if key in res or idx in used:
                continue
            res[key] = v; used.add(idx)
        for n in range(1, 13):
            res.setdefault(f'h{n:02d}', None)
        return res

    od_h = assign(cands_od)
    os_h = assign(cands_os)
    return {**{f'od_{k}': v for k, v in od_h.items()},
            **{f'os_{k}': v for k, v in os_h.items()}}


# ── 3. CV 룰 ─────────────────────────────────────────────────
def cross_validate(eye, quad, clk, avg_rnfl):
    flags = []
    qv = [quad[f'{eye}_{d}'] for d in 'STIN']
    cv = [clk[f'{eye}_h{n:02d}'] for n in range(1, 13)]
    qv_ok = [x for x in qv if x is not None]
    cv_ok = [x for x in cv if x is not None]

    if avg_rnfl and len(qv_ok) >= 2:
        m = sum(qv_ok)/len(qv_ok)
        if abs(m - avg_rnfl) > 10:
            flags.append(f'CV_quad({len(qv_ok)}of4): mean={m:.1f} vs avg_rnfl={avg_rnfl} (diff={abs(m-avg_rnfl):.1f})')

    if avg_rnfl and len(cv_ok) >= 6:
        m = sum(cv_ok)/len(cv_ok)
        if abs(m - avg_rnfl) > 10:
            flags.append(f'CV_clk({len(cv_ok)}of12): mean={m:.1f} vs avg_rnfl={avg_rnfl} (diff={abs(m-avg_rnfl):.1f})')

    # CV3: S quadrant ≈ mean(h10, h11, h12, h01, h02) (위쪽 5시간)
    if quad[f'{eye}_S']:
        s_clk = [clk[f'{eye}_h{n:02d}'] for n in (10, 11, 12, 1, 2)]
        s_clk_ok = [x for x in s_clk if x is not None]
        if len(s_clk_ok) >= 3:
            m = sum(s_clk_ok)/len(s_clk_ok)
            if abs(m - quad[f'{eye}_S']) > 15:
                flags.append(f'CV_S: quad_S={quad[f"{eye}_S"]} vs clk_top_mean={m:.1f}')

    if quad[f'{eye}_I']:
        i_clk = [clk[f'{eye}_h{n:02d}'] for n in (4, 5, 6, 7, 8)]
        i_clk_ok = [x for x in i_clk if x is not None]
        if len(i_clk_ok) >= 3:
            m = sum(i_clk_ok)/len(i_clk_ok)
            if abs(m - quad[f'{eye}_I']) > 15:
                flags.append(f'CV_I: quad_I={quad[f"{eye}_I"]} vs clk_bot_mean={m:.1f}')

    return flags


# ── 4. Run ───────────────────────────────────────────────────
def run(pairs_csv, mode='verify', out_csv=None):
    """mode: 'verify' (샘플 10개 출력) or 'full' (전체 csv 저장)"""
    ml_full = list(csv.DictReader(open(ML_FULL, encoding='utf-8-sig')))
    rnfl_map = {(r['patient_id'], r['eye'], r['vf_date']): r['rnfl_dir'] for r in ml_full}

    oct_idx = {}
    for r in csv.DictReader(open(OCT, encoding='utf-8-sig')):
        oct_idx[(r['patient_id'], r['eye'])] = r

    pair_rows = list(csv.DictReader(open(pairs_csv, encoding='utf-8-sig')))

    if mode == 'verify':
        pair_rows = pair_rows[:10]

    QKEYS = ['S', 'T', 'I', 'N']
    HKEYS = [f'h{n:02d}' for n in range(1, 13)]
    FIELDS = (
        ['patient_id', 'eye', 'vf_date', 'oct_date'] +
        [f'rnfl_q_{d.lower()}' for d in QKEYS] +
        [f'rnfl_{h}' for h in HKEYS] +
        ['cv_flags', 'notes']
    )

    # path별 캐시 (양안 통합 리포트라 OD에서 OS 결과 같이 추출됨)
    path_cache = {}

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(pair_rows, 1):
        pid, eye, vfd, octd = r['patient_id'], r['eye'], r['vf_date'], r['oct_date']
        rnfl_dir = rnfl_map.get((pid, eye, vfd), '')
        notes = []

        if not rnfl_dir:
            notes.append('no_rnfl_dir')
            row = {k: '' for k in FIELDS}
            row.update({'patient_id': pid, 'eye': eye, 'vf_date': vfd, 'oct_date': octd,
                        'notes': '|'.join(notes)})
            out_rows.append(row)
            continue

        if rnfl_dir not in path_cache:
            quad_path = os.path.join(rnfl_dir, 'quadrants.png')
            clk_path  = os.path.join(rnfl_dir, 'clockhours.png')
            t1 = time.time()
            try:
                q = parse_quadrants(quad_path)
                c = parse_clockhours(clk_path)
                path_cache[rnfl_dir] = (q, c, None)
            except Exception as e:
                path_cache[rnfl_dir] = (None, None, str(e))
            if mode == 'verify':
                print(f'[{i}] {pid} {eye} ({time.time()-t1:.1f}s)')

        q, c, err = path_cache[rnfl_dir]
        if err:
            notes.append(f'ocr_error:{err[:40]}')
            row = {k: '' for k in FIELDS}
            row.update({'patient_id': pid, 'eye': eye, 'vf_date': vfd, 'oct_date': octd,
                        'notes': '|'.join(notes)})
            out_rows.append(row)
            continue

        eye_l = eye.lower()
        avg = None
        oct_r = oct_idx.get((pid, eye), {})
        try:
            avg = float(oct_r.get(f'{eye_l}_avg_rnfl', '') or 0) or None
        except: pass

        cv_flags = cross_validate(eye_l, q, c, avg)

        # 누락 카운트
        n_q_miss = sum(1 for d in QKEYS if q[f'{eye_l}_{d}'] is None)
        n_c_miss = sum(1 for h in HKEYS if c[f'{eye_l}_{h}'] is None)
        if n_q_miss >= 2: notes.append(f'quad_miss={n_q_miss}/4')
        if n_c_miss >= 3: notes.append(f'clk_miss={n_c_miss}/12')

        row = {
            'patient_id': pid, 'eye': eye, 'vf_date': vfd, 'oct_date': octd,
            'cv_flags': '|'.join(cv_flags),
            'notes': '|'.join(notes),
        }
        for d in QKEYS:
            row[f'rnfl_q_{d.lower()}'] = q[f'{eye_l}_{d}'] if q[f'{eye_l}_{d}'] is not None else ''
        for h in HKEYS:
            row[f'rnfl_{h}'] = c[f'{eye_l}_{h}'] if c[f'{eye_l}_{h}'] is not None else ''
        out_rows.append(row)

    if mode == 'full':
        target = out_csv or OUT_90
        with open(target, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(out_rows)
        print(f'\n저장: {target} ({len(out_rows)}행)')

    # 요약
    print(f'\n=== {mode} 요약 ({len(out_rows)} 페어, {time.time()-t0:.1f}s) ===')
    q_total = c_total = 0
    q_filled = c_filled = 0
    n_cv_flagged = 0
    n_notes = 0
    for r in out_rows:
        for d in 'STIN':
            q_total += 1
            if r[f'rnfl_q_{d.lower()}'] != '':
                q_filled += 1
        for n in range(1, 13):
            c_total += 1
            if r[f'rnfl_h{n:02d}'] != '':
                c_filled += 1
        if r['cv_flags']: n_cv_flagged += 1
        if r['notes']:    n_notes += 1
    print(f'Quadrant 충원율  : {q_filled}/{q_total} ({q_filled/q_total*100:.1f}%)')
    print(f'Clock-hour 충원율: {c_filled}/{c_total} ({c_filled/c_total*100:.1f}%)')
    print(f'CV flag 발생     : {n_cv_flagged}/{len(out_rows)}')
    print(f'notes 발생       : {n_notes}/{len(out_rows)}')
    return out_rows


if __name__ == '__main__':
    mode  = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    which = sys.argv[2] if len(sys.argv) > 2 else '90d'
    if which == '180d':
        run(ML_180D, mode, OUT_180)
    else:
        run(ML_90D, mode, OUT_90)
