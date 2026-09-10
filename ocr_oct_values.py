"""
OCT 수치 OCR 추출 v5
- Tier 2 필드만 추출 (20개 수치) + cv-only 보조 추출
- RANGES sanity check (범위 밖 → None + flag)
- Cross-validation 4개 룰 (silent error 감지)
- binary thr 확장 (70,100,120,150) → gray text 처리
- 반전 contrast 추가 → dark bg + light text 처리
"""
import csv, os, re, math, sys, time
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageChops

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from paths import ROOT
BASE    = str(ROOT)
ML_CSV  = os.path.join(BASE, 'ml_dataset.csv')
OUT_CSV = os.path.join(BASE, 'oct_values.csv')

# ── Sanity check 범위 ────────────────────────────────────────
RANGES = {
    'avg_rnfl': (40,  130),
    'min_gcl':  (20,  100),
    'avg_gcl':  (40,  100),
    'vert_cd':  (0.0, 0.99),
    '_s_':      (20,  200),
}

def apply_sanity(key, val, flagged=None):
    if val is None:
        return None
    for pattern, (lo, hi) in RANGES.items():
        if pattern in key:
            if val < lo or val > hi:
                if flagged is not None:
                    flagged.append((key, val, lo, hi))
                return None
    return val


# ── 공통 OCR 유틸 ────────────────────────────────────────────
_OCR_CFG8 = '--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789.'
_OCR_CFG7 = '--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.'
_OCR_CFG6 = '--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.'
_PAD = 10

def _pad(pil_l):
    out = Image.new('L', (pil_l.width + 2*_PAD, pil_l.height + 2*_PAD), 255)
    out.paste(pil_l, (_PAD, _PAD))
    return out

def _rgb_binary(crop_rgb, thr=70):
    r, g, b = crop_rgb.split()
    mx = ImageChops.lighter(ImageChops.lighter(r, g), b)
    return mx.point(lambda x: 0 if x < thr else 255, 'L')

def _try_val(pil_img, cfg, min_val=None, max_val=None):
    txt = pytesseract.image_to_string(pil_img, config=cfg).strip()
    m = re.search(r'\d+\.?\d*', txt)
    if not m:
        return None
    v = float(m.group())
    if min_val is not None and v < min_val:
        return None
    if max_val is not None and v > max_val:
        return None
    return v

def ocr_cell(img_rgb, cx, cy, hw, hh, hh_below=None, psm8_first=False,
             min_val=None, max_val=None):
    """
    Full pipeline:
    1. RGB binary (thr=70,100,120,150)
    2. Grayscale contrast (3.0,2.0,4.0,1.5) + PSM6 fallback
    3. Inverted binary
    4. Inverted contrast (dark bg + light text)
    """
    hb = hh_below if hh_below is not None else hh
    x0, y0 = max(0, cx-hw), max(0, cy-hh)
    x1, y1 = min(img_rgb.width, cx+hw), min(img_rgb.height, cy+hb)
    crop     = img_rgb.crop((x0, y0, x1, y1))
    big_rgb  = crop.resize((crop.width*4, crop.height*4), Image.LANCZOS)
    big_gray = big_rgb.convert('L')
    kw = dict(min_val=min_val, max_val=max_val)
    s1 = ((_OCR_CFG8,), (_OCR_CFG7,)) if psm8_first else ((_OCR_CFG7,), (_OCR_CFG8,))

    # 1. RGB 이진화 (thr 확장 → gray text 대응)
    for (cfg,) in s1:
        for thr in (70, 100, 120, 150):
            v = _try_val(_pad(_rgb_binary(big_rgb, thr=thr)), cfg, **kw)
            if v is not None:
                return v

    # 2. 그레이스케일 대비 강화
    for contrast in (3.0, 2.0, 4.0, 1.5):
        c = _pad(ImageEnhance.Contrast(big_gray).enhance(contrast))
        for (cfg,) in s1:
            v = _try_val(c, cfg, **kw)
            if v is not None:
                return v
        v = _try_val(c, _OCR_CFG6, **kw)
        if v is not None:
            return v

    # 3. 반전 이진화
    inv = _pad(ImageOps.invert(_rgb_binary(big_rgb, thr=60)))
    for (cfg,) in s1:
        v = _try_val(inv, cfg, **kw)
        if v is not None:
            return v

    # 4. 반전 그레이스케일 (dark bg + light text)
    for contrast in (3.0, 4.0, 2.0):
        ig = _pad(ImageOps.invert(ImageEnhance.Contrast(big_gray).enhance(contrast)))
        for (cfg,) in s1:
            v = _try_val(ig, cfg, **kw)
            if v is not None:
                return v

    return None


# ── 1. GCL 두께표 ────────────────────────────────────────────
def parse_gcl_table(path):
    empty = {k: None for k in ['avg_gcl_od', 'avg_gcl_os', 'min_gcl_od', 'min_gcl_os']}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    cells = {
        'avg_gcl_od': (int(W*0.64), int(H*0.42), 45, 22),
        'avg_gcl_os': (int(W*0.85), int(H*0.42), 45, 22),
        'min_gcl_od': (int(W*0.64), int(H*0.81), 45, 22),
        'min_gcl_os': (int(W*0.85), int(H*0.81), 45, 22),
    }
    result = {}
    for key, (cx, cy, hw, hh) in cells.items():
        x0, y0 = max(0, cx-hw), max(0, cy-hh)
        x1, y1 = min(W, cx+hw), min(H, cy+hh)
        crop = img.crop((x0, y0, x1, y1))
        big  = crop.resize((crop.width*4, crop.height*4), Image.LANCZOS)
        v = None
        for contrast in (3.0, 2.0, 1.5):
            c = ImageEnhance.Contrast(big.convert('L')).enhance(contrast)
            v = _try_val(c, _OCR_CFG6, max_val=150)
            if v is not None:
                break
            v = _try_val(c, _OCR_CFG8, max_val=150)
            if v is not None:
                break
        result[key] = int(v) if v is not None else None
    return result


# ── 2. GCA 섹터 (6섹터) ──────────────────────────────────────
def parse_sectors(path, eye='OD'):
    keys  = ['s_sup', 's_sup_t', 's_inf_t', 's_inf', 's_inf_n', 's_sup_n']
    empty = {f'{eye.lower()}_{k}': None for k in keys}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    cx_c, cy_c = W // 2, H // 2
    r = min(W, H) * 0.35

    if eye.upper() == 'OD':
        angles_deg = [90, 30, 330, 270, 210, 150]
    else:
        angles_deg = [90, 150, 210, 270, 330, 30]

    result = {}
    for key, ang in zip(keys, angles_deg):
        rad = math.radians(ang)
        px  = int(cx_c + r * math.cos(rad))
        py  = int(cy_c - r * math.sin(rad))
        # top/bottom: 수평 텍스트라 hw 크게, hh 작게
        if ang in (90, 270):
            hw, hh = 55, 20
        else:
            hw, hh = 55, 30
        v = ocr_cell(img, px, py, hw, hh, psm8_first=True, min_val=15, max_val=200)
        result[f'{eye.lower()}_{key}'] = int(v) if v is not None else None
    return result


# ── 3. RNFL 요약표 (avg_rnfl + vert_cd만) ───────────────────
def parse_rnfl_summary(path):
    empty = {'od_avg_rnfl': None, 'os_avg_rnfl': None,
             'od_vert_cd':  None, 'os_vert_cd':  None}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    od_x = int(W * 0.56)
    os_x = int(W * 0.85)
    hw, hh = 55, 14
    cells = {
        'od_avg_rnfl': (od_x, int(H * 0.18), 30,  200),
        'os_avg_rnfl': (os_x, int(H * 0.18), 30,  200),
        'od_vert_cd':  (od_x, int(H * 0.83), None, 0.99),
        'os_vert_cd':  (os_x, int(H * 0.83), None, 0.99),
    }
    result = {}
    for key, (cx, cy, min_val, max_val) in cells.items():
        v = ocr_cell(img, cx, cy, hw, hh, min_val=min_val, max_val=max_val)
        result[key] = v
    return result


# ── 4. Quadrants (CV 전용) ───────────────────────────────────
def parse_quadrants_cv(path):
    """Cross-validation 전용. PSM11 스캔 + 위치 기반 할당."""
    empty = {'od_S': None, 'od_T': None, 'od_I': None, 'od_N': None,
             'os_S': None, 'os_T': None, 'os_I': None, 'os_N': None}
    if not path or not os.path.exists(path):
        return empty
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    big = img.resize((W*2, H*2), Image.LANCZOS)
    enh = ImageEnhance.Contrast(big.convert('L')).enhance(3.0)
    cfg = '--psm 11 --oem 1 -c tessedit_char_whitelist=0123456789.'
    data = pytesseract.image_to_data(enh, config=cfg, output_type=pytesseract.Output.DICT)
    cands = []
    for i, txt in enumerate(data['text']):
        m = re.search(r'\d+', txt.strip())
        if not m or data['conf'][i] < 20:
            continue
        v = float(m.group())
        if v < 30 or v > 250:
            continue
        cx = (data['left'][i] + data['width'][i]//2) // 2
        cy = (data['top'][i]  + data['height'][i]//2) // 2
        cands.append((v, cx, cy))
    targets = {
        'od_S': (W*0.21, H*0.13), 'od_T': (W*0.049, H*0.63),
        'od_I': (W*0.21, H*0.87), 'od_N': (W*0.364, H*0.63),
        'os_S': (W*0.79, H*0.13), 'os_N': (W*0.660, H*0.63),
        'os_I': (W*0.79, H*0.87), 'os_T': (W*0.972, H*0.63),
    }
    result = {}
    used = set()
    for key, (tx, ty) in targets.items():
        best_v, best_d, best_i = None, 50, -1
        for idx, (v, cx, cy) in enumerate(cands):
            if idx in used:
                continue
            d = math.sqrt((cx-tx)**2 + (cy-ty)**2)
            if d < best_d:
                best_d, best_v, best_i = d, v, idx
        if best_i >= 0:
            result[key] = best_v
            used.add(best_i)
        else:
            result[key] = None
    return result


# ── 5. Clock hours (CV 전용) ─────────────────────────────────
def parse_clockhours_cv(path):
    """Cross-validation 전용. PSM11 스캔 + 각도 기반 할당."""
    empty_od = {f'od_h{i+1:02d}': None for i in range(12)}
    empty_os = {f'os_h{i+1:02d}': None for i in range(12)}
    if not path or not os.path.exists(path):
        return {**empty_od, **empty_os}
    img = Image.open(path).convert('RGB')
    W, H = img.width, img.height
    cx_od = int(W * 0.206)
    cx_os = int(W * 0.817)
    cy    = int(H * 0.522)
    r_min, r_max = 60, 135
    enh = ImageEnhance.Contrast(img.convert('L')).enhance(6.0)
    cfg = '--psm 11 --oem 1 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(enh, config=cfg, output_type=pytesseract.Output.DICT)
    cands_od, cands_os = [], []
    for i, txt in enumerate(data['text']):
        m = re.search(r'\d+', txt.strip())
        if not m or data['conf'][i] < 20:
            continue
        v = int(m.group())
        if v < 5 or v > 300:
            continue
        bx = data['left'][i] + data['width'][i]//2
        by = data['top'][i]  + data['height'][i]//2
        d_od = math.sqrt((bx-cx_od)**2 + (by-cy)**2)
        d_os = math.sqrt((bx-cx_os)**2 + (by-cy)**2)
        if d_od < d_os and r_min <= d_od <= r_max:
            cands_od.append((v, math.degrees(math.atan2(-(by-cy), bx-cx_od)), d_od))
        elif d_os < d_od and r_min <= d_os <= r_max:
            cands_os.append((v, math.degrees(math.atan2(-(by-cy), bx-cx_os)), d_os))

    def assign(cands):
        target = {f'h{n:02d}': 90 - n*30 for n in range(1, 13)}
        res = {}; used = set()
        for key, tgt in target.items():
            best_v, best_d, best_i = None, 20, -1
            for idx, (v, ang, _) in enumerate(cands):
                if idx in used:
                    continue
                diff = abs((ang - tgt + 180) % 360 - 180)
                if diff < best_d:
                    best_d, best_v, best_i = diff, v, idx
            if best_i >= 0:
                res[key] = best_v; used.add(best_i)
            else:
                res[key] = None
        return res

    od_h = assign(cands_od)
    os_h = assign(cands_os)
    return {**{f'od_{k}': v for k, v in od_h.items()},
            **{f'os_{k}': v for k, v in os_h.items()}}


# ── Cross-validation ─────────────────────────────────────────
_CV_THR = 5   # 허용 차이 (μm)

def cross_validate(norm_row, quad, clk):
    """4개 룰로 silent error 감지. 위반 시 flag 문자열 리스트 반환."""
    flags = []

    def _mean(vals):
        v = [x for x in vals if x is not None]
        return sum(v)/len(v) if v else None

    sec_keys = ['sup', 'sup_t', 'inf_t', 'inf', 'inf_n', 'sup_n']

    for eye in ('od', 'os'):
        # Rule 1: avg_gcl vs mean(6섹터)
        avg_gcl = norm_row.get(f'avg_gcl_{eye}')
        secs    = [norm_row.get(f'{eye}_s_{k}') for k in sec_keys]
        m_sec   = _mean(secs)
        if avg_gcl is not None and m_sec is not None and len([x for x in secs if x]) >= 3:
            diff = abs(avg_gcl - m_sec)
            if diff > _CV_THR:
                flags.append(f'CV1_{eye}: avg_gcl={avg_gcl} vs sec_mean={m_sec:.1f} (diff={diff:.1f})')

        # Rule 2: avg_rnfl vs mean(4쿼드런트)
        avg_rnfl = norm_row.get(f'{eye}_avg_rnfl')
        q_vals   = [quad.get(f'{eye}_{d}') for d in ('S', 'T', 'I', 'N')]
        m_quad   = _mean(q_vals)
        if avg_rnfl is not None and m_quad is not None and len([x for x in q_vals if x]) >= 2:
            diff = abs(avg_rnfl - m_quad)
            if diff > _CV_THR:
                flags.append(f'CV2_{eye}: avg_rnfl={avg_rnfl} vs quad_mean={m_quad:.1f} (diff={diff:.1f})')

        # Rule 3: min_gcl_table vs min(섹터)
        min_gcl = norm_row.get(f'min_gcl_{eye}')
        if min_gcl is not None and m_sec is not None:
            min_sec = min(x for x in secs if x is not None) if any(x for x in secs if x) else None
            if min_sec is not None and min_gcl > min_sec + _CV_THR:
                flags.append(f'CV3_{eye}: min_gcl_table={min_gcl} > min_sec={min_sec}')

        # Rule 4: avg_rnfl vs mean(clock hours 12개)
        clk_vals = [clk.get(f'{eye}_h{n:02d}') for n in range(1, 13)]
        m_clk    = _mean(clk_vals)
        n_clk    = len([x for x in clk_vals if x])
        if avg_rnfl is not None and m_clk is not None and n_clk >= 6:
            diff = abs(avg_rnfl - m_clk)
            if diff > _CV_THR:
                flags.append(f'CV4_{eye}: avg_rnfl={avg_rnfl} vs clk_mean={m_clk:.1f} (diff={diff:.1f},n={n_clk})')

    return flags


# ── 추출 메인 ────────────────────────────────────────────────
def extract_one(ml_row):
    gca  = ml_row.get('gca_dir', '')
    rnfl = ml_row.get('rnfl_dir', '')
    gcl    = parse_gcl_table(os.path.join(gca,  'gcl_thickness_table.png') if gca else '')
    od_s   = parse_sectors(os.path.join(gca, 'od_sectors.png') if gca else '', 'OD')
    os_s   = parse_sectors(os.path.join(gca, 'os_sectors.png') if gca else '', 'OS')
    rnfl_s = parse_rnfl_summary(os.path.join(rnfl, 'rnfl_summary_table.png') if rnfl else '')
    # CV 전용 보조 추출
    quad   = parse_quadrants_cv(os.path.join(rnfl, 'quadrants.png') if rnfl else '')
    clk    = parse_clockhours_cv(os.path.join(rnfl, 'clockhours.png') if rnfl else '')
    return {**gcl, **od_s, **os_s, **rnfl_s}, quad, clk


FIELDS = (
    ['patient_id', 'eye'] +
    ['avg_gcl_od', 'avg_gcl_os', 'min_gcl_od', 'min_gcl_os'] +
    [f'od_s_{k}' for k in ['sup', 'sup_t', 'inf_t', 'inf', 'inf_n', 'sup_n']] +
    [f'os_s_{k}' for k in ['sup', 'sup_t', 'inf_t', 'inf', 'inf_n', 'sup_n']] +
    ['od_avg_rnfl', 'os_avg_rnfl', 'od_vert_cd', 'os_vert_cd', 'cv_flags']
)

def normalize_row(pid, eye, vals, quad, clk, flagged_log=None):
    r = {'patient_id': pid, 'eye': eye}
    if flagged_log is None:
        flagged_log = []
    for k in FIELDS:
        if k in ('patient_id', 'eye', 'cv_flags'):
            continue
        r[k] = apply_sanity(k, vals.get(k), flagged_log)
    cv = cross_validate(r, quad, clk)
    r['cv_flags'] = '|'.join(cv) if cv else ''
    return r


if __name__ == '__main__':
    ml = list(csv.DictReader(open(ML_CSV, encoding='utf-8-sig')))
    seen = set()
    ok = err = 0
    total = len(ml)

    done_keys = set()
    if os.path.exists(OUT_CSV):
        for ex in csv.DictReader(open(OUT_CSV, encoding='utf-8-sig')):
            done_keys.add((ex['patient_id'], ex['eye']))
        print(f'기존 {len(done_keys)}건 → 이어서 진행', flush=True)

    write_header = not os.path.exists(OUT_CSV) or len(done_keys) == 0
    f_out = open(OUT_CSV, 'a' if done_keys else 'w', newline='', encoding='utf-8-sig')
    writer = csv.DictWriter(f_out, fieldnames=FIELDS, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    t0 = time.time()
    for row in ml:
        pid, eye = row['patient_id'], row['eye']
        key = (pid, eye)
        if key in seen or key in done_keys:
            continue
        seen.add(key)

        t1 = time.time()
        print(f'[{ok+err+1:3d}/{total}] {pid} {eye} ...', end=' ', flush=True)
        try:
            sanity_flags = []
            vals, quad, clk = extract_one(row)
            norm_row = normalize_row(pid, eye, vals, quad, clk, sanity_flags)
            writer.writerow(norm_row)
            f_out.flush()
            n_ok    = sum(1 for k, v in norm_row.items()
                          if k not in ('patient_id', 'eye', 'cv_flags') and v is not None)
            elapsed = time.time() - t1
            cv_str  = f'  CV:{norm_row["cv_flags"]}' if norm_row['cv_flags'] else ''
            san_str = f'  sanity:{len(sanity_flags)}' if sanity_flags else ''
            print(f'ok {n_ok}/20  ({elapsed:.1f}s){san_str}{cv_str}', flush=True)
            ok += 1
        except Exception as e:
            print(f'ERR: {e}', flush=True)
            err += 1

    f_out.close()
    total_t = time.time() - t0
    print(f'\n=== 완료 === 성공:{ok}  에러:{err}  총소요:{total_t/60:.1f}분')
    print(f'저장: {OUT_CSV}')
