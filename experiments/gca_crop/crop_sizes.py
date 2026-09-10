#!/usr/bin/env python3
"""분석 코호트 277안의 GCA/RNFL crop 크기를 한 번 재서 캐시한다.

crop 규칙은 image_preprocessing._crop_gca_thickness / _crop_rnfl_thickness 와
같다. 그 모듈은 torch 를 import 하므로 env aaa 에서 쓸 수 없어, 그림 스크립트
(scripts/make_fig_pipeline.py) 와 같은 방식으로 두 함수만 복제한다.

안 목록은 fusion_eval_common.load_all() 이 정하는 그대로다(XGB + 5백본 키 교집합).
읽기 전용. 출력: experiments/gca_crop/crop_sizes.json
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT))
from paths import rewrite_data_path            # noqa: E402
from fusion_eval_common import load_all        # noqa: E402


def gca_box(img):
    a = np.array(img.convert('RGB')).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > r + 40) & (b > g + 40) & (b > 80)
    rows = np.where(blue.sum(axis=1) > 30)[0]
    cols = np.where(blue.sum(axis=0) > 30)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img.size                     # crop 실패 시 원본 그대로
    return int(cols[-1]) - int(cols[0]) + 1, int(rows[-1]) - int(rows[0]) + 1


def rnfl_box(img, top_margin=16):
    W, H = img.size
    sq = H - top_margin
    a = np.array(img.convert('RGB')).astype(float)
    mx, mn = a.max(axis=2), a.min(axis=2)
    cols = np.where((((mx - mn) / (mx + 1e-5) > 0.2) & (mx > 30)).any(axis=0))[0]
    if len(cols) == 0:
        return img.size
    right = int(cols[-1]) + 1
    return right - max(0, right - sq), H - top_margin


def index():
    idx = {}
    with open(ROOT / 'ml_dataset.csv', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            d = {}
            for tag, col in (('gca', 'gca_dir'), ('rnfl', 'rnfl_dir')):
                base = rewrite_data_path(row.get(col, ''))
                p = os.path.join(base, f"{row['eye'].lower()}_thickness_map.png") if base else ''
                if p and os.path.exists(p):
                    d[tag] = p
            idx[(row['patient_id'], row['eye'], row['vf_date'])] = d
    return idx


def main() -> int:
    oof, tst, _, _ = load_all()
    keys = {'OOF': sorted({k for j in range(5) for k in oof[j]['keys']}),
            'TEST': sorted(tst[0]['keys'])}
    idx = index()
    out = {}
    for split, ks in keys.items():
        rec = {}
        for k in ks:
            p = idx[tuple(k)]
            gw, gh = gca_box(Image.open(p['gca']))
            rw, rh = rnfl_box(Image.open(p['rnfl']))
            rec['|'.join(k)] = dict(gca_w=gw, gca_h=gh, rnfl_w=rw, rnfl_h=rh)
        out[split] = rec
        g = np.array([[v['gca_w'], v['gca_h']] for v in rec.values()])
        r = np.array([[v['rnfl_w'], v['rnfl_h']] for v in rec.values()])
        print(f'{split}: {len(rec)}안  GCA 고유 {len({tuple(x) for x in g})}종 '
              f'W {g[:,0].min()}~{g[:,0].max()} H {g[:,1].min()}~{g[:,1].max()}  '
              f'RNFL 고유 {len({tuple(x) for x in r})}종')
    p = ROOT / 'experiments/gca_crop/crop_sizes.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'저장: {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
