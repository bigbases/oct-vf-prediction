#!/usr/bin/env python3
"""반대편 눈 GCA 섹터의 좌표 프레임 판별 — 가정하지 않고 데이터로 결정한다.

배경: flip CSV 의 flip 은 `eye=='OS'` 행에만, `os_*` 섹터에만 적용된다
(scripts/apply_vf_neg1_to_zero.py:118). 따라서 study 눈 피처는 전부 OD-normalized 지만
반대편 눈 섹터는 한쪽 행 그룹에서 프레임이 어긋난다. **어느 그룹이 어긋나는지는 원본
컬럼 이름이 해부학 기준인지 화면 위치 기준인지에 따라 정반대가 되므로 추측하면 안 된다.**

판별 원리: 한 사람의 두 눈은 해부학적으로 상관된다. 프레임이 맞으면
  corr(study_sup_t, fellow_sup_t) > corr(study_sup_t, fellow_sup_n)
가 성립해야 한다. 이 부등호를 OD 행과 OS 행에서 따로 재면, 어긋난 그룹에서만 부등호가
뒤집힌다. 그 그룹의 fellow 섹터에 t/n 스왑을 적용하는 것이 교정이다.

읽기 전용. 출력: docs/fellow_eye_frame_check.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [('s_sup_t', 's_sup_n'), ('s_inf_t', 's_inf_n')]


def load(name: str):
    return list(csv.DictReader(open(ROOT / name, encoding='utf-8-sig')))


def fnum(row, col):
    v = row.get(col, '')
    try:
        return float(v) if v not in ('', None) else np.nan
    except ValueError:
        return np.nan


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float('nan'), int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def dedupe_by_patient(rows):
    seen, out = set(), []
    for r in rows:
        p = r['patient_id']
        if p not in seen:
            seen.add(p)
            out.append(r)
    return out


def probe(rows, eye: str, swap_fellow: bool):
    """study eye = `eye`. fellow 섹터에 t/n 스왑을 걸지 말지 선택해 부등호를 잰다."""
    sub = dedupe_by_patient([r for r in rows if r['eye'] == eye])
    sp = 'od' if eye == 'OD' else 'os'
    fp = 'os' if eye == 'OD' else 'od'
    lines = []
    for t_suf, n_suf in PAIRS:
        s_t = [fnum(r, f'{sp}_{t_suf}') for r in sub]
        f_t_col, f_n_col = f'{fp}_{t_suf}', f'{fp}_{n_suf}'
        if swap_fellow:
            f_t_col, f_n_col = f_n_col, f_t_col
        f_t = [fnum(r, f_t_col) for r in sub]
        f_n = [fnum(r, f_n_col) for r in sub]
        c_same, n1 = corr(s_t, f_t)
        c_cross, _ = corr(s_t, f_n)
        lines.append(dict(sector=t_suf, n=n1, same=c_same, cross=c_cross,
                          aligned=bool(c_same > c_cross)))
    return len(sub), lines


def report(title, rows):
    out = [f'### {title}', '']
    verdicts = {}
    for eye in ('OD', 'OS'):
        for swap in (False, True):
            n, ls = probe(rows, eye, swap)
            tag = 'fellow 스왑 적용' if swap else '원본 그대로'
            out.append(f'- **study {eye}** ({n}환자), {tag}')
            for d in ls:
                mark = 'OK' if d['aligned'] else '뒤집힘'
                out.append(f"    - {d['sector']}: same t-t rho={d['same']:+.3f}, "
                           f"cross t-n rho={d['cross']:+.3f} → {mark} (n={d['n']})")
            verdicts[(eye, swap)] = all(d['aligned'] for d in ls)
    out.append('')
    return out, verdicts


def main() -> None:
    md = ['# 반대편 눈 GCA 섹터 프레임 판별', '',
          '`scripts/check_fellow_eye_frame.py` 산출. 읽기 전용 검사.', '',
          '판별 원리: 한 사람의 두 눈은 해부학적으로 상관되므로 프레임이 맞으면',
          '`corr(study_sup_t, fellow_sup_t) > corr(study_sup_t, fellow_sup_n)` 이어야 한다.',
          '이 부등호가 뒤집힌 행 그룹이 프레임이 어긋난 쪽이다.', '']

    all_verdicts = {}
    for name, title in (
        ('ml_final_90d_excl_empty.csv', 'non-flip CSV (원본 추출 프레임, 참조)'),
        ('ml_final_90d_excl_empty_flip.csv', 'flip CSV (정본 학습이 쓰는 파일)'),
    ):
        if not (ROOT / name).exists():
            md += [f'### {title}', '', f'- 파일 없음: `{name}`', '']
            continue
        block, v = report(title, load(name))
        md += block
        all_verdicts[name] = v

    fl = all_verdicts.get('ml_final_90d_excl_empty_flip.csv', {})
    md += ['## 판정', '']
    if fl:
        od_raw, os_raw = fl.get(('OD', False)), fl.get(('OS', False))
        od_sw, os_sw = fl.get(('OD', True)), fl.get(('OS', True))
        md.append(f'- flip CSV, 원본 그대로: study OD 정합={od_raw}, study OS 정합={os_raw}')
        md.append(f'- flip CSV, fellow 스왑: study OD 정합={od_sw}, study OS 정합={os_sw}')
        md.append('')
        if od_raw and not os_raw:
            rule = ('**OS 행의 fellow(`od_*`) 섹터에 t/n 스왑을 적용한다.** '
                    'OD 행은 손대지 않는다. 원본 컬럼 이름이 해부학 기준이라는 뜻이다.')
        elif os_raw and not od_raw:
            rule = ('**OD 행의 fellow(`os_*`) 섹터에 t/n 스왑을 적용한다.** '
                    'OS 행은 손대지 않는다. 원본 컬럼 이름이 화면 위치 기준이라는 뜻이다.')
        elif od_raw and os_raw:
            rule = ('두 그룹 모두 원본에서 정합이다. **교정이 필요 없다.** '
                    '반대편 눈 섹터를 그대로 써도 프레임이 일치한다.')
        else:
            rule = ('두 그룹 모두 원본에서 뒤집혔다. 섹터 이름 규약 자체를 다시 확인해야 한다. '
                    '**이 상태에서는 반대편 눈 섹터를 쓰지 않는다.**')
        md += [f'- 결론: {rule}', '']

    md += ['## 주의', '',
           '- 이 검사는 GCA 상/하 t·n 섹터에만 적용된다. 방향성이 없는 `avg_gcl`, `min_gcl`,',
           '  `avg_rnfl`, `vert_cd`, 그리고 `s_sup`·`s_inf` 는 스왑 대상이 아니다.',
           '- 부등호가 뒤집힌 그룹을 "버그"로 단정하기 전에 이 표를 먼저 본다. 이 프로젝트에는',
           '  정상 데이터를 버그로 오진해 정답을 손상시킨 전례가 있다',
           '  (`docs/flip_convention_FINAL.md` §7).', '']

    out = ROOT / 'docs/fellow_eye_frame_check.md'
    out.write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
