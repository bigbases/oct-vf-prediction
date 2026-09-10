"""
clockhours.csv / clockhours_180d.csv 의 OS→OD flip 버전 생성.

OS 행만 좌우 반전 적용:
  clock hours (수평 반전, h01=1시·h12=12시 컨벤션 / ocr_rnfl_detail.py 기준):
    H12(12시), H06(6시) = 수직축 고정점
    H01↔H11, H02↔H10, H03↔H09, H04↔H08, H05↔H07

사분면은 반전하지 않는다 (2026-08-27).
  이전에는 rnfl_q_t ↔ rnfl_q_n 을 여기서 맞바꿨는데, 그것이 이중 적용이었다.
  시계시간 h01..h12 는 위치 이름이라 OD/OS 어느 쪽이든 같은 화면 좌표를
  가리키고, 따라서 여기서 한 번 반전해야 해부학적 프레임이 맞는다. 반면
  사분면 이름 S/T/I/N 은 `ocr_rnfl_detail.py:80-89` 에서 이미 안별로 다른
  좌표에서 읽는다 — od_T 는 x=0.040(왼쪽 끝), os_T 는 x=0.985(오른쪽 끝).
  즉 OCR 단계에서 이미 양안 모두 T=측두 로 붙어 나온다. 여기서 다시 바꾸면
  OS 행의 rnfl_q_t 에 비측 값이 들어간다.

  근거는 행 단위 항등식으로 확인된다: 각 사분면은 자기 3개 시계시간의
  평균이다. 수정 후 양안 모두 q_t↔h08-10, q_n↔h02-04 에서 중앙값 0.33 µm
  로 맞는다. 감사 기록은 docs/LATERALITY_AUDIT.md.

원본을 덮어쓰지 않으려면 --out-dir 로 다른 디렉터리를 준다. import 만으로
파일이 쓰이지 않도록 __main__ 가드 아래에서만 실행된다.
"""
import argparse
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))

QUAD_SWAP = []  # 위 docstring 참조. OCR 단계에서 이미 안별로 붙었다
HOUR_SWAP = [('rnfl_h01', 'rnfl_h11'), ('rnfl_h02', 'rnfl_h10'),
             ('rnfl_h03', 'rnfl_h09'), ('rnfl_h04', 'rnfl_h08'),
             ('rnfl_h05', 'rnfl_h07')]
# h06, h12 고정

PAIRS = [('clockhours.csv', 'clockhours_flip.csv'),
         ('clockhours_180d.csv', 'clockhours_180d_flip.csv')]


def flip_row(row: dict) -> dict:
    r = dict(row)
    for a, b in QUAD_SWAP + HOUR_SWAP:
        r[a], r[b] = row[b], row[a]
    return r


def process(src_name: str, dst_name: str, src_dir: Path = None, out_dir: Path = None):
    src_dir = src_dir or ROOT
    out_dir = out_dir or ROOT
    rows = list(csv.DictReader(open(src_dir / src_name, encoding='utf-8-sig')))
    fieldnames = list(rows[0].keys())
    out, n_os = [], 0
    for row in rows:
        if row.get('eye', '').upper() == 'OS':
            out.append(flip_row(row))
            n_os += 1
        else:
            out.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / dst_name, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f'{out_dir / dst_name}: {len(rows)}행 저장 (OS flip {n_os}건)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=Path, default=ROOT)
    ap.add_argument('--out-dir', type=Path, default=ROOT,
                    help='기본값은 ROOT — 정본을 덮어쓴다. 대조용이면 다른 곳을 준다')
    args = ap.parse_args()
    for src, dst in PAIRS:
        process(src, dst, args.src_dir, args.out_dir)
    print('완료')


if __name__ == '__main__':
    main()
