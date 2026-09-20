#!/usr/bin/env python3
"""원고 sections/*.tex의 모든 수치를 근거 산출물과 대조한다.

근거 풀:
  1. docs/CANONICAL_SPEC.md                      (정본 명세 §1~§4 발췌)
  2. runs/*.json                                 (기계가 만든 산출물)
  3. 이 파일의 STRUCTURAL 허용목록            (프로토콜·하이퍼파라미터 상수)

출력은 세 갈래:
  [OK]   근거를 찾음 — 어느 파일에서 찾았는지 같이 찍는다
  [CITE] \\cite가 붙은 줄의 수치 — 문헌 인용값이므로 자동검증 대상 아님
  [MISS] 근거 없음 — file:line과 함께 보고. **이게 이 스크립트의 존재 이유다.**

이 스크립트는 보고 도구가 아니라 **게이트**다. 결과는 종료 코드로 드러난다:

  0  위반 없음. 또는 남은 위반이 전부 verify/exceptions.yaml의 승인된 예외다.
  1  위반이 남았다 — MISS / STALE 앵커 / 플레이스홀더 중 하나라도.
  2  검사 자체를 못 돌렸다 — 근거 파일 없음, YAML 파싱 실패, 예외 설정 오류 등.
     **2를 0으로 오인하면 안 된다.** 검사 불능은 통과가 아니다.

부속 파일 (전부 verify/ 아래):
  verify/exceptions.yaml            승인된 예외. {id, reason, approved_by, date} 필수
  verify/placeholder_patterns.yaml  플레이스홀더 문자열 목록 (코드에 박지 않는다)
  verify/logs/run_YYYYMMDD_HHMMSS.log  실행 로그. stdout과 같은 내용을 남긴다

env: hvf. PyYAML이 있으면 쓰고, 없으면 내장 최소 파서로 폴백한다
     (CHECK_NUMBERS_NO_YAML=1 로 폴백 경로를 강제할 수 있다 — 회귀 테스트용).
사용:
  python scripts/check_numbers.py            # 게이트 실행 (요약 + 위반 목록)
  python scripts/check_numbers.py --all      # OK까지 전부
  python scripts/check_numbers.py --file 04_results.tex
  python scripts/check_numbers.py --selftest # 숫자 매칭 경계조건 회귀 테스트
  python scripts/check_numbers.py --json out.json   # 위반 목록을 기계가 읽게 덤프
  python scripts/check_numbers.py --root /tmp/fixture  # 임시 픽스처 대상으로 실행
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from collections import namedtuple
from pathlib import Path

# --- 종료 코드 --------------------------------------------------------------
# 게이트의 계약. 셋은 절대 섞이면 안 된다.
EXIT_PASS = 0    # 위반 없음, 또는 남은 위반이 전부 승인된 예외
EXIT_FAIL = 1    # 위반이 남음 (MISS / STALE / 플레이스홀더)
EXIT_ERROR = 2   # 검사 불능 (파일 없음, 파싱 실패, 예외 설정 오류)

EXCEPTION_MAX_AGE_DAYS = 30   # 이보다 오래된 예외는 경고만 하고 통과시킨다


class GateError(Exception):
    """검사를 수행할 수 없다.

    이 예외는 EXIT_ERROR로 나간다. 검사를 못 돌린 것을 '위반 없음'으로
    오인하는 것이 이 게이트의 가장 위험한 실패 모드이므로, 통과(0)는
    물론이고 위반(1)과도 코드를 구분한다.
    """


def _spec_default(root: Path) -> Path:
    """SPEC 을 어디서 찾을지. 저장소 사본이 우선이다.

    이 저장소에는 원고 트리가 없다(README '무엇이 빠져 있나' 참조). 그래서
    §1~§4 발췌를 `docs/CANONICAL_SPEC.md` 로 동봉한다 — 프로토콜 상수와
    §2.1/§2.2 표 앵커가 거기 있어야 복제본에서도 게이트가 돈다.
    `docs/journal_manuscript/` 배치는 원고 트리와 같이 쓰는 로컬 작업용으로
    남겨 두고, 둘 다 있으면 저장소 사본을 쓴다(--manuscript 는 여전히
    원고 쪽 것으로 덮는다).
    """
    here = root / 'docs/CANONICAL_SPEC.md'
    return here if here.exists() else root / 'docs/journal_manuscript/CANONICAL_SPEC.md'


ROOT = Path(__file__).resolve().parent.parent
SEC = ROOT / 'docs/journal_manuscript/sections'
SPEC = _spec_default(ROOT)
RUNS = ROOT / 'runs'
MAIN_TEX = ROOT / 'docs/journal_manuscript/main.tex'
VERIFY = ROOT / 'verify'
EXCEPTIONS_FILE = VERIFY / 'exceptions.yaml'
PATTERNS_FILE = VERIFY / 'placeholder_patterns.yaml'
LOG_DIR = VERIFY / 'logs'


def configure(root):
    """검사 대상 트리를 바꾼다. 회귀 테스트가 임시 픽스처를 가리킬 때 쓴다.

    정본 파일을 건드리지 않고 게이트를 시험하려면 경로가 한 곳에서
    갈라져야 한다. 모듈 전역을 다시 묶는 방식이라 --root 를 준 프로세스
    안에서만 유효하다.
    """
    global ROOT, SEC, SPEC, RUNS, MAIN_TEX
    global VERIFY, EXCEPTIONS_FILE, PATTERNS_FILE, LOG_DIR
    ROOT = Path(root).resolve()
    SEC = ROOT / 'docs/journal_manuscript/sections'
    SPEC = _spec_default(ROOT)
    RUNS = ROOT / 'runs'
    MAIN_TEX = ROOT / 'docs/journal_manuscript/main.tex'
    VERIFY = ROOT / 'verify'
    EXCEPTIONS_FILE = VERIFY / 'exceptions.yaml'
    PATTERNS_FILE = VERIFY / 'placeholder_patterns.yaml'
    LOG_DIR = VERIFY / 'logs'


def configure_runs(path):
    """근거 산출물 디렉터리만 갈아끼운다. 원고와 검사 설정은 그대로 둔다.

    배포본(release/)에는 runs/ 가 없다 — 그 디렉터리는 PHI 를 포함해서
    공개 트리에서 빠졌고, 공개된 근거는 paper/results_frozen/ 다. 트리
    전체를 옮기는 --root 로는 verify/ 까지 따라가버리므로 근거 경로만
    따로 받는다.
    """
    global RUNS
    d = Path(path).resolve()
    if not d.is_dir():
        raise GateError(f'근거 디렉터리가 아니다: {d}')
    if not sorted(d.glob('*.json')):
        raise GateError(f'근거 디렉터리에 *.json 이 없다: {d}')
    RUNS = d


def configure_manuscript(path):
    """원고 경로만 갈아끼운다. 산출물(runs/)과 검사 설정(verify/)은 그대로 둔다.

    정본 원고는 이 저장소에 없다 (Overleaf). 저장소 안의
    docs/journal_manuscript/sections/ 는 낡은 사본이고, 그것을 검사하면
    게이트가 통과해도 정본을 검증한 것이 아니다. 그래서 원고 경로는
    --root(트리 전체 이동)와 분리해서 따로 받는다.

    받는 것은 *.tex 가 들어 있는 디렉터리다. 같은 디렉터리에 main.tex /
    CANONICAL_SPEC.md 가 있으면 그것도 같이 쓴다 (없으면 저장소 것을 유지).
    """
    global SEC, SPEC, MAIN_TEX
    d = Path(path).resolve()
    if not d.is_dir():
        raise GateError(f'원고 디렉터리가 아니다: {d}')
    if not sorted(d.glob('*.tex')):
        raise GateError(f'원고 디렉터리에 *.tex 가 없다: {d}')
    SEC = d
    # main.tex / CANONICAL_SPEC.md 는 sections/ 안이 아니라 그 부모에 있는
    # 배치가 흔하다 (이 저장소가 그렇다). 두 곳을 다 본다.
    for base in (d, d.parent):
        if (base / 'main.tex').exists():
            MAIN_TEX = base / 'main.tex'
            break
    for base in (d, d.parent):
        if (base / 'CANONICAL_SPEC.md').exists():
            SPEC = base / 'CANONICAL_SPEC.md'
            break

# ---------------------------------------------------------------------------
# 구조 상수 허용목록.
# 근거 산출물이 아니라 프로토콜·모델 설계에서 오는 값. CANONICAL_SPEC §2/§3에
# 산문으로 적혀 있어 숫자 대조로는 안 잡히는 것들만 여기 둔다.
# 새 값을 넣을 때는 반드시 근거 한 줄을 같이 적을 것. 빈칸 메우기 금지.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 앵커. 값만 대조하면 위양성이 크다. **왜 큰지는 2026-09-17 에 다시 쟀고,
# 그 전에 적어 둔 "10~27%" 는 틀렸다** — 풀 4,683개에 대해 원고 수치 하나가
# 여러 곳과 맞을 확률은 최대 1.6% 다. 위험은 확률이 아니라 **주제 불일치**다.
# scan_numbers 는 `where = paths[0]`, 즉 풀에서 먼저 맞는 값이 이긴다. 그래서
# [OK] 는 "근거가 있다"가 아니라 "풀 어딘가에 인쇄 자릿수까지 같은 값이
# 있다"는 뜻일 뿐이고, 실제로 severity 행수 235 는 case_profile.json 의
# 무관한 값으로, 표 3 Pooled Δ 여섯 칸은 서로 다른 여섯 파일의 잔값으로
# 통과하고 있었다.
# 그래서 중요한 값은 **JSON 경로로 못 박는다.** 여기 있는 값이 원고와 다르면
# 그건 우연 일치가 아니라 진짜 불일치다.
# 형식: (runs 파일, 점 표기 경로, 사람이 읽을 이름)
# ---------------------------------------------------------------------------
SPEC_ANCHOR = 'CANONICAL_SPEC.md §2.1'


def spec_md_table():
    """SPEC §2.1의 277쌍 기준 MD 통계 표를 읽는다.

    앵커 근거를 이 스크립트 안에 상수로 박으면 정본이 두 벌이 된다.
    (그게 이번에 280쌍 오류가 통과한 원인의 축소판이다.) 그래서 값은
    항상 SPEC에서 파싱한다. 표 형식은 `| Quantity | 280 pairs | **277 pairs** |`
    이고
    정본은 굵게 표시한 세 번째 칸이다. `range` 행은 두 기준이 같아
    앵커로 쓰지 않는다.
    """
    if not SPEC.exists():
        raise GateError(f'정본 명세가 없다: {SPEC}')
    text = SPEC.read_text(encoding='utf-8')
    m = re.search(r'^### 2\.1 .*?$(.*?)^(?=### |## )', text, re.S | re.M)
    if not m:
        raise GateError(f'{SPEC.name}에 §2.1 절이 없다. 표를 파싱할 수 없다.')
    out = {}
    for line in m.group(1).split('\n'):
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) != 3 or not cells[2].startswith('**'):
            continue
        # 헤더 행. 세 번째 칸이 굵어서 걸린다. 공개본은 영어라 둘 다 본다.
        if cells[0] in ('Quantity', '항목'):
            continue
        num = re.search(r'-?\d+(?:\.\d+)?', cells[2].strip('*'))
        if num:
            # 정수는 int로 둔다. float로 두면 271이 "271.0"으로 조회돼
            # 원고의 "271 pairs"를 못 찾는다.
            t = num.group()
            out[cells[0]] = float(t) if '.' in t else int(t)
    # 행 이름 별칭. 공개 발췌본은 영어이고 원고 트리의 전문은 한국어라 SPEC 이
    # 두 벌 돌아다닌다. 어느 쪽을 --manuscript 로 집어도 앵커가 풀리게 한다
    # (안 그러면 전문 쪽에서 NOKEY → exit 2 로 멎는다. 2026-09-17 실측).
    for a, b in (('pairs with MD', 'MD 있는 쌍'),):
        if a not in out and b in out:
            out[a] = out[b]
        elif b not in out and a in out:
            out[b] = out[a]
    return out


ANCHORS = [
    # 코호트 구조
    ('skeleton_numbers.json', 'labels.n_eyes', '전체 280안'),
    ('skeleton_numbers.json', 'labels.n_patients', '전체 145명'),
    ('skeleton_numbers.json', 'split_structure.n_eyes_total', 'split 대상 280안'),
    ('skeleton_numbers.json', 'split_structure.cv_fold_eye_counts.0', 'fold0 47안'),
    ('skeleton_numbers.json', 'split_structure.cv_fold_eye_counts.1', 'fold1 50안'),
    ('skeleton_numbers.json', 'split_structure.cv_fold_eye_counts.4', 'fold4 51안'),
    ('skeleton_numbers.json', 'split_structure.cv_fold_eye_counts.test', 'held-out 38안'),
    # labels.n_cells_52pt(14560) 은 뺐다. 구본에만 있던 문장이고 제출본에는
    # 없다 — 앵커로 남겨두면 영구 STALE 이다 (2026-09-17, 사용자 승인).
    ('skeleton_numbers.json', 'oct_vf_gap_days.mean', 'OCT-VF 간격 평균'),
    ('skeleton_numbers.json', 'oct_vf_gap_days.sd', 'OCT-VF 간격 SD'),
    ('skeleton_numbers.json', 'oct_vf_gap_days.pct_same_day', '당일 촬영 80.0%'),
    ('skeleton_numbers.json', 'cohort_severity_EMR_MD.n_missing', 'MD 조인 실패 6안'),
    # skeleton_numbers.json의 cohort_severity_EMR_MD는 280쌍 기준이다.
    # 원고는 277쌍 기준이므로 앵커 근거를 SPEC §2.1로 옮겼다.
    # JSON은 runs/* 수정 금지 규약에 따라 그대로 둔다.
    (SPEC_ANCHOR, 'pairs with MD', 'MD 있는 271쌍'),
    (SPEC_ANCHOR, 'mean', 'MD 평균 -7.39'),
    (SPEC_ANCHOR, 'SD', 'MD SD 8.70'),
    (SPEC_ANCHOR, 'median', 'MD 중앙값 -3.97'),
    (SPEC_ANCHOR, 'normal (MD > -3)', 'normal 40.6%'),
    (SPEC_ANCHOR, 'early (-6 < MD <= -3)', 'early 22.5%'),
    (SPEC_ANCHOR, 'moderate (-12 < MD <= -6)', 'moderate 12.5%'),
    (SPEC_ANCHOR, 'advanced (MD <= -12)', 'advanced 24.4%'),
    # 주 결과 (IR-v2)
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.n_eyes', 'OOF 240안'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.xgb.rmse', 'OOF XGB RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.xgb.mae', 'OOF XGB MAE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.cnn.rmse', 'OOF CNN RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.cnn.mae', 'OOF CNN MAE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.fusion.rmse', 'OOF fusion RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.fusion.mae', 'OOF fusion MAE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.w_xgb', 'w=0.47'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.n_eyes', 'held-out 37안'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.xgb.rmse', 'test XGB RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.cnn.rmse', 'test CNN RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.fusion.rmse', 'test fusion RMSE'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.p_fus_vs_xgb', 'test p=0.031'),
    # 환자 수준
    ('skeleton_numbers.json', 'patient_level_ir_v2.oof.n_patients', 'OOF 125명'),
    ('skeleton_numbers.json', 'patient_level_ir_v2.oof.p_fus_vs_xgb', 'OOF 환자수준 p'),
    ('skeleton_numbers.json', 'patient_level_ir_v2.test.n_patients', 'test 19명'),
    ('skeleton_numbers.json', 'patient_level_ir_v2.test.p_fus_vs_xgb', 'test 환자수준 p'),
    # 5-backbone ensemble (C4 null)
    ('skeleton_numbers.json', 'ensemble_5backbone.oof.ens_cnn.rmse', 'ensemble image RMSE'),
    ('skeleton_numbers.json', 'ensemble_5backbone.oof.ens_fusion_w047.rmse', 'ensemble fusion RMSE'),
    # held-out 앙상블 p(0.446) 는 제출본이 뺐다. 같은 자리의 문장은 이제 OOF
    # 비교다 — "not significant (p = 0.54, fusion better in 120 of 240 eyes)".
    #
    # 옮길 곳을 고르는 데 함정이 있었다. skeleton_numbers.json 의
    # ensemble_5backbone.oof.p_fus_vs_ensCNN 은 **0.593 / 115of240** 이고
    # 원고의 0.54 / 120of240 이 아니다. 원고가 보고하는 건 sixth_backbone.json
    # 의 basisB 비교(5백본+XGB fusion vs 5백본 앙상블)다. 파일명만 보고
    # skeleton 의 oof 로 옮겼으면 0.59 를 0.54 로 읽는 셈이었다 (2026-09-17).
    ('sixth_backbone.json', 'comparisons_basisB.1.p_wilcoxon', 'ensemble OOF p=0.54'),
    ('sixth_backbone.json', 'comparisons_basisB.1.n_a_better', 'ensemble OOF 120/240'),
    # severity 표의 행수와 고유 안구 수 (2026-09-17 추가).
    # 235 는 그동안 무관한 case_profile.json 의 235 에 **우연히** 맞아 통과했고,
    # 230 은 아무 데도 없어 MISS 였다. 이제 build_severity_region.py 가 둘 다
    # 기록한다. 층별 n(93+54+29+59) 합과도 맞는다.
    # 표 1 Ridge 행의 held-out MAE. 원고는 8.00 으로 인쇄돼 있었고 근거 풀의
    # 무관한 문자열 속 '8.0' 에 맞아 통과하고 있었다 — 같은 행의 나머지 셋
    # (10.07/7.42/10.50)은 근거와 정확히 일치한다. 진짜 값은 7.94 (2026-09-17).
    ('trivial_baselines.json', 'rows.2.test.mae', '표 1 Ridge held-out MAE 7.94'),
    ('severity_region.json', 'n_rows_severity', 'severity 행 235'),
    # region 표 n 열 = **안구당 점 수**(24-2 52점 격자 분할: 상 26·하 26·
    # 중심 16·주변 36). 여태 .md 표에만 찍히고 JSON 에는 없어서 셋 다 우연히
    # 통과하고 있었다 — 36 은 case_profile.json 의 md_percentile_rank=35.74,
    # 16 은 bias_by_bin 류 39개 값 중 아무거나였다. 하반구 26 은 상반구와
    # 같은 값이라 따로 걸지 않는다.
    ('severity_region.json', 'region.superior hemifield.n_points_per_eye',
     'region 상반구 26점'),
    ('severity_region.json', 'region.central ≤9°.n_points_per_eye',
     'region 중심 16점'),
    ('severity_region.json', 'region.peripheral.n_points_per_eye',
     'region 주변 36점'),
    # 좌우 미러링 절제 (§4.4). 네 수치 전부 osflip_compare.json 에 있었는데
    # 동결이 안 돼 있었다. p 둘은 그대로 있고 delta 둘은 DERIVED(delta).
    ('osflip_compare.json', 'paired_wilcoxon_flip_vs_noflip.oof_cnn.p',
     '미러링 OOF CNN p=0.79'),
    ('osflip_compare.json', 'paired_wilcoxon_flip_vs_noflip.test_cnn.p',
     '미러링 held-out CNN p=0.19'),
    # floor label 제거 후 held-out fusion vs summary (§4.6). 원고의 1.9e-4 는
    # 여태 reliability_sensitivity.json 의 **vgg16 OOF fusion_vs_cnn**
    # 1.852e-4 로 통과했다 — 백본·split·비교쌍이 전부 다른 값이다.
    ('neg1_mask_sensitivity.json', 'splits.test.paired_excl.fusion_vs_xgb.p',
     'floor 제거 held-out p=1.9e-4'),
    ('severity_region.json', 'n_eyes_severity_distinct', 'severity 고유 안구 230'),
    # 신뢰도 필터
    ('reliability_sensitivity.json',
     'backbones.inception_resnet_v2.OOF.crA_FL.n_eyes', '신뢰도 필터 A 181안'),
]


# ---------------------------------------------------------------------------
# 파생 앵커. 산출물은 **분율**(0~1)로 저장하는데 원고는 **백분율**로 인쇄하고,
# 때로는 여집합으로 뒤집어 쓴다. 예: 재학습 모델의 14 dB 미만 예측 분율
# 0.00345 를 원고는 "99.7%가 14 dB 이상"으로 쓴다. 값 대조만으로는 못 잡아
# MISS 로 떨어졌다(2026-08-26). 여기서 명시적으로 변환해 근거 풀에 넣는다.
#
# 규칙: 여기 넣는 값은 반드시 **JSON 경로로 못 박은 것**만. 일반적인 1-x 스캔은
# 금지다 — 근거 풀에 3천 개가 넘는 수치가 있어서 위양성이 폭증한다.
# 형식: (runs 파일, 점 표기 경로, kind, 사람이 읽을 이름)
#   kind='pct'      -> v * 100
#   kind='pct_comp' -> (1 - v) * 100
#   kind='diff'     -> |a - b|. 경로를 'pathA|pathB' 로 준다.
# ---------------------------------------------------------------------------
DERIVED = [
    # 표 3 'Pooled Δ' 열 — **여섯 칸 전부** 저장돼 있지 않았고 여섯 개가 전부
    # 우연히 통과하고 있었다 (2026-09-17). 열의 정의는 cnn_pooled - xgb_pooled
    # 이고 skeleton_numbers.json 하나에 양변이 다 있다. 원고가 쓴 자릿수는
    # 이 파일의 반올림된 값에서 나온다 — 앙상블 -0.545 는 8.115-8.660 이고,
    # 원시 정밀도(8.11471-8.66028)로는 -0.546 이 된다.
    ('skeleton_numbers.json',
     'backbones.inception_v3.oof.cnn.rmse|backbones.inception_v3.oof.xgb.rmse',
     'delta', 'Pooled Δ Inception-v3 +0.450'),
    ('skeleton_numbers.json',
     'backbones.inception_resnet_v2.oof.cnn.rmse|'
     'backbones.inception_resnet_v2.oof.xgb.rmse',
     'delta', 'Pooled Δ IR-v2 -0.120'),
    ('skeleton_numbers.json',
     'backbones.vgg16.oof.cnn.rmse|backbones.vgg16.oof.xgb.rmse',
     'delta', 'Pooled Δ VGG16 +0.505'),
    ('skeleton_numbers.json',
     'backbones.xception.oof.cnn.rmse|backbones.xception.oof.xgb.rmse',
     'delta', 'Pooled Δ Xception -0.308'),
    ('skeleton_numbers.json',
     'backbones.densenet121.oof.cnn.rmse|backbones.densenet121.oof.xgb.rmse',
     'delta', 'Pooled Δ DenseNet121 +0.009'),
    ('skeleton_numbers.json',
     'ensemble_5backbone.oof.ens_cnn.rmse|'
     'backbones.inception_resnet_v2.oof.xgb.rmse',
     'delta', 'Pooled Δ 5-backbone 앙상블 -0.545'),
    # 그림 3(a) 캡션의 네 구간 격차 "1.59, 1.34, 1.38 and 1.76". 영상분기와
    # 요약분기의 구간별 평균 잔차 차이이고, 저장된 건 두 배열뿐이라 차 자체는
    # 어디에도 없었다. 캡션이 앙상블을 쓴다고 명시하므로 passes.B.ENSEMBLE.
    ('bias_by_bin.json',
     'passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.0|'
     'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.0',
     'delta', '구간 [0,10) 격차 1.59'),
    ('bias_by_bin.json',
     'passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.1|'
     'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.1',
     'delta', '구간 [10,20) 격차 1.34'),
    ('bias_by_bin.json',
     'passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.2|'
     'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.2',
     'delta', '구간 [20,30) 격차 1.38'),
    ('bias_by_bin.json',
     'passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.3|'
     'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.3',
     'delta', '구간 [30,inf) 격차 1.76'),
    # 좌우 미러링 절제의 두 효과크기 (flip - noflip).
    ('osflip_compare.json', 'flip.oof_cnn.rmse|noflip.oof_cnn.rmse',
     'delta', '미러링 OOF CNN +0.147'),
    ('osflip_compare.json', 'flip.test_cnn.rmse|noflip.test_cnn.rmse',
     'delta', '미러링 held-out CNN -0.261'),
    # 14 dB floor 재학습이 바꾼 예측 분포 (04_results.tex:216-221, 05_discussion.tex:183)
    ('stepsize_sensitivity.json', 'clamp_retrain.stratified_fusion.frac_pred_below_floor',
     'pct_comp', '재학습 fusion 예측의 99.7%가 14 dB 이상'),
    ('stepsize_sensitivity.json', 'clamp_label.clamp14.stratified_fusion.frac_pred_below_floor',
     'pct_comp', '주 모델(fusion) 예측의 87.5%가 14 dB 이상'),
    ('stepsize_sensitivity.json', 'label_distribution.frac_below_14dB',
     'pct', '실측 감도의 20.4%가 14 dB 미만'),
    ('stepsize_sensitivity.json', 'label_distribution.frac_below_14dB',
     'pct_comp', '실측 감도의 79.6%가 14 dB 이상'),
    # 감도 구간별 지점 분포 (04_results.tex:332)
    ('fusion_sensitivity_trend.json', 'bin_distribution.OOF.frac_by_bin.0', 'pct', '[0,10) 17.5%'),
    ('fusion_sensitivity_trend.json', 'bin_distribution.OOF.frac_by_bin.1', 'pct', '[10,20) 9.1%'),
    ('fusion_sensitivity_trend.json', 'bin_distribution.OOF.frac_by_bin.2', 'pct', '[20,30) 48.4%'),
    ('fusion_sensitivity_trend.json', 'bin_distribution.OOF.frac_by_bin.3', 'pct', '[30,inf) 25.0%'),
    # early stopping 낙관 검사 — 원고의 0.055 는 두 이득의 차이다
    # (03_methods.tex:217, 05_discussion.tex:280). 두 항은 각각 근거가 있으나
    # 차이 자체는 어디에도 저장돼 있지 않아 MISS 로 떨어졌다(2026-08-26).
    ('skeleton_numbers.json',
     'backbones.inception_resnet_v2.oof.delta_fus_minus_xgb_rmse|'
     'backbones.inception_resnet_v2.test.delta_fus_minus_xgb_rmse',
     'diff', 'OOF 이득 0.589 - held-out 이득 0.534 = 0.055 dB'),
    # 부호 규약: 산출물은 차이(fusion - xgb)라 음수로 저장하고, 원고는 이득을
    # 양의 크기로 쓴다. 같은 양이고 부호 규약만 반대다.
    # (03_methods.tex:216-217, 05_discussion.tex:25, 05_discussion.tex:297-298)
    #
    # 등록 이유는 0.534 가 MISS 로 떴기 때문만이 아니다. 짝인 0.589 는
    # **우연히 초록으로 떠 있었다** — fusion_sensitivity_trend.json 의 무관한
    # slope_ci_hi 와 SPEC 산문의 "Primary effect 0.589 dB" 두 경로다. 둘 다
    # 근거가 아니다. 후자는 순환이라 이번에 풀에서 제외했고, 전자는 우연이므로
    # 여기서 경로로 못 박는다.
    #
    # 전수 확인(2026-08-29): skeleton_numbers.json 의 delta_fus_minus_* 20개
    # 중 원고가 크기로 인쇄하는 것은 아래 둘뿐이다. 나머지 18개는 원고에
    # 나오지 않는다. 다른 산출물의 같은 양(fusion_contribution_per_backbone,
    # xgb_diversity_control)은 이 둘과 값이 같은 중복 경로다.
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.oof.delta_fus_minus_xgb_rmse',
     'gain', 'OOF fusion 이득 0.589 dB (근거 -0.589, fus-xgb)'),
    ('skeleton_numbers.json', 'backbones.inception_resnet_v2.test.delta_fus_minus_xgb_rmse',
     'gain', 'held-out fusion 이득 0.534 dB (근거 -0.534, fus-xgb)'),
    # 분위수 뒤집힘 (results.tex:411-412). 같은 문장의 짝인 p50(+1.088 -> "rises
    # by 1.09")은 부호가 그대로라 통과하는데, p95 는 **-3.118 로 저장되고 원고는
    # "falls by 3.12" 로 크기만 쓴다.** 위의 delta_fus_minus_* 와 똑같은 부호
    # 규약 문제다. 이 값은 원고가 3.10 으로 잘못 적고 있던 자리이기도 하다
    # (2026-09-17 정정).
    ('heldout_mae_reversal.json', 'inception_resnet_v2.quantile_delta.p95',
     'gain', '95백분위 하락 3.118 dB (근거 -3.118, fusion-CNN)'),
]


# ---------------------------------------------------------------------------
# 앵커의 등록 위치.  anchor_id -> (섹션 파일, 문맥 정규식)
#
# 왜 필요한가. 원고가 앵커보다 **낮은 자릿수로** 인쇄하는 값이 있다. 근거
# 6.66일이 §3.1에 "6.7"로, 근거 p=0.113이 §4.2에 "0.11"로 적혀 있다.
# 문서 전체 스캔으로는 둘 중 하나를 고를 수 없다 — 선언 정밀도를 요구하면
# 이 넷이 STALE로 뜨고(과교정), 자릿수를 낮춰 찾으면 무관한 문단의 "0.1"과
# "0.5"에 걸린다(내부 결함 등록부 #3 의 원래 결함 — 등록부는 비공개).
#
# 그래서 **자릿수는 원고가 정하고 위치가 오탐을 막는다.** 등록 위치 근방의
# 토큰을 뽑아, 그 토큰의 인쇄 자릿수로 근거값을 반올림해 비교한다.
# 여기 등록되지 않은 앵커는 종전대로 문서 전체 스캔 + 선언 정밀도로 간다.
#
# 위치는 줄 번호가 아니라 **문맥 문자열**로 적는다. 5단계에서 원고를 고치면
# 줄 번호는 밀리지만 문장은 남는다. 문맥이 안 맞으면 NOKEY(= exit 2)라
# 조용히 통과하지 않는다.
# ---------------------------------------------------------------------------
ANCHOR_LOC = {
    # §3.1 "The interval was short in practice: $6.7 \pm 18.2$ days"
    # 근거는 6.66 / 18.18. 원고가 소수 1자리로 인쇄한다.
    'anchor:skeleton_numbers.json#oct_vf_gap_days.mean':
        ('methods.tex', r'pairs acquired on the same day'),
    'anchor:skeleton_numbers.json#oct_vf_gap_days.sd':
        ('methods.tex', r'pairs acquired on the same day'),
    # §4.2 "the patient-level test over 19 patients gives $p = 0.11$"
    # 근거는 0.113. 본문 곳곳의 "0.1"에 걸리면 안 되므로 위치를 못 박는다.
    'anchor:skeleton_numbers.json#patient_level_ir_v2.test.p_fus_vs_xgb':
        ('results.tex', r'patient-level test over 19 patients'),
    # §4.6 "the per-eye difference is not significant ($p = 0.54$,
    # fusion better in 120 of 240 eyes)". 근거는 0.5380 / 120.
    # 위치를 못 박지 않으면 0.54 는 **바로 위 표의 가중치 열 0.54** 에 걸린다.
    'anchor:sixth_backbone.json#comparisons_basisB.1.p_wilcoxon':
        ('results.tex', r'difference is not significant'),
    'anchor:sixth_backbone.json#comparisons_basisB.1.n_a_better':
        ('results.tex', r'fusion better in'),
    # §4.6 표 캡션 "the severity rows total 235" / "the 235 rows come from
    # 230 distinct eyes". 235 는 원고에 두 번 나오고 둘 다 같은 캡션이다.
    'anchor:trivial_baselines.json#rows.2.test.mae':
        ('results.tex', r'Ridge on summary parameters'),
    'anchor:severity_region.json#region.superior hemifield.n_points_per_eye':
        ('results.tex', r'Superior & '),
    'anchor:severity_region.json#region.central ≤9°.n_points_per_eye':
        ('results.tex', r'Central \(\$\\leq'),
    'anchor:severity_region.json#region.peripheral.n_points_per_eye':
        ('results.tex', r'Peripheral & '),
    'anchor:osflip_compare.json#paired_wilcoxon_flip_vs_noflip.oof_cnn.p':
        ('results.tex', r'out-of-fold CNN'),
    'anchor:osflip_compare.json#paired_wilcoxon_flip_vs_noflip.test_cnn.p':
        ('results.tex', r'held-out CNN'),
    # 캡션은 2자리로 인쇄한다(1.59). delta 의 기본 자릿수는 3자리라
    # 위치를 등록해서 원고가 자릿수를 정하게 한다.
    'derived:bias_by_bin.json#passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.0|'
    'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.0#delta':
        ('results.tex', r'sits above the summary branch by'),
    'derived:bias_by_bin.json#passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.1|'
    'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.1#delta':
        ('results.tex', r'sits above the summary branch by'),
    'derived:bias_by_bin.json#passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.2|'
    'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.2#delta':
        ('results.tex', r'sits above the summary branch by'),
    'derived:bias_by_bin.json#passes.B.ENSEMBLE.OOF.cnn_residual_mean_by_bin.3|'
    'passes.B.ENSEMBLE.OOF.xgb_residual_mean_by_bin.3#delta':
        ('results.tex', r'sits above the summary branch by'),
    'anchor:severity_region.json#n_rows_severity':
        ('results.tex', r'severity rows total'),
    'anchor:severity_region.json#n_eyes_severity_distinct':
        ('results.tex', r'rows come from'),
    # §4.4 "the 95th percentile falls by 3.12 dB". 근거는 3.118 이고 원고는
    # 소수 2자리로 인쇄한다. 위치를 못 박아야 자릿수를 원고가 정한다.
    'derived:heldout_mae_reversal.json#inception_resnet_v2.quantile_delta.p95#gain':
        ('results.tex', r'95th percentile falls by'),
}

LOC_WINDOW = 160       # 문맥 매치 앞뒤로 볼 문자 수. 한 문장이 들어갈 만큼만.


STRUCTURAL = {
    24: 'Humphrey 24-2 프로토콜 이름',
    2: '24-2의 -2 / 두께맵 2장 / 맹점 2곳',
    52: '맹점 제외 검사지점 수 (SPEC §2)',
    54: '맹점 포함 24-2 전체 지점 수',
    26: '장비 요약 정량지표 개수 (SPEC §3)',
    46: 'fellow-eye arm의 변수 개수 (26 + 20)',
    161: '두께맵 처리 해상도 (SPEC §3)',
    322: '가로 결합 입력 폭 = 161x2 (SPEC §3)',
    3: '입력 채널 수 / 시드 3개 / 자릿수',
    1024: 'regression head 1층 (SPEC §3)',
    512: 'regression head 2층 (SPEC §3)',
    256: 'regression head 3층 (SPEC §3)',
    300: '최대 epoch (SPEC §3)',
    100: 'early stopping patience (SPEC §3) / 백분율',
    64: 'batch size (SPEC §3)',
    42: 'seed (SPEC §3, train.py:46-52)',
    43: '재현 seed 2 (multiseed_fusion)',
    44: '재현 seed 3 (multiseed_fusion)',
    90: 'OCT-VF 간격 상한 90일 (SPEC §2)',
    5: '5-fold / 5 backbone',
    4: 'fold 0-4',
    0: 'fold 0 / 0 dB floor',
    1: '단수 표현 / 1.0 gradient clipping',
    10: '지수 표기 밑수',
    5000: 'bootstrap 반복수 (patient_cluster_test.py:29)',
    95: '95% 신뢰구간',
    2026: '연도',
    2011: '코호트 등록기간 시작 연도 (03_methods.tex, Cohort)',
    1000: "산문 표현 '1000-plus dimensions' (05_discussion.tex:91). 측정값 아님",
    # 장비 스캔 프로토콜. Cirrus 시신경유두 큐브 200x200, 황반 큐브 512x128.
    # 512 는 위에 regression head 로 이미 있다. 128 은 그동안 direct_test.json
    # 에 **우연히** 맞아 통과하고 있었다 (2026-09-17).
    200: '시신경유두 큐브 축당 A-scan 수 (장비 프로토콜, methods.tex)',
    128: '황반 큐브 B-scan 수 512x128 (장비 프로토콜, methods.tex)',
    # 산문 근사. "With roughly 190 pairs in each training fold" (discussion.tex:94).
    # 235 x 4/5 = 188 을 저자가 반올림한 값이고 저장된 측정값이 아니다.
    190: "산문 근사 'roughly 190 pairs in each training fold' (discussion.tex:94)",
}

# 소프트웨어 버전 문자열. **수치가 아니다.**
#
# 왜 따로 빼는가. 토큰 추출기는 "Python 3.10.18" 에서 '3.10' 을, "PyTorch
# 1.12.1" 에서 '1.12' 를 뽑는다. 그렇게 뽑힌 여섯 개가 전부 무관한 산출물에
# 우연히 맞아 [OK] 로 떠 있었고(1.12<-bias_by_bin_refit, 11.3<-fusion_error_
# structure, 3.2<-aggregation_comparability, 1.26/1.15<-bias_by_bin,
# 1.7<-aggregation_comparability), 우연을 못 찾은 '3.10' 하나만 MISS 로
# 떨어졌다. 즉 게이트가 버전 줄 전체를 "검증했다"고 말하고 있었다.
# 게다가 그 '3.10' 은 results.tex 의 진짜 3.10 문제와 화면에서 섞였다.
# (2026-09-17)
#
# 두 갈래로만 지운다 — 넓히면 진짜 측정값을 먹는다. 원고에는 "gives 7.934",
# "rises by 1.09" 처럼 **낱말 뒤에 오는 수치**가 널려 있어서, 일반적인
# '낱말+점찍힌수' 규칙은 쓸 수 없다.
#   (a) 3성분 이상(semver): 1.12.1 · 3.10.18 · 1.26.4 — 측정값은 이 모양이 없다
#   (b) 알려진 소프트웨어 이름 바로 뒤의 2성분: "CUDA 11.3" 한 건뿐이다
VERSION_SEMVER = re.compile(r'(?<![0-9.])[0-9]+(?:\.[0-9]+){2,}(?![0-9])')
VERSION_NAMED = re.compile(
    r'(?<![A-Za-z])(?:PyTorch|CUDA|cuDNN|timm|XGBoost|Python|NumPy|SciPy|'
    r'scikit-learn|scikit-image|pandas|statsmodels|matplotlib|LightGBM|'
    r'CatBoost|OpenCV|Pillow)\s+v?[0-9]+(?:\.[0-9]+)+(?![0-9])')

# LaTeX 문법에서 나오는 수치 — 내용이 아니다.
LATEX_NOISE = re.compile(
    r'\\(?:cite|citep|citet|ref|label|includegraphics|input|include|'
    r'newcommand|orcidauthor\w*|hspace|vspace|columnwidth|textwidth|'
    r'linewidth|begin|end)\s*\{[^}]*\}'
)
SCI = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*\\times\s*10\^\{?(-?[0-9]+)\}?')

# 저자가 넣은 줄바꿈이 지수 표기 한가운데를 끊는 경우.
#   04_results.tex:163  ... at $p \leq 1.4 \times
#   04_results.tex:164  10^{-10}$ in every seed out-of-fold, ...
# extract() 는 한 줄씩 보므로 SCI 가 못 잡고, 남은 '10' 과 '-10' 이 근거 없는
# MISS 로 뜬다. 줄바꿈을 **지우지 않고 뒤로 옮긴다** — 줄 수가 보존돼야
# 보고되는 행번호가 밀리지 않는다.
SCI_WRAP = [
    re.compile(r'(\\times)[ \t]*\n[ \t]*(10\^\{?-?[0-9]+\}?)'),
    re.compile(r'([0-9])[ \t]*\n[ \t]*(\\times\s*10\^\{?-?[0-9]+\}?)'),
]


def join_wrapped_sci(text):
    """줄바꿈으로 갈라진 지수 표기를 한 줄로 붙인다. 줄 수는 그대로다."""
    for rx in SCI_WRAP:
        text = rx.sub(r'\1 \2\n', text)
    return text


def load_sources():
    """(값, 출처) 목록과 문자열 풀을 만든다."""
    num_src: dict[float, list[str]] = {}
    str_pool: dict[str, list[str]] = {}

    def add_num(v, where):
        num_src.setdefault(float(v), [])
        if where not in num_src[float(v)]:
            num_src[float(v)].append(where)

    def add_str(s, where):
        str_pool.setdefault(s, [])
        if where not in str_pool[s]:
            str_pool[s].append(where)

    def walk(o, where):
        if isinstance(o, dict):
            for v in o.values():
                walk(v, where)
        elif isinstance(o, list):
            for v in o:
                walk(v, where)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            add_num(o, where)
        elif isinstance(o, str):
            add_str(o.strip(), where)
            for m in re.findall(r'-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?', o):
                try:
                    add_num(float(m), where)
                except ValueError:
                    pass

    if not RUNS.is_dir():
        raise GateError(f'근거 디렉터리가 없다: {RUNS}')
    run_files = sorted(RUNS.glob('*.json'))
    if not run_files:
        raise GateError(f'근거 산출물이 하나도 없다: {RUNS}/*.json')
    for f in run_files:
        try:
            walk(json.loads(f.read_text()), f'runs/{f.name}')
        except Exception:
            continue

    # 파생 앵커 — 분율 -> 백분율 / 여집합 백분율 / 두 값의 차
    for fname, path, kind, label in DERIVED:
        jf = RUNS / fname
        if not jf.exists():
            continue
        try:
            v = derive(json.loads(jf.read_text()), path, kind)
        except Exception:
            continue
        add_num(v, f'runs/{fname}:{path} ({kind})')

    # SPEC 은 **표 행만** 근거로 쓴다. 산문 줄은 제외한다.
    #
    # 이유는 순환이다. SPEC 산문은 원고를 보고 사람이 옮겨 적은 문장이고,
    # 원고는 그 SPEC 을 근거로 검증된다. 실제 피해가 있었다: 원고의
    # "0.589 dB" 는 산출물 근거(-0.589)가 부호 규약 때문에 안 맞는데도
    # SPEC §4 의 "Primary effect 0.589 dB" 한 줄 때문에 통과하고 있었다
    # (내부 결함 등록부 #8). 산출물이 갱신돼 값이 바뀌어도
    # SPEC 산문이 옛 숫자를 들고 있으면 게이트는 계속 초록이다.
    #
    # 표 행(`| ... |`)은 다르다. §2.1 MD 통계표처럼 정본이 SPEC 에만 있는
    # 구조화된 값이고, spec_md_table() 이 이미 앵커로 파싱하는 대상이다.
    if SPEC.exists():
        rows = [l for l in SPEC.read_text().split('\n') if l.lstrip().startswith('|')]
        txt = '\n'.join(rows)
        for m in re.findall(r'-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?', txt):
            try:
                add_num(float(m), 'CANONICAL_SPEC.md (표)')
            except ValueError:
                pass
        for m in re.findall(r'[0-9]+/[0-9]+', txt):
            add_str(m, 'CANONICAL_SPEC.md (표)')

    return num_src, str_pool


def render_match(src: float, printed: str) -> bool:
    """원고에 인쇄된 자릿수로 반올림(또는 절사)했을 때 같은가."""
    import math as _m
    if not _m.isfinite(src):
        return False
    # extract()가 "1,924"를 한 토큰으로 넘긴다. float()이 못 읽으므로 여기서 뗀다.
    p = printed.strip().replace(',', '')
    if 'e' in p.lower():
        try:
            v = float(p)
        except ValueError:
            return False
        if v == 0 or src == 0:
            return src == v
        # 원고가 6.1e-11로 인쇄했으면 근거 6.09e-11과 맞다고 봐야 한다.
        # 인쇄된 가수의 자릿수로 반올림해서 비교한다.
        mant = p.lower().split('e')[0]
        nd = len(mant.split('.')[1]) if '.' in mant else 0
        return f'{src:.{nd}e}' == f'{v:.{nd}e}'
    if '.' in p:
        nd = len(p.split('.')[1])
        if f'{src:.{nd}f}' == f'{float(p):.{nd}f}':
            return True
        # 절사도 허용한다. 원고가 실제로 절사를 쓴다 —
        # 04_results.tex:317 "124 to 967"은 fusion_precision_mde의
        # 124.04 / 967.58을 반올림이 아니라 잘라서 쓴 값이다.
        #
        # ⚠️ 부호를 보존한다. 양변에 abs()를 씌우면 근거가 -0.589인데 원고가
        # 0.58로 인쇄해도 통과했다 (내부 결함 등록부 #2).
        # 이 원고에는 부호가 의미를 갖는 값이 많다 — bin별 평균 차, CI 하한,
        # delta_fus_minus_xgb. 절사는 0 쪽으로 자른다(trunc), floor가 아니다:
        # floor(-58.9) = -59라 음수에서 자릿수 절사가 아니라 반내림이 된다.
        f = 10 ** nd
        return _m.trunc(src * f) / f == float(p)
    try:
        v = float(p)
        if float(src) == v:
            return True
        # 정수도 소수 분기와 같게 **반올림과 절사를 모두** 받는다. 소수 쪽은
        # 처음부터 둘 다 받았는데 정수 쪽만 절사뿐이어서, 근거 959.52를 원고가
        # 960으로 반올림해 적은 자리가 MISS로 떴다 (제출본 §4.6 표본수 3개,
        # 2026-09-17). 근거가 없는 게 아니라 인쇄 관례가 반대였다.
        return _m.trunc(float(src)) == v or float(f'{float(src):.0f}') == v
    except ValueError:
        return False


def extract(line: str):
    """한 줄에서 (인쇄문자열, 값) 목록을 뽑는다."""
    # LaTeX 주석 제거. **이스케이프한 백분율 기호는 주석이 아니다.**
    # 그냥 split('%') 하면 "80.0\% of pairs ... (mean $6.7 \pm 18.2$)" 에서
    # \% 뒤가 통째로 잘려 6.7 과 18.2 가 **한 번도 검사되지 않는다**
    # (2026-09-17, 제출본 §3.1 에서 발견. 앵커 두 개가 STALE 로 떠서 드러났다).
    s = re.split(r'(?<!\\)%', line)[0]
    s = VERSION_NAMED.sub(' ', s)              # "CUDA 11.3"
    s = VERSION_SEMVER.sub(' ', s)             # "3.10.18" -> '3.10' 방지
    s = LATEX_NOISE.sub(' ', s)                # \cite{...} 등 제거
    out = []
    for mant, exp in SCI.findall(s):
        out.append((f'{mant}e{exp}', float(mant) * 10 ** int(exp)))
    s = SCI.sub(' ', s)
    s = re.sub(r'\\[A-Za-z]+', ' ', s)         # 남은 매크로 이름
    s = re.sub(r'[A-Za-z_]+[-_]?[0-9]+(?:[._][0-9A-Za-z]+)*', ' ', s)  # v2, fig_1 같은 식별자
    # 천 단위 구분자를 한 토큰으로 읽는다. 이 대안이 없으면 원고의
    # "1,924 locations"가 1과 924 두 수로 쪼개져 1924는 **한 번도 검사되지
    # 않고** 924는 근거 없는 MISS가 된다 (내부 결함 등록부 #4).
    # 쉼표 뒤 세 자리를 강제하므로 "124, 967"이나 "1, 2, 3" 같은 나열은
    # 쉼표 뒤 공백 때문에 걸리지 않는다.
    # 앞의 '-?' 가 부호 하이픈이다. 이게 없으면 원고의 음수가 **한 번도
    # 추출되지 않는다** — lookbehind가 '-'를 막아 '-1.47'이 통째로 사라졌다
    # (내부 결함 등록부 #6). 이번 갱신에서 바뀌는 값에 음수가 많다.
    #
    # 부호로 인정하는 조건은 lookbehind 하나가 전부다: '-' 앞이 영숫자·점·
    # 밑줄·하이픈이 아닐 때. 즉 행머리·공백·여는 괄호·'$'·'=' 뒤에서만 부호다.
    #   fold-1  -> 'd' 가 앞이라 차단 (애초에 식별자 제거로도 걸린다)
    #   24-2    -> '4' 가 앞이라 차단. '24' 만 나온다
    #   IR-v2   -> 'R' 이 앞이라 차단
    #   $-7.39$ -> '$' 가 앞이라 부호. LaTeX 수식 마이너스가 여기서 처리된다
    #   1--2    -> '-' 가 앞이라 차단 (LaTeX en-dash 범위를 음수로 읽지 않는다)
    for tok in re.findall(
            r'(?<![A-Za-z0-9._-])'
            r'(-?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?))'
            r'(?![0-9]*[A-Za-z])', s):
        try:
            out.append((tok, float(tok.replace(',', ''))))
        except ValueError:
            pass
    return out


def dig(obj, dotted):
    """점 표기 경로로 JSON을 판다. 없으면 KeyError."""
    cur = obj
    for k in dotted.split('.'):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def derive(obj, path, kind):
    """DERIVED 한 줄을 실제 값으로 바꾼다. 실패하면 예외를 그대로 올린다."""
    if kind == 'diff':
        a, b = path.split('|')
        return abs(float(dig(obj, a)) - float(dig(obj, b)))
    if kind == 'delta':
        # diff 와 달리 **부호를 살린다**. 원고가 부호로 방향을 말하는 열이
        # 있어서다 — 표 3 'Pooled Δ' 는 +0.450 과 -0.120 이 각각 "요약분기보다
        # 나쁘다/낫다"를 뜻한다. abs 를 씌우면 방향이 뒤집혀도 통과한다.
        a, b = path.split('|')
        return float(dig(obj, a)) - float(dig(obj, b))
    v = float(dig(obj, path))
    if kind == 'gain':
        # 부호 규약 변환. 근거가 진짜 그 방향인지 여기서 확인한다 —
        # 산출물이 어느 날 부호를 뒤집어 저장하기 시작하면 조용히 통과하는
        # 대신 터져야 한다. 0 은 방향이 없으므로 함께 막는다.
        if v >= 0:
            raise ValueError(
                f"kind='gain' 은 음수(fusion - baseline)를 기대한다: {path} = {v}")
        return -v
    return v * 100 if kind == 'pct' else (1 - v) * 100


ANCHOR_MAX_ND = 3      # 원고가 소수 3자리를 넘겨 인쇄하는 값은 없다


def declared_nd(val, cap=ANCHOR_MAX_ND):
    """앵커 값이 **스스로 밝히는** 소수 자릿수. 8.7 -> 1, 0.113 -> 3.

    앵커의 정밀도는 값 자신이 선언한다. CANONICAL_SPEC 표에서 온 값은 표에
    적힌 자릿수 그대로 파싱되고(-7.39 -> 2), JSON 원시 float은 자릿수를
    선언하지 않으므로 원고 관례인 cap까지 요구한다.

    이 함수가 없으면 소수 1자리까지 폴백해서 p=0.113이 본문의 "0.1"로
    통과한다 (내부 결함 등록부 #3).
    """
    for nd in range(cap + 1):
        if abs(round(val, nd) - val) < 1e-12:
            return nd
    return cap


def anchor_patterns(val):
    """앵커 값이 원고에 인쇄됐는지 찾을 정규식 목록.

    ⚠️ 경계 조건이 핵심이다. 경계가 없으면 **틀린 값이 맞는 값의
    부분문자열일 때 통과한다.** 실제로 있었던 일: SD 8.67을 소수 1자리로
    줄인 "8.7"이 정정된 값 "8.70" 안에 들어 있어서 STALE이 안 떴다
    (2026-08-27). 앞은 숫자·소수점, 뒤는 숫자를 금지한다.
    소수점은 뒤쪽에서 막지 않는다 — 문장 끝 마침표가 정상이다.
    """
    if isinstance(val, float) and 0 < abs(val) < 1e-3:
        # p값: 지수 표기로 인쇄된다
        out = []
        mant, exp = f'{val:.1e}'.split('e')
        exp = int(exp)
        for m in {mant, f'{val:.2e}'.split('e')[0]}:
            out += [rf'{m}\s*\\times\s*10\^\{{?{exp}\}}?',
                    rf'{m}\\times10\^\{{?{exp}\}}?']
        return out
    if isinstance(val, float):
        # 앵커가 선언한 정밀도 이상만 후보로 둔다. 아래로 내려가면 유효자릿수가
        # 뭉개진 값도 "원고에 있다"고 인정하게 된다.
        return [r'(?<![0-9.])' + re.escape(f'{val:.{nd}f}') + r'(?![0-9])'
                for nd in range(declared_nd(val), ANCHOR_MAX_ND + 1)]
    return [rf'(?<![0-9.]){int(val)}(?![0-9.])']


def loc_match(val, tok):
    """등록 위치에서 뽑은 토큰 하나와 앵커 값을 비교한다.

    자릿수는 **원고가 정한다** — 토큰이 인쇄한 소수 자릿수로 근거값을
    반올림해서 맞춰본다. 절사는 허용하지 않는다. 위치를 이미 좁혔으므로
    문서 전체 스캔처럼 관대할 이유가 없다.

    유효숫자를 전부 잃는 토큰은 근거로 인정하지 않는다 — 근거 0.113을
    문단 안의 "0"에 맞춰 통과시키면 위치를 못 박은 의미가 없다.
    """
    p_ = tok.replace(',', '')
    if 'e' in p_.lower():
        return render_match(val, tok)          # 지수 표기는 기존 경로가 맞다
    try:
        pv = float(p_)
    except ValueError:
        return False
    nd = len(p_.split('.')[1]) if '.' in p_ else 0
    r = f'{float(val):.{nd}f}'
    if val != 0 and float(r) == 0:
        return False
    return r == f'{pv:.{nd}f}'


def anchor_near(val, loc, files):
    """등록 위치 근방에서 앵커 값을 찾는다. 반환 (found, err).

    err 가 비어 있지 않으면 검사 불능이다 — 호출자가 NOKEY로 올린다.
    문맥이 사라졌는데 조용히 STALE/FOUND 로 떨어지면 안 된다.
    """
    fname, ctx = loc
    # 파일명은 판본마다 다르다. 내부 판본은 'NN_' 접두사로 순서를 박아 뒀고
    # (03_methods.tex), Overleaf 로 넘어간 제출본은 접두사 없이 methods.tex 다.
    # 등록부는 **어느 절이냐**를 가리키지 파일명 관례를 가리키지 않으므로,
    # 접두사를 떼고 맞춘다. 2026-09-17 제출본으로 갈아끼웠을 때 앵커 4개가
    # 전부 NOKEY 로 떨어진 것이 이 때문이었다 — 근거가 없어서가 아니었다.
    role = lambda n: re.sub(r'^\d+_', '', n)
    hit = [f for f in files if role(f.name) == role(fname)]
    if not hit:
        return False, f'등록 위치의 파일이 없다: {fname}'
    if len(hit) > 1:
        # 두 관례가 한 트리에 섞여 있으면 어느 쪽을 읽었는지 알 수 없다.
        return False, f'등록 위치가 모호하다: {[f.name for f in hit]}'
    text = join_wrapped_sci(hit[0].read_text())
    ms = list(re.finditer(ctx, text))
    if not ms:
        return False, f'등록 문맥을 찾지 못했다: {ctx!r} in {fname}'
    for m in ms:
        w = text[max(0, m.start() - LOC_WINDOW): m.end() + LOC_WINDOW]
        if any(loc_match(val, tok) for tok, _ in extract(w)):
            return True, ''
    return False, ''


def selftest():
    """회귀 테스트. 비교 로직 결함을 다시 열지 않기 위한 것.

    세 묶음이다 — 앵커 패턴(부분문자열·정밀도), render_match(부호·절사),
    extract(토큰화). 각각 실제로 일어난 결함이다: 앵커가 소수 1자리까지
    폴백해 p=0.113 이 "0.1" 로 통과했고, render_match 의 절사 분기가 부호를
    무시했고, extract 가 "1,924" 를 1 과 924 로 쪼개고 음수 부호를
    lookbehind 로 막아 버렸다. 번호는 비공개 결함 등록부 #2·#3·#4·#6 이다.
    """
    def hit(val, text):
        return any(re.search(p_, text) for p_ in anchor_patterns(val))

    cases = [
        # (값, 본문, 기대, 설명)
        (8.67, '$-7.39 \\pm 8.70$ dB', False,
         '8.67 -> "8.7"이 "8.70"의 부분문자열이어도 miss여야 한다'),
        (8.70, '$-7.39 \\pm 8.70$ dB', True, '정확히 인쇄된 값은 hit'),
        (8.67, '$-7.42 \\pm 8.67$ dB', True, '옛 값도 옛 본문에서는 hit'),
        (8.67, 'SD is 8.6789 dB', False, '뒤에 숫자가 더 붙으면 miss'),
        (8.67, 'ratio 18.67 percent', False, '앞이 숫자면 miss'),
        (8.7, 'SD 8.7 dB.', True, '문장 끝 마침표는 막지 않는다'),
        (-7.39, 'mean of $-7.39 \\pm 8.70$', True, '음수 hit'),
        (-7.39, 'mean of $-7.392$', False, '음수도 뒤 숫자면 miss'),
        (271, 'available for 271 pairs', True, '정수 hit'),
        (271, 'available for 2718 pairs', False, '정수도 뒤 숫자면 miss'),
        (27, 'available for 271 pairs', False, '정수 부분문자열 miss'),
        (240, 'all 240 out-of-fold pairs', True, 'OOF 240'),
        # ── 정밀도 폴백 (결함 등록부 #3) ──
        (0.113, 'the p value was 0.1 overall', False,
         '0.113이 소수 1자리 "0.1"로 통과하면 안 된다'),
        (0.113, 'the p value was 0.11 here', False,
         '0.113이 소수 2자리 "0.11"로도 통과하면 안 된다'),
        (0.113, 'the p value was 0.113 here', True,
         '선언된 정밀도 그대로면 hit'),
        (0.46, 'w was fixed at 0.5 throughout', False,
         '0.46이 본문의 흔한 "0.5"로 통과하면 안 된다'),
        (0.46, 'the weight 0.46 was used', True, '0.46은 자기 자릿수로 hit'),
        (0.46, 'the weight 0.460 was used', True, '뒤에 0을 더 붙인 표기도 hit'),
        (8.7, 'SD 8.70 dB', True, '1자리 앵커는 2자리 인쇄도 hit'),
        (7.964, 'RMSE of 7.96 dB', False,
         '3자리 앵커를 2자리로 줄인 인쇄는 STALE 이어야 한다'),
    ]
    bad = []
    for val, text, want, why in cases:
        got = hit(val, text)
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append((val, text, want, got, why))
        print(f'  [{mark}] {val!r:>8} in {text!r:44s} -> {got}   {why}')

    # ── render_match: 부호 (결함 등록부 #2) ──
    print()
    rm_cases = [
        (-0.589, '0.58', False, '음수 근거가 양수 인쇄를 통과하면 안 된다 (절사 분기)'),
        (-0.589, '0.589', False, '반올림 분기도 부호가 다르면 miss'),
        (-0.589, '-0.58', True, '부호까지 같으면 절사 통과'),
        (-0.589, '-0.589', True, '부호까지 같으면 반올림 통과'),
        (0.589, '0.58', True, '양수 절사는 그대로 통과'),
        (-1.47, '1.47', False, 'bin별 평균 차의 부호가 뒤집히면 miss'),
        (967.578, '967', True, '정수부 절사 — 원고가 실제로 쓰는 표기'),
        (-967.578, '967', False, '정수부 절사도 부호를 본다'),
        # ── 천 단위 구분자 (결함 등록부 #4) ──
        (1924, '1,924', True, '쉼표가 든 토큰이 그대로 매칭돼야 한다'),
        (1924.0, '1,924', True, 'float 근거도 같다'),
        (1925, '1,924', False, '쉼표를 떼도 값이 다르면 miss'),
    ]
    for src, printed, want, why in rm_cases:
        got = render_match(src, printed)
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append((src, printed, want, got, why))
        print(f'  [{mark}] render_match({src!r}, {printed!r:>9}) -> {got}   {why}')

    # ── extract: 토큰화 (결함 등록부 #4) ──
    print()
    ex_cases = [
        ('1,924 locations', ['1,924'], '천 단위 구분자는 한 토큰이다'),
        ('12,345,678 cells', ['12,345,678'], '구분자가 둘이어도 한 토큰'),
        ('124 to 967 points', ['124', '967'], '구분자 없는 수는 그대로'),
        ('bins 1, 2, 3 were used', ['1', '2', '3'],
         '쉼표 뒤 공백이면 나열이다 — 붙이면 안 된다'),
        ('at 240, 125 of them', ['240', '125'], '쉼표+공백 나열은 그대로 둘로'),
        ('1,9245 odd', ['1', '9245'],
         '쉼표 뒤 네 자리는 천 단위가 아니다 — 1924로 읽으면 안 된다'),
        # ── 부호 하이픈 (결함 등록부 #6) ──
        ('a shift of -1.47 dB', ['-1.47'], '공백 뒤 하이픈은 부호다'),
        ('the CI was $-7.39$ dB', ['-7.39'], 'LaTeX 수식 마이너스도 부호다'),
        ('($-0.261$ dB, $p = 0.19$)', ['-0.261', '0.19'],
         '여는 괄호 뒤 부호와 양수가 함께 나온다'),
        ('x=-0.5 here', ['-0.5'], "'=' 뒤 하이픈도 부호다"),
        ('fold-1 was held out', [], 'fold-1 은 식별자다 — -1 이 아니다'),
        ('24-2 SITA-Standard perimetry', ['24'],
         '24-2 는 프로토콜 이름이다 — -2 가 나오면 안 된다'),
        ('IR-v2 backbone', [], 'IR-v2 는 백본 이름이다 — 수가 아니다'),
        ('range 10-20 dB', ['10'], '영숫자 뒤 하이픈은 계속 막는다'),
        ('pages 1--2', ['1'], 'LaTeX en-dash 를 음수로 읽지 않는다'),
    ]
    for line, want_toks, why in ex_cases:
        got = [t for t, _ in extract(line)]
        mark = 'ok  ' if got == want_toks else 'FAIL'
        if got != want_toks:
            bad.append((line, want_toks, got, why))
        print(f'  [{mark}] extract({line!r:28s}) -> {got}   {why}')

    # ── 등록 위치 기반 앵커 조회 (결함 등록부 #3 의 과교정) ──
    # 자릿수는 원고가 정하고 위치가 오탐을 막는다. 여기서는 loc_match 만
    # 본다 — 실제 원고 문맥 매칭은 anchor_near 가 하고 본 실행에서 걸린다.
    print()
    lm_cases = [
        (6.66, '6.7', True, '원고가 1자리로 인쇄하면 1자리로 반올림해 맞춘다'),
        (18.18, '18.2', True, '같은 문장의 SD'),
        (0.113, '0.11', True, '원고 §4.2 의 2자리 인쇄'),
        (0.464, '0.46', True, '원고 §4.3 의 2자리 인쇄'),
        (0.113, '0.5', False, '무관한 0.5 에는 걸리면 안 된다'),
        (0.113, '0.12', False, '반올림해도 다른 값이면 miss'),
        (0.113, '0', False, '유효숫자를 다 잃는 토큰은 근거가 아니다'),
        (6.66, '6', False, '절사는 허용하지 않는다 — 반올림이면 7 이다'),
        (6.66, '7', True, '정수 자리 반올림은 인정한다'),
        (-0.589, '-0.59', True, '음수도 원고 자릿수로 반올림'),
        (-0.589, '0.59', False, '부호가 다르면 miss'),
    ]
    for val, tok, want, why in lm_cases:
        got = loc_match(val, tok)
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append((val, tok, want, got, why))
        print(f'  [{mark}] loc_match({val!r:>8}, {tok!r:>8}) -> {got}   {why}')

    # ── 줄바꿈으로 갈라진 지수 표기 (04_results.tex:163-164) ──
    print()
    wrap_src = ('branch at $p \\leq 1.4 \\times\n'
                '10^{-10}$ in every seed out-of-fold\n'
                'tail\n')
    wrapped = join_wrapped_sci(wrap_src)
    w_cases = [
        (wrap_src.count('\n') == wrapped.count('\n'), True,
         '줄 수가 보존돼야 보고 행번호가 안 밀린다'),
        ([t for t, _ in extract(wrapped.splitlines()[0])] == ['1.4e-10'], True,
         '갈라진 지수 표기가 첫 줄에서 한 토큰으로 잡힌다'),
        (any(t in ('10', '-10') for t in
             (t for l in wrapped.splitlines() for t, _ in extract(l))), False,
         "'10' 과 '-10' 조각이 남으면 안 된다"),
        (join_wrapped_sci('fold-1\n10^{-3}$ x\n') ==
         'fold-1\n10^{-3}$ x\n', True,
         '\\times 가 없으면 건드리지 않는다'),
    ]
    for got, want, why in w_cases:
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append(('sci-wrap', want, got, why))
        print(f'  [{mark}] join_wrapped_sci -> {got}   {why}')

    # ── 부호 규약 변환 (DERIVED kind='gain') ──
    print()
    g_obj = {'a': {'delta_fus_minus_xgb_rmse': -0.589},
             'b': {'delta_fus_minus_xgb_rmse': 0.589},
             'c': {'delta_fus_minus_xgb_rmse': 0.0}}
    def _gain(path):
        try:
            return derive(g_obj, path, 'gain')
        except ValueError:
            return 'ValueError'
    g_cases = [
        (_gain('a.delta_fus_minus_xgb_rmse'), 0.589,
         '음수 근거를 양의 크기로 뒤집는다'),
        (_gain('b.delta_fus_minus_xgb_rmse'), 'ValueError',
         '근거가 이미 양수면 규약이 깨진 것이다 — 조용히 통과하면 안 된다'),
        (_gain('c.delta_fus_minus_xgb_rmse'), 'ValueError',
         '0 은 방향이 없다'),
        (render_match(0.589, '0.589'), True,
         '뒤집은 값이 원고의 0.589 와 맞는다'),
        (render_match(-0.589, '0.589'), False,
         '뒤집지 않으면 여전히 안 맞는다 — 변환이 실제로 필요하다'),
    ]
    for got, want, why in g_cases:
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append(('gain', want, got, why))
        print(f'  [{mark}] gain -> {got!r:>10}   {why}')

    # ── 부호를 살리는 차 (DERIVED kind='delta') ──
    print()
    d_obj = {'cnn': {'rmse': 9.11}, 'xgb': {'rmse': 8.66},
             'ens': {'rmse': 8.115}}
    def _delta(path):
        return derive(d_obj, path, 'delta')
    d_cases = [
        (round(_delta('cnn.rmse|xgb.rmse'), 3), 0.45,
         '표 3 Pooled Δ 는 cnn - xgb 이고 양수면 영상분기가 더 나쁘다'),
        (round(_delta('ens.rmse|xgb.rmse'), 3), -0.545,
         '앙상블은 음수 — 부호가 방향을 말한다'),
        (derive(d_obj, 'cnn.rmse|xgb.rmse', 'diff') ==
         derive(d_obj, 'xgb.rmse|cnn.rmse', 'diff'), True,
         'diff 는 abs 라 순서를 못 가린다 — delta 가 따로 필요한 이유'),
        (_delta('cnn.rmse|xgb.rmse') == _delta('xgb.rmse|cnn.rmse'), False,
         'delta 는 순서가 뒤집히면 다른 값이다'),
        (render_match(-0.545, '-0.545'), True, '음수 그대로 원고와 맞는다'),
    ]
    for got, want, why in d_cases:
        mark = 'ok  ' if got == want else 'FAIL'
        if got != want:
            bad.append(('delta', want, got, why))
        print(f'  [{mark}] delta -> {got!r:>10}   {why}')

    n = (len(cases) + len(rm_cases) + len(ex_cases) + len(lm_cases)
         + len(w_cases) + len(g_cases) + len(d_cases))
    print()
    if bad:
        print(f'  {len(bad)}건 실패 / {n}건.')
        return 1
    print(f'  {n}건 전부 통과.')
    return 0


def anchor_id(fname, path, kind=None):
    """앵커 항목의 안정된 id. exceptions.yaml이 이 문자열로 항목을 가리킨다.

    값이 아니라 **위치**로 만든다. 라테랄리티 수정으로 앵커 값이 갱신돼도
    id는 그대로 남아야 예외가 값 변화에 딸려 조용히 풀리지 않는다.
    """
    if kind is None:
        return f'anchor:{fname}#{path}'
    return f'derived:{fname}#{path}#{kind}'


def check_anchors(sections=None):
    """앵커 값이 원고에 실제로 인쇄돼 있는지 확인한다.

    값 대조와 방향이 반대다. 여기서는 **근거에서 출발해서 원고를 찾는다.**
    근거 값이 원고 어디에도 없으면 그건 숫자가 갱신됐는데 원고가 안 따라온
    것이므로 [STALE]로 잡힌다.

    반환: dict 목록. status는 FOUND / STALE / NOFILE / NOKEY.
      FOUND         통과
      STALE         위반 (exit 1, 예외 승인 가능)
      NOFILE/NOKEY  검사 불능 (exit 2, 예외로 덮을 수 없다)
    """
    files = sorted(SEC.glob('*.tex')) if sections is None else list(sections)
    if not files:
        raise GateError(f'원고 섹션을 찾지 못했다: {SEC}/*.tex')
    body = '\n'.join(f.read_text() for f in files)
    rows = []

    def row(status, aid, label, src, shown):
        rows.append({'status': status, 'id': aid, 'label': label,
                     'src': src, 'shown': shown})

    spec_tbl = spec_md_table()
    for fname, path, label in ANCHORS:
        aid = anchor_id(fname, path)
        src = f'{fname}:{path}'
        if fname == SPEC_ANCHOR:
            if path not in spec_tbl:
                row('NOKEY', aid, label, src, 'KeyError')
                continue
            val = spec_tbl[path]
        else:
            jf = RUNS / fname
            if not jf.exists():
                row('NOFILE', aid, label, src, '')
                continue
            try:
                val = dig(json.loads(jf.read_text()), path)
            except Exception as e:
                row('NOKEY', aid, label, src, type(e).__name__)
                continue

        # 등록 위치가 있으면 그 근방에서만 찾는다. 자릿수는 원고가 정한다.
        loc = ANCHOR_LOC.get(aid)
        if loc:
            found, err = anchor_near(val, loc, files)
            if err:
                row('NOKEY', aid, label, src, err)
                continue
        else:
            found = any(re.search(pat, body) for pat in anchor_patterns(val))
        shown = (f'{val:.1e}' if isinstance(val, float) and 0 < abs(val) < 1e-3
                 else str(val))
        row('FOUND' if found else 'STALE', aid, label, src, shown)

    for fname, path, kind, label in DERIVED:
        aid = anchor_id(fname, path, kind)
        src = f'{fname}:{path} ({kind})'
        jf = RUNS / fname
        if not jf.exists():
            row('NOFILE', aid, label, src, '')
            continue
        try:
            v = derive(json.loads(jf.read_text()), path, kind)
        except Exception as e:
            row('NOKEY', aid, label, src, type(e).__name__)
            continue
        # 원고는 백분율을 % 기호 없이 쓰기도 한다("17.5, 9.1 ... percent",
        # 04_results.tex:332). 그래서 기호를 강제하지 않고 독립 토큰으로 찾는다.
        # gain 은 원고가 소수 3자리로 인쇄한다(0.589 / 0.534). 2자리로도
        # 찾으면 본문의 "0.59"·"0.53" 에 걸릴 수 있어 3자리로 못 박는다.
        #
        # 등록 위치가 있으면 위쪽 ANCHORS 루프와 똑같이 위치로 간다. 자릿수를
        # kind 로 못 박는 위 규칙은 **원고가 그보다 적게 인쇄하면 STALE 을
        # 낸다** — p95 하락 3.118 을 원고가 "3.12" 로 쓰는 자리가 그랬다
        # (2026-09-17). 위치를 등록한 것만 예외로 두고, 등록 안 한 것은
        # 종전의 엄격한 자릿수를 그대로 쓴다.
        loc = ANCHOR_LOC.get(aid)
        if loc:
            found, err = anchor_near(v, loc, files)
            if err:
                row('NOKEY', aid, label, src, err)
                continue
        else:
            nds = ((3, 4) if kind == 'diff'
                   else (3,) if kind in ('gain', 'delta') else (1, 2))
            pats = [r'(?<![0-9.])' + re.escape(f'{v:.{nd}f}') + r'(?![0-9])'
                    for nd in nds]
            found = any(re.search(p_, body) for p_ in pats)
        shown = (f'{v:.4f}' if kind == 'diff'
                 else f'+{v:.3f}' if kind == 'gain'
                 else f'{v:+.3f}' if kind == 'delta' else f'{v:.2f}%')
        row('FOUND' if found else 'STALE', aid, label, src, shown)
    return rows


# ---------------------------------------------------------------------------
# YAML. PyYAML이 있으면 쓰고, 없으면 최소 파서로 폴백한다.
# 이 스크립트는 원래 표준 라이브러리만으로 도는 게 규약이었다. 부속 파일을
# YAML로 두되 의존성을 새로 강제하지 않기 위해 두 경로를 모두 지원하고,
# 회귀 테스트에서 두 경로가 같은 판정을 내는지 확인한다.
# ---------------------------------------------------------------------------
def _strip_comment(line):
    """따옴표 밖의 ` #` 이후를 자른다. id에 '#'가 들어가므로 인용을 존중한다."""
    out = []
    quote = None
    for i, c in enumerate(line):
        if quote:
            out.append(c)
            if c == quote:
                quote = None
            continue
        if c in '"\'':
            quote = c
            out.append(c)
        elif c == '#' and (i == 0 or line[i - 1] in ' \t'):
            break
        else:
            out.append(c)
    return ''.join(out)


def _scalar(tok):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in '"\'':
        return tok[1:-1]
    if tok in ('[]', '{}', '~', 'null', 'Null', 'NULL'):
        return None
    return tok


def _mini_yaml(text):
    """지원 범위: 최상위 매핑 -> 리스트 -> 평평한 매핑(스칼라 값).

    exceptions.yaml / placeholder_patterns.yaml 두 파일의 모양만 읽으면 된다.
    그 밖의 문법이 나오면 조용히 넘기지 않고 GateError로 올린다.
    """
    doc = {}
    key = None
    item = None
    for no, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if indent == 0:
            if not body.endswith(':') and ': ' not in body:
                raise GateError(f'YAML {no}행을 해석할 수 없다: {raw.strip()!r}')
            if body.endswith(':'):
                key = body[:-1].strip()
                doc[key] = []
                item = None
            else:
                k, _, v = body.partition(':')
                doc[k.strip()] = _scalar(v)
                key = None
                item = None
            continue
        if key is None:
            raise GateError(f'YAML {no}행: 최상위 키 없이 들여쓴 줄이다.')
        if body.startswith('- '):
            item = {}
            doc[key].append(item)
            body = body[2:].strip()
        elif body == '-':
            item = {}
            doc[key].append(item)
            continue
        if item is None:
            raise GateError(f'YAML {no}행: 리스트 항목 밖의 줄이다.')
        if ':' not in body:
            raise GateError(f'YAML {no}행을 해석할 수 없다: {raw.strip()!r}')
        k, _, v = body.partition(':')
        item[k.strip()] = _scalar(v)
    return doc


def load_yaml_doc(path, what):
    """부속 YAML 하나를 읽는다. 없거나 깨졌으면 GateError (= exit 2)."""
    if not path.exists():
        raise GateError(f'{what} 파일이 없다: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as e:
        raise GateError(f'{what} 파일을 읽을 수 없다: {path} ({e})')
    if os.environ.get('CHECK_NUMBERS_NO_YAML'):
        data = _mini_yaml(text)
    else:
        try:
            import yaml
        except ImportError:
            data = _mini_yaml(text)
        else:
            try:
                data = yaml.safe_load(text)
            except Exception as e:
                raise GateError(f'{what} 파싱 실패: {path} ({e})')
            if data is None:
                data = {}
    if not isinstance(data, dict):
        raise GateError(f'{what} 최상위가 매핑이 아니다: {path}')
    return data


# ---------------------------------------------------------------------------
# 플레이스홀더 스캔
# ---------------------------------------------------------------------------
def load_placeholder_patterns():
    doc = load_yaml_doc(PATTERNS_FILE, '플레이스홀더 패턴')
    items = doc.get('patterns')
    if not isinstance(items, list) or not items:
        raise GateError(
            f'{PATTERNS_FILE}: patterns 목록이 비었다. '
            '패턴 없는 스캔을 통과로 처리하지 않는다.')
    out = []
    seen = set()
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict):
            raise GateError(f'{PATTERNS_FILE}: {n}번째 patterns 항목이 매핑이 아니다.')
        pid = str(it.get('id') or '').strip()
        text = it.get('text')
        text = '' if text is None else str(text)
        if not pid:
            raise GateError(f'{PATTERNS_FILE}: {n}번째 항목에 id가 없다.')
        if not text.strip():
            raise GateError(f'{PATTERNS_FILE}: 패턴 {pid!r}에 text가 없다.')
        if pid in seen:
            raise GateError(f'{PATTERNS_FILE}: id가 중복이다: {pid!r}')
        seen.add(pid)
        as_regex = str(it.get('regex', '')).strip().lower() in ('1', 'true', 'yes')
        try:
            rx = re.compile(text if as_regex else re.escape(text))
        except re.error as e:
            raise GateError(f'{PATTERNS_FILE}: 패턴 {pid!r} 정규식 오류: {e}')
        out.append((pid, text, rx))
    return out


def placeholder_targets(sections=None):
    """스캔 대상 원고 소스. main.tex + sections/*.tex."""
    files = []
    if sections is None:
        if MAIN_TEX.exists():
            files.append(MAIN_TEX)
        files += sorted(SEC.glob('*.tex'))
    else:
        files = list(sections)
    if not files:
        raise GateError(f'스캔할 원고 소스가 없다: {SEC}/*.tex')
    return files


def scan_placeholders(patterns, sections=None):
    """플레이스홀더를 찾아 (파일, 행번호, 행 전체)로 보고한다."""
    hits = []
    for f in placeholder_targets(sections):
        try:
            rel = str(f.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(f)
        for ln, line in enumerate(f.read_text(encoding='utf-8').splitlines(), 1):
            for pid, text, rx in patterns:
                if rx.search(line):
                    hits.append(Finding(
                        kind='PLACEHOLDER',
                        id=f'placeholder:{pid}:{rel}:{ln}',
                        label=f'{text!r}',
                        where=f'{rel}:{ln}',
                        detail=line))
    return hits


# ---------------------------------------------------------------------------
# 예외 (verify/exceptions.yaml)
# ---------------------------------------------------------------------------
Finding = namedtuple('Finding', 'kind id label where detail')
Exception_ = namedtuple('Exception_', 'id reason approved_by date stale')
REQUIRED_EXC_FIELDS = ('id', 'reason', 'approved_by', 'date')


def load_exceptions(known_ids):
    """승인된 예외를 읽는다.

    반환: (approved, invalid, warnings)
      approved  {id: Exception_}   — 네 필드가 전부 채워진 유효한 예외
      invalid   [(id표시, 사유)]    — 무효. 해당 항목은 다시 위반으로 떨어진다
      warnings  [문자열]            — 30일 초과 등, 통과는 시키되 알린다

    known_ids에 없는 id가 적혀 있으면 GateError(= exit 2)다. 오타 하나로
    예외가 조용히 아무것도 덮지 않는 상태를 통과로 두지 않는다.
    """
    if not EXCEPTIONS_FILE.exists():
        return {}, [], [f'예외 파일이 없다: {EXCEPTIONS_FILE} — 예외 없이 검사한다.']
    doc = load_yaml_doc(EXCEPTIONS_FILE, '예외')
    items = doc.get('exceptions')
    if items is None:
        items = []
    if not isinstance(items, list):
        raise GateError(f'{EXCEPTIONS_FILE}: exceptions는 목록이어야 한다.')

    approved, invalid, warnings, unknown = {}, [], [], []
    today = _dt.date.today()
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict):
            invalid.append((f'<{n}번째 항목>', '매핑이 아니다'))
            continue
        vals = {}
        missing = []
        for f in REQUIRED_EXC_FIELDS:
            raw = it.get(f)
            txt = '' if raw is None else str(raw).strip()
            if not txt:
                missing.append(f)
            vals[f] = txt
        shown_id = vals['id'] or f'<{n}번째 항목: id 없음>'
        if missing:
            invalid.append((shown_id, '필수 항목 누락/공백: ' + ', '.join(missing)))
            continue
        try:
            d = _dt.date.fromisoformat(vals['date'][:10])
        except ValueError:
            invalid.append((shown_id, f'date 형식이 YYYY-MM-DD가 아니다: {vals["date"]!r}'))
            continue
        if vals['id'] in approved:
            invalid.append((shown_id, '같은 id의 예외가 중복이다'))
            continue
        if vals['id'] not in known_ids:
            unknown.append(vals['id'])
            continue
        age = (today - d).days
        stale = age > EXCEPTION_MAX_AGE_DAYS
        if stale:
            warnings.append(
                f'예외가 {age}일 지났다 (>{EXCEPTION_MAX_AGE_DAYS}일): {vals["id"]} '
                f'— {vals["date"]} {vals["approved_by"]}. 재승인할 것. '
                '(경고만 하고 통과시킨다.)')
        if age < 0:
            warnings.append(f'예외 날짜가 미래다: {vals["id"]} ({vals["date"]})')
        approved[vals['id']] = Exception_(vals['id'], vals['reason'],
                                          vals['approved_by'], vals['date'], stale)
    if unknown:
        raise GateError(
            f'{EXCEPTIONS_FILE}: 검사 목록에 없는 id가 예외에 적혀 있다:\n  '
            + '\n  '.join(unknown)
            + '\n  (CANONICAL_SPEC 앵커에도, 이번 실행의 위반 목록에도 없다. '
              '오타이거나, 원고가 바뀌어 행 번호가 옮겨갔거나, 이미 해결된 항목이다.)')
    return approved, invalid, warnings


# ---------------------------------------------------------------------------
# 로그. stdout에 찍는 것과 **같은 내용**을 파일에도 남긴다.
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def open_log():
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f'run_{_dt.datetime.now():%Y%m%d_%H%M%S}.log'
        return path, path.open('w', encoding='utf-8')
    except OSError as e:
        raise GateError(f'로그 파일을 열 수 없다: {LOG_DIR} ({e})')


def scan_numbers(files, num_src):
    """섹션의 수치를 근거 풀과 대조한다. (misses, oks, 카운트) 반환."""
    n_ok = n_cite = n_struct = 0
    misses, oks = [], []
    LIT = re.compile(r'\\cite|[A-Z][a-z]+\s+et\s+al')

    for f in files:
        lines = join_wrapped_sci(f.read_text()).splitlines()
        # 인용 범위는 줄이 아니라 **문단**이다. 앞뒤 2줄 창으로는 놓친다:
        #   02_related_work.tex:101의 2.93/2.49는 \cite가 3줄 위에 있고,
        #   05_discussion.tex:152의 2220은 그 문단에 \cite가 아예 없이
        #   "Park et al."로만 지칭한다(인용은 02_related_work.tex:77).
        # 그래서 빈 줄로 끊은 문단 안에 \cite 또는 "<Name> et al"이 있으면
        # 그 문단의 수치는 문헌값 후보로 본다.
        para_lit = [False] * (len(lines) + 1)
        start = 0
        for i in range(len(lines) + 1):
            if i == len(lines) or not lines[i].strip():
                blob = '\n'.join(lines[start:i])
                if LIT.search(blob):
                    for j in range(start, i):
                        para_lit[j + 1] = True
                start = i + 1

        for ln, line in enumerate(lines, 1):
            has_cite = para_lit[ln]
            for printed, val in extract(line):
                if val in STRUCTURAL and float(val).is_integer():
                    n_struct += 1
                    continue
                where = None
                for src, paths in num_src.items():
                    if render_match(src, printed):
                        where = paths[0]
                        break
                if where:
                    n_ok += 1
                    oks.append((f.name, ln, printed, where))
                elif has_cite:
                    n_cite += 1
                else:
                    misses.append(Finding(
                        kind='MISS',
                        id=f'miss:{f.name}:{ln}#{printed}',
                        label=printed,
                        where=f'{f.name}:{ln}',
                        detail=line.strip()))
    return misses, oks, n_ok, n_cite, n_struct


def run_gate(args, log_path):
    """게이트 본체. 위반이 남으면 EXIT_FAIL, 검사 불능이면 GateError."""
    print('=== check_numbers 게이트 ===')
    print(f'  시각      : {_dt.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'  대상 트리 : {ROOT}')
    print(f'  원고      : {SEC}'
          + ('' if getattr(args, 'manuscript', None)
             else '   ** 저장소 안의 사본이다. 정본은 Overleaf 에 있다 —'
                  ' 정본을 검사하려면 --manuscript DIR 로 지정할 것. **'))
    print(f'  근거      : {RUNS}')
    print(f'  로그      : {log_path}')
    print(f'  예외      : {EXCEPTIONS_FILE}')
    print(f'  패턴      : {PATTERNS_FILE}')
    print()

    # 설정 파일 먼저. 설정 오류를 검사보다 앞에서 드러낸다.
    patterns = load_placeholder_patterns()

    files = sorted(SEC.glob('*.tex'))
    if not files:
        raise GateError(f'원고 섹션을 찾지 못했다: {SEC}/*.tex')
    partial = False
    if args.file:
        files = [f for f in files if f.name == args.file]
        if not files:
            raise GateError(f'그런 섹션이 없다: {args.file}')
        partial = True
        print(f'  ** 부분 검사: {args.file} 만 본다. 통과해도 전체 통과가 아니다. **')
        print()

    num_src, _ = load_sources()
    misses, oks, n_ok, n_cite, n_struct = scan_numbers(files, num_src)

    # 앵커는 **항상 원고 전체**를 본다. --file로 좁히면 다른 섹션에 인쇄된
    # 값이 전부 STALE로 뜬다 — 부분 검사의 목적이 아니다.
    anchors = check_anchors()
    unrunnable = [r for r in anchors if r['status'] in ('NOFILE', 'NOKEY')]
    stale = [r for r in anchors if r['status'] == 'STALE']

    ph = scan_placeholders(patterns, sections=files if partial else None)

    findings = list(misses)
    findings += [Finding(kind='STALE', id=r['id'], label=r['label'],
                         where=r['src'], detail=f"근거값 {r['shown']} 이 원고에 없다")
                 for r in stale]
    findings += ph

    # 예외가 가리킬 수 있는 id 전체. 앵커는 통과 여부와 무관하게 등록한다
    # (라테랄리티 수정 뒤 STALE로 바뀔 항목을 미리 승인해 둘 수 있어야 한다).
    known_ids = {r['id'] for r in anchors} | {f.id for f in findings}
    approved, invalid, warnings = load_exceptions(known_ids)

    excepted = [f for f in findings if f.id in approved]
    blocking = [f for f in findings if f.id not in approved]

    if args.all:
        print('=== [OK] 근거 확인 ===')
        for fn, ln, pr, w in oks:
            print(f'  {fn}:{ln}  {pr:>12}  <- {w}')
        print()

    print('=== [MISS] 근거 없음 ===')
    print('  runs/*.json + CANONICAL_SPEC.md 어디에도 근거가 없는 수치.')
    print('  2026-08-26 전수 추적 결과, 아래 대부분은 배포 대상이 아닌 '
          '내부 산문 문서에만 있고 기계 산출물에는 없다.')
    print('  즉 **재계산 근거가 없는 값**이다.')
    print('  88명 편입 시 이 값들은 자동으로 갱신되지 않으므로 손으로 다시 내야 한다.')
    print()
    live_miss = [f for f in misses if f.id not in approved]
    if not live_miss:
        print('  없음.' if not misses else '  전부 승인된 예외.')
    for f in live_miss:
        print(f'  {f.where}  {f.label:>12}   | {f.detail[:90]}')
        print(f'       id: {f.id}')

    print()
    print('=== [ANCHOR] 근거 -> 원고 역방향 대조 ===')
    print('  (값 대조만으로는 출처를 모른다 — 먼저 맞는 값이 이긴다. '
          '아래는 JSON 경로 / SPEC 표로 못 박은 것.)')
    live_stale = [r for r in stale if r['id'] not in approved]
    if not live_stale and not unrunnable:
        print(f'  {len(anchors)}개 앵커 전부 원고에서 확인.')
    for r in live_stale:
        print(f"  [STALE] {r['label']}: {r['src']} = {r['shown']} — 원고에 없음")
        print(f"       id: {r['id']}")

    print()
    print('=== [PLACEHOLDER] 원고에 남은 자리표시자 ===')
    live_ph = [f for f in ph if f.id not in approved]
    if not live_ph:
        print('  없음.' if not ph else '  전부 승인된 예외.')
    for f in live_ph:
        print(f'  {f.where}: {f.label}')
        print(f'    | {f.detail}')
        print(f'    id: {f.id}')

    if excepted:
        print()
        print('=== [EXCEPTED] 승인된 예외로 통과시킨 항목 ===')
        for f in excepted:
            e = approved[f.id]
            mark = ' (기한 초과)' if e.stale else ''
            print(f'  [{f.kind}] {f.id}{mark}')
            print(f'    사유: {e.reason}')
            print(f'    승인: {e.approved_by} / {e.date}')

    if invalid:
        print()
        print('=== [INVALID EXCEPTION] 인정하지 않은 예외 ===')
        print('  네 필드(id, reason, approved_by, date)가 전부 채워져야 예외다.')
        print('  아래 항목은 무효이므로, 덮으려던 위반은 그대로 살아 있다.')
        for eid, why in invalid:
            print(f'  {eid}: {why}')

    if warnings:
        print()
        print('=== [WARN] ===')
        for w in warnings:
            print(f'  {w}')

    if unrunnable:
        print()
        print('=== [ERROR] 검사 불능 — 예외로 덮을 수 없다 ===')
        for r in unrunnable:
            print(f"  [{r['status']}] {r['label']}: {r['src']} ({r['shown']})")

    n_block = len(blocking)
    n_miss_b = sum(1 for f in blocking if f.kind == 'MISS')
    n_stale_b = sum(1 for f in blocking if f.kind == 'STALE')
    n_ph_b = sum(1 for f in blocking if f.kind == 'PLACEHOLDER')

    if unrunnable:
        code = EXIT_ERROR
    elif n_block:
        code = EXIT_FAIL
    else:
        code = EXIT_PASS

    bar = '  ' + '-' * 58
    print()
    print('=== 요약 ===')
    print(bar)
    print(f'  {"항목":<22}{"수":>8}   설명')
    print(bar)
    print(f'  {"[OK]":<22}{n_ok:>8}   근거 산출물/SPEC에서 확인')
    print(f'  {"[STRUCT]":<22}{n_struct:>8}   프로토콜 상수 허용목록')
    print(f'  {"[CITE]":<22}{n_cite:>8}   \\cite 문단의 수치 — 자동검증 대상 아님')
    print(f'  {"[ANCHOR-FOUND]":<22}{len(anchors) - len(stale) - len(unrunnable):>8}'
          f'   /{len(anchors)} 근거값이 원고에서 확인됨')
    print(bar)
    print(f'  {"[MISS]":<22}{n_miss_b:>8}   근거 없음 (미승인)')
    print(f'  {"[STALE]":<22}{n_stale_b:>8}   앵커가 원고에 없음 (미승인)')
    print(f'  {"[PLACEHOLDER]":<22}{n_ph_b:>8}   자리표시자 잔존 (미승인)')
    print(f'  {"[EXCEPTED]":<22}{len(excepted):>8}   승인된 예외로 통과')
    print(f'  {"[INVALID EXC]":<22}{len(invalid):>8}   무효 예외 (덮지 못함)')
    print(f'  {"[UNRUNNABLE]":<22}{len(unrunnable):>8}   검사 불능')
    print(bar)
    print(f'  {"위반 합계":<21}{n_block:>8}')
    print(bar)
    print(f'  근거 풀: runs/*.json {len(list(RUNS.glob("*.json")))}개 + '
          f'CANONICAL_SPEC.md, distinct 수치 {len(num_src)}개')
    verdict = {EXIT_PASS: 'PASS', EXIT_FAIL: 'FAIL', EXIT_ERROR: 'ERROR'}[code]
    if partial and code == EXIT_PASS:
        verdict = 'PASS (부분 검사)'
    print(f'  판정: {verdict}  (exit {code})')
    print(bar)

    if args.json:
        payload = {
            'root': str(ROOT),
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            'exit_code': code,
            'counts': {'ok': n_ok, 'struct': n_struct, 'cite': n_cite,
                       'miss': n_miss_b, 'stale': n_stale_b,
                       'placeholder': n_ph_b, 'excepted': len(excepted),
                       'invalid_exceptions': len(invalid),
                       'unrunnable': len(unrunnable)},
            'blocking': [f._asdict() for f in blocking],
            'excepted': [f._asdict() for f in excepted],
            'invalid_exceptions': [{'id': a, 'why': b} for a, b in invalid],
            'unrunnable': unrunnable,
            'known_ids': sorted(known_ids),
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  기계용 덤프: {args.json}')

    return code


def main():
    ap = argparse.ArgumentParser(
        description='원고 수치 검증 게이트. exit 0=통과 / 1=위반 / 2=검사 불능')
    ap.add_argument('--all', action='store_true', help='OK 항목까지 전부 출력')
    ap.add_argument('--file', default=None, help='특정 섹션만 검사 (부분 검사)')
    ap.add_argument('--selftest', action='store_true',
                    help='숫자 매칭 경계 조건 회귀 테스트만 실행')
    ap.add_argument('--json', default=None, help='위반 목록을 JSON으로 덤프할 경로')
    ap.add_argument('--root', default=None,
                    help='검사 대상 트리 (기본: 저장소 루트). 임시 픽스처 시험용')
    ap.add_argument('--runs', default=None, metavar='DIR',
                    help='근거 산출물 *.json 디렉터리. 배포본에는 runs/ 가 '
                         '없다 — paper/results_frozen 을 여기에 준다.')
    ap.add_argument('--manuscript', default=None, metavar='DIR',
                    help='검사할 원고 *.tex 디렉터리. 정본 원고는 이 저장소에 없다 '
                         '(Overleaf) — 내려받은 경로를 여기에 준다. 생략하면 '
                         'docs/journal_manuscript/sections/ 의 낡은 사본을 검사한다.')
    args = ap.parse_args()

    if args.root:
        configure(args.root)
    if args.runs:
        try:
            configure_runs(args.runs)
        except GateError as e:
            print(f'[ERROR] {e}', file=sys.stderr)
            print(f'[ERROR] 검사 불능 — 통과가 아니다 (exit {EXIT_ERROR}).', file=sys.stderr)
            return EXIT_ERROR
    if args.manuscript:
        try:
            configure_manuscript(args.manuscript)
        except GateError as e:
            print(f'[ERROR] {e}', file=sys.stderr)
            print(f'[ERROR] 검사 불능 — 통과가 아니다 (exit {EXIT_ERROR}).', file=sys.stderr)
            return EXIT_ERROR

    if args.selftest:
        print('=== 숫자 매칭 경계 조건 회귀 테스트 ===')
        return EXIT_FAIL if selftest() else EXIT_PASS

    try:
        log_path, fh = open_log()
    except GateError as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        print(f'[ERROR] 검사 불능 — 통과가 아니다 (exit {EXIT_ERROR}).', file=sys.stderr)
        return EXIT_ERROR

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, fh)
    try:
        code = run_gate(args, log_path)
    except GateError as e:
        print()
        print('=== [ERROR] 검사 불능 ===')
        print(f'  {e}')
        print(f'  판정: ERROR (exit {EXIT_ERROR}) — 이것은 통과가 아니다.')
        code = EXIT_ERROR
        print(f'[ERROR] {e}', file=sys.stderr)
    except Exception:
        import traceback
        print()
        print('=== [ERROR] 예상치 못한 실행 오류 ===')
        traceback.print_exc(file=sys.stdout)
        print(f'  판정: ERROR (exit {EXIT_ERROR}) — 이것은 통과가 아니다.')
        code = EXIT_ERROR
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout = real_stdout
        fh.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
