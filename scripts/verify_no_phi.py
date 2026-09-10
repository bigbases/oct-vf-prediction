#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-push PHI verification — each layer is judged separately.

공개(push) 직전 PHI 무결성 검증. 자매 저장소
(`oct-hvf-extraction-validation`)의 `scripts/verify_no_phi.py` 를 이 저장소에
맞춰 옮긴 것이다. 판정 전에 **양성 대조**로 패턴이 실제 작동하는지 증명하고,
"깨끗하다"고 쓸 때 어느 층이 깨끗한지 반드시 명시한다.

  L0  파일명    — 이름에 PHI 가 박힌 파일(바이너리도 여기서 걸린다)
  L1  작업 트리 — gitignore 제외분을 뺀, 공개될 파일 집합
  L1b 컨테이너  — xlsx/docx(zip+XML), pdf(압축 스트림)를 풀어서 검사
  L1c 보조 도구 흔적 — 파일 내용 + 커밋 메시지·저자 메타데이터
  L1d 산문 검사 — .md 의 내부 상태 서술
  L1e 한글 검사 — 독자가 읽어야 하는 산문·인용 메타데이터에 한글이 남았는지
  L1f 사설 블록리스트 — 기관명·연구자 실명·IRB 번호·실제 환자ID
  L1g 준식별자  — 가명(P-\\d+)과 laterality/검사일이 같은 행에 함께 나오는 경우
  L2  HEAD      — 지금 커밋돼 있는 내용
  L3  이력 전체 — 모든 커밋의 모든 blob

**기관명·연구자 실명·실제 환자ID 는 이 소스에 적지 않는다.** 적으면 이 파일
자체가 적출 대상이 된다. 대신 두 개의 사설 파일을 런타임에 읽는다(둘 다 트리
밖, 커밋 대상 아님):

  HVF_PHI_BLOCKLIST   기본 ~/hvf_private/phi_blocklist.txt  한 줄에 리터럴 하나
  HVF_PSEUDONYM_MAP   기본 ~/hvf_private/pseudonym_map.csv  real_patient_id 열

둘 중 하나라도 없으면 그 층은 **UNVERIFIED 로 보고하고 종료코드를 1 로 올린다**
— "검사 안 했음"이 "깨끗함"으로 읽히지 않게 하기 위해서다.

종료 코드: L0·L1·L1b·L1c·L1d·L1e·L1f·L1g·L2 중 하나라도 적출되면 1, 아니면 0.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/verify_no_phi.py'

# ── 패턴 ────────────────────────────────────────────────────────────────
# 경계에 \b 를 쓰지 않는다 — 밑줄이 단어문자라 `_19999999_` 를 놓친다.
PATTERNS = {
    # 이 코호트의 식별자는 5~8자리다. 'patient'/'pid'/'환자' 문맥을 요구해
    # 소수부·SHA 조각 같은 오탐을 피한다.
    '환자ID(문맥)': re.compile(
        b'(?:patient|pid|' + '환자'.encode('utf-8') + b')'
        rb'[\s:=_#]{0,3}(?<![0-9])\d{5,8}(?![0-9])', re.I),
    '환자ID(8자리)': re.compile(rb'(?<![0-9.])1[0-9]{7}(?![0-9])'),
    # 소수점 뒤를 배제한다 — 부동소수 가수의 끝 13자리가 우연히 이 형태를
    # 만든다(예: -0.3057941198349). 주민등록번호가 마침표 바로 뒤에
    # 오는 표기는 없으므로 패턴이 느슨해지지 않는다.
    '주민등록번호형': re.compile(rb'(?<![0-9.])\d{6}-?[1-4]\d{6}(?![0-9])'),
    'DOB/검사일': re.compile(rb'(?<![0-9.])(?:19[0-9]{2}|20[0-2][0-9])'
                             rb'(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])(?![0-9])'),
    # 하이픈/슬래시 표기도 검사일이다. 작업 일자 주석이 많아 문맥이 있을 때만.
    '검사일(문맥)': re.compile(
        b'(?:patient|exam|vf_date|scan_date|' + '환자'.encode('utf-8')
        + b'|' + '검사'.encode('utf-8') + b')'
        rb'[^\n]{0,30}?(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])', re.I),
    '개인 경로(Win)': re.compile(rb'C:[\\/]{1,2}Users[\\/]{1,2}(?!\{user\}|<user>)\w+'),
    '개인 경로(Unix)': re.compile(rb'/(?:home|Users)/(?!\{user\}|<user>)[a-z][a-z0-9_-]{1,30}'),
    '이메일': re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'IP 주소': re.compile(rb'(?<![0-9.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}'
                          rb'(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![0-9.])'),
    '실명(폴더명 규칙)': re.compile(rb'(?<![0-9])1[0-9]{7}__([a-z][a-z_ -]{2,30}?)__', re.I),
}
# 확인된 오탐만 개별 문자열로 면제한다(패턴을 느슨하게 만들지 않는다).
ALLOW = (
    b'anonymous@example.org',        # 커밋 저자로 쓰는 익명 주소
    b'noreply@example.org',
    rb'C:\Users\{user}', rb'C:\\Users\\{user}',   # paths.py 의 플레이스홀더
    rb'C:\Program Files\Tesseract-OCR',           # 표준 설치 경로(개인 계정 아님)
    b'/home/{user}', b'/home/<user>',
    # 이 파일 자신의 양성 대조 프로브(합성값). 실제 값이 아니다.
    b'19999999', b'test_person', b'19010101', b'20200101',
    b'patient 999999', b'exam 2000-01-01', b'900101-1234567',
    rb'C:\Users\nobody', rb'C:\\Users\\nobody', b'/home/nobody',
    b'nobody@example.invalid', b'203.0.113.9',
    # 아래 넷은 작업 산출물의 파일명·디렉터리명에 박힌 **작업 일시**다.
    # 검사일이 아니다. 각 출현 위치를 개별 확인했다:
    #   backup_vf_neg1_fix_20260530_193547  (백업 디렉터리 이름)
    #     experiments/forest_robustness/compute_rows.py, scripts/neg1_mask_sensitivity.py,
    #     scripts/verify_skeleton_numbers.py
    #   robustness_gains_20260711.md        (보고서 파일명)
    #     scripts/check_numbers.py
    #   2026-08-31                          (ResNet50 예측이 저장된 날, 산출물 note)
    #     paper/results_frozen/resnet50_late_fusion.json
    b'20260530_193547', b'20260711', b'2026-08-31',
)
TOOL_PATTERNS = {
    '보조도구 이름': re.compile(
        rb'claude|anthropic|chatgpt|gpt-4|openai|copilot|cursor|windsurf|'
        rb'codeium|tabnine', re.I),
    '커밋 트레일러': re.compile(rb'co-authored-by|generated\s+with\s+\[?', re.I),
    '내부 문서 참조': re.compile(rb'CURSOR_BRIEFING|CLAUDE\.md|MEMORY\.md'),
}
TOOL_ALLOW = (
    b'Jean-Claude',
)
PROSE_PATTERNS = {
    '미해결·TODO': ('TODO', 'FIXME', 'XXX:', 'STUB', '미해결', '미구현', '확보 필요',
                    '확정 필요', '이관 예정', '진행 중'),
    '재현 실패 서술': ('재현안됨', '부분재현', '재현 실패', '드리프트', '워킹트리',
                     '삭제됨', '불일치 확인', '격차'),
    '내부 인프라': ('conda', 'Ubuntu', '서버', 'GPU 서버'),
}
# 확인된 오탐만 개별 문자열로 면제한다(단어 목록을 줄이지 않는다).
# 아래는 전부 독자용 설치 안내다 — 내부 장비·환경을 가리키지 않는다.
PROSE_ALLOW = (
    'the equivalent conda\nspecification',            # README.md
    'the equivalent conda\nspecification.',           # ENVIRONMENT.md
    'conda env create -f environment.yml && conda activate hvf',
    'The conda environment used for the reported runs',
    '$CONDA_PREFIX/lib',
    # ENVIRONMENT.md — 모듈 docstring 의 `env:` 태그가 무엇인지 설명하는 절.
    # 두 이름 모두 environment.yml 로 재현되는 공개 사양이다.
    'These are the two\nconda environments the work was done in',
)
def prose_hit(word: str, low: str) -> bool:
    """산문 낱말 하나를 찾는다. 영문은 낱말 경계를 요구한다.

    부분일치로 두면 'secondary' 안의 'conda' 처럼 뜻이 전혀 다른 자리가
    걸린다. 한글·기호가 섞인 항목은 경계 개념이 없으므로 그대로 찾는다.
    """
    w = word.lower()
    if re.fullmatch(r'[a-z0-9 ]+', w):
        return re.search(r'(?<![a-z0-9])%s(?![a-z0-9])' % re.escape(w),
                         low) is not None
    return w in low


HANGUL = re.compile('[가-힣ᄀ-ᇿ㄰-㆏]')
HANGUL_TARGETS = {'.md', '.txt', '.cff', '.rst', '.yaml', '.yml'}
HANGUL_INFO_ONLY = {'.py', '.json', '.sh', '.toml', '.cfg', '.ini', ''}

# 준식별자: 가명 단독은 허용, 가명 + laterality 또는 가명 + 날짜의 동시 출현은 적출.
QUASI_PSEUDO = re.compile(rb'P-\d+')
QUASI_LAT = re.compile(rb'\b(?:OD|OS)\b')
QUASI_DATE = re.compile(rb'(?:19|20)\d{2}[-/]?\d{2}[-/]?\d{2}')

FILENAME_ALLOW = ()
# 인용 메타데이터(CITATION.cff)에는 저자 실명·소속이 드는 것이 정상이므로
# 여기 넣으면 차단이 아니라 면제로 보고한다. 지금은 저자 목록이 확정되지
# 않아 파일 자체를 넣지 않았다 — 확정되면 파일과 함께 이 집합에 되돌린다.
BLOCKLIST_FILE_ALLOW = set()

TEXT_EXT = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.sh', '.tex', '.cfg',
            '.ini', '.toml', '.cff', '.csv', '.bat', '.bib', '.log', ''}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache'}


def scan(body: bytes, apply_allow: bool = True):
    if apply_allow:
        for a in ALLOW:
            body = body.replace(a, b'')
    out = {}
    for name, rx in PATTERNS.items():
        hits = rx.findall(body)
        if hits:
            out[name] = len(set(hits))
    return out


def git(*args, stdin: bytes = None) -> bytes:
    return subprocess.run(['git', '-c', 'core.quotepath=false', *args],
                          cwd=str(ROOT), input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def id_needle(term: str):
    """블록리스트 항목 하나를 바이트 정규식으로 만든다(리터럴은 저장하지 않는다).

    숫자로만 된 항목 — 실제 환자번호가 그렇다 — 은 앞뒤가 숫자면 일치로 치지
    않는다. 소수점 바로 뒤도 배제한다. 부동소수 가수 안에서 6자리 숫자가
    우연히 겹치는 일이 실제로 있었고(paper/results_frozen 의 통계값), 그걸
    유출로 세면 검사기가 늑대소년이 된다. 숫자가 아닌 항목(기관·실명)은
    경계 없이 그대로 찾는다.
    """
    b = re.escape(term.lower().encode('utf-8'))
    if term.isdigit():
        return re.compile(rb'(?<![0-9.])' + b + rb'(?![0-9])')
    return re.compile(b)


def loose_hits(needle, blob: bytes) -> int:
    """경계 조건 때문에 제외된 우연 일치 수 — 보고에만 쓴다."""
    core = needle.pattern
    core = core.replace(rb'(?<![0-9.])', b'').replace(rb'(?![0-9])', b'')
    return len(re.findall(core, blob)) if core != needle.pattern else 0


def blob_hits(data: bytes, needles) -> dict:
    """커밋된 blob 하나를 본다 — 패턴 + 사설 블록리스트."""
    h = dict(scan(data))
    low = data.lower()
    for nd, kind in needles:
        if nd.search(low):
            h[kind] = h.get(kind, 0) + 1
    return h


def load_private():
    """블록리스트와 가명 매핑표를 읽는다. 값은 절대 출력하지 않는다."""
    bl_path = Path(os.environ.get('HVF_PHI_BLOCKLIST',
                                  Path.home() / 'hvf_private' / 'phi_blocklist.txt'))
    map_path = Path(os.environ.get('HVF_PSEUDONYM_MAP',
                                   Path.home() / 'hvf_private' / 'pseudonym_map.csv'))
    terms = []
    if bl_path.exists():
        for ln in bl_path.read_text(encoding='utf-8').splitlines():
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                terms.append(ln)
    ids = []
    if map_path.exists():
        with map_path.open(encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh):
                v = (row.get('real_patient_id') or '').strip()
                if v:
                    ids.append(v)
    return terms, ids, bl_path.exists(), map_path.exists()


def positive_control(terms, ids) -> bool:
    print('=' * 74)
    print(' 0. 양성 대조 — 검사기 자체가 작동하는지 먼저 증명')
    print('=' * 74)
    ok = True
    probe = ('cirrus_out/by_case/19999999__test_person__19999999_19010101_'
             '20200101_OPT/x.png C:\\Users\\nobody\\OneDrive /home/nobody '
             'patient 999999 exam 2000-01-01 900101-1234567 '
             'nobody@example.invalid 203.0.113.9').encode('utf-8')
    got = scan(probe, apply_allow=False)
    for name in PATTERNS:
        hit = name in got
        print('  %-22s %s' % (name, '검출 O' if hit else '검출 실패 X'))
        ok &= hit
    underscore_ok = '환자ID(8자리)' in scan(b'_19999999_', apply_allow=False)
    print('  %-22s %s' % ('밑줄 인접 ID', '검출 O' if underscore_ok else '검출 실패 X'))
    ok &= underscore_ok
    fp = scan(b'sha256 aa4a75bba04464120e2b5a9 / 2026-07-23 patch / 8.6531234 '
              b'/ 0.12345678 / -0.3057941198349 / 7.5025954246521',
              apply_allow=False)
    print('  %-22s %s' % ('SHA·작업일자·소수 오탐',
                          '없음 O' if not fp else '오탐 %s X' % list(fp)))
    ok &= not fp
    clean = not scan(b'anonymous@example.org / C:\\Users\\{user}\\x')
    print('  %-22s %s' % ('오탐 면제', '정상 O' if clean else '면제 실패 X'))
    ok &= clean
    # 블록리스트 항목 자체는 출력하지 않는다. 첫 숫자 항목 하나로
    # 경계 규칙만 증명한다 — 맨몸으로 나오면 잡고, 소수 가수 안에 파묻히면
    # 넘긴다.
    num = next((i for i in ids if i.isdigit()), None)
    if num is None:
        print('  %-22s %s' % ('ID 경계 규칙', '숫자 ID 없음 -'))
    else:
        nd = id_needle(num)
        bare = bool(nd.search((' pid ' + num + ' ').encode()))
        buried = bool(nd.search(('0.12' + num + '34').encode()))
        print('  %-22s %s' % ('ID 경계 규칙',
                              '검출 O / 소수 내부 무시 O' if bare and not buried
                              else '규칙 실패 X'))
        ok &= bare and not buried
    prose_ok = (prose_hit('conda', 'conda activate hvf')
                and not prose_hit('conda', 'a secondary source')
                and prose_hit('미해결', '미해결 잔재'))
    print('  %-22s %s' % ('산문 낱말 경계',
                          '검출 O / 부분일치 무시 O' if prose_ok else '규칙 실패 X'))
    ok &= prose_ok
    q = QUASI_PSEUDO.search(b'P-19 OS') and QUASI_LAT.search(b'P-19 OS')
    print('  %-22s %s' % ('준식별자 조합', '검출 O' if q else '검출 실패 X'))
    ok &= bool(q)
    print('  %-22s %s' % ('사설 블록리스트',
                          '%d 항목 로드' % len(terms) if terms else '없음 X'))
    print('  %-22s %s' % ('가명 매핑표', '%d ID 로드' % len(ids) if ids else '없음 X'))
    print('\n  => 검사기 %s\n' % ('정상 — 아래 판정을 신뢰할 수 있다' if ok
                                  else '고장 — 아래 판정은 무의미하다'))
    return ok


def ignored_set(rels, tracked):
    """제외될 경로의 집합. 부정 규칙(!)에 걸린 경로는 제외가 아니다.

    git 2.17 의 check-ignore 는 "마지막으로 일치한 규칙"이 부정 규칙일 때도
    그 경로를 출력한다. 그래서 -v 없이 쓰면 `!paper/results_frozen/*.json`
    처럼 공개하려고 되살린 파일이 "제외됨"으로 잡혀 검사에서 통째로 빠진다.
    실제로 그렇게 빠져 있었다 — 되살린 파일이 곧 공개 대상인데 그 층만
    검사되지 않았다. 규칙 문자열을 받아 부정 규칙이면 되살린다.

    추적 중인 파일은 어떤 규칙에 걸리든 push 되므로 항상 검사한다.
    """
    out = git('check-ignore', '-v', '--stdin',
              stdin='\n'.join(rels).encode('utf-8'))
    got = set()
    for line in out.decode('utf-8').split('\n'):
        if '\t' not in line:
            continue
        rule, path = line.split('\t', 1)
        if rule.rsplit(':', 1)[-1].startswith('!'):   # 되살린 경로
            continue
        got.add(path)
    return got - set(tracked)


def report(title, rows, note=''):
    print('=' * 74)
    print(' %s' % title)
    print('=' * 74)
    if note:
        print(' %s' % note)
    if not rows:
        print('  적출 0건 OK\n')
        return 0
    for rel, hits in sorted(rows):
        print('  %-50s %s' % (rel[:50], ', '.join('%s %d' % kv for kv in hits.items())))
    print('  => 적출 %d개 파일\n' % len(rows))
    return len(rows)


def main():
    terms, ids, have_bl, have_map = load_private()
    if not positive_control(terms, ids):
        print('양성 대조 실패 — 검증을 중단한다.')
        return 2

    files = [p for p in ROOT.rglob('*')
             if p.is_file() and not (SKIP_DIRS & set(p.parts))]
    rels = [p.relative_to(ROOT).as_posix() for p in files]
    tracked = {x for x in git('ls-files').decode('utf-8').split('\n') if x.strip()}
    # 블록리스트 바늘은 L1f 뿐 아니라 L2·L3 에서도 쓴다. 커밋했다가 지운
    # 파일의 기관명·실명은 작업 트리에 없으므로 L1f 로는 영영 안 잡힌다.
    needles = [(id_needle(t), '블록리스트') for t in terms if t] + \
              [(id_needle(i), '실제 환자ID') for i in ids if i]
    ign = ignored_set(rels, tracked)
    live = [(p, r) for p, r in zip(files, rels) if r not in ign]
    missed = sorted(tracked - {r for _, r in live})
    print(' 검사 대상 %d개 (추적 %d · 제외 %d · 미추적이나 검사 %d)'
          % (len(live), len(tracked), len(ign),
             len([r for _, r in live if r not in tracked])))
    if missed:
        print(' ! 추적 중인데 검사에서 빠졌다 — 검증 무효 (%d개)' % len(missed))
        for r in missed[:10]:
            print('   %s' % r)
        return 2
    print()

    l0 = [(r, h) for _, r in live if r not in FILENAME_ALLOW
          for h in [scan(r.encode('utf-8'))] if h]
    n0 = report('L0. 파일명 (gitignore 제외분 제외)', l0,
                '이름만으로 환자를 특정할 수 있는 파일. 바이너리도 여기서 걸린다.')

    l1, bins = [], []
    for p, r in live:
        if p.suffix.lower() not in TEXT_EXT:
            bins.append(r)
            continue
        h = scan(p.read_bytes())
        if h:
            l1.append((r, h))
    n1 = report('L1. 작업 트리 내용 (gitignore 제외분 제외)', l1,
                '이 층이 곧 공개 대상이다. 여기가 깨끗해야 push 가능.')

    l1c = []
    for p, r in live:
        if p.suffix.lower() not in TEXT_EXT or r == SELF:
            continue
        body = p.read_bytes()
        for a in TOOL_ALLOW:
            body = body.replace(a, b'')
        h = {n: len(set(rx.findall(body))) for n, rx in TOOL_PATTERNS.items()
             if rx.search(body)}
        if h:
            l1c.append((r, h))
    meta = git('log', '--format=%B%n%an%n%ae%n%cn%n%ce')
    for a in TOOL_ALLOW:
        meta = meta.replace(a, b'')
    mh = {n: len(set(rx.findall(meta))) for n, rx in TOOL_PATTERNS.items()
          if rx.search(meta)}
    if mh:
        l1c.append(('(커밋 메시지·저자 메타데이터)', mh))
    probe_ok = all(rx.search(b'co-authored-by claude cursor CLAUDE.md')
                   for rx in TOOL_PATTERNS.values())
    n1c = report('L1c. 보조 도구 흔적 (파일 내용 + 커밋 메시지·저자)', l1c,
                 'PHI 는 아니지만 공개 저장소에 남기지 않는다. '
                 '패턴 자기검사: %s' % ('정상 O' if probe_ok else '고장 X'))

    l1d = []
    for p, r in live:
        if p.suffix.lower() != '.md' or r == SELF:
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore')
        for a in PROSE_ALLOW:
            txt = txt.replace(a, '')
        h = {}
        low = txt.lower()
        for name, words in PROSE_PATTERNS.items():
            found = sorted({w for w in words if prose_hit(w, low)})
            if found:
                h[name] = len(found)
        if h:
            l1d.append((r, h))
    n1d = report('L1d. 산문 검사 (.md 의 내부 상태 서술)', l1d,
                 '자동 판정이 아니라 "이 문서를 사람이 다시 읽으라"는 신호다.')

    l1e = []
    for p, r in live:
        if p.suffix.lower() not in HANGUL_TARGETS or r == SELF:
            continue
        n = sum(1 for ln in p.read_text(encoding='utf-8', errors='ignore').split('\n')
                if HANGUL.search(ln))
        if n:
            l1e.append((r, {'한글 포함 줄': n}))
    n1e = report('L1e. 한글 검사 — 적출 대상: %s' % ' '.join(sorted(HANGUL_TARGETS)),
                 l1e,
                 '독자가 저장소를 쓰기 위해 읽는 산문·설정·인용 메타데이터만 적출한다.\n'
                 ' 소스 주석의 한글은 대상이 아니다(아래 참고 수치).')
    info = {}
    for p, r in live:
        if p.suffix.lower() in HANGUL_TARGETS or p.suffix.lower() not in HANGUL_INFO_ONLY:
            continue
        n = sum(1 for ln in p.read_text(encoding='utf-8', errors='ignore').split('\n')
                if HANGUL.search(ln))
        if n:
            k = p.suffix.lower() or '(확장자 없음)'
            f_cnt, l_cnt = info.get(k, (0, 0))
            info[k] = (f_cnt + 1, l_cnt + n)
    if info:
        print(' 참고 — 적출 대상 밖의 한글(의도적으로 남김, README 에 명시):')
        for k in sorted(info, key=lambda x: -info[x][1]):
            print('   %-16s %3d개 파일  %5d줄' % (k, info[k][0], info[k][1]))
        print('   합계 %d줄\n' % sum(v[1] for v in info.values()))

    # ── L1f 사설 블록리스트 ─────────────────────────────────────────────
    l1f, l1f_exempt = [], []
    collisions = 0
    if have_bl or have_map:
        for p, r in live:
            if p.suffix.lower() not in TEXT_EXT or r == SELF:
                continue
            low = p.read_bytes().lower()
            h = {}
            for nd, kind in needles:
                if not nd.search(low):
                    collisions += loose_hits(nd, low)
                    continue
                if r in BLOCKLIST_FILE_ALLOW and kind == '블록리스트':
                    kind = '블록리스트(인용 메타데이터·면제)'
                h[kind] = h.get(kind, 0) + 1
            if h:
                (l1f_exempt if set(h) == {'블록리스트(인용 메타데이터·면제)'}
                 else l1f).append((r, h))
        # 파일명도 본다
        for _, r in live:
            low = r.lower().encode('utf-8')
            h = {k: 1 for nd, k in needles if nd.search(low)}
            if h:
                l1f.append(('(파일명) ' + r, h))
    n1f = report('L1f. 사설 블록리스트 (기관·연구자 실명·IRB·실제 환자ID)', l1f,
                 '리터럴은 이 소스에 없다 — %s / %s 에서 읽는다.'
                 % ('블록리스트 O' if have_bl else '블록리스트 없음 X',
                    '가명매핑표 O' if have_map else '가명매핑표 없음 X'))
    if collisions:
        print(' 참고 — 숫자 경계 밖 우연 일치 %d건(소수 가수 안의 숫자열). '
              '유출이 아니라 자릿수 충돌이다.\n' % collisions)
    if l1f_exempt:
        print(' 면제(차단 아님) — 인용 메타데이터에 저자 실명·소속이 드는 것은 정상:')
        for r, h in l1f_exempt:
            print('   %-46s %s' % (r, ', '.join('%s %d' % kv for kv in h.items())))
        print()
    unverified = not (have_bl and have_map)

    # ── L1g 준식별자 (가명 + laterality/날짜 동시 출현) ────────────────
    l1g = []
    for p, r in live:
        if p.suffix.lower() not in TEXT_EXT or r == SELF:
            continue
        h = {}
        for i, raw in enumerate(p.read_bytes().split(b'\n'), 1):
            if not QUASI_PSEUDO.search(raw):
                continue
            if QUASI_LAT.search(raw):
                h['가명+laterality'] = h.get('가명+laterality', 0) + 1
            if QUASI_DATE.search(raw):
                h['가명+날짜'] = h.get('가명+날짜', 0) + 1
        if QUASI_PSEUDO.search(r.encode('utf-8')) and (
                QUASI_LAT.search(r.encode('utf-8')) or QUASI_DATE.search(r.encode('utf-8'))):
            h['파일명 조합'] = 1
        if h:
            l1g.append((r, h))
    n1g = report('L1g. 준식별자 (가명 P-n 과 laterality/검사일의 동시 출현)', l1g,
                 '가명 단독은 허용. 조합은 재식별 위험이므로 적출한다.')

    l2 = []
    for rel in sorted(tracked):
        if Path(rel).suffix.lower() not in TEXT_EXT:
            continue
        h = blob_hits(git('show', 'HEAD:' + rel), needles)
        if h:
            l2.append((rel, h))
    n2 = report('L2. HEAD — 지금 커밋돼 있는 내용', l2,
                '작업 트리가 아니라 커밋된 blob 을 읽는다.')

    pairs = []
    for line in git('rev-list', '--all', '--objects').decode('utf-8', 'replace').split('\n'):
        if ' ' in line.strip():
            sha, path = line.strip().split(' ', 1)
            pairs.append((sha, path))
    l3 = {}
    if pairs:
        checks = git('cat-file', '--batch-check',
                     stdin='\n'.join(s for s, _ in pairs).encode()).decode().split('\n')
        isblob = {c.split()[0]: (c.split()[1] == 'blob')
                  for c in checks if len(c.split()) >= 2}
        seen = set()
        for sha, path in pairs:
            if not isblob.get(sha) or (sha, path) in seen:
                continue
            seen.add((sha, path))
            if Path(path).suffix.lower() not in TEXT_EXT:
                continue
            h = blob_hits(git('cat-file', 'blob', sha), needles)
            if h:
                cur = l3.setdefault(path, {})
                for k, v in h.items():
                    cur[k] = max(cur.get(k, 0), v)
    report('L3. 이력 전체 (%d 커밋)' % len(git('rev-list', '--all').decode().split()),
           list(l3.items()),
           '이 저장소는 이력을 새로 시작했으므로 L3 도 깨끗해야 정상이다.')

    l1b, opaque = [], []
    for rel in bins:
        p = ROOT / rel
        low = rel.lower()
        try:
            if low.endswith(('.xlsx', '.docx', '.pptx')):
                import zipfile
                z = zipfile.ZipFile(str(p))
                body = b''.join(z.read(n) for n in z.namelist())
            elif low.endswith('.pdf'):
                import zlib
                raw = p.read_bytes()
                parts = []
                for m in re.finditer(rb'stream\r?\n(.*?)endstream', raw, re.S):
                    try:
                        parts.append(zlib.decompress(m.group(1)))
                    except Exception:
                        parts.append(m.group(1))
                body = re.sub(rb'\.\d+', b'', b''.join(parts))
            else:
                opaque.append(rel)
                continue
        except Exception:
            opaque.append(rel)
            continue
        h = scan(body)
        if h:
            l1b.append((rel, h))
    n1b = report('L1b. 바이너리 컨테이너 (xlsx/docx/pdf 압축 해제 후)', l1b,
                 'zip+XML 과 PDF 스트림을 풀어서 본다. 원시 바이트로는 안 잡히는 층.')

    print('=' * 74)
    print(' 참고. 불투명 바이너리 %d개 — 자동 판정 불가(육안 확인 필요)' % len(opaque))
    print('=' * 74)
    for b in sorted(opaque)[:20]:
        print('   %s' % b)
    if len(opaque) > 20:
        print('   ... 외 %d개' % (len(opaque) - 20))

    print('\n' + '=' * 74)
    remote = git('remote', '-v').decode().strip()
    print(' 원격: %s' % (remote if remote else '없음 (push 이력 없음)'))
    bad = n0 or n1 or n1b or n1c or n1d or n1e or n1f or n1g or n2 or l3 or unverified
    print(' 판정: L0 %d / L1 %d / L1b %d / L1c %d / L1d %d / L1e %d / L1f %d / L1g %d'
          ' / L2 %d / L3 %d' % (n0, n1, n1b, n1c, n1d, n1e, n1f, n1g, n2, len(l3)))
    if unverified:
        print(' L1f UNVERIFIED — 사설 파일이 없어 기관·실명·실제ID 를 검사하지 못했다.')
    print(' => %s' % ('PUSH 불가' if bad else '전부 깨끗 — PUSH 가능'))
    print('=' * 74)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
