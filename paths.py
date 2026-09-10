"""저장소 트리와 원자료 디렉터리의 위치를 한 곳에서 해석한다.

코드에 개인 계정 경로(`C:\\Users\\<name>\\...`)를 하드코딩하지 않기 위해 도입했다.
원자료(`cirrus_out/` 등)는 식별자가 파일명에 남아 있어 저장소 트리 밖으로 옮길
예정이므로, 코드가 참조하는 지점을 여기로 모은다.

환경변수
--------
HVF_ROOT
    저장소 루트. 기본값은 이 파일이 있는 디렉터리.
HVF_DATA_ROOT
    원자료 루트(`cirrus_out/`의 부모). 기본값은 HVF_ROOT — 아직 트리 안에
    있는 현재 상태에서 기존 동작을 그대로 유지한다. 트리 밖으로 옮긴 뒤에는
    `export HVF_DATA_ROOT=/home/<user>/hvf_private/data` 처럼 지정한다.
HVF_LEGACY_ROOTS
    CSV의 경로 컬럼에 남아 있는 옛 절대경로 접두사들. `os.pathsep`(리눅스 `:`)로
    구분한다. 지정하지 않으면 아래 기본 목록을 쓴다.
HVF_POPPLER_PATH
    Windows에서 pdf2image가 쓰는 poppler bin 경로. 리눅스에서는 불필요.

`rewrite_data_path()`
---------------------
`ml_dataset.csv`의 `gca_dir`/`rnfl_dir`/`sfa_dir` 값은 데이터를 만든 Windows
머신의 절대경로다. 옛 접두사를 DATA_ROOT로 갈아끼워 현재 머신에서 열 수 있게
한다. 접두사가 안 맞으면 입력을 그대로 돌려준다(상대경로·이미 정규화된 경로).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent))

# 원자료 루트. 기본값 = ROOT (cirrus_out/이 아직 트리 안에 있는 상태)
DATA_ROOT = Path(os.environ.get('HVF_DATA_ROOT', ROOT))

# CSV에 박제된 옛 절대경로 접두사. 사용자명은 포함하지 않는다.
_DEFAULT_LEGACY_ROOTS = (
    r'C:\Users\{user}\OneDrive\바탕 화면\hvf_project',
)


def _legacy_roots():
    env = os.environ.get('HVF_LEGACY_ROOTS')
    if env:
        return [p for p in env.split(os.pathsep) if p]
    # 사용자명을 코드에 남기지 않기 위해 CSV에서 실제 접두사를 추론한다.
    return list(_DEFAULT_LEGACY_ROOTS)


def rewrite_data_path(value: str) -> str:
    """CSV의 경로 문자열을 현재 머신의 DATA_ROOT 기준으로 바꾼다."""
    if not value:
        return value
    norm = value.replace('\\', '/')
    # 접두사를 모르는 경우에도 'cirrus_out/' 이하만 잘라내면 복원할 수 있다.
    marker = 'cirrus_out/'
    idx = norm.find(marker)
    if idx >= 0:
        return str(DATA_ROOT / norm[idx:])
    for legacy in _legacy_roots():
        ln = legacy.replace('\\', '/')
        if norm.startswith(ln):
            return str(DATA_ROOT) + norm[len(ln):]
    return value


POPPLER_PATH = os.environ.get('HVF_POPPLER_PATH') or None
