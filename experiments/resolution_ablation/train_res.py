#!/usr/bin/env python3
"""공간 해상도 절제 파일럿 런처 — train.py 를 import 해서 입력 변환만 하나 끼운다.

train.py 를 복사하지 않는다. 모듈로 import 한 뒤 image_preprocessing.get_transforms 를
감싸, ToTensor 와 Normalize 사이에 블록 평균(계단 함수) 변환을 넣는다.
따라서 ImageNet 정규화는 블록 평균 '이후'에 적용된다.

블록 경계는 패널 하나(161x161) 안에서만 잡는다. VFDataset 이 4장을 각각 따로
transform 하므로 GCA/RNFL 이 섞일 여지가 없다 (concat 은 모델 forward 에서 일어난다).

한 변당 블록 수 n:
  edges = round(linspace(0, 161, n+1)) 로 겹치지 않는 분할을 만들고 블록별 평균으로 채운다.
  n >= 161 이면 항등(원본) — 정본 재현 게이트용.

그 외 설정(백본, optimizer, epochs, patience, batch, seed, 폴드)은 전부 명령행으로
정본과 동일하게 준다. 산출물은 --out_dir 절대경로에만 쓴다. 정본 runs/ 와 원고 미변경.
env: hvf
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
from torchvision import transforms          # noqa: E402

import image_preprocessing as IP            # noqa: E402
import train as T                           # noqa: E402

assert T.ROOT == REPO, f'ROOT 불일치: {T.ROOT}'

PANEL = 161


def block_matrix(size: int, n: int) -> torch.Tensor:
    """M[p,q] = 1/|block(p)| if q in block(p) else 0.  M @ x @ M.T = 블록 평균 계단 함수."""
    edges = np.round(np.linspace(0, size, n + 1)).astype(int)
    assert edges[0] == 0 and edges[-1] == size
    widths = np.diff(edges)
    assert (widths > 0).all(), f'빈 블록 발생: n={n}, size={size}'
    M = torch.zeros(size, size, dtype=torch.float32)
    for i in range(n):
        a, b = int(edges[i]), int(edges[i + 1])
        M[a:b, a:b] = 1.0 / (b - a)
    return M


class BlockAverage:
    """[C,H,W] 텐서를 패널 안에서 n x n 블록 평균 계단 함수로 바꾼다."""

    def __init__(self, n: int):
        self.n = n
        self.identity = n >= PANEL
        self._cache = {}

    def _M(self, size: int) -> torch.Tensor:
        if size not in self._cache:
            self._cache[size] = block_matrix(size, self.n)
        return self._cache[size]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x
        C, H, W = x.shape
        Mh = self._M(H)
        Mw = self._M(W)
        return torch.matmul(torch.matmul(Mh, x), Mw.t())

    def __repr__(self):
        return f'BlockAverage(n={self.n}, identity={self.identity})'


BLOCKS_N = None
_orig_get_transforms = IP.get_transforms


def get_transforms(mode='train', image_size=IP.IMG_SIZE):
    """정본 파이프라인을 그대로 만든 뒤 ToTensor 와 Normalize 사이에 블록 평균을 삽입."""
    base = _orig_get_transforms(mode, image_size)
    if BLOCKS_N is None:
        return base
    ts = list(base.transforms)
    i_tt = [k for k, t in enumerate(ts) if isinstance(t, transforms.ToTensor)]
    i_nm = [k for k, t in enumerate(ts) if isinstance(t, transforms.Normalize)]
    assert len(i_tt) == 1 and len(i_nm) == 1, f'예상과 다른 transform 구성: {ts}'
    assert i_nm[0] == i_tt[0] + 1, f'ToTensor 바로 뒤가 Normalize 가 아님: {ts}'
    ts.insert(i_tt[0] + 1, BlockAverage(BLOCKS_N))
    return transforms.Compose(ts)


def main():
    global BLOCKS_N
    argv = sys.argv[1:]
    if '--blocks_n' not in argv:
        raise SystemExit('--blocks_n <int> 필수 (한 변당 블록 수; 161 = 원본)')
    k = argv.index('--blocks_n')
    BLOCKS_N = int(argv[k + 1])
    del argv[k:k + 2]
    sys.argv = [sys.argv[0]] + argv

    IP.get_transforms = get_transforms
    print(f'=== 블록 평균: n={BLOCKS_N} (패널당 {BLOCKS_N**2} 블록, '
          f'identity={BLOCKS_N >= PANEL}) ===', flush=True)
    T.main()


if __name__ == '__main__':
    main()
