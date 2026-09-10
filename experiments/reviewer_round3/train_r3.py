#!/usr/bin/env python3
"""(3) 여섯 번째 CNN 백본 학습 런처 — train.py 를 import 해서 백본만 하나 추가한다.

train.py 를 복사하지 않는다. 모듈로 import 한 뒤
  - BACKBONE_CHOICES 에 'resnet50' 을 더하고
  - build_image_backbone 을 감싸 resnet50 분기만 추가한다 (나머지는 원본 호출).
그 외 설정(head, optimizer, epochs, patience, early_stop_metric, 증강 없음, 폴드,
seed)은 전부 명령행으로 기존 5백본과 동일하게 준다 — 기본값을 새로 정하지 않는다.

resnet50 은 densenet121 과 같은 방식으로 torchvision ImageNet 가중치를 쓰고
분류 head 를 Identity 로 바꾼다.

산출물은 --out_dir 로 준 절대경로에만 쓴다. 정본 runs/ 와 원고 미변경.
env: hvf (torch + timm)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch.nn as nn                     # noqa: E402
from torchvision import models            # noqa: E402

import train as T                         # noqa: E402

assert T.ROOT == REPO, f'ROOT 불일치: {T.ROOT}'

_orig_build = T.build_image_backbone


def build_image_backbone(name: str):
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_feat
    if name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, in_feat
    return _orig_build(name)


T.build_image_backbone = build_image_backbone
T.BACKBONE_CHOICES = tuple(T.BACKBONE_CHOICES) + ('resnet50', 'efficientnet_b0')

if __name__ == '__main__':
    T.main()
