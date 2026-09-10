"""Loader for config/params.yaml — the recorded value of every reported constant.

Kept deliberately small: the analysis code does not import it (see the scope
note in the YAML), and `tests/test_config.py` uses it to check that the recorded
values still match the literals in the source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(os.environ.get(
    'HVF_CONFIG', Path(__file__).resolve().parent / 'config' / 'params.yaml'))

_cache: dict | None = None


def load_config(path: Path | str | None = None) -> dict:
    global _cache
    if path is not None:
        return yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if _cache is None:
        _cache = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    return _cache


def get(*keys: str, default: Any = None) -> Any:
    node: Any = load_config()
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def seed() -> int:
    return int(get('seed'))
