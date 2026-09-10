"""config/params.yaml records the constants; the code still hard-codes them.

Every value in the YAML is read back out of the source file it was copied from,
by parsing that file rather than importing it, so these tests run without torch,
xgboost or any data. If someone edits a constant in the code and not in the YAML
(or the other way round), one of these fails.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from hvf_config import get, load_config, seed  # noqa: E402


def source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding='utf-8')


def kwargs_of(rel: str, func: str) -> dict:
    """Literal keyword arguments of the dict(...) call inside `func`."""
    tree = ast.parse(source(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'dict':
                    out = {}
                    for kw in call.keywords:
                        try:
                            out[kw.arg] = ast.literal_eval(kw.value)
                        except ValueError:
                            pass
                    return out
    raise AssertionError(f'{func} not found in {rel}')


def test_config_loads():
    cfg = load_config()
    assert cfg['constants'] and cfg['paths'] and cfg['environment']


def test_seed():
    assert seed() == 42
    assert 42 in get('seeds_reported')
    assert re.search(r'def set_seed\(seed: int = 42\)', source('train.py'))
    assert re.search(r"--seed'[^)]*default=42", source('train.py'), re.S)


@pytest.mark.parametrize('producer', ['baseline_xgb.py', 'scripts/export_xgb_oof.py'])
def test_xgb_hyperparameters(producer):
    """Both branches of the summary model must carry the Section 3.6 settings."""
    code = kwargs_of(producer, '_xgb_kwargs')
    want = get('constants', 'xgb')
    for key in ('n_estimators', 'learning_rate', 'max_depth', 'subsample',
                'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state'):
        assert code[key] == want[key], f'{producer}: {key} {code[key]} != {want[key]}'


def test_blind_points():
    m = re.search(r'BLIND_POINTS = frozenset\(\{([^}]*)\}\)',
                  source('image_preprocessing.py'))
    assert m
    code = sorted(ast.literal_eval('[' + m.group(1) + ']'))
    assert code == sorted(get('constants', 'vf', 'blind_points'))
    assert get('constants', 'vf', 'n_points_total') == 54
    assert (get('constants', 'vf', 'n_points_total')
            - len(code) == get('constants', 'vf', 'n_points_used'))


def test_input_geometry():
    """--input_geometry paper: two 161x161 panels concatenated into 161x322."""
    code = source('train.py')
    inp = get('constants', 'inputs')
    assert f"args.image_h, args.image_w = {inp['panel_h']}, {inp['panel_w']}" in code
    assert f"args.model_h, args.model_w = {inp['model_h']}, {inp['model_w']}" in code
    assert inp['model_w'] == 2 * inp['panel_w']
    assert inp['use_deviation'] is False


def test_fusion_weight_grid():
    n = get('constants', 'fusion', 'grid_points')
    for rel in ('scripts/recompute_late_fusion.py', 'scripts/fusion_eval_common.py'):
        assert re.search(rf'np\.linspace\(0(?:\.0)?, ?1(?:\.0)?, ?{n}\)', source(rel)), rel


def test_xgb_regressor_count_matches_used_points():
    assert get('constants', 'xgb', 'n_regressors') == get('constants', 'vf', 'n_points_used')


def test_cv_folds():
    assert get('constants', 'split', 'cv_folds') == 5
    assert '[0, 1, 2, 3, 4]' in source('train.py')
