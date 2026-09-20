"""paper/results_frozen/ must stay parseable, de-identified, and quotable.

These are the numbers the manuscript prints. The tests check that the files are
there, that the headline values are what the text says, and that the three
transforms the freezer applies -- absolute paths removed, the representative
eye's quasi-identifier removed, the prose translated -- are still in force.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'paper' / 'results_frozen'


def frozen_files():
    return sorted(FROZEN.glob('*.json'))


def expected_names():
    """The freezer's own source list, so the two cannot drift apart."""
    import sys

    sys.path.insert(0, str(ROOT / 'scripts'))
    from freeze_paper_results import SOURCES

    return set(SOURCES)


def test_all_present():
    assert {p.name for p in frozen_files()} == expected_names()


def test_readme_states_the_number_of_files():
    """The count in the repository layout table is quoted, so it goes stale."""
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    m = re.search(r'The (\d+) JSON files the manuscript', text)
    assert m, 'the README no longer states how many frozen files there are'
    assert int(m.group(1)) == len(frozen_files())


@pytest.mark.parametrize('p', frozen_files(), ids=lambda p: p.name)
def test_parses(p):
    json.loads(p.read_text(encoding='utf-8'))


@pytest.mark.parametrize('p', frozen_files(), ids=lambda p: p.name)
def test_no_absolute_home_path(p):
    text = p.read_text(encoding='utf-8')
    for m in re.findall(r'/(?:home|Users)/[A-Za-z0-9_.-]+', text):
        assert m.endswith('/<user>'), f'{p.name}: {m}'


def test_representative_eye_is_not_identifiable():
    """Pseudonym, laterality and examination date are re-identifying together."""
    doc = json.loads((FROZEN / 'case_profile.json').read_text(encoding='utf-8'))
    case = doc['case']
    for key in ('pseudonym', 'eye', 'vf_date'):
        assert key not in case, key
    # the part the manuscript actually cites survives
    assert doc['distribution_position']['rmse_fusion_minus_summary']['n'] == 240


def test_headline_out_of_fold_values():
    """Section 4: XGB 8.66, image branch 8.54, late fusion 8.07 (IR-v2, OOF)."""
    doc = json.loads((FROZEN / 'final_model_comparison.json').read_text(encoding='utf-8'))
    row = next(r for r in doc['rows'] if r['backbone'] == 'inception_resnet_v2')
    assert row['w_xgb'] == 0.47
    assert row['oof']['n_eyes'] == 240
    assert row['test']['n_eyes'] == 37
    assert row['oof']['xgb']['rmse'] == 8.66
    assert row['oof']['cnn']['rmse'] == 8.54
    assert row['oof']['fusion']['rmse'] == 8.07


def test_backbone_set_matches_config():
    import sys
    sys.path.insert(0, str(ROOT))
    from hvf_config import get
    want = set(get('constants', 'backbones', 'ensemble5'))
    doc = json.loads((FROZEN / 'backbone_matrix_table.json').read_text(encoding='utf-8'))
    assert {r['backbone'] for r in doc['rows']} == want

    # the same five, under their display labels
    labels = get('constants', 'backbones', 'labels')
    doc = json.loads((FROZEN / 'fusion_consistency_matrix.json').read_text(encoding='utf-8'))
    assert {r['backbone'] for r in doc['rows']} == {labels[b] for b in want}


def freezer():
    import sys

    sys.path.insert(0, str(ROOT / 'scripts'))
    import freeze_paper_results

    return freeze_paper_results


HANGUL = re.compile(r'[\uac00-\ud7a3]')


@pytest.mark.parametrize('p', frozen_files(), ids=lambda p: p.name)
def test_prose_is_in_english(p):
    """The producers annotate in Korean; the freezer translates on the way out."""
    text = p.read_text(encoding='utf-8')
    assert not HANGUL.search(text), f'{p.name}: untranslated prose'


def test_translation_table_translates():
    """Keys are what a producer wrote, values are what a reader gets."""
    prose = freezer().PROSE
    for ko, en in prose.items():
        assert HANGUL.search(ko), ko
        assert not HANGUL.search(en), en


def test_translation_moves_no_number():
    """The condition on the third transform: it rewrites strings, nothing else.

    Every entry of the table is fed through with numbers beside it, including
    the shapes that turn up in these files -- integers, negatives, exponents,
    and values whose repr is not their literal.
    """
    fr = freezer()
    doc = {
        'prose': list(fr.PROSE),
        'nested': [{'note': ko, 'rmse': 8.0832} for ko in fr.PROSE],
        'numbers': {'w': 0.47, 'n': 240, 'delta': -0.089, 'lr': 1e-4,
                    'p': 1.886e-4, 'zero': 0, 'flag': True},
    }
    before = fr.numbers(doc)
    out = fr.scrub(doc)
    assert fr.numbers(out) == before
    assert not HANGUL.search(json.dumps(out, ensure_ascii=False))
