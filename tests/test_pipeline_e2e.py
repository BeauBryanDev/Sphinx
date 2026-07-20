"""End-to-end pipeline tests on canonical fixtures (real ONNX model).

Expected numbers from CLAUDE.md (letterbox + no enhance). Detection
counts drift slightly across model versions, so assertions use floors,
not exact counts — except cartouche identity, which must hold.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_unas_columns_finds_both_wnjs(sphinx_pipeline, unas_image):
    result = sphinx_pipeline.run(str(unas_image), layout='columns')
    assert result['layout'] == 'columns'
    assert result['n_detections'] >= 90
    assert result['n_cartouches'] == 2
    translits = [c['translit'] for c in result['cartouches']]
    assert translits == ['wnjs', 'wnjs']


def test_glyph_wall_negative_control(sphinx_pipeline, glyph_wall_image):
    """No cartouches in scene — must find none (overfit guard)."""
    result = sphinx_pipeline.run(str(glyph_wall_image), layout='columns')
    assert result['n_cartouches'] == 0
    assert result['n_detections'] >= 40


def test_result_dict_contract(sphinx_pipeline, glyph_wall_image):
    result = sphinx_pipeline.run(str(glyph_wall_image), layout='columns')
    for key in ('layout', 'direction', 'image_shape', 'n_detections',
                'n_cartouches', 'outer', 'cartouches'):
        assert key in result, key
    outer = result['outer']
    for key in ('slots', 'boundary_hints', 'n_synthetic', 'correction'):
        assert key in outer, key
    assert len(outer['correction']['flat_corrected_seq']) == \
        len(outer['slots'])
