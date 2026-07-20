"""Pydantic contract tests for app/schemas/."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.predict import PredictResponse
from app.schemas.reverse_translation import ReverseTranslateRequest
from app.schemas.transliterations import TextContext, TransliterateRequest


def test_text_context_defaults_all_unknown():
    ctx = TextContext()
    assert all(v == 'unknown' for v in ctx.model_dump().values())


def test_text_context_rejects_bad_period():
    with pytest.raises(ValidationError):
        TextContext(period='bronze_age')


def test_text_context_accepts_valid_vocab():
    ctx = TextContext(period='new_kingdom', text_type='stela',
                      support='limestone', location_type='temple',
                      site='Karnak')
    assert ctx.period == 'new_kingdom' and ctx.site == 'Karnak'


def test_transliterate_request_defaults():
    req = TransliterateRequest(codes=['G17', 'N35'])
    assert req.confidences == [] and req.direction == 'rtl'
    assert req.layout == 'rows'
    assert req.context == TextContext()


def test_transliterate_request_rejects_bad_layout():
    with pytest.raises(ValidationError):
        TransliterateRequest(codes=['G17'], layout='diagonal')


def test_reverse_request_text_bounds():
    assert ReverseTranslateRequest(text='a').text == 'a'
    assert len(ReverseTranslateRequest(text='x' * 1000).text) == 1000
    with pytest.raises(ValidationError):
        ReverseTranslateRequest(text='')
    with pytest.raises(ValidationError):
        ReverseTranslateRequest(text='x' * 1001)


def test_reverse_request_register_vocab():
    assert ReverseTranslateRequest(text='hi', register='monumental')
    with pytest.raises(ValidationError):
        ReverseTranslateRequest(text='hi', register='casual')


def test_predict_response_roundtrip():
    from tests.conftest import make_raw_result
    raw = make_raw_result()
    # mirror InferenceService mapping shape without the service
    data = {
        'layout': raw['layout'], 'direction': raw['direction'],
        'image_shape': raw['image_shape'],
        'n_detections': raw['n_detections'],
        'n_cartouches': raw['n_cartouches'],
        'outer': raw['outer'], 'cartouches': raw['cartouches'],
    }
    resp = PredictResponse(**data)
    assert resp.transliteration is None and resp.annotated_image is None
    assert resp.cartouches[0].translit == 'wnjs'
