"""Claim decomposition: determinism, idempotence, spans, edge cases."""

from __future__ import annotations

import pytest

from citelock import decompose_claims


def test_basic_sentence_split():
    claims = decompose_claims("The sky is blue. Grass is green.")
    assert [c.text for c in claims] == ["The sky is blue.", "Grass is green."]
    assert all(c.decomposed_by == "deterministic" and c.is_deterministic for c in claims)


def test_idempotent():
    text = "First fact here. Second one too! And a third?"
    a = [c.text for c in decompose_claims(text)]
    b = [c.text for c in decompose_claims(text)]
    assert a == b


def test_spans_recover_text():
    text = "Alpha beta. Gamma delta."
    for c in decompose_claims(text):
        assert c.origin_span is not None
        s, e = c.origin_span
        assert text[s:e] == c.text


def test_empty_and_whitespace_yield_no_claims():
    assert decompose_claims("") == []
    assert decompose_claims("    \n\t ") == []


def test_unique_ids():
    claims = decompose_claims("A. B. C. D.")
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids))


def test_llm_requires_fn():
    with pytest.raises(ValueError):
        decompose_claims("text", method="llm")


def test_unknown_method():
    with pytest.raises(ValueError):
        decompose_claims("text", method="bogus")
