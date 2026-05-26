"""Backend contract: stub determinism, fixture replay, NLIResult shape."""

from __future__ import annotations

import pytest

from citelock.backends import FixtureBackend, LexicalStubBackend
from citelock.backends.base import NLIResult


def test_stub_is_deterministic():
    b = LexicalStubBackend()
    r1 = b.classify("The sky is blue.", "The sky is blue.")
    r2 = b.classify("The sky is blue.", "The sky is blue.")
    assert (r1.entailment, r1.contradiction, r1.neutral) == (
        r2.entailment,
        r2.contradiction,
        r2.neutral,
    )
    assert b.is_deterministic is True


def test_stub_entails_on_high_overlap():
    b = LexicalStubBackend()
    r = b.classify("Paris is the capital of France.", "Paris is the capital of France.")
    assert r.entailment > r.contradiction
    assert r.entailment > r.neutral


def test_stub_detects_negation():
    b = LexicalStubBackend()
    r = b.classify("The treaty was signed in 1990.", "The treaty was not signed in 1990.")
    assert r.contradiction > r.entailment


def test_stub_neutral_on_no_overlap():
    b = LexicalStubBackend()
    r = b.classify("Quantum chromodynamics is complex.", "Bananas are yellow.")
    assert r.neutral > r.entailment


def test_classify_batch_default_loops():
    b = LexicalStubBackend()
    out = b.classify_batch([("a b c", "a b c"), ("x y z", "p q r")])
    assert len(out) == 2
    assert all(isinstance(r, NLIResult) for r in out)


def test_fixture_roundtrip(tmp_path):
    records = [
        {
            "premise": "p1",
            "hypothesis": "h1",
            "entailment": 0.9,
            "contradiction": 0.05,
            "neutral": 0.05,
        },
    ]
    import json

    path = tmp_path / "fx.json"
    path.write_text(json.dumps({"backend_id": "local:test", "records": records}))
    b = FixtureBackend.from_file(path)
    r = b.classify("p1", "h1")
    assert r.entailment == 0.9
    assert "local:test" in b.backend_id


def test_fixture_missing_pair_raises(tmp_path):
    b = FixtureBackend({}, backend_id="empty")
    with pytest.raises(KeyError):
        b.classify("nope", "nope")
