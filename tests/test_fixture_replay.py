"""Offline gate-logic replay against scores captured from the real NLI model.

These run with no network and no torch: the scores were recorded once from
``cross-encoder/nli-deberta-v3-base`` into tests/fixtures/deberta_pairs.json.
This is how citelock's gate behaviour stays reproducible in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citelock import gate
from citelock.backends.fixture import FixtureBackend

FIXTURE = Path(__file__).parent / "fixtures" / "deberta_pairs.json"


@pytest.fixture
def backend() -> FixtureBackend:
    return FixtureBackend.from_file(FIXTURE)


def test_entailed_answer_allowed(backend):
    r = gate(
        "Paris is the capital of France.",
        ["Paris is the capital of France."],
        backend=backend,
    )
    assert r.decision == "allow"
    assert r.claim_verdicts[0].verdict == "entailed"
    assert r.claim_verdicts[0].max_entailment > 0.9


def test_contradicted_answer_denied(backend):
    r = gate(
        "The treaty was never signed.",
        ["The treaty was signed in 1990."],
        backend=backend,
    )
    assert r.decision == "deny"
    assert r.claim_verdicts[0].verdict == "contradicted"


def test_unrelated_citation_denied(backend):
    # An unrelated passage must not support the claim -> deny (fail-closed).
    r = gate(
        "The stock market rose sharply today.",
        ["Bananas are a yellow fruit grown in tropical regions."],
        backend=backend,
    )
    assert r.decision == "deny"


def test_fixture_backend_is_deterministic(backend):
    assert backend.is_deterministic
    assert "cross-encoder/nli-deberta-v3-base" in backend.backend_id
