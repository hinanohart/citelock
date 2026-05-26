"""LLM-judge backend parsing and LLM-assisted decomposition (no network)."""

from __future__ import annotations

import pytest

from citelock import decompose_claims, gate
from citelock.backends.llm import LLMJudgeBackend


def _fake_completion(label: str, confidence: float = 0.9):
    def fn(prompt: str) -> str:
        return f'Sure! {{"label": "{label}", "confidence": {confidence}}}'

    return fn


def test_llm_backend_entailment():
    b = LLMJudgeBackend(_fake_completion("entailment", 0.8), model_label="fake")
    r = b.classify("premise", "hypothesis")
    assert r.entailment == 0.8
    assert b.backend_id == "llm-judge:fake"
    assert b.is_deterministic is False


def test_llm_backend_contradiction_and_neutral():
    c = LLMJudgeBackend(_fake_completion("contradiction")).classify("p", "h")
    assert c.contradiction > c.entailment
    n = LLMJudgeBackend(_fake_completion("neutral")).classify("p", "h")
    assert n.neutral > n.entailment


def test_llm_backend_bad_json_raises_then_gate_fails_closed():
    b = LLMJudgeBackend(lambda prompt: "no json here")
    with pytest.raises(ValueError):
        b.classify("p", "h")
    # And through the gate, that error must produce a deny, never an allow.
    r = gate("A claim.", ["passage"], backend=b)
    assert r.decision == "deny"
    assert r.claim_verdicts[0].rule_applied == "fail-closed:backend-error"


def test_llm_backend_unknown_label_raises():
    b = LLMJudgeBackend(_fake_completion("maybe"))
    with pytest.raises(ValueError):
        b.classify("p", "h")


def test_llm_decomposition_marks_nondeterministic():
    def llm_fn(answer: str) -> list[str]:
        return ["Claim one.", "Claim two.", "  "]

    claims = decompose_claims("anything", method="llm", llm_fn=llm_fn)
    assert [c.text for c in claims] == ["Claim one.", "Claim two."]
    assert all(c.decomposed_by == "llm" and not c.is_deterministic for c in claims)


def test_llm_decomposition_flips_determinism_flag():
    from citelock.backends.stub import LexicalStubBackend

    r = gate(
        "Paris is the capital of France.",
        ["Paris is the capital of France."],
        backend=LexicalStubBackend(),
        decompose="llm",
        llm_fn=lambda a: ["Paris is the capital of France."],
    )
    assert r.is_fully_deterministic is False
