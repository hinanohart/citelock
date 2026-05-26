"""Relevance filtering: distractor citations must not flip a supported claim.

This guards the fix for the gate's biggest false-deny source: NLI models score
unrelated passages as confident contradictions, and aggregating over a RAG
retriever's noisy passages then denies correct answers. Only on-topic citations
(>= policy.min_relevance content-word overlap) may vote.
"""

from __future__ import annotations

from citelock import GatePolicy, gate
from citelock.backends.base import NLIBackend, NLIResult
from citelock.textutil import content_words, relevance


class SupportSkyContradictElse(NLIBackend):
    """Entails passages mentioning 'sky'; confidently contradicts everything else.

    Models the real failure mode: an off-topic passage gets a high contradiction
    score it does not deserve.
    """

    @property
    def backend_id(self) -> str:
        return "test-sky"

    @property
    def is_deterministic(self) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        if "sky" in premise.lower():
            return NLIResult(entailment=0.9, contradiction=0.02, neutral=0.08)
        return NLIResult(entailment=0.02, contradiction=0.9, neutral=0.08)


ANSWER = "The sky is blue."
CITS = ["The sky is blue and clear.", "Mount Everest is the tallest mountain."]


def test_distractor_is_filtered_by_default():
    # Default min_relevance: the off-topic (and falsely-contradicting) citation
    # is filtered, so the supported claim is allowed.
    r = gate(ANSWER, CITS, backend=SupportSkyContradictElse())
    assert r.decision == "allow"
    assert r.claim_verdicts[0].verdict == "entailed"


def test_min_relevance_zero_restores_old_overdeny():
    # With the filter off, the distractor's false contradiction denies (R-A).
    r = gate(ANSWER, CITS, backend=SupportSkyContradictElse(), policy=GatePolicy(min_relevance=0.0))
    assert r.decision == "deny"
    assert r.claim_verdicts[0].verdict == "contradicted"


def test_evidence_records_relevance(entail_backend):
    r = gate(ANSWER, CITS, backend=entail_backend)
    ev = {e.citation_id: e for e in r.claim_verdicts[0].evidence}
    # Both citations appear in evidence; only the on-topic one is marked relevant.
    relevants = [e for e in ev.values() if e.relevant]
    assert len(relevants) == 1
    assert all(0.0 <= e.relevance <= 1.0 for e in ev.values())


def test_no_relevant_citation_denies(entail_backend):
    # Every citation is off-topic -> no citation may vote -> fail-closed deny.
    r = gate("The sky is blue.", ["Bananas are a yellow fruit."], backend=entail_backend)
    assert r.decision == "deny"
    assert r.claim_verdicts[0].rule_applied == "R-C:no-relevant-citation"


def test_relevance_scores():
    assert (
        relevance("Paris is the capital of France.", "Paris is the capital city of France.") == 1.0
    )
    assert relevance("Paris is the capital of France.", "Saturn has rings.") == 0.0
    # partial overlap is between 0 and 1
    r = relevance("Mount Everest is the shortest peak.", "Everest is a peak.")
    assert 0.0 < r < 1.0


def test_content_words_strips_stopwords():
    assert content_words("The cat is on the mat.") == {"cat", "mat"}
    assert content_words("not no never") == set()
