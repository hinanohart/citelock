"""Gate logic + the fail-closed invariant (the property that matters most)."""

from __future__ import annotations

from citelock import GatePolicy, gate

CITS = ["The sky is blue.", "Grass is green."]


def test_all_entailed_allows(entail_backend):
    r = gate("The sky is blue. Grass is green.", CITS, backend=entail_backend)
    assert r.decision == "allow"
    assert r.allowed
    assert r.n_claims == 2 and r.n_entailed == 2
    assert r.is_fully_deterministic is True


def test_one_contradicted_denies(contradict_backend):
    r = gate("The sky is blue.", CITS, backend=contradict_backend)
    assert r.decision == "deny"
    assert r.n_contradicted == 1
    assert r.claim_verdicts[0].rule_applied == "R-A"


def test_neutral_is_baseless_and_denies(neutral_backend):
    r = gate("The sky is blue.", CITS, backend=neutral_backend)
    assert r.decision == "deny"
    assert r.n_baseless == 1
    assert r.claim_verdicts[0].verdict == "baseless"


def test_backend_exception_fails_closed(raising_backend):
    # A backend that raises must DENY, never allow.
    r = gate("The sky is blue.", CITS, backend=raising_backend)
    assert r.decision == "deny"
    assert r.claim_verdicts[0].rule_applied == "fail-closed:backend-error"
    assert r.claim_verdicts[0].error is not None


def test_empty_answer_fails_closed(entail_backend):
    r = gate("   ", CITS, backend=entail_backend)
    assert r.decision == "deny"
    assert r.n_claims == 0
    assert "no verifiable claims" in r.reason


def test_too_few_citations_fails_closed(entail_backend):
    r = gate("The sky is blue.", [], backend=entail_backend)
    assert r.decision == "deny"
    assert "citation" in r.reason


def test_min_citations_policy(entail_backend):
    pol = GatePolicy(min_citations=3)
    r = gate("The sky is blue.", CITS, backend=entail_backend, policy=pol)
    assert r.decision == "deny"


def test_contradiction_wins_over_entailment():
    # If one citation entails and another contradicts, R-A (contradiction) wins.
    from citelock.backends.base import NLIBackend, NLIResult

    class Mixed(NLIBackend):
        @property
        def backend_id(self) -> str:
            return "mixed"

        @property
        def is_deterministic(self) -> bool:
            return True

        def classify(self, premise, hypothesis):
            if "entail" in premise:
                return NLIResult(entailment=0.95, contradiction=0.02, neutral=0.03)
            return NLIResult(entailment=0.02, contradiction=0.95, neutral=0.03)

    # min_relevance=0: this test isolates aggregation, not the relevance filter.
    r = gate(
        "A claim.",
        ["entail passage", "contradict passage"],
        backend=Mixed(),
        policy=GatePolicy(min_relevance=0.0),
    )
    assert r.decision == "deny"
    assert r.claim_verdicts[0].verdict == "contradicted"


def test_nondeterministic_backend_flips_flag(nondet_entail_backend):
    r = gate("The sky is blue.", CITS, backend=nondet_entail_backend)
    assert r.decision == "allow"
    assert r.is_fully_deterministic is False


def test_input_digest_stable_and_sensitive(entail_backend):
    r1 = gate("The sky is blue.", CITS, backend=entail_backend)
    r2 = gate("The sky is blue.", CITS, backend=entail_backend)
    r3 = gate("The sky is green.", CITS, backend=entail_backend)
    assert r1.input_digest == r2.input_digest
    assert r1.input_digest != r3.input_digest


def test_threshold_boundary():
    from citelock.backends.base import NLIBackend, NLIResult

    class Exact(NLIBackend):
        @property
        def backend_id(self) -> str:
            return "exact"

        @property
        def is_deterministic(self) -> bool:
            return True

        def classify(self, premise, hypothesis):
            return NLIResult(entailment=0.5, contradiction=0.0, neutral=0.5)

    # entailment exactly at tau_entail (0.5) -> entailed (>= is inclusive).
    # min_relevance=0: "p" shares no words with the claim; this test checks the
    # threshold boundary, not relevance filtering.
    r = gate("A claim.", ["p"], backend=Exact(), policy=GatePolicy(min_relevance=0.0))
    assert r.claim_verdicts[0].verdict == "entailed"
