"""Judge-sanity harness behaviour with controllable backends."""

from __future__ import annotations

from citelock.backends.base import NLIBackend, NLIResult
from citelock.sanity import SanityCase, run_sanity


class ReadsCitations(NLIBackend):
    """Entails only when the citation shares the claim's first content word.

    A backend that genuinely depends on the citation text: dropping the right
    citation flips the verdict, so drop/negate sanity should pass.
    """

    @property
    def backend_id(self) -> str:
        return "reads-citations"

    @property
    def is_deterministic(self) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        hwords = hypothesis.lower().split()
        key = next((w for w in hwords if w not in {"it", "is", "the", "not", "case", "that"}), "")
        if "not the case" in hypothesis.lower():
            return NLIResult(entailment=0.1, contradiction=0.6, neutral=0.3)
        if key and key in premise.lower():
            return NLIResult(entailment=0.9, contradiction=0.02, neutral=0.08)
        return NLIResult(entailment=0.1, contradiction=0.05, neutral=0.85)


class IgnoresCitations(NLIBackend):
    """Always entails regardless of citation — the failure mode sanity catches."""

    @property
    def backend_id(self) -> str:
        return "ignores-citations"

    @property
    def is_deterministic(self) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return NLIResult(entailment=0.95, contradiction=0.02, neutral=0.03)


CASES = [
    SanityCase(
        answer="Photosynthesis occurs in plants.",
        citations=["Photosynthesis is a process in plants.", "Rocks are minerals."],
    ),
    SanityCase(
        answer="Gravity attracts masses.",
        citations=["Gravity is a force that attracts masses.", "Birds can fly."],
    ),
]


def test_good_backend_passes_sanity():
    report = run_sanity(CASES, backend=ReadsCitations(), seed=1)
    assert report.n_entailed_claims >= 2
    assert report.shuffle_invariance_rate == 1.0
    assert report.drop_flip_rate >= 0.5
    assert report.passed


def test_ignoring_backend_fails_drop_sanity():
    report = run_sanity(CASES, backend=IgnoresCitations(), seed=1)
    # It always entails, so dropping a citation never flips -> sanity fails.
    assert report.drop_flip_rate == 0.0
    assert not report.passed
    assert any("not be reading" in n for n in report.notes)


def test_bootstrap_ci_is_ordered():
    report = run_sanity(CASES, backend=ReadsCitations(), seed=2)
    lo, hi = report.drop_flip_ci
    assert 0.0 <= lo <= hi <= 1.0
