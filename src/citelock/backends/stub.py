"""A deterministic, dependency-free NLI backend for tests and demos.

This is a transparent lexical heuristic, NOT a real entailment model. It exists
so the gate, ledger, CLI and sanity harness can be exercised offline (no torch,
no network, no model download) and so the test suite is fully reproducible.

Do not gate production traffic with it. Use ``LocalCrossEncoderBackend`` (the
``[nli]`` extra) for real entailment. ``backend_id`` makes the choice auditable
in the ledger.
"""

from __future__ import annotations

import re

from .base import NLIBackend, NLIResult

_WORD = re.compile(r"[a-z0-9]+")
_NEGATIONS = frozenset({"not", "no", "never", "none", "cannot", "without", "n't", "nor", "neither"})
# Words carrying no entailment signal; excluded from overlap.
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "with",
        "by",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "from",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
    }
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _has_negation(text: str) -> bool:
    toks = set(_WORD.findall(text.lower()))
    return bool(toks & _NEGATIONS) or "n't" in text.lower()


class LexicalStubBackend(NLIBackend):
    """Coverage + negation heuristic. Deterministic, offline, test-only."""

    @property
    def backend_id(self) -> str:
        return "stub-lexical-v1"

    @property
    def is_deterministic(self) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        hyp = _content_words(hypothesis)
        prem = _content_words(premise)
        if not hyp:
            # Nothing to support -> neutral (the gate folds this into baseless).
            return NLIResult(entailment=0.0, contradiction=0.0, neutral=1.0)

        coverage = len(hyp & prem) / len(hyp)
        neg_mismatch = _has_negation(premise) != _has_negation(hypothesis)

        if coverage >= 0.6 and neg_mismatch:
            # Same content, opposite polarity -> contradiction.
            return NLIResult(entailment=0.05, contradiction=0.85, neutral=0.10)
        if coverage >= 0.6:
            scaled = min(1.0, 0.55 + 0.45 * coverage)
            rest = 1.0 - scaled
            return NLIResult(entailment=scaled, contradiction=rest * 0.2, neutral=rest * 0.8)
        # Insufficient lexical support -> neutral (-> baseless at the gate).
        return NLIResult(
            entailment=0.15 * coverage, contradiction=0.05, neutral=1.0 - 0.15 * coverage - 0.05
        )
