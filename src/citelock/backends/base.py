"""NLI backend abstraction.

A backend answers one question, deterministically or not: *does this premise
(a cited passage) entail, contradict, or stay neutral toward this hypothesis
(a claim)?* The gate depends only on this interface, never on a concrete model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class NLIResult:
    """Three-way NLI probabilities for a (premise, hypothesis) pair.

    The three values are expected to be non-negative and sum to ~1.0; backends
    should apply softmax before returning. They are not re-normalised here so a
    backend bug (e.g. scores that do not sum to 1) stays visible to callers.
    """

    entailment: float
    contradiction: float
    neutral: float


class NLIBackend(ABC):
    """Abstract NLI backend."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable identifier recorded in every GateResult and ledger entry."""

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """True iff repeated calls with identical inputs give identical outputs.

        A local model run in eval mode with no sampling is deterministic; an LLM
        judge reached over an API is not. The gate propagates this into
        ``GateResult.is_fully_deterministic``.
        """

    @abstractmethod
    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify a single (premise, hypothesis) pair."""

    def classify_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Classify many pairs. Override for true batching; default loops."""
        return [self.classify(p, h) for p, h in pairs]
