"""Judge-sanity harness: does the gate actually read the citations?

A citation gate is only meaningful if removing the supporting passage changes
the verdict. We perturb inputs and measure how often the verdict responds the
way it must:

  * drop    — remove the best supporting citation for an entailed claim.
              The claim should STOP being entailed. A low drop-flip rate means
              the NLI model is ignoring the passages (concept failure).
  * negate  — negate the claim. It should STOP being entailed.
  * shuffle — reorder the citations. The verdict MUST be unchanged; the gate
              aggregates with a max, so order independence is a logic property,
              not a heuristic. Anything below 1.0 is a bug.

Rates come with bootstrap confidence intervals (fixed seed -> reproducible). We
report; we never print "calibrated".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from .backends.base import NLIBackend
from .gate import _classify_claim, _coerce_citations
from .policy import DEFAULT_POLICY, GatePolicy
from .types import Claim


@dataclass(frozen=True)
class SanityCase:
    answer: str
    citations: Sequence[Any]


@dataclass(frozen=True)
class SanityReport:
    n_entailed_claims: int
    drop_flip_rate: float
    drop_flip_ci: tuple[float, float]
    negate_flip_rate: float
    negate_flip_ci: tuple[float, float]
    shuffle_invariance_rate: float
    passed: bool
    notes: tuple[str, ...]


def _bootstrap_ci(
    indicators: list[int], rng: Random, n: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    if not indicators:
        return (0.0, 0.0)
    k = len(indicators)
    samples = []
    for _ in range(n):
        resample = [indicators[rng.randrange(k)] for _ in range(k)]
        samples.append(sum(resample) / k)
    samples.sort()
    lo = samples[int((alpha / 2) * n)]
    hi = samples[min(n - 1, int((1 - alpha / 2) * n))]
    return (round(lo, 4), round(hi, 4))


def _negate(text: str) -> str:
    return f"It is not the case that {text[0].lower()}{text[1:]}" if text else text


def run_sanity(
    cases: Sequence[SanityCase],
    *,
    backend: NLIBackend,
    policy: GatePolicy = DEFAULT_POLICY,
    n_bootstrap: int = 1000,
    seed: int = 0,
    min_drop_flip: float = 0.5,
    min_negate_flip: float = 0.5,
) -> SanityReport:
    rng = Random(seed)
    drop_flips: list[int] = []
    negate_flips: list[int] = []
    shuffle_invariant: list[int] = []
    notes: list[str] = []

    from .decompose import decompose_claims

    for case in cases:
        cits = _coerce_citations(case.citations)
        claims = decompose_claims(case.answer)
        for claim in claims:
            base = _classify_claim(claim, cits, backend, policy)
            if base.verdict != "entailed":
                continue  # drop/negate are defined on the entailed population.

            # drop: remove the best supporting citation.
            kept = [c for c in cits if c.id != base.best_citation_id]
            dropped = _classify_claim(claim, kept, backend, policy)
            drop_flips.append(1 if dropped.verdict != "entailed" else 0)

            # negate: the negated claim should no longer be entailed.
            neg_claim = Claim(
                id=claim.id + "-neg",
                text=_negate(claim.text),
                origin_span=None,
                decomposed_by=claim.decomposed_by,
                is_deterministic=claim.is_deterministic,
            )
            negated = _classify_claim(neg_claim, cits, backend, policy)
            negate_flips.append(1 if negated.verdict != "entailed" else 0)

            # shuffle: verdict must be invariant to citation order.
            shuffled = list(cits)
            rng.shuffle(shuffled)
            reshuf = _classify_claim(claim, shuffled, backend, policy)
            shuffle_invariant.append(1 if reshuf.verdict == base.verdict else 0)

    n = len(drop_flips)
    drop_rate = sum(drop_flips) / n if n else 0.0
    negate_rate = sum(negate_flips) / n if n else 0.0
    shuffle_rate = sum(shuffle_invariant) / len(shuffle_invariant) if shuffle_invariant else 1.0

    if n == 0:
        notes.append("no entailed claims in the provided cases; drop/negate undefined.")
    if shuffle_rate < 1.0:
        notes.append(
            "BUG: gate verdict changed under citation reordering (should be order-invariant)."
        )
    if n and drop_rate < min_drop_flip:
        notes.append(
            f"WARNING: low drop-flip rate ({drop_rate:.2f}) — the backend may not "
            "be reading the citations."
        )

    passed = (
        n > 0
        and shuffle_rate == 1.0
        and drop_rate >= min_drop_flip
        and negate_rate >= min_negate_flip
    )

    return SanityReport(
        n_entailed_claims=n,
        drop_flip_rate=round(drop_rate, 4),
        drop_flip_ci=_bootstrap_ci(drop_flips, rng, n_bootstrap),
        negate_flip_rate=round(negate_rate, 4),
        negate_flip_ci=_bootstrap_ci(negate_flips, rng, n_bootstrap),
        shuffle_invariance_rate=round(shuffle_rate, 4),
        passed=passed,
        notes=tuple(notes),
    )
