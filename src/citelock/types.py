"""Typed intermediate representation for citelock.

Every value that crosses the public API boundary is a pydantic model so that
untrusted RAG output is validated once, at the edge, and the ledger has a single
canonical serialization. The gate never trusts a dict it did not build itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A claim is judged against the *cited passages only*. "baseless" means the
# passages neither entail nor contradict it (NLI-neutral is folded into
# baseless: an unsupported claim is denied, never silently allowed).
Verdict = Literal["entailed", "contradicted", "baseless"]

# The gate is binary and fail-closed: anything that is not "all claims entailed"
# is a deny.
Decision = Literal["allow", "deny"]

DecomposedBy = Literal["deterministic", "llm", "manual"]


class Citation(BaseModel):
    """A passage the answer claims to be supported by."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str


class Claim(BaseModel):
    """One atomic statement extracted from the answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str
    # (start, end) character offsets into the original answer, when known.
    origin_span: tuple[int, int] | None = None
    decomposed_by: DecomposedBy
    is_deterministic: bool


class PairEvidence(BaseModel):
    """NLI scores for one (claim, citation) pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_id: str
    entailment: float
    contradiction: float
    neutral: float


class ClaimVerdict(BaseModel):
    """The gate's decision for a single claim, with the evidence that drove it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    claim_text: str
    verdict: Verdict
    # Which fail-closed rule produced this verdict: "R-A" (contradiction),
    # "R-B" (entailment), "R-C" (baseless), or "fail-closed:<cause>".
    rule_applied: str
    best_citation_id: str | None
    max_entailment: float
    max_contradiction: float
    evidence: tuple[PairEvidence, ...] = Field(default_factory=tuple)
    error: str | None = None


class GateResult(BaseModel):
    """The outcome of gating one (answer, citations) pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Decision
    n_claims: int
    n_entailed: int
    n_contradicted: int
    n_baseless: int
    claim_verdicts: tuple[ClaimVerdict, ...]
    backend_id: str
    # True only if BOTH the backend and every claim decomposition were
    # deterministic. A single LLM-decomposed claim or an LLM-judge backend
    # flips this to False — citelock never claims reproducibility it cannot keep.
    is_fully_deterministic: bool
    policy_id: str
    # blake3 digest of the canonical (answer, citations, policy, backend) input.
    input_digest: str
    citelock_version: str
    # Human-readable explanation of the allow/deny.
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"
