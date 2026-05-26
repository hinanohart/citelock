"""citelock — a deterministic, fail-closed citation gate for RAG answers.

citelock decides whether an answer is *supported by the passages it cites*. It
splits the answer into claims, asks a local NLI model whether each claim is
entailed / contradicted / unsupported by the cited passages, and **denies
(fail-closed) any answer with an unsupported or contradicted claim**, emitting a
per-claim ledger for CI and audit.

It checks textual entailment against the provided passages only — NOT real-world
truth. It is a gate, not a fact-checker, and it inherits its NLI backend's
errors. With the default local backend it uses no LLM API and no network at
inference time, and its deterministic path is reproducible.
"""

from __future__ import annotations

from .backends.base import NLIBackend, NLIResult
from .backends.stub import LexicalStubBackend
from .decompose import decompose_claims
from .gate import Citelock, gate
from .ledger import JsonlLedger
from .policy import DEFAULT_POLICY, GatePolicy
from .sanity import SanityCase, SanityReport, run_sanity
from .types import (
    Citation,
    Claim,
    ClaimVerdict,
    Decision,
    GateResult,
    PairEvidence,
    Verdict,
)

try:  # version from installed metadata; falls back when running from a checkout
    from importlib.metadata import version as _version

    __version__ = _version("citelock")
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "gate",
    "Citelock",
    "GatePolicy",
    "DEFAULT_POLICY",
    "GateResult",
    "ClaimVerdict",
    "Claim",
    "Citation",
    "PairEvidence",
    "Verdict",
    "Decision",
    "decompose_claims",
    "JsonlLedger",
    "run_sanity",
    "SanityCase",
    "SanityReport",
    "NLIBackend",
    "NLIResult",
    "LexicalStubBackend",
    "__version__",
]
