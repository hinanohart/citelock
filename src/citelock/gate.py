"""The gate: turn (answer, citations) into a fail-closed allow/deny decision.

Fail-closed is the whole point. Every path that is not "all claims are entailed
by the cited passages" ends in a deny: a contradicted claim, an unsupported
claim, zero claims, too few citations, a backend that raised, a decomposition
that raised. There is no code path that turns an error into an allow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .backends.base import NLIBackend
from .decompose import decompose_claims
from .hashing import hash_obj
from .policy import DEFAULT_POLICY, GatePolicy
from .types import Citation, Claim, ClaimVerdict, Decision, GateResult, PairEvidence, Verdict


def _citelock_version() -> str:
    try:
        from importlib.metadata import version

        return version("citelock")
    except Exception:  # pragma: no cover - only before install
        return "0.0.0+unknown"


def _coerce_citations(citations: Iterable[Any]) -> list[Citation]:
    out: list[Citation] = []
    for i, c in enumerate(citations):
        if isinstance(c, Citation):
            out.append(c)
        elif isinstance(c, dict):
            out.append(Citation(**c))
        elif isinstance(c, str):
            out.append(Citation(id=f"cit{i}", text=c))
        else:
            raise TypeError(f"citation {i} must be Citation|dict|str, got {type(c).__name__}")
    return out


def _classify_claim(
    claim: Claim,
    citations: Sequence[Citation],
    backend: NLIBackend,
    policy: GatePolicy,
) -> ClaimVerdict:
    if not citations:
        # No passages to support the claim -> baseless -> deny.
        return ClaimVerdict(
            claim_id=claim.id,
            claim_text=claim.text,
            verdict="baseless",
            rule_applied="R-C",
            best_citation_id=None,
            max_entailment=0.0,
            max_contradiction=0.0,
        )
    try:
        results = backend.classify_batch([(c.text, claim.text) for c in citations])
    except Exception as e:  # fail-closed: a backend error denies, never allows.
        return ClaimVerdict(
            claim_id=claim.id,
            claim_text=claim.text,
            verdict="baseless",
            rule_applied="fail-closed:backend-error",
            best_citation_id=None,
            max_entailment=0.0,
            max_contradiction=0.0,
            error=f"{type(e).__name__}: {e}",
        )

    evidence: list[PairEvidence] = []
    max_e = -1.0
    max_c = -1.0
    best_e_cid: str | None = None
    best_c_cid: str | None = None
    for cit, res in zip(citations, results, strict=True):
        evidence.append(
            PairEvidence(
                citation_id=cit.id,
                entailment=res.entailment,
                contradiction=res.contradiction,
                neutral=res.neutral,
            )
        )
        if res.entailment > max_e:
            max_e = res.entailment
            best_e_cid = cit.id
        if res.contradiction > max_c:
            max_c = res.contradiction
            best_c_cid = cit.id

    # R-A (contradiction) is checked first: contradiction always wins over
    # entailment, so a claim that some passage contradicts can never be allowed.
    verdict: Verdict
    if max_c >= policy.tau_contra:
        verdict, rule, best = "contradicted", "R-A", best_c_cid
    elif max_e >= policy.tau_entail:
        verdict, rule, best = "entailed", "R-B", best_e_cid
    else:
        verdict, rule, best = "baseless", "R-C", best_e_cid

    return ClaimVerdict(
        claim_id=claim.id,
        claim_text=claim.text,
        verdict=verdict,
        rule_applied=rule,
        best_citation_id=best,
        max_entailment=max(max_e, 0.0),
        max_contradiction=max(max_c, 0.0),
        evidence=tuple(evidence),
    )


def _result(
    *,
    decision: Decision,
    verdicts: Sequence[ClaimVerdict],
    backend_id: str,
    is_det: bool,
    policy: GatePolicy,
    input_digest: str,
    reason: str,
) -> GateResult:
    n_e = sum(1 for v in verdicts if v.verdict == "entailed")
    n_c = sum(1 for v in verdicts if v.verdict == "contradicted")
    n_b = sum(1 for v in verdicts if v.verdict == "baseless")
    return GateResult(
        decision=decision,
        n_claims=len(verdicts),
        n_entailed=n_e,
        n_contradicted=n_c,
        n_baseless=n_b,
        claim_verdicts=tuple(verdicts),
        backend_id=backend_id,
        is_fully_deterministic=is_det,
        policy_id=policy.policy_id(),
        input_digest=input_digest,
        citelock_version=_citelock_version(),
        reason=reason,
    )


def gate(
    answer: str,
    citations: Iterable[Any],
    *,
    backend: NLIBackend,
    policy: GatePolicy = DEFAULT_POLICY,
    decompose: str = "deterministic",
    llm_fn: Callable[[str], list[str]] | None = None,
    ledger: Any | None = None,
) -> GateResult:
    """Gate ``answer`` against ``citations``. Returns a fail-closed GateResult.

    ``citations`` accepts ``Citation`` objects, dicts (``{"id","text"}``) or bare
    strings. If ``ledger`` is given (a ``JsonlLedger``), every claim verdict and
    a gate summary are appended to it.
    """
    cits = _coerce_citations(citations)
    input_digest = hash_obj(
        {
            "answer": answer,
            "citations": [c.model_dump() for c in cits],
            "policy": policy.model_dump(),
            "backend": backend.backend_id,
        }
    )
    decompose_is_det = decompose == "deterministic"
    is_det = backend.is_deterministic and decompose_is_det

    if len(cits) < policy.min_citations:
        result = _result(
            decision="deny",
            verdicts=(),
            backend_id=backend.backend_id,
            is_det=is_det,
            policy=policy,
            input_digest=input_digest,
            reason=(
                f"fail-closed: {len(cits)} citation(s) provided, "
                f"policy requires at least {policy.min_citations}."
            ),
        )
        if ledger is not None:
            ledger.record(result)
        return result

    try:
        claims = decompose_claims(answer, method=decompose, llm_fn=llm_fn)
    except Exception as e:  # fail-closed: cannot decompose -> deny.
        result = _result(
            decision="deny",
            verdicts=(),
            backend_id=backend.backend_id,
            is_det=is_det,
            policy=policy,
            input_digest=input_digest,
            reason=f"fail-closed: claim decomposition failed ({type(e).__name__}: {e}).",
        )
        if ledger is not None:
            ledger.record(result)
        return result

    if not claims:
        result = _result(
            decision="deny",
            verdicts=(),
            backend_id=backend.backend_id,
            is_det=is_det,
            policy=policy,
            input_digest=input_digest,
            reason="fail-closed: no verifiable claims extracted from the answer.",
        )
        if ledger is not None:
            ledger.record(result)
        return result

    verdicts = [_classify_claim(c, cits, backend, policy) for c in claims]
    is_det = is_det and all(c.is_deterministic for c in claims)

    n_c = sum(1 for v in verdicts if v.verdict == "contradicted")
    n_b = sum(1 for v in verdicts if v.verdict == "baseless")
    all_entailed = n_c == 0 and n_b == 0
    decision: Decision = "allow" if all_entailed else "deny"
    if all_entailed:
        reason = f"allow: all {len(verdicts)} claim(s) entailed by the cited passages."
    else:
        reason = f"deny: {n_c} contradicted, {n_b} unsupported of {len(verdicts)} claim(s)."

    result = _result(
        decision=decision,
        verdicts=verdicts,
        backend_id=backend.backend_id,
        is_det=is_det,
        policy=policy,
        input_digest=input_digest,
        reason=reason,
    )
    if ledger is not None:
        ledger.record(result)
    return result


class Citelock:
    """Reusable gate bound to a backend and policy.

    >>> lock = Citelock(backend=my_backend)
    >>> lock.gate("The sky is blue.", ["The sky appears blue."]).allowed
    """

    def __init__(
        self,
        *,
        backend: NLIBackend,
        policy: GatePolicy = DEFAULT_POLICY,
        decompose: str = "deterministic",
        llm_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.decompose = decompose
        self.llm_fn = llm_fn

    def gate(
        self,
        answer: str,
        citations: Iterable[Any],
        *,
        ledger: Any | None = None,
    ) -> GateResult:
        return gate(
            answer,
            citations,
            backend=self.backend,
            policy=self.policy,
            decompose=self.decompose,
            llm_fn=self.llm_fn,
            ledger=ledger,
        )
