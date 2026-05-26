"""citelock command-line interface.

Exit codes are the contract that makes citelock a CI gate:
    0  gate allowed (or ledger verified, or command succeeded)
    2  gate DENIED (or ledger tampered) — the "block the build" signal
    3  usage / runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .backends.base import NLIBackend
from .gate import gate
from .ledger import JsonlLedger
from .policy import GatePolicy

EXIT_OK = 0
EXIT_DENY = 2
EXIT_ERROR = 3

_BACKEND_LICENSES = [
    ("stub", "stub-lexical-v1", "n/a (built-in, test/demo only — do not gate with it)"),
    ("local", "cross-encoder/nli-deberta-v3-base", "Apache-2.0 (default, recommended)"),
    (
        "local --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "MoritzLaurer/...mnli-fever-anli",
        "MIT weights; training data includes ANLI (CC-BY-NC-4.0) — verify for your use",
    ),
    (
        "local --model lytang/MiniCheck-DeBERTa-v3-Large",
        "lytang/MiniCheck-DeBERTa-v3-Large",
        "CC-BY-NC-4.0 — NON-COMMERCIAL only",
    ),
    ("llm", "(injected, not via CLI)", "your provider's terms; non-deterministic, not for gating"),
]


def _read_text(value: str) -> str:
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return f.read()
    return value


def _load_citations(args: argparse.Namespace) -> list[dict[str, Any] | str]:
    cits: list[dict[str, Any] | str] = []
    if args.citations_file:
        with open(args.citations_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--citations-file must contain a JSON list")
        cits.extend(data)
    for c in args.citation or []:
        cits.append(c)
    return cits


def _make_backend(args: argparse.Namespace) -> NLIBackend:
    if args.backend == "stub":
        from .backends.stub import LexicalStubBackend

        return LexicalStubBackend()
    if args.backend == "local":
        from .backends.deberta import DEFAULT_MODEL, LocalCrossEncoderBackend

        return LocalCrossEncoderBackend(model_name=args.model or DEFAULT_MODEL)
    if args.backend == "fixture":
        from .backends.fixture import FixtureBackend

        if not args.fixture:
            raise ValueError("--backend fixture requires --fixture PATH")
        return FixtureBackend.from_file(args.fixture)
    raise ValueError(f"unknown backend: {args.backend}")


def _cmd_gate(args: argparse.Namespace) -> int:
    answer = _read_text(args.answer)
    citations = _load_citations(args)
    backend = _make_backend(args)
    policy = GatePolicy(
        tau_entail=args.tau_entail,
        tau_contra=args.tau_contra,
        min_citations=args.min_citations,
        min_relevance=args.min_relevance,
    )
    ledger = JsonlLedger(args.ledger) if args.ledger else None
    result = gate(
        answer,
        citations,
        backend=backend,
        policy=policy,
        decompose=args.decompose,
        ledger=ledger,
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"decision: {result.decision.upper()}  ({result.reason})")
        print(
            f"claims: {result.n_claims}  entailed={result.n_entailed} "
            f"contradicted={result.n_contradicted} unsupported={result.n_baseless}"
        )
        print(f"backend: {result.backend_id}  deterministic={result.is_fully_deterministic}")
        print(f"policy: {result.policy_id}  input_digest: {result.input_digest[:16]}…")
        for v in result.claim_verdicts:
            mark = {"entailed": "✓", "contradicted": "✗", "baseless": "?"}[v.verdict]
            print(f"  [{mark}] {v.verdict:13s} ({v.rule_applied}) {v.claim_text}")
    return EXIT_OK if result.allowed else EXIT_DENY


def _cmd_verify_ledger(args: argparse.Namespace) -> int:
    ledger = JsonlLedger(args.ledger)
    ok, reason = ledger.verify()
    if ok:
        print(f"ledger OK: {len(ledger.entries())} entries, chain intact.")
        return EXIT_OK
    print(f"ledger TAMPERED: {reason}", file=sys.stderr)
    return EXIT_DENY


def _cmd_sanity(args: argparse.Namespace) -> int:
    from .sanity import SanityCase, run_sanity

    with open(args.cases_file, encoding="utf-8") as f:
        raw = json.load(f)
    cases = [SanityCase(answer=c["answer"], citations=c["citations"]) for c in raw]
    backend = _make_backend(args)
    report = run_sanity(cases, backend=backend, seed=args.seed)
    print(
        json.dumps(
            {
                "n_entailed_claims": report.n_entailed_claims,
                "drop_flip_rate": report.drop_flip_rate,
                "drop_flip_ci": report.drop_flip_ci,
                "negate_flip_rate": report.negate_flip_rate,
                "negate_flip_ci": report.negate_flip_ci,
                "shuffle_invariance_rate": report.shuffle_invariance_rate,
                "passed": report.passed,
                "notes": list(report.notes),
            },
            indent=2,
        )
    )
    return EXIT_OK if report.passed else EXIT_DENY


def _cmd_backends(_args: argparse.Namespace) -> int:
    print("citelock NLI backends and licenses:\n")
    for selector, model, lic in _BACKEND_LICENSES:
        print(f"  {selector}")
        print(f"      model:   {model}")
        print(f"      license: {lic}\n")
    print(
        "Default is the Apache-2.0 local model. Opt-in models with restrictive "
        "training data print a warning before loading."
    )
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citelock", description=__doc__)
    p.add_argument("--version", action="version", version=f"citelock {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_backend_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--backend", default="stub", choices=["stub", "local", "fixture"])
        sp.add_argument("--model", default=None, help="model name for --backend local")
        sp.add_argument("--fixture", default=None, help="fixture path for --backend fixture")

    g = sub.add_parser("gate", help="gate an answer against its citations")
    g.add_argument("--answer", required=True, help="answer text, or @path to a file")
    g.add_argument("--citation", action="append", help="a citation passage (repeatable)")
    g.add_argument("--citations-file", default=None, help="JSON list of citations")
    add_backend_args(g)
    g.add_argument("--tau-entail", type=float, default=0.5)
    g.add_argument("--tau-contra", type=float, default=0.5)
    g.add_argument("--min-citations", type=int, default=1)
    g.add_argument(
        "--min-relevance",
        type=float,
        default=0.2,
        help="min content-word overlap for a citation to vote (0 disables filtering)",
    )
    g.add_argument("--decompose", default="deterministic", choices=["deterministic"])
    g.add_argument("--ledger", default=None, help="append a per-claim ledger here")
    g.add_argument("--json", action="store_true", help="emit the full GateResult as JSON")
    g.set_defaults(func=_cmd_gate)

    v = sub.add_parser("verify-ledger", help="verify a ledger's hash-chain")
    v.add_argument("--ledger", required=True)
    v.set_defaults(func=_cmd_verify_ledger)

    s = sub.add_parser("sanity", help="run the judge-sanity harness")
    s.add_argument("--cases-file", required=True, help="JSON list of {answer, citations}")
    add_backend_args(s)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_sanity)

    b = sub.add_parser("backends", help="list backends and their licenses")
    b.set_defaults(func=_cmd_backends)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as e:
        print(f"error: file not found: {e}", file=sys.stderr)
        return EXIT_ERROR
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
