"""citelock evaluation harness.

Reports REAL, reproducible numbers — never placeholders. It evaluates any
labeled set passed via --cases (JSON list of {answer, citations, gold}); the
shipped sets are eval/labeled_cases.json (clear-cut) and eval/distractor_cases.json
(noisy retrieval). This is what the README table reports.

Public benchmarks (RAGTruth, ExpertQA; both MIT) are NOT redistributed here and
there is no built-in loader for them yet: convert them to the --cases JSON format
yourself to evaluate on them. We do not publish numbers we have not run and
stamped.

Every run prints an environment stamp (hardware/OS/Python/date/seed/backend) and
bootstrap confidence intervals on the headline rates.

Usage:
    python eval/run_eval.py --backend stub
    python eval/run_eval.py --backend local                              # real NLI model
    python eval/run_eval.py --backend local --cases eval/distractor_cases.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import date
from pathlib import Path

from citelock import GatePolicy, gate
from citelock.backends.base import NLIBackend

HERE = Path(__file__).parent


def _make_backend(name: str, model: str | None) -> NLIBackend:
    if name == "stub":
        from citelock.backends.stub import LexicalStubBackend

        return LexicalStubBackend()
    if name == "local":
        from citelock.backends.deberta import DEFAULT_MODEL, LocalCrossEncoderBackend

        return LocalCrossEncoderBackend(model_name=model or DEFAULT_MODEL)
    raise ValueError(f"unknown backend {name!r}")


def _bootstrap_ci(values: list[int], rng: random.Random, n: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    k = len(values)
    samples = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(samples[int(0.025 * n)], 4), round(samples[min(n - 1, int(0.975 * n))], 4)]


def evaluate(cases: list[dict], backend: NLIBackend, seed: int, min_relevance: float) -> dict:
    rng = random.Random(seed)
    policy = GatePolicy(min_relevance=min_relevance)
    # Confusion matrix on the DENY class (deny is the "positive"/safety class).
    tp = fp = tn = fn = 0
    correct: list[int] = []
    false_deny: list[int] = []  # gold allow but predicted deny
    for c in cases:
        result = gate(c["answer"], c["citations"], backend=backend, policy=policy)
        pred = result.decision
        gold = c["gold"]
        correct.append(1 if pred == gold else 0)
        if gold == "allow":
            false_deny.append(1 if pred == "deny" else 0)
        if pred == "deny" and gold == "deny":
            tp += 1
        elif pred == "deny" and gold == "allow":
            fp += 1
        elif pred == "allow" and gold == "allow":
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(correct) / len(correct) if correct else 0.0
    fd_rate = sum(false_deny) / len(false_deny) if false_deny else 0.0

    return {
        "n_cases": len(cases),
        "deny_precision": round(precision, 4),
        "deny_recall": round(recall, 4),
        "deny_f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "accuracy_ci95": _bootstrap_ci(correct, rng),
        "false_deny_rate": round(fd_rate, 4),
        "false_deny_rate_ci95": _bootstrap_ci(false_deny, rng),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def env_stamp(backend: NLIBackend, seed: int) -> dict:
    return {
        "date": date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "seed": seed,
        "backend_id": backend.backend_id,
        "backend_deterministic": backend.is_deterministic,
        "citelock_version": __import__("citelock").__version__,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="citelock evaluation harness")
    p.add_argument("--backend", default="stub", choices=["stub", "local"])
    p.add_argument("--model", default=None)
    p.add_argument("--cases", default=str(HERE / "labeled_cases.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-relevance", type=float, default=0.2)
    p.add_argument("--json", default=None, help="also write the report JSON here")
    args = p.parse_args(argv)

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    backend = _make_backend(args.backend, args.model)
    stamp = env_stamp(backend, args.seed)
    stamp["min_relevance"] = args.min_relevance
    report = {"env": stamp, "metrics": evaluate(cases, backend, args.seed, args.min_relevance)}

    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
