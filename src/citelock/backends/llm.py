"""LLM-judge NLI backend — for comparison and evaluation only.

This backend exists so you can benchmark a non-deterministic LLM judge against
the deterministic NLI backend on the same data. It is NOT recommended for
production gating: it is non-deterministic, costs money/latency, and needs
network. citelock bundles no SDK; you inject a ``completion_fn(prompt) -> str``
(wrap OpenAI, Anthropic, a local server — your choice). ``is_deterministic``
defaults to False; set it True only if you have genuinely pinned the decode
(temperature 0 and a fixed model), and even then the gate records the backend so
the choice is auditable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from .base import NLIBackend, NLIResult

_PROMPT = """\
You are a strict natural-language-inference labeler. Given a PREMISE and a
HYPOTHESIS, decide whether the premise ENTAILS, CONTRADICTS, or is NEUTRAL
toward the hypothesis. Judge only from the premise text, not outside knowledge.
Reply with ONLY a JSON object:
{{"label": "entailment|contradiction|neutral", "confidence": 0.0-1.0}}

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> tuple[str, float]:
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"LLM judge returned no JSON object: {raw!r}")
    obj = json.loads(m.group(0))
    label = str(obj["label"]).strip().lower()
    if label not in {"entailment", "contradiction", "neutral"}:
        raise ValueError(f"LLM judge returned unknown label: {label!r}")
    conf = float(obj.get("confidence", 1.0))
    conf = min(1.0, max(0.0, conf))
    return label, conf


class LLMJudgeBackend(NLIBackend):
    def __init__(
        self,
        completion_fn: Callable[[str], str],
        *,
        model_label: str = "unknown",
        is_deterministic: bool = False,
    ) -> None:
        self._fn = completion_fn
        self._label = model_label
        self._is_det = is_deterministic

    @property
    def backend_id(self) -> str:
        return f"llm-judge:{self._label}"

    @property
    def is_deterministic(self) -> bool:
        return self._is_det

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        raw = self._fn(_PROMPT.format(premise=premise, hypothesis=hypothesis))
        label, conf = _parse(raw)  # parse failure -> raises -> gate fail-closes.
        rest = (1.0 - conf) / 2.0
        if label == "entailment":
            return NLIResult(entailment=conf, contradiction=rest, neutral=rest)
        if label == "contradiction":
            return NLIResult(entailment=rest, contradiction=conf, neutral=rest)
        return NLIResult(entailment=rest, contradiction=rest, neutral=conf)
