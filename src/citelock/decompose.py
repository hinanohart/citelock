"""Claim decomposition: split an answer into atomic claims.

The default is a deterministic sentence segmentation (``syntok``): the same
answer always yields the same claim list, byte for byte, which is what lets
``GateResult.is_fully_deterministic`` stay True. Sentence granularity is a
deliberate, honest simplification for the alpha — a sentence with two facts is
one claim. Finer, LLM-assisted decomposition is opt-in and, by definition, not
deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import syntok.segmenter as segmenter

from .types import Claim


def _deterministic_split(answer: str) -> list[Claim]:
    claims: list[Claim] = []
    idx = 0
    for paragraph in segmenter.process(answer):
        for sentence in paragraph:
            tokens = [t for t in sentence]
            if not tokens:
                continue
            start = tokens[0].offset
            last = tokens[-1]
            end = last.offset + len(last.value)
            text = answer[start:end]
            if not text.strip():
                continue
            claims.append(
                Claim(
                    id=f"c{idx}",
                    text=text,
                    origin_span=(start, end),
                    decomposed_by="deterministic",
                    is_deterministic=True,
                )
            )
            idx += 1
    return claims


def _llm_split(answer: str, llm_fn: Callable[[str], list[str]]) -> list[Claim]:
    raw = llm_fn(answer)
    claims: list[Claim] = []
    for idx, text in enumerate(raw):
        text = text.strip()
        if not text:
            continue
        claims.append(
            Claim(
                id=f"c{idx}",
                text=text,
                origin_span=None,  # an LLM may rephrase; offsets are not reliable.
                decomposed_by="llm",
                is_deterministic=False,
            )
        )
    return claims


def decompose_claims(
    answer: str,
    *,
    method: str = "deterministic",
    llm_fn: Callable[[str], list[str]] | None = None,
) -> list[Claim]:
    """Decompose ``answer`` into claims.

    Returns an empty list for empty/whitespace input; the gate treats "no
    claims" as fail-closed (a deny), never as a vacuous allow.
    """
    if method == "deterministic":
        return _deterministic_split(answer)
    if method == "llm":
        if llm_fn is None:
            raise ValueError("method='llm' requires an llm_fn callable")
        return _llm_split(answer, llm_fn)
    raise ValueError(f"unknown decomposition method: {method!r}")
