"""Deterministic lexical relevance between a claim and a citation.

Used to decide whether a citation is *on-topic enough* to vote on a claim. NLI
models confidently mislabel unrelated passages (a passage about Saturn can score
0.97 "contradiction" against a claim about Paris); aggregating that across the
several noisy passages a RAG retriever returns produces large false-deny rates.
Requiring a minimum content-word overlap before a citation may entail or
contradict a claim filters out those distractors deterministically.

This is intentionally simple and language-agnostic-ish (whitespace + alnum
tokens, English stopwords). It is a *gate on participation*, not a semantic
similarity score.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
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
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "nor",
        "neither",
    }
)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def relevance(claim: str, citation: str) -> float:
    """Fraction of the claim's content words that appear in the citation.

    Returns 0.0 for a claim with no content words (which the gate then treats as
    having no relevant citation -> fail-closed deny).
    """
    claim_cw = content_words(claim)
    if not claim_cw:
        return 0.0
    return len(claim_cw & content_words(citation)) / len(claim_cw)
