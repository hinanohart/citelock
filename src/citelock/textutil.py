"""Deterministic lexical relevance between a claim and a citation.

Used to decide whether a citation is *on-topic enough* to vote on a claim. NLI
models confidently mislabel unrelated passages (a passage about Saturn can score
0.97 "contradiction" against a claim about Paris); aggregating that across the
several noisy passages a RAG retriever returns produces large false-deny rates.
Requiring a minimum content-word overlap before a citation may entail or
contradict a claim filters out those distractors deterministically.

It is a *gate on participation*, not a semantic similarity score, and it has two
honest limitations callers must know about:

1. The filter suppresses a citation's *contradiction* vote as well as its
   entailment vote. A genuine contradiction phrased with little shared
   vocabulary is therefore dropped too, which can let a wrong answer through
   (see README "Limitations" / the contradiction-recall eval). The filter trades
   contradiction recall for a large drop in distractor-driven false-denies.
2. Tokenization is Unicode word characters split on non-word boundaries. This
   works for whitespace-delimited scripts (Latin incl. accents, Cyrillic, Greek,
   …). Scripts written without spaces between words (CJK, Thai, …) are *not*
   segmented — a run of such characters becomes one token — so lexical overlap
   is unreliable there and you likely need ``min_relevance=0`` plus a backend
   you trust, or a custom tokenizer. Stopwords are English-only.
"""

from __future__ import annotations

import re

# Unicode word characters (so non-ASCII, whitespace-delimited scripts are not
# silently reduced to the empty set, which would deny every answer). NOTE: this
# does not segment scripts without word spacing (CJK/Thai); see module docstring.
_WORD = re.compile(r"\w+", re.UNICODE)
# English stopwords, including negations. Negations are dropped here on purpose:
# relevance is about *topic* overlap, and "X happened" vs "X did not happen"
# share a topic and should both be allowed to vote (the NLI backend, not this
# filter, decides entail vs contradict). backends/stub.py keeps its OWN, smaller
# stopword set that deliberately *excludes* negations because it does cheap
# negation detection; the two sets are intentionally not shared.
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
