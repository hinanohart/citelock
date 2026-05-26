"""NLI backends for citelock.

Importing this package never imports torch: ``LocalCrossEncoderBackend`` loads
torch/transformers lazily, only when a model is actually run.
"""

from .base import NLIBackend, NLIResult
from .deberta import DEFAULT_MODEL, LocalCrossEncoderBackend
from .fixture import FixtureBackend
from .llm import LLMJudgeBackend
from .stub import LexicalStubBackend

__all__ = [
    "NLIBackend",
    "NLIResult",
    "LexicalStubBackend",
    "FixtureBackend",
    "LocalCrossEncoderBackend",
    "DEFAULT_MODEL",
    "LLMJudgeBackend",
]
