"""Local cross-encoder NLI backend (the ``[nli]`` extra).

Default model: ``cross-encoder/nli-deberta-v3-base`` (Apache-2.0, trained on
SNLI + MultiNLI). Runs on CPU; no GPU required. The model is loaded lazily so
importing citelock never pulls in torch.

Determinism: inference is a single forward pass in ``eval`` mode under
``torch.no_grad()`` with no sampling, so it is deterministic for a fixed model
and hardware. Scores may differ in their last digits across hardware/dtype;
because the gate compares against thresholds this almost never flips a verdict,
but we do not claim bit-identical scores across machines.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .base import NLIBackend, NLIResult

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

# Models whose weights or training data carry usage restrictions. citelock never
# selects these by default; if a user names one, we warn before loading.
_RESTRICTED = {
    "moritzlaurer/deberta-v3-base-mnli-fever-anli": (
        "weights are MIT but training data includes ANLI (CC-BY-NC-4.0); "
        "verify it is compatible with your deployment"
    ),
    "lytang/minicheck-deberta-v3-large": "CC-BY-NC-4.0 — non-commercial use only",
}


class LocalCrossEncoderBackend(NLIBackend):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._max_length = max_length
        self._tok: Any = None
        self._model: Any = None
        self._label_index: dict[str, int] | None = None
        self._warn_if_restricted(model_name)

    @staticmethod
    def _warn_if_restricted(name: str) -> None:
        for key, msg in _RESTRICTED.items():
            if key in name.lower():
                warnings.warn(
                    f"NLI model {name!r}: {msg}.",
                    stacklevel=3,
                )

    @property
    def backend_id(self) -> str:
        return f"local:{self.model_name}"

    @property
    def is_deterministic(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover - depends on extra
            raise ImportError(
                "LocalCrossEncoderBackend needs the 'nli' extra: pip install 'citelock[nli]'"
            ) from e

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        if self._device:
            model.to(self._device)
        self._model = model
        # Map label name -> logit index from the model config (do not hardcode).
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        self._label_index = {v.lower(): k for k, v in id2label.items()}

    def classify_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        if not pairs:
            return []
        import torch

        self._ensure_loaded()
        assert self._tok is not None and self._model is not None
        assert self._label_index is not None
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        enc = self._tok(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        if self._device:
            enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        ei = self._label_index.get("entailment")
        ci = self._label_index.get("contradiction")
        ni = self._label_index.get("neutral")
        out: list[NLIResult] = []
        for row in probs:
            out.append(
                NLIResult(
                    entailment=float(row[ei]) if ei is not None else 0.0,
                    contradiction=float(row[ci]) if ci is not None else 0.0,
                    neutral=float(row[ni]) if ni is not None else 0.0,
                )
            )
        return out

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return self.classify_batch([(premise, hypothesis)])[0]

    def record_fixture(self, pairs: list[tuple[str, str]], path: str | Path) -> None:
        """Run ``pairs`` and save scores to a JSON fixture for offline replay.

        Used to make the test suite reproducible without re-downloading the
        model or hitting the network (see backends/fixture.py).
        """
        results = self.classify_batch(pairs)
        records = [
            {
                "premise": p,
                "hypothesis": h,
                "entailment": r.entailment,
                "contradiction": r.contradiction,
                "neutral": r.neutral,
            }
            for (p, h), r in zip(pairs, results, strict=True)
        ]
        Path(path).write_text(
            json.dumps({"backend_id": self.backend_id, "records": records}, indent=2),
            encoding="utf-8",
        )
