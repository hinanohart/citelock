"""Replay backend: serve pre-recorded NLI scores from a fixture.

Lets the gate logic be tested and the eval-of-logic be run fully offline, with
no model download and no network, against realistic scores captured once from a
real backend via ``LocalCrossEncoderBackend.record_fixture``. A pair that is not
in the fixture raises ``KeyError`` — which the gate turns into a fail-closed
deny, never a silent allow.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import NLIBackend, NLIResult


class FixtureBackend(NLIBackend):
    def __init__(
        self,
        records: dict[tuple[str, str], NLIResult],
        *,
        backend_id: str = "fixture",
        is_deterministic: bool = True,
    ) -> None:
        self._records = records
        self._backend_id = backend_id
        self._is_det = is_deterministic

    @classmethod
    def from_file(cls, path: str | Path) -> FixtureBackend:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = {
            (r["premise"], r["hypothesis"]): NLIResult(
                entailment=r["entailment"],
                contradiction=r["contradiction"],
                neutral=r["neutral"],
            )
            for r in data["records"]
        }
        return cls(records, backend_id=f"fixture:{data.get('backend_id', 'unknown')}")

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def is_deterministic(self) -> bool:
        return self._is_det

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        key = (premise, hypothesis)
        if key not in self._records:
            raise KeyError(f"no fixture entry for pair {key!r}")
        return self._records[key]
