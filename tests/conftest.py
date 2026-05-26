"""Shared test fixtures: controllable NLI backends for precise gate-logic tests."""

from __future__ import annotations

import pytest

from citelock.backends.base import NLIBackend, NLIResult


class _Const(NLIBackend):
    def __init__(self, result: NLIResult, backend_id: str, deterministic: bool = True):
        self._r = result
        self._id = backend_id
        self._det = deterministic

    @property
    def backend_id(self) -> str:
        return self._id

    @property
    def is_deterministic(self) -> bool:
        return self._det

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return self._r


class RaisingBackend(NLIBackend):
    @property
    def backend_id(self) -> str:
        return "raising"

    @property
    def is_deterministic(self) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        raise RuntimeError("backend boom")


@pytest.fixture
def entail_backend() -> NLIBackend:
    return _Const(NLIResult(entailment=0.92, contradiction=0.03, neutral=0.05), "always-entail")


@pytest.fixture
def contradict_backend() -> NLIBackend:
    return _Const(NLIResult(entailment=0.03, contradiction=0.92, neutral=0.05), "always-contradict")


@pytest.fixture
def neutral_backend() -> NLIBackend:
    return _Const(NLIResult(entailment=0.10, contradiction=0.05, neutral=0.85), "always-neutral")


@pytest.fixture
def nondet_entail_backend() -> NLIBackend:
    return _Const(
        NLIResult(entailment=0.92, contradiction=0.03, neutral=0.05),
        "nondet-entail",
        deterministic=False,
    )


@pytest.fixture
def raising_backend() -> NLIBackend:
    return RaisingBackend()
