"""Hashing canonicalization and policy id stability."""

from __future__ import annotations

from citelock import GatePolicy
from citelock.hashing import HASH_ALGO, canonical_bytes, hash_chain, hash_obj


def test_canonical_is_key_order_independent():
    assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})


def test_hash_obj_stable_and_sensitive():
    assert hash_obj({"x": 1}) == hash_obj({"x": 1})
    assert hash_obj({"x": 1}) != hash_obj({"x": 2})


def test_hash_chain_depends_on_prev():
    a = hash_chain("0" * 64, {"p": 1})
    b = hash_chain("1" * 64, {"p": 1})
    assert a != b


def test_hash_algo_known():
    assert HASH_ALGO in {"blake3", "blake2b"}


def test_policy_id_stable():
    assert GatePolicy().policy_id() == GatePolicy().policy_id()


def test_policy_id_changes_with_threshold():
    assert GatePolicy(tau_entail=0.5).policy_id() != GatePolicy(tau_entail=0.7).policy_id()


def test_policy_rejects_out_of_range():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GatePolicy(tau_entail=1.5)
