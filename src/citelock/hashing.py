"""Canonical hashing for the input digest and the ledger hash-chain.

Prefers BLAKE3; falls back to stdlib BLAKE2b if the ``blake3`` wheel is not
installed. ``HASH_ALGO`` records which was actually used so the ledger is
self-describing and the README never over-claims the algorithm.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from blake3 import blake3 as _blake3

    HASH_ALGO = "blake3"

    def _hash_bytes(data: bytes) -> str:
        return _blake3(data).hexdigest()

except ImportError:  # pragma: no cover - exercised only without the blake3 wheel
    import hashlib

    HASH_ALGO = "blake2b"

    def _hash_bytes(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=32).hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 serialization (sorted keys, no insignificant space)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def hash_obj(obj: Any) -> str:
    return _hash_bytes(canonical_bytes(obj))


def hash_chain(prev_hash: str, payload: Any) -> str:
    """Hash of ``prev_hash`` bound to ``payload`` — the chain link."""
    return _hash_bytes(prev_hash.encode("utf-8") + b"\n" + canonical_bytes(payload))
