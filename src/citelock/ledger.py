"""Per-claim, append-only, tamper-evident ledger.

Each ``gate()`` call appends one entry per claim verdict plus a gate-summary
entry. Entries are chained: ``entry_hash = H(prev_hash || entry_core)``, so any
edit to a past entry breaks every later link and ``verify()`` catches it.

Honest scope: this is *single-writer* tamper-evidence. It detects after-the-fact
edits to the file; it is not a distributed log, not multi-writer safe, and uses
no ``fsync`` (best-effort durability). It does not prove *when* an entry was
made beyond the recorded timestamp.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ulid import ULID

from .hashing import HASH_ALGO, hash_chain
from .types import GateResult

GENESIS = "0" * 64


class JsonlLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["entry_hash"]
        return last

    def _append(self, payload: dict[str, Any], prev: str) -> str:
        uid = str(ULID())
        ts = datetime.now(timezone.utc).isoformat()
        core = {"ulid": uid, "ts": ts, "payload": payload}
        entry_hash = hash_chain(prev, core)
        entry = {
            "ulid": uid,
            "ts": ts,
            "hash_algo": HASH_ALGO,
            "prev_hash": prev,
            "entry_hash": entry_hash,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:  # best-effort, no fsync
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry_hash

    def record(self, result: GateResult) -> str:
        """Append all claim verdicts + a gate summary. Returns the new head hash."""
        prev = self._last_hash()
        for v in result.claim_verdicts:
            payload = {"kind": "claim", "input_digest": result.input_digest}
            payload.update(v.model_dump())
            prev = self._append(payload, prev)
        prev = self._append(
            {
                "kind": "gate",
                "decision": result.decision,
                "n_claims": result.n_claims,
                "n_entailed": result.n_entailed,
                "n_contradicted": result.n_contradicted,
                "n_baseless": result.n_baseless,
                "backend_id": result.backend_id,
                "is_fully_deterministic": result.is_fully_deterministic,
                "policy_id": result.policy_id,
                "input_digest": result.input_digest,
                "citelock_version": result.citelock_version,
                "reason": result.reason,
            },
            prev,
        )
        return prev

    def verify(self) -> tuple[bool, str | None]:
        """Re-walk the chain. Returns (ok, reason_if_broken)."""
        if not self.path.exists():
            return True, None
        prev = GENESIS
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as e:
                    return False, f"line {lineno}: invalid JSON ({e})"
                if entry.get("hash_algo") != HASH_ALGO:
                    return False, (
                        f"line {lineno}: hash algo mismatch "
                        f"(ledger={entry.get('hash_algo')}, this build={HASH_ALGO})"
                    )
                if entry.get("prev_hash") != prev:
                    return False, f"line {lineno}: broken chain link (prev_hash mismatch)"
                core = {"ulid": entry["ulid"], "ts": entry["ts"], "payload": entry["payload"]}
                if hash_chain(prev, core) != entry["entry_hash"]:
                    return False, f"line {lineno}: entry hash mismatch (tampered)"
                prev = entry["entry_hash"]
        return True, None

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
