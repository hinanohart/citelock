"""Ledger: per-claim recording, chain verification, tamper detection."""

from __future__ import annotations

import json

from citelock import JsonlLedger, gate


def test_records_per_claim_plus_summary(tmp_path, entail_backend):
    path = tmp_path / "run.jsonl"
    ledger = JsonlLedger(path)
    gate("The sky is blue. Grass is green.", ["a", "b"], backend=entail_backend, ledger=ledger)
    entries = ledger.entries()
    kinds = [e["payload"]["kind"] for e in entries]
    assert kinds.count("claim") == 2
    assert kinds.count("gate") == 1


def test_verify_ok(tmp_path, entail_backend):
    path = tmp_path / "run.jsonl"
    ledger = JsonlLedger(path)
    gate("The sky is blue.", ["a"], backend=entail_backend, ledger=ledger)
    ok, reason = ledger.verify()
    assert ok and reason is None


def test_chain_links_across_calls(tmp_path, entail_backend):
    path = tmp_path / "run.jsonl"
    ledger = JsonlLedger(path)
    gate("One claim.", ["a"], backend=entail_backend, ledger=ledger)
    gate("Two claim.", ["b"], backend=entail_backend, ledger=ledger)
    ok, _ = ledger.verify()
    assert ok
    assert len(ledger.entries()) == 4  # 2 calls * (1 claim + 1 gate)


def test_tamper_detected(tmp_path, entail_backend):
    path = tmp_path / "run.jsonl"
    ledger = JsonlLedger(path)
    # Citation is relevant to the claim so the verdict is "entailed".
    gate("The sky is blue.", ["The sky is blue."], backend=entail_backend, ledger=ledger)
    lines = path.read_text().splitlines()
    # Flip the decision inside a recorded entry without fixing the hash.
    tampered = lines[0].replace('"entailed"', '"contradicted"')
    assert tampered != lines[0]
    path.write_text("\n".join([tampered, *lines[1:]]) + "\n")
    ok, reason = ledger.verify()
    assert not ok
    assert reason is not None and "tampered" in reason.lower()


def test_empty_ledger_verifies(tmp_path):
    ok, reason = JsonlLedger(tmp_path / "absent.jsonl").verify()
    assert ok and reason is None


def _make_ledger(tmp_path, entail_backend):
    path = tmp_path / "run.jsonl"
    ledger = JsonlLedger(path)
    gate(
        "The sky is blue. Grass is green.",
        ["The sky is blue.", "Grass is green."],
        backend=entail_backend,
        ledger=ledger,
    )
    return path, ledger


def test_invalid_json_line_detected(tmp_path, entail_backend):
    path, ledger = _make_ledger(tmp_path, entail_backend)
    with path.open("a", encoding="utf-8") as f:
        f.write("this is not json\n")
    ok, reason = ledger.verify()
    assert not ok
    assert reason is not None and "invalid JSON" in reason


def test_hash_algo_mismatch_detected(tmp_path, entail_backend):
    path, ledger = _make_ledger(tmp_path, entail_backend)
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["hash_algo"] = "fake-algo-1"
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    ok, reason = ledger.verify()
    assert not ok
    assert reason is not None and "hash algo mismatch" in reason


def test_broken_chain_link_detected(tmp_path, entail_backend):
    path, ledger = _make_ledger(tmp_path, entail_backend)
    lines = path.read_text().splitlines()
    assert len(lines) >= 2  # 2 claims + 1 gate summary
    second = json.loads(lines[1])
    second["prev_hash"] = "f" * 64  # break the link to the previous entry
    path.write_text("\n".join([lines[0], json.dumps(second), *lines[2:]]) + "\n")
    ok, reason = ledger.verify()
    assert not ok
    assert reason is not None and "broken chain link" in reason
