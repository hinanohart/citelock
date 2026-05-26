"""CLI exit-code contract: 0 allow, 2 deny, 3 error."""

from __future__ import annotations

import json

from citelock.cli import EXIT_DENY, EXIT_ERROR, EXIT_OK, main


def test_gate_allow_exit0(capsys):
    code = main(
        [
            "gate",
            "--answer",
            "Paris is the capital of France.",
            "--citation",
            "Paris is the capital of France.",
            "--backend",
            "stub",
        ]
    )
    assert code == EXIT_OK
    assert "ALLOW" in capsys.readouterr().out


def test_gate_deny_exit2(capsys):
    code = main(
        [
            "gate",
            "--answer",
            "The moon is made of cheese.",
            "--citation",
            "The capital of Japan is Tokyo.",
            "--backend",
            "stub",
        ]
    )
    assert code == EXIT_DENY
    assert "DENY" in capsys.readouterr().out


def test_gate_json_output(capsys):
    code = main(
        [
            "gate",
            "--answer",
            "Paris is the capital of France.",
            "--citation",
            "Paris is the capital of France.",
            "--backend",
            "stub",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["decision"] in {"allow", "deny"}
    assert "input_digest" in data
    assert code in {EXIT_OK, EXIT_DENY}


def test_gate_too_few_citations_denies():
    code = main(["gate", "--answer", "A claim.", "--backend", "stub"])
    assert code == EXIT_DENY


def test_ledger_roundtrip_and_verify(tmp_path):
    led = tmp_path / "run.jsonl"
    main(
        [
            "gate",
            "--answer",
            "Paris is the capital of France.",
            "--citation",
            "Paris is the capital of France.",
            "--backend",
            "stub",
            "--ledger",
            str(led),
        ]
    )
    assert led.exists()
    code = main(["verify-ledger", "--ledger", str(led)])
    assert code == EXIT_OK


def test_verify_tampered_ledger_exit2(tmp_path):
    led = tmp_path / "run.jsonl"
    main(
        [
            "gate",
            "--answer",
            "Paris is the capital of France.",
            "--citation",
            "Paris is the capital of France.",
            "--backend",
            "stub",
            "--ledger",
            str(led),
        ]
    )
    text = led.read_text()
    led.write_text(text.replace("entailed", "contradicted"))
    code = main(["verify-ledger", "--ledger", str(led)])
    assert code == EXIT_DENY


def test_backends_lists_licenses(capsys):
    code = main(["backends"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Apache-2.0" in out
    assert "CC-BY-NC-4.0" in out


def test_missing_citations_file_is_error():
    code = main(
        ["gate", "--answer", "A.", "--citations-file", "/nonexistent/x.json", "--backend", "stub"]
    )
    assert code == EXIT_ERROR
