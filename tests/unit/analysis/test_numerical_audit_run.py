"""Tests for numerical_audit.run_audit() and main() (lines 158-257)."""

from __future__ import annotations

import pytest

from qie_research.analysis.numerical_audit import _set_seeds, run_audit


# _set_seeds (lines 62-63)

def test_set_seeds_smoke():
    import numpy as np
    _set_seeds(1)
    a = np.random.rand()
    _set_seeds(1)
    b = np.random.rand()
    assert a == b


# run_audit — full execution (lines 158-233)

def test_run_audit_returns_dict(tmp_path):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        report = run_audit()
    finally:
        audit_mod.OUTPUT_PATH = orig

    assert isinstance(report, dict)
    assert report["phase"] == 2
    assert "dataset" in report
    assert "encodings" in report


def test_run_audit_three_encodings(tmp_path):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        report = run_audit()
    finally:
        audit_mod.OUTPUT_PATH = orig

    assert len(report["encodings"]) == 3
    names = [e["encoding"] for e in report["encodings"]]
    assert names == ["amplitude", "angle", "basis"]


def test_run_audit_invariants_pass(tmp_path):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        report = run_audit()
    finally:
        audit_mod.OUTPUT_PATH = orig

    for enc in report["encodings"]:
        assert enc["invariant_check"]["passed"] is True, \
            f"{enc['encoding']} invariant failed: {enc['invariant_check']['details']}"


def test_run_audit_writes_json(tmp_path):
    import json
    import qie_research.analysis.numerical_audit as audit_mod
    out_path = tmp_path / "audit.json"
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = out_path
    try:
        run_audit()
    finally:
        audit_mod.OUTPUT_PATH = orig

    assert out_path.exists()
    with out_path.open() as f:
        data = json.load(f)
    assert "encodings" in data


def test_run_audit_fields_present(tmp_path):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        report = run_audit()
    finally:
        audit_mod.OUTPUT_PATH = orig

    for enc in report["encodings"]:
        assert "effective_rank" in enc
        assert "condition_number" in enc
        assert "singular_values" in enc
        assert "noise_stability" in enc
        assert enc["effective_rank"] > 0
        assert enc["condition_number"] > 0


# numerical_audit main() (lines 239-246)

def test_numerical_audit_main(tmp_path, capsys):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        audit_mod.main()
    finally:
        audit_mod.OUTPUT_PATH = orig

    out = capsys.readouterr().out
    assert "Numerical Audit Summary" in out
    assert "amplitude" in out
    assert "angle" in out
    assert "basis" in out


def test_numerical_audit_main_accepts_argv(tmp_path, capsys):
    import qie_research.analysis.numerical_audit as audit_mod
    orig = audit_mod.OUTPUT_PATH
    audit_mod.OUTPUT_PATH = tmp_path / "audit.json"
    try:
        audit_mod.main(["--dataset", "high_rank_noise", "--n-features", "32", "--noise-std", "1.5"])
    finally:
        audit_mod.OUTPUT_PATH = orig

    out = capsys.readouterr().out
    assert "Numerical Audit Summary" in out
