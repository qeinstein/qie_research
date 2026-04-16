"""
Determinism Smoke Test

Verifies that the runner produces identical metrics, output dimensions, and
encoding parameters across two independent runs on the same config.

This is the Phase 1 exit gate.  It must pass before Phase 1 can be closed.

Run with:
    pytest tests/test_determinism.py -v
"""

from pathlib import Path

import pytest

from qie_research.runner import run


CONFIG_PATH = Path("configs/smoke_test.yaml")


@pytest.fixture(scope="module")
def two_runs():
    """Execute the smoke test config twice and return both result dicts."""
    first = run(CONFIG_PATH)
    second = run(CONFIG_PATH)
    return first, second


def test_config_exists():
    """Smoke test config file must exist before any run."""
    assert CONFIG_PATH.exists(), f"Config file not found: {CONFIG_PATH}"


def test_same_number_of_encoding_results(two_runs):
    """Both runs must produce results for the same number of encodings."""
    first, second = two_runs
    assert len(first["results"]) == len(second["results"])


def test_same_encoding_names(two_runs):
    """Encoding names must appear in the same order across both runs."""
    first, second = two_runs
    names_first = [r["encoding"] for r in first["results"]]
    names_second = [r["encoding"] for r in second["results"]]
    assert names_first == names_second


def test_same_encoding_params(two_runs):
    """Encoding hyperparameters must be identical across both runs."""
    first, second = two_runs
    for r1, r2 in zip(first["results"], second["results"]):
        assert r1["encoding_params"] == r2["encoding_params"], (
            f"Encoding params differ for '{r1['encoding']}': "
            f"{r1['encoding_params']} vs {r2['encoding_params']}"
        )


def test_same_output_dimensions(two_runs):
    """Output dimensionality must be deterministic for each encoding."""
    first, second = two_runs
    for r1, r2 in zip(first["results"], second["results"]):
        assert r1["output_dim"] == r2["output_dim"], (
            f"output_dim differs for '{r1['encoding']}': "
            f"{r1['output_dim']} vs {r2['output_dim']}"
        )


def test_same_accuracy(two_runs):
    """Accuracy must be identical across both runs."""
    first, second = two_runs
    for r1, r2 in zip(first["results"], second["results"]):
        assert r1["metrics"]["accuracy"] == r2["metrics"]["accuracy"], (
            f"Accuracy differs for '{r1['encoding']}': "
            f"{r1['metrics']['accuracy']} vs {r2['metrics']['accuracy']}"
        )


def test_same_f1(two_runs):
    """F1 macro must be identical across both runs."""
    first, second = two_runs
    for r1, r2 in zip(first["results"], second["results"]):
        assert r1["metrics"]["f1_macro"] == r2["metrics"]["f1_macro"], (
            f"F1 differs for '{r1['encoding']}': "
            f"{r1['metrics']['f1_macro']} vs {r2['metrics']['f1_macro']}"
        )


def test_same_dataset_info(two_runs):
    """Dataset split sizes must be identical across both runs."""
    first, second = two_runs
    assert first["dataset"] == second["dataset"]


def test_same_seed(two_runs):
    """Both runs must have used the same seed."""
    first, second = two_runs
    assert first["run"]["seed"] == second["run"]["seed"]
