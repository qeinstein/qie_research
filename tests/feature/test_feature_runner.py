"""Feature tests for the config-driven runner — end-to-end integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from qie_research.runner import run


# helpers

def _make_config(tmp_path: Path, extra: dict | None = None) -> Path:
    # minimal valid config using the in-memory wine dataset
    cfg = {
        "run": {
            "name": "test_run",
            "seed": 42,
            "output_dir": str(tmp_path / "results"),
        },
        "dataset": {"name": "wine", "test_size": 0.2},
        "encodings": [
            {"name": "amplitude", "pad_to_power_of_two": True},
            {"name": "angle", "scale": 3.141592653589793, "standardize": True},
            {"name": "basis", "n_bits": 4},
        ],
        "model": {"name": "logistic_regression", "max_iter": 200, "C": 1.0},
    }
    if extra:
        cfg.update(extra)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# basic run smoke test

def test_run_produces_output_file(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    output_file = Path(result["run"]["config_path"]).parent / f"{result['run']['name']}.json"
    # the runner writes to output_dir / name.json
    expected = Path(tmp_path / "results" / "test_run.json")
    assert expected.exists()


def test_run_returns_dict_with_required_keys(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    assert "run" in result
    assert "dataset" in result
    assert "results" in result
    assert "baselines" in result


def test_run_dataset_info_correct(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    ds = result["dataset"]
    assert ds["name"] == "wine"
    assert ds["n_features"] == 13
    assert ds["n_classes"] == 3
    assert ds["n_train"] + ds["n_test"] == 178


def test_run_all_encodings_present(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    names = [r["encoding"] for r in result["results"]]
    assert names == ["amplitude", "angle", "basis"]


def test_run_metrics_in_bounds(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    for r in result["results"]:
        assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
        assert 0.0 <= r["metrics"]["f1_macro"] <= 1.0


def test_run_output_dims_positive(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    for r in result["results"]:
        assert r["output_dim"] > 0
        assert r["input_dim"] == 13


def test_run_timing_nonnegative(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    for r in result["results"]:
        t = r["timing_seconds"]
        assert t["encoding"] >= 0
        assert t["training"] >= 0
        assert t["total"] >= 0


def test_run_memory_bytes_nonneg(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    for r in result["results"]:
        m = r["memory_bytes"]
        assert m["encoding_peak"] >= 0
        assert m["training_peak"] >= 0


# output JSON is valid and loadable

def test_run_output_json_valid(tmp_path):
    cfg_path = _make_config(tmp_path)
    run(cfg_path)
    json_path = tmp_path / "results" / "test_run.json"
    with json_path.open() as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "results" in data


# seed in run block is recorded correctly

def test_run_seed_in_output(tmp_path):
    cfg_path = _make_config(tmp_path)
    result = run(cfg_path)
    assert result["run"]["seed"] == 42


# determinism: two runs with same seed produce identical metrics

def test_run_determinism(tmp_path):
    cfg_path = _make_config(tmp_path)
    r1 = run(cfg_path)
    r2 = run(cfg_path)
    for enc1, enc2 in zip(r1["results"], r2["results"]):
        assert enc1["metrics"]["accuracy"] == enc2["metrics"]["accuracy"]
        assert enc1["metrics"]["f1_macro"] == enc2["metrics"]["f1_macro"]


# config not found raises FileNotFoundError

def test_run_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        run(tmp_path / "does_not_exist.yaml")


# unknown dataset raises ValueError

def test_run_unknown_dataset_raises(tmp_path):
    cfg = {
        "run": {"name": "bad", "seed": 0, "output_dir": str(tmp_path)},
        "dataset": {"name": "nonexistent_dataset"},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="Unknown dataset"):
        run(p)


# unknown encoding raises ValueError

def test_run_unknown_encoding_raises(tmp_path):
    cfg = {
        "run": {"name": "bad_enc", "seed": 0, "output_dir": str(tmp_path)},
        "dataset": {"name": "wine"},
        "encodings": [{"name": "mystery_encoding"}],
        "model": {"name": "logistic_regression"},
    }
    p = tmp_path / "bad_enc.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="Unknown encoding"):
        run(p)


# baselines integration: logistic regression baseline included

def test_run_with_baseline(tmp_path):
    cfg_path = _make_config(tmp_path, extra={
        "baselines": [
            {
                "name": "raw_linear",
                "model": {"name": "logistic_regression", "max_iter": 100},
            }
        ]
    })
    result = run(cfg_path)
    assert len(result["baselines"]) == 1
    bl = result["baselines"][0]
    assert bl["name"] == "raw_linear"
    assert 0.0 <= bl["metrics"]["accuracy"] <= 1.0


def test_run_with_rff_baseline(tmp_path):
    cfg_path = _make_config(tmp_path, extra={
        "baselines": [
            {
                "name": "rff_baseline",
                "feature_map": {"name": "rff", "n_components": "auto"},
                "model": {"name": "logistic_regression", "max_iter": 100},
            }
        ]
    })
    result = run(cfg_path)
    bl = result["baselines"][0]
    assert bl["feature_map"] == "rff"
    assert 0.0 <= bl["metrics"]["accuracy"] <= 1.0


def test_run_with_polynomial_baseline(tmp_path):
    cfg_path = _make_config(tmp_path, extra={
        "baselines": [
            {
                "name": "poly_baseline",
                "feature_map": {"name": "polynomial", "degree": 2},
                "model": {"name": "logistic_regression", "max_iter": 200},
            }
        ]
    })
    result = run(cfg_path)
    bl = result["baselines"][0]
    assert bl["feature_map"] == "polynomial"


def test_run_with_pca_baseline(tmp_path):
    cfg_path = _make_config(tmp_path, extra={
        "baselines": [
            {
                "name": "pca_baseline",
                "feature_map": {"name": "pca", "n_components": 5},
                "model": {"name": "logistic_regression", "max_iter": 200},
            }
        ]
    })
    result = run(cfg_path)
    bl = result["baselines"][0]
    assert bl["feature_map"] == "pca"
    assert bl["feature_dim"] == 5


# high_dim_parity dataset end-to-end

def test_run_high_dim_parity(tmp_path):
    cfg = {
        "run": {"name": "parity_test", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "high_dim_parity", "n_samples": 200, "n_features": 10},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression", "max_iter": 200},
    }
    p = tmp_path / "parity.yaml"
    p.write_text(yaml.dump(cfg))
    result = run(p)
    assert result["dataset"]["n_features"] == 10
    assert result["dataset"]["n_classes"] == 2


# high_rank_noise dataset end-to-end

def test_run_high_rank_noise(tmp_path):
    cfg = {
        "run": {"name": "noise_test", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "high_rank_noise", "n_samples": 200, "n_features": 20},
        "encodings": [{"name": "angle"}],
        "model": {"name": "logistic_regression", "max_iter": 200},
    }
    p = tmp_path / "noise.yaml"
    p.write_text(yaml.dump(cfg))
    result = run(p)
    assert result["dataset"]["n_classes"] == 2
