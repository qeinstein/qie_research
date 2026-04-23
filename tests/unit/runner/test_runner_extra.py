"""Tests for uncovered runner.py branches: rff gamma, torch_mlp baseline, torch head, main()."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from qie_research.runner import _build_rff_map, _run_baseline, run


# _build_rff_map with explicit float gamma (line 424)

def test_build_rff_map_explicit_float_gamma():
    from sklearn.pipeline import Pipeline
    fm = _build_rff_map({"n_components": 8, "gamma": 0.5}, n_components_auto=8, seed=0, n_features=4)
    assert isinstance(fm, Pipeline)
    rff = fm.named_steps["rff"]
    assert rff.gamma == pytest.approx(0.5)


def test_build_rff_map_gamma_none_uses_auto():
    from sklearn.pipeline import Pipeline
    fm = _build_rff_map({"n_components": 8, "gamma": None}, n_components_auto=8, seed=0, n_features=10)
    assert isinstance(fm, Pipeline)
    rff = fm.named_steps["rff"]
    assert rff.gamma == pytest.approx(1.0 / 10)


# torch_mlp baseline via _run_baseline (lines 710-725)

def _make_xy(n=80, d=6, seed=30):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_run_baseline_torch_mlp():
    X_tr, y_tr = _make_xy(80, 6, seed=30)
    X_te, y_te = _make_xy(20, 6, seed=31)
    bl_cfg = {
        "name": "torch_mlp_test",
        "model": {
            "name": "torch_mlp",
            "hidden_layer_sizes": [16, 8],
            "epochs": 3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
        },
    }
    result = _run_baseline(bl_cfg, X_tr, y_tr, X_te, y_te, seed=0, n_components_auto=8)
    assert result["model"] == "torch_mlp"
    assert "accuracy" in result["metrics"]
    assert result["feature_map"] is None


def test_run_baseline_torch_mlp_with_max_samples():
    X_tr, y_tr = _make_xy(200, 6, seed=32)
    X_te, y_te = _make_xy(20, 6, seed=33)
    bl_cfg = {
        "name": "torch_mlp_subsample",
        "max_samples": 50,
        "model": {
            "name": "torch_mlp",
            "hidden_layer_sizes": [16],
            "epochs": 2,
        },
    }
    result = _run_baseline(bl_cfg, X_tr, y_tr, X_te, y_te, seed=0, n_components_auto=8)
    assert result["model"] == "torch_mlp"


# torch linear head in run() via config with torch: block (lines 866-867)

def test_run_with_torch_linear_head(tmp_path):
    cfg = {
        "run": {
            "name": "torch_head_test",
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
            "torch": {
                "epochs": 3,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 64,
            },
        },
        "dataset": {"name": "wine", "test_size": 0.2},
        "encodings": [{"name": "amplitude", "pad_to_power_of_two": True}],
        "model": {"name": "logistic_regression", "max_iter": 100},
    }
    p = tmp_path / "torch_cfg.yaml"
    p.write_text(yaml.dump(cfg))
    result = run(p)
    enc_result = result["results"][0]
    assert enc_result["torch_linear_head"] is not None
    assert "metrics" in enc_result["torch_linear_head"]
    assert "training_curves" in enc_result["torch_linear_head"]


def test_run_torch_only_skips_non_torch_baselines(tmp_path):
    cfg = {
        "run": {"name": "torch_only_skip_baselines", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "wine", "test_size": 0.2},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression", "max_iter": 100},
        "baselines": [
            {"name": "raw_linear", "model": {"name": "logistic_regression", "max_iter": 100}},
            {"name": "rbf_svm", "model": {"name": "rbf_svm", "C": 1.0}},
        ],
    }
    p = tmp_path / "torch_only_skip_baselines.yaml"
    p.write_text(yaml.dump(cfg))

    result = run(p, torch_only=True)
    assert result["baselines"] == []
    assert len(result["results"]) == 1
    assert result["results"][0]["encoding"] == "amplitude"


# runner main() CLI function (lines 940-978)

def test_main_prints_results(tmp_path, capsys):
    cfg = {
        "run": {"name": "main_test", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "wine", "test_size": 0.2},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression", "max_iter": 100},
    }
    p = tmp_path / "main_cfg.yaml"
    p.write_text(yaml.dump(cfg))

    from qie_research.runner import main
    with patch("sys.argv", ["runner", str(p)]):
        main()

    out = capsys.readouterr().out
    assert "QIE Encodings" in out
    assert "amplitude" in out


def test_main_prints_baselines(tmp_path, capsys):
    cfg = {
        "run": {"name": "main_bl_test", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "wine", "test_size": 0.2},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression", "max_iter": 100},
        "baselines": [
            {"name": "raw_linear", "model": {"name": "logistic_regression", "max_iter": 100}}
        ],
    }
    p = tmp_path / "main_bl.yaml"
    p.write_text(yaml.dump(cfg))

    from qie_research.runner import main
    with patch("sys.argv", ["runner", str(p)]):
        main()

    out = capsys.readouterr().out
    assert "Classical Baselines" in out
    assert "raw_linear" in out


# main() with subsampled baseline prints subsample info

def test_main_prints_subsample(tmp_path, capsys):
    cfg = {
        "run": {"name": "sub_test", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"name": "high_dim_parity", "n_samples": 200, "n_features": 8},
        "encodings": [{"name": "amplitude"}],
        "model": {"name": "logistic_regression", "max_iter": 100},
        "baselines": [
            {
                "name": "sub_bl",
                "max_samples": 50,
                "model": {"name": "logistic_regression", "max_iter": 100},
            }
        ],
    }
    p = tmp_path / "sub_cfg.yaml"
    p.write_text(yaml.dump(cfg))

    from qie_research.runner import main
    with patch("sys.argv", ["runner", str(p)]):
        main()

    out = capsys.readouterr().out
    assert "subsample=50" in out
