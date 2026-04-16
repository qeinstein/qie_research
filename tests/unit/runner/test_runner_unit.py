"""Unit tests for runner.py utility functions and registries."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.runner import (
    DATASET_REGISTRY,
    ENCODING_REGISTRY,
    FEATURE_MAP_REGISTRY,
    MODEL_REGISTRY,
    _build_logistic_regression,
    _build_mlp,
    _build_pca_map,
    _build_poly_map,
    _build_rbf_svm,
    _build_rff_map,
    _build_scaler_map,
    _encode,
    _load_breast_cancer,
    _load_high_dim_parity,
    _load_high_rank_noise,
    _load_wine,
    _run_baseline,
    _set_seeds,
    _train_and_evaluate,
)


# registry completeness

def test_encoding_registry_keys():
    assert set(ENCODING_REGISTRY.keys()) == {"amplitude", "angle", "basis"}


def test_dataset_registry_contains_required():
    required = {"wine", "breast_cancer", "high_dim_parity", "high_rank_noise"}
    assert required.issubset(set(DATASET_REGISTRY.keys()))


def test_model_registry_keys():
    assert set(MODEL_REGISTRY.keys()) == {"logistic_regression", "rbf_svm", "mlp"}


def test_feature_map_registry_keys():
    assert set(FEATURE_MAP_REGISTRY.keys()) == {"scaler", "rff", "polynomial", "pca"}


# _set_seeds — just check it runs without error and that numpy is seeded

def test_set_seeds_determinism():
    _set_seeds(0)
    a = np.random.rand()
    _set_seeds(0)
    b = np.random.rand()
    assert a == b


# dataset loaders that don't need external files

def test_load_wine_shape():
    X, y = _load_wine({})
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 13


def test_load_breast_cancer_shape():
    X, y = _load_breast_cancer({})
    assert X.shape == (569, 30)
    assert set(np.unique(y)) == {0, 1}


def test_load_high_dim_parity_defaults():
    X, y = _load_high_dim_parity({})
    assert X.shape == (2000, 50)
    assert set(np.unique(y)) == {0, 1}


def test_load_high_dim_parity_custom():
    X, y = _load_high_dim_parity({"n_samples": 100, "n_features": 10, "seed": 1})
    assert X.shape == (100, 10)


def test_load_high_rank_noise_defaults():
    X, y = _load_high_rank_noise({})
    assert X.shape == (2000, 100)
    assert set(np.unique(y)) == {0, 1}


def test_load_high_rank_noise_custom():
    X, y = _load_high_rank_noise({"n_samples": 50, "n_features": 20, "seed": 7})
    assert X.shape == (50, 20)


# file-backed loaders raise FileNotFoundError when cache is absent

def test_load_fashion_mnist_missing_cache(tmp_path):
    from qie_research.runner import _load_fashion_mnist
    with pytest.raises(FileNotFoundError, match="Fashion-MNIST cache"):
        _load_fashion_mnist({"data_home": str(tmp_path)})


def test_load_dry_bean_missing_cache(tmp_path):
    from qie_research.runner import _load_dry_bean
    with pytest.raises(FileNotFoundError, match="Dry Bean cache"):
        _load_dry_bean({"data_home": str(tmp_path)})


# feature map builders

def test_build_scaler_map_returns_scaler():
    from sklearn.preprocessing import StandardScaler
    fm = _build_scaler_map({}, 16, 42)
    assert isinstance(fm, StandardScaler)


def test_build_rff_map_auto_n_components():
    from sklearn.pipeline import Pipeline
    fm = _build_rff_map({"n_components": "auto"}, 8, 42, n_features=4)
    assert isinstance(fm, Pipeline)


def test_build_rff_map_explicit_n_components():
    from sklearn.pipeline import Pipeline
    fm = _build_rff_map({"n_components": 10}, 8, 42, n_features=4)
    assert isinstance(fm, Pipeline)


def test_build_poly_map_default_degree():
    from sklearn.pipeline import Pipeline
    fm = _build_poly_map({}, 8, 42)
    assert isinstance(fm, Pipeline)


def test_build_pca_map_explicit():
    from sklearn.decomposition import PCA
    fm = _build_pca_map({"n_components": 5}, 8, 42)
    assert isinstance(fm, PCA)
    assert fm.n_components == 5


def test_build_pca_map_auto():
    from sklearn.decomposition import PCA
    fm = _build_pca_map({"n_components": "auto"}, 6, 42)
    assert isinstance(fm, PCA)
    assert fm.n_components == 6


# model builders

def test_build_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    m = _build_logistic_regression({"max_iter": 200, "C": 0.5, "_seed": 0})
    assert isinstance(m, LogisticRegression)


def test_build_rbf_svm():
    from sklearn.pipeline import Pipeline
    m = _build_rbf_svm({"C": 2.0, "_seed": 1})
    assert isinstance(m, Pipeline)


def test_build_mlp():
    from sklearn.pipeline import Pipeline
    m = _build_mlp({"hidden_layer_sizes": [64, 32], "_seed": 2})
    assert isinstance(m, Pipeline)


# _encode

def test_encode_returns_correct_shapes():
    from qie_research.encodings.amplitude_encoding import AmplitudeEncoding
    enc = AmplitudeEncoding()
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((10, 5))
    X_test = rng.standard_normal((4, 5))
    X_tr_enc, X_te_enc, elapsed, peak = _encode(enc, X_train, X_test)
    assert X_tr_enc.shape[0] == 10
    assert X_te_enc.shape[0] == 4
    assert X_tr_enc.shape[1] == X_te_enc.shape[1]
    assert elapsed >= 0
    assert peak >= 0


# _train_and_evaluate

def test_train_and_evaluate_keys():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X_tr = rng.standard_normal((50, 4))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((20, 4))
    y_te = (X_te[:, 0] > 0).astype(int)
    model = LogisticRegression(random_state=0)
    metrics, elapsed, peak = _train_and_evaluate(model, X_tr, y_tr, X_te, y_te)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert elapsed >= 0
    assert peak >= 0


# _run_baseline with logistic regression and no feature map

def test_run_baseline_raw_linear():
    rng = np.random.default_rng(4)
    X_tr = rng.standard_normal((60, 6))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((20, 6))
    y_te = (X_te[:, 0] > 0).astype(int)

    bl_cfg = {
        "name": "raw_linear",
        "model": {"name": "logistic_regression", "max_iter": 100, "C": 1.0},
    }
    result = _run_baseline(bl_cfg, X_tr, y_tr, X_te, y_te, seed=0, n_components_auto=8)
    assert result["name"] == "raw_linear"
    assert "accuracy" in result["metrics"]
    assert result["feature_map"] is None


def test_run_baseline_with_scaler():
    rng = np.random.default_rng(5)
    X_tr = rng.standard_normal((60, 6))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((20, 6))
    y_te = (X_te[:, 0] > 0).astype(int)

    bl_cfg = {
        "name": "scaled_linear",
        "feature_map": {"name": "scaler"},
        "model": {"name": "logistic_regression", "max_iter": 100},
    }
    result = _run_baseline(bl_cfg, X_tr, y_tr, X_te, y_te, seed=0, n_components_auto=8)
    assert result["feature_map"] == "scaler"
    assert "accuracy" in result["metrics"]


def test_run_baseline_unknown_feature_map_raises():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((10, 3))
    y = np.zeros(10, dtype=int)
    bl_cfg = {
        "name": "bad",
        "feature_map": {"name": "nonexistent_map"},
        "model": {"name": "logistic_regression"},
    }
    with pytest.raises(ValueError, match="Unknown feature map"):
        _run_baseline(bl_cfg, X, y, X, y, seed=0, n_components_auto=4)


def test_run_baseline_unknown_model_raises():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((10, 3))
    y = np.zeros(10, dtype=int)
    bl_cfg = {
        "name": "bad_model",
        "model": {"name": "mystery_model"},
    }
    with pytest.raises(ValueError, match="Unknown model"):
        _run_baseline(bl_cfg, X, y, X, y, seed=0, n_components_auto=4)


def test_run_baseline_subsampling():
    rng = np.random.default_rng(8)
    X_tr = rng.standard_normal((200, 4))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((40, 4))
    y_te = (X_te[:, 0] > 0).astype(int)

    bl_cfg = {
        "name": "subsampled",
        "max_samples": 50,
        "model": {"name": "logistic_regression", "max_iter": 100},
    }
    result = _run_baseline(bl_cfg, X_tr, y_tr, X_te, y_te, seed=0, n_components_auto=4)
    assert result["subsampled_train_n"] == 50
