"""Unit tests for models/torch_trainer.py."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.models.torch_trainer import _device, _grad_norm, train_linear_head, train_mlp


# _device

def test_device_returns_cpu_or_cuda():
    import torch
    d = _device()
    assert d.type in ("cpu", "cuda")


# _grad_norm

def test_grad_norm_zero_when_no_grads():
    import torch.nn as nn
    model = nn.Linear(4, 2)
    # no backward call, grads are None
    assert _grad_norm(model) == 0.0


def test_grad_norm_positive_after_backward():
    import torch
    import torch.nn as nn
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 0])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    assert _grad_norm(model) > 0.0


# train_linear_head — output schema

@pytest.fixture()
def binary_encoded_data():
    rng = np.random.default_rng(20)
    X_tr = rng.standard_normal((60, 8)).astype(np.float32)
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((20, 8)).astype(np.float32)
    y_te = (X_te[:, 0] > 0).astype(int)
    return X_tr, y_tr, X_te, y_te


def test_train_linear_head_returns_required_keys(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=3, seed=0)
    assert "metrics" in result
    assert "training_curves" in result
    assert "gradients_through_encoding" in result
    assert "encoding_note" in result
    assert "timing_seconds" in result
    assert "memory_bytes" in result


def test_train_linear_head_metrics_in_bounds(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=3, seed=0)
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["f1_macro"] <= 1.0


def test_train_linear_head_gradients_through_encoding_false(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0)
    assert result["gradients_through_encoding"] is False


def test_train_linear_head_training_curves_length(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    n = 5
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=n, seed=0)
    curves = result["training_curves"]
    assert len(curves["epochs"]) == n
    assert len(curves["train_loss"]) == n
    assert len(curves["grad_norm"]) == n
    assert curves["epochs"] == list(range(1, n + 1))


def test_train_linear_head_basis_note(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=2, encoding_name="basis")
    assert "discontinuous" in result["encoding_note"]


def test_train_linear_head_non_basis_note(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=2, encoding_name="amplitude")
    assert "amplitude" in result["encoding_note"]


def test_train_linear_head_timing_nonneg(binary_encoded_data):
    X_tr, y_tr, X_te, y_te = binary_encoded_data
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0)
    assert result["timing_seconds"] >= 0.0
    assert result["memory_bytes"] >= 0


def test_train_linear_head_multiclass():
    rng = np.random.default_rng(21)
    X_tr = rng.standard_normal((90, 8)).astype(np.float32)
    y_tr = rng.integers(0, 3, size=90)
    X_te = rng.standard_normal((30, 8)).astype(np.float32)
    y_te = rng.integers(0, 3, size=30)
    result = train_linear_head(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0)
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


# train_mlp — output schema

@pytest.fixture()
def raw_binary_data():
    rng = np.random.default_rng(22)
    X_tr = rng.standard_normal((80, 6))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.standard_normal((20, 6))
    y_te = (X_te[:, 0] > 0).astype(int)
    return X_tr, y_tr, X_te, y_te


def test_train_mlp_returns_required_keys(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=3, seed=0, hidden_layer_sizes=[16, 8])
    assert "metrics" in result
    assert "training_curves" in result
    assert result["gradients_through_encoding"] is True
    assert result["encoding_note"] is None


def test_train_mlp_metrics_in_bounds(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=3, seed=0, hidden_layer_sizes=[16])
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


def test_train_mlp_training_curves_length(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    n = 4
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=n, seed=0, hidden_layer_sizes=[16])
    assert len(result["training_curves"]["epochs"]) == n
    assert len(result["training_curves"]["train_loss"]) == n


def test_train_mlp_no_subsampled_key_when_not_needed(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0, hidden_layer_sizes=[8])
    assert "subsampled_train_n" not in result


def test_train_mlp_subsampled_key_when_max_samples(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0,
                       hidden_layer_sizes=[8], max_samples=30)
    assert result["subsampled_train_n"] == 30


def test_train_mlp_max_samples_no_effect_when_small(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    # max_samples > n_train → no subsampling
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0,
                       hidden_layer_sizes=[8], max_samples=9999)
    assert "subsampled_train_n" not in result


def test_train_mlp_timing_nonneg(raw_binary_data):
    X_tr, y_tr, X_te, y_te = raw_binary_data
    result = train_mlp(X_tr, y_tr, X_te, y_te, n_epochs=2, seed=0, hidden_layer_sizes=[8])
    assert result["timing_seconds"] >= 0.0
    assert result["memory_bytes"] >= 0
