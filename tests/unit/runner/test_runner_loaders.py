"""Tests for file-backed dataset loaders and covertype with mocking."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.runner import (
    _load_cifar10,
    _load_covertype,
    _load_credit_card_fraud,
    _load_dry_bean,
    _load_fashion_mnist,
    _load_higgs,
)


# helpers that write fake npy caches to tmp_path

def _write_npy(tmp_path, name_X, name_y, n=20, d=10):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int32)
    np.save(tmp_path / name_X, X)
    np.save(tmp_path / name_y, y)
    return X, y


# fashion_mnist — cache exists, no subsampling

def test_load_fashion_mnist_with_cache(tmp_path):
    X_saved, y_saved = _write_npy(tmp_path, "fashion_mnist_X.npy", "fashion_mnist_y.npy", n=50, d=784)
    X, y = _load_fashion_mnist({"data_home": str(tmp_path)})
    assert X.shape == (50, 784)
    assert y.shape == (50,)


# fashion_mnist — max_samples subsampling path (lines 100-110)

def test_load_fashion_mnist_max_samples(tmp_path):
    _write_npy(tmp_path, "fashion_mnist_X.npy", "fashion_mnist_y.npy", n=50, d=784)
    X, y = _load_fashion_mnist({"data_home": str(tmp_path), "max_samples": 20, "seed": 1})
    assert X.shape[0] == 20
    assert y.shape[0] == 20


# dry_bean — cache exists, no subsampling

def test_load_dry_bean_with_cache(tmp_path):
    _write_npy(tmp_path, "dry_bean_X.npy", "dry_bean_y.npy", n=30, d=16)
    X, y = _load_dry_bean({"data_home": str(tmp_path)})
    assert X.shape == (30, 16)


# dry_bean — max_samples subsampling path (lines 224-234)

def test_load_dry_bean_max_samples(tmp_path):
    _write_npy(tmp_path, "dry_bean_X.npy", "dry_bean_y.npy", n=60, d=16)
    X, y = _load_dry_bean({"data_home": str(tmp_path), "max_samples": 25, "seed": 2})
    assert X.shape[0] == 25


# dry_bean — missing cache raises

def test_load_dry_bean_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dry Bean"):
        _load_dry_bean({"data_home": str(tmp_path)})


# credit_card_fraud — cache exists

def test_load_credit_card_fraud_with_cache(tmp_path):
    _write_npy(tmp_path, "credit_card_fraud_X.npy", "credit_card_fraud_y.npy", n=40, d=30)
    X, y = _load_credit_card_fraud({"data_home": str(tmp_path)})
    assert X.shape == (40, 30)


# credit_card_fraud — max_samples (lines 255-275)

def test_load_credit_card_fraud_max_samples(tmp_path):
    _write_npy(tmp_path, "credit_card_fraud_X.npy", "credit_card_fraud_y.npy", n=100, d=30)
    X, y = _load_credit_card_fraud({"data_home": str(tmp_path), "max_samples": 30})
    assert X.shape[0] == 30


# credit_card_fraud — missing cache raises

def test_load_credit_card_fraud_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Credit Card Fraud"):
        _load_credit_card_fraud({"data_home": str(tmp_path)})


# cifar10 — cache exists

def test_load_cifar10_with_cache(tmp_path):
    _write_npy(tmp_path, "cifar10_X.npy", "cifar10_y.npy", n=60, d=3072)
    X, y = _load_cifar10({"data_home": str(tmp_path)})
    assert X.shape == (60, 3072)


# cifar10 — max_samples (lines 294-314)

def test_load_cifar10_max_samples(tmp_path):
    _write_npy(tmp_path, "cifar10_X.npy", "cifar10_y.npy", n=100, d=3072)
    X, y = _load_cifar10({"data_home": str(tmp_path), "max_samples": 40})
    assert X.shape[0] == 40


# cifar10 — missing cache raises

def test_load_cifar10_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="CIFAR-10"):
        _load_cifar10({"data_home": str(tmp_path)})


# higgs — cache exists, max_samples default keeps all when small enough

def test_load_higgs_with_cache(tmp_path):
    _write_npy(tmp_path, "higgs_X.npy", "higgs_y.npy", n=20, d=21)
    X, y = _load_higgs({"data_home": str(tmp_path), "max_samples": 500_000})
    assert X.shape == (20, 21)


# higgs — max_samples subsampling when dataset smaller than default 500k (lines 336-356)

def test_load_higgs_max_samples_subsets(tmp_path):
    _write_npy(tmp_path, "higgs_X.npy", "higgs_y.npy", n=100, d=21)
    X, y = _load_higgs({"data_home": str(tmp_path), "max_samples": 30})
    assert X.shape[0] == 30


# higgs — missing cache raises

def test_load_higgs_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="HIGGS"):
        _load_higgs({"data_home": str(tmp_path)})


# covertype — max_samples path via monkeypatching sklearn.datasets.fetch_covtype

def test_load_covertype_max_samples(monkeypatch):
    rng = np.random.default_rng(0)

    class FakeCovtype:
        data = rng.standard_normal((200, 54)).astype(float)
        target = rng.integers(1, 8, size=200).astype(int)

    import sklearn.datasets as skds
    monkeypatch.setattr(skds, "fetch_covtype", lambda: FakeCovtype())

    from qie_research.runner import _load_covertype
    X, y = _load_covertype({"max_samples": 50, "seed": 0})
    assert X.shape[0] == 50
