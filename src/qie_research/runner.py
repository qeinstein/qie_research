"""
Config-Driven Runner

Executes a full benchmark run for all encodings specified in a YAML config
file.  A single command produces one JSON results file containing metrics,
timing, and memory measurements for every encoding.

Usage
-----
    python -m qie_research.runner <path/to/config.yaml>

Example
-------
    python -m qie_research.runner configs/smoke_test.yaml

Output
------
A JSON file written to the run's output_dir:

    {
        "run":      { name, seed, config_path, timestamp },
        "dataset":  { name, n_train, n_test, n_features, n_classes },
        "results":  [
            {
                "encoding":         "amplitude",
                "encoding_params":  { ... },
                "input_dim":        13,
                "output_dim":       16,
                "metrics":          { accuracy, f1_macro },
                "timing_seconds":   { encoding, training, total },
                "memory_bytes":     { encoding_peak, training_peak }
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import fetch_openml, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from qie_research.encodings import ENCODING_REGISTRY

# Dataset registry

def _load_wine(params: dict) -> tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    return data.data, data.target


def _load_fashion_mnist(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST from numpy cache files.

    The cache files (fashion_mnist_X.npy, fashion_mnist_y.npy) must be
    generated once using the helper script:

        python -m qie_research.datasets.prepare_fashion_mnist

    Raw pixel values are normalised to [0, 1].  The full dataset is 70,000
    samples x 784 features (flattened 28x28 images).  The config may specify
    max_samples to subsample for faster runs during development.

    Config keys
    -----------
    data_home : str, default "data/raw/"
    max_samples : int, optional
    seed : int, default 42
    """
    data_home = Path(params.get("data_home", "data/raw/"))
    cache_X = data_home / "fashion_mnist_X.npy"
    cache_y = data_home / "fashion_mnist_y.npy"

    if not cache_X.exists() or not cache_y.exists():
        raise FileNotFoundError(
            f"Fashion-MNIST cache not found at {data_home}. "
            "Run: python -m qie_research.datasets.prepare_fashion_mnist"
        )

    X = np.load(cache_X)
    y = np.load(cache_y)

    max_samples = params.get("max_samples", None)
    if max_samples is not None:
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


def _load_high_dim_parity(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic high-dimensional parity dataset.

    Each sample is a vector of d continuous features drawn from U[-1, 1].
    The label is the parity of the signs of the first k features:

        y = (sign(x_1) * sign(x_2) * ... * sign(x_k) > 0) ? 1 : 0

    This is provably hard for linear models at high dimension and provides
    a controlled test of representational capacity.

    Config keys
    -----------
    n_samples : int, default 2000
    n_features : int, default 50
    n_parity_bits : int, default 5   (number of features that determine label)
    seed : int, default 42
    """
    seed = params.get("seed", 42)
    n_samples = int(params.get("n_samples", 2000))
    n_features = int(params.get("n_features", 50))
    n_parity_bits = int(params.get("n_parity_bits", 5))

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    parity = np.prod(np.sign(X[:, :n_parity_bits]), axis=1)
    y = (parity > 0).astype(int)

    return X, y


def _load_high_rank_noise(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic high-rank noise stress-test dataset.

    A low-rank signal matrix is constructed from k latent components,
    then isotropic Gaussian noise is added at a controlled signal-to-noise
    ratio.  The resulting feature matrix has high intrinsic rank, making it
    a genuine stress test for encodings that collapse to low-dimensional
    representations (e.g. amplitude encoding).

    The classification task separates two Gaussian clusters embedded in
    this high-dimensional noisy space.

    Config keys
    -----------
    n_samples : int, default 2000
    n_features : int, default 100
    n_signal_components : int, default 5
    noise_std : float, default 1.0
    seed : int, default 42
    """
    seed = params.get("seed", 42)
    n_samples = int(params.get("n_samples", 2000))
    n_features = int(params.get("n_features", 100))
    n_signal = int(params.get("n_signal_components", 5))
    noise_std = float(params.get("noise_std", 1.0))

    rng = np.random.default_rng(seed)

    # Low-rank signal: two clusters separated along the first signal component
    latent = rng.standard_normal((n_samples, n_signal))
    basis = rng.standard_normal((n_signal, n_features))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)

    y = (latent[:, 0] > 0).astype(int)
    latent[y == 1, 0] += 2.0   # shift cluster 1 along first component

    X = latent @ basis + rng.normal(0, noise_std, size=(n_samples, n_features))

    return X, y


def _load_breast_cancer(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    UCI Breast Cancer Wisconsin dataset.

    569 samples, 30 features, binary classification (malignant/benign).
    Loaded directly from sklearn — no download required.
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data, data.target


def _load_dry_bean(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    UCI Dry Bean dataset.

    13,611 samples, 16 features, 7 classes.  Loaded from numpy cache.
    Cache must be generated once using:

        python -m qie_research.datasets.prepare_dry_bean

    Config keys
    -----------
    data_home : str, default "data/raw/"
    max_samples : int, optional
    seed : int, default 42
    """
    data_home = Path(params.get("data_home", "data/raw/"))
    cache_X = data_home / "dry_bean_X.npy"
    cache_y = data_home / "dry_bean_y.npy"

    if not cache_X.exists() or not cache_y.exists():
        raise FileNotFoundError(
            f"Dry Bean cache not found at {data_home}. "
            "Run: python -m qie_research.datasets.prepare_dry_bean"
        )

    X = np.load(cache_X)
    y = np.load(cache_y)

    max_samples = params.get("max_samples", None)
    if max_samples is not None:
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


def _load_credit_card_fraud(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Credit Card Fraud Detection dataset.

    284,807 samples, 30 PCA-transformed features, binary (highly imbalanced).
    Loaded from numpy cache.  Cache must be generated once using:

        python -m qie_research.datasets.prepare_credit_card_fraud

    The raw CSV is available from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    Config keys
    -----------
    data_home : str, default "data/raw/"
    max_samples : int, optional
    seed : int, default 42
    """
    data_home = Path(params.get("data_home", "data/raw/"))
    cache_X = data_home / "credit_card_fraud_X.npy"
    cache_y = data_home / "credit_card_fraud_y.npy"

    if not cache_X.exists() or not cache_y.exists():
        raise FileNotFoundError(
            f"Credit Card Fraud cache not found at {data_home}. "
            "Run: python -m qie_research.datasets.prepare_credit_card_fraud"
        )

    X = np.load(cache_X)
    y = np.load(cache_y)

    max_samples = params.get("max_samples", None)
    if max_samples is not None:
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


def _load_cifar10(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    CIFAR-10 dataset, flattened to 3072-dimensional vectors.

    60,000 samples, 3,072 features (32x32x3 flattened), 10 classes.
    Pixel values normalised to [0, 1].
    Loaded from numpy cache.  Cache must be generated once using:

        python -m qie_research.datasets.prepare_cifar10

    Config keys
    -----------
    data_home : str, default "data/raw/"
    max_samples : int, optional
    seed : int, default 42
    """
    data_home = Path(params.get("data_home", "data/raw/"))
    cache_X = data_home / "cifar10_X.npy"
    cache_y = data_home / "cifar10_y.npy"

    if not cache_X.exists() or not cache_y.exists():
        raise FileNotFoundError(
            f"CIFAR-10 cache not found at {data_home}. "
            "Run: python -m qie_research.datasets.prepare_cifar10"
        )

    X = np.load(cache_X)
    y = np.load(cache_y)

    max_samples = params.get("max_samples", None)
    if max_samples is not None:
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


def _load_higgs(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    HIGGS dataset (500k subset).

    500,000 samples, 21 features, binary classification.
    Canonical large-scale benchmark in the quantum ML literature.
    Loaded from numpy cache.  Cache must be generated once using:

        python -m qie_research.datasets.prepare_higgs

    The raw dataset is available from the UCI ML Repository:
    https://archive.ics.uci.edu/dataset/280/higgs

    Config keys
    -----------
    data_home : str, default "data/raw/"
    max_samples : int, default 500000
    seed : int, default 42
    """
    data_home = Path(params.get("data_home", "data/raw/"))
    cache_X = data_home / "higgs_X.npy"
    cache_y = data_home / "higgs_y.npy"

    if not cache_X.exists() or not cache_y.exists():
        raise FileNotFoundError(
            f"HIGGS cache not found at {data_home}. "
            "Run: python -m qie_research.datasets.prepare_higgs"
        )

    X = np.load(cache_X)
    y = np.load(cache_y)

    max_samples = params.get("max_samples", 500_000)
    if max_samples is not None and int(max_samples) < len(X):
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


def _load_covertype(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Forest Covertype dataset.

    581,012 samples, 54 features, 7 classes.
    Loaded via sklearn fetch_covtype — no manual download required.
    sklearn caches the download automatically.

    Config keys
    -----------
    max_samples : int, optional
    seed : int, default 42
    """
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    X = data.data.astype(float)
    y = (data.target - 1).astype(int)   # sklearn returns 1-indexed labels

    max_samples = params.get("max_samples", None)
    if max_samples is not None:
        rng = np.random.default_rng(params.get("seed", 42))
        idx = rng.choice(len(X), size=int(max_samples), replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    return X, y


DATASET_REGISTRY: dict[str, callable] = {
    "wine": _load_wine,
    "breast_cancer": _load_breast_cancer,
    "dry_bean": _load_dry_bean,
    "credit_card_fraud": _load_credit_card_fraud,
    "fashion_mnist": _load_fashion_mnist,
    "cifar10": _load_cifar10,
    "higgs": _load_higgs,
    "covertype": _load_covertype,
    "high_dim_parity": _load_high_dim_parity,
    "high_rank_noise": _load_high_rank_noise,
}


# Model registry

def _build_logistic_regression(params: dict):
    return LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        C=params.get("C", 1.0),
        random_state=params.get("_seed"),  # injected by runner
    )


MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": _build_logistic_regression,
}


# Seed control

def _set_seeds(seed: int) -> None:
    """Set all relevant random seeds before any data or model operations."""
    random.seed(seed)
    np.random.seed(seed)


# Timed + memory-tracked encode step

def _encode(
    encoder,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fit on X_train and transform both splits.

    Returns
    -------
    X_train_enc, X_test_enc, elapsed_seconds, peak_memory_bytes
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    encoder.fit(X_train)
    X_train_enc = encoder.transform(X_train)
    X_test_enc = encoder.transform(X_test)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X_train_enc, X_test_enc, elapsed, peak


# Timed + memory-tracked train + evaluate step

def _train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, float, int]:
    """
    Fit model on training data and evaluate on test data.

    Returns
    -------
    metrics, elapsed_seconds, peak_memory_bytes
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 6),
    }
    return metrics, elapsed, peak


# Main runner

def run(config_path: str | Path) -> dict:
    """
    Execute a full benchmark run from a YAML config file.

    Parameters
    ----------
    config_path : str or Path

    Returns
    -------
    results : dict
        The full results dictionary, also written to disk as JSON.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    # Seed control — must happen before anything else
    seed = cfg["run"]["seed"]
    _set_seeds(seed)

    # 2. Load dataset and split into train/test
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    X, y = DATASET_REGISTRY[dataset_name](dataset_cfg)
    test_size = dataset_cfg.get("test_size", 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    dataset_info = {
        "name": dataset_name,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }

    # Run each encoding
    encoding_results = []

    for enc_cfg in cfg["encodings"]:
        enc_name = enc_cfg["name"]

        if enc_name not in ENCODING_REGISTRY:
            raise ValueError(
                f"Unknown encoding '{enc_name}'. "
                f"Available: {list(ENCODING_REGISTRY.keys())}"
            )

        # Build encoder, pass all config keys except 'name'
        enc_params = {k: v for k, v in enc_cfg.items() if k != "name"}
        encoder = ENCODING_REGISTRY[enc_name](**enc_params)

        # Encode
        X_train_enc, X_test_enc, enc_time, enc_mem = _encode(
            encoder, X_train, X_test
        )

        # Build model, inject seed for reproducibility
        model_cfg = cfg["model"]
        model_params = {k: v for k, v in model_cfg.items() if k != "name"}
        model_params["_seed"] = seed
        model = MODEL_REGISTRY[model_cfg["name"]](model_params)

        # Train and evaluate
        metrics, train_time, train_mem = _train_and_evaluate(
            model, X_train_enc, y_train, X_test_enc, y_test
        )

        encoding_results.append({
            "encoding": enc_name,
            "encoding_params": enc_params,
            "input_dim": int(X_train.shape[1]),
            "output_dim": int(encoder.output_dim_),
            "metrics": metrics,
            "timing_seconds": {
                "encoding": round(enc_time, 6),
                "training": round(train_time, 6),
                "total": round(enc_time + train_time, 6),
            },
            "memory_bytes": {
                "encoding_peak": enc_mem,
                "training_peak": train_mem,
            },
        })

    # Assemble and write results
    output = {
        "run": {
            "name": cfg["run"]["name"],
            "seed": seed,
            "config_path": str(config_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "dataset": dataset_info,
        "results": encoding_results,
    }

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg['run']['name']}.json"

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {output_path}")
    return output


# CLI entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a QIE benchmark from a YAML config file."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file (e.g. configs/smoke_test.yaml).",
    )
    args = parser.parse_args()

    results = run(args.config)

    print("\nSummary")
    print("-" * 50)
    for r in results["results"]:
        print(
            f"  {r['encoding']:<12}"
            f"  accuracy={r['metrics']['accuracy']:.4f}"
            f"  f1={r['metrics']['f1_macro']:.4f}"
            f"  enc={r['timing_seconds']['encoding']:.4f}s"
            f"  train={r['timing_seconds']['training']:.4f}s"
            f"  d_out={r['output_dim']}"
        )


if __name__ == "__main__":
    main()
