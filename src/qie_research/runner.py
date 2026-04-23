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
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

    return X, y

def _load_higgs(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    HIGGS dataset (500k subset).

    500,000 samples, 28 features (21 low-level + 7 high-level derived),
    binary classification.
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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

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
        X, y = _stratified_subsample(X, y, int(max_samples), params.get("seed", 42))

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

# Feature-map registry (for classical baselines that use an explicit transform)

def _build_rff_map(params: dict, n_components_auto: int, seed: int,
                   n_features: int = 1):
    """
    Random Fourier Features approximation of an RBF kernel.

    Pipeline: StandardScaler → RBFSampler.  Scaling first means the gamma
    heuristic makes sense: ``gamma='auto'`` (the default) sets
    ``gamma = 1 / n_features`` which is equivalent to sklearn's ``'scale'``
    heuristic on standardised data and gives a kernel that is neither
    vanishingly narrow nor trivially flat.

    n_components defaults to n_components_auto (mean QIE d_out for the run)
    when the config specifies ``n_components: auto`` or omits the key.
    """
    n = params.get("n_components", "auto")
    if n == "auto" or n is None:
        n = n_components_auto
    gamma_cfg = params.get("gamma", "auto")
    if gamma_cfg == "auto" or gamma_cfg is None:
        gamma = 1.0 / max(n_features, 1)
    else:
        gamma = float(gamma_cfg)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rff", RBFSampler(n_components=int(n), gamma=gamma, random_state=seed)),
    ])

def _build_poly_map(params: dict, n_components_auto: int, seed: int,
                    n_features: int = 1):
    """
    Polynomial feature expansion followed by standardisation.

    Pipeline: PolynomialFeatures → StandardScaler.  The scaler is essential:
    cross-terms and powers have wildly different magnitudes without it, which
    causes the downstream logistic regression to fail to converge.

    degree defaults to 2.  include_bias=False avoids collinearity with the
    logistic regression intercept term.
    """
    degree = int(params.get("degree", 2))
    interaction_only = bool(params.get("interaction_only", False))
    return Pipeline([
        ("poly", PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False,
        )),
        ("scaler", StandardScaler()),
    ])

def _build_pca_map(params: dict, n_components_auto: int, seed: int,
                   n_features: int = 1):
    """
    PCA projection as a learned linear embedding baseline.

    n_components defaults to n_components_auto when the config specifies
    ``n_components: auto`` or omits the key.  The runner caps the value at
    min(n_features - 1, n_samples - 1) before constructing this object so
    sklearn never receives an infeasible request.
    """
    n = params.get("n_components", "auto")
    if n == "auto" or n is None:
        n = n_components_auto
    return PCA(n_components=int(n), random_state=seed)

def _build_scaler_map(params: dict, n_components_auto: int, seed: int,
                      n_features: int = 1):
    """
    StandardScaler as a standalone feature map.

    Used for the 'raw_linear' baseline (scaled linear): applies mean-variance
    normalisation to raw features before logistic regression.  This is the
    'scaled linear' comparator required by the Phase 0 scope lock.
    """
    return StandardScaler()

FEATURE_MAP_REGISTRY: dict[str, callable] = {
    "scaler": _build_scaler_map,
    "rff": _build_rff_map,
    "polynomial": _build_poly_map,
    "pca": _build_pca_map,
}

# Model registry

def _build_logistic_regression(params: dict):
    return LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        C=params.get("C", 1.0),
        random_state=params.get("_seed"),  # injected by runner
    )

def _build_rbf_svm(params: dict):
    """
    SVM with RBF kernel.

    Pipeline: StandardScaler → SVC.  Scaling is mandatory for gamma='scale'
    to have the intended effect.  For large datasets, pass ``max_samples``
    in the baseline config block to subsample before training.
    """
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C, gamma=gamma,
                    random_state=params.get("_seed"))),
    ])

def _build_mlp(params: dict):
    """
    Multi-layer perceptron baseline.

    Pipeline: StandardScaler → MLPClassifier.  early_stopping=True prevents
    runaway training on large datasets.  hidden_layer_sizes defaults to
    [256, 128] — a non-trivially shallow architecture that is not
    intentionally underpowered.
    """
    hidden = params.get("hidden_layer_sizes", [256, 128])
    max_iter = int(params.get("max_iter", 500))
    alpha = float(params.get("alpha", 1e-4))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(hidden),
            max_iter=max_iter,
            alpha=alpha,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=params.get("_seed"),
        )),
    ])

MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": _build_logistic_regression,
    "rbf_svm": _build_rbf_svm,
    "mlp": _build_mlp,
}
TORCH_BASELINE_MODELS = {"torch_mlp"}

# Seed control

def _set_seeds(seed: int) -> None:
    """Set all relevant random seeds before any data or model operations."""
    random.seed(seed)
    np.random.seed(seed)  # covers sklearn internals that use the legacy global state
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# Stratified subsampling helper

def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a stratified random subsample of (X, y) with exactly max_samples rows.

    Stratification preserves class proportions, which is critical for
    imbalanced datasets (e.g. credit_card_fraud with ~0.17% positive rate).
    Falls back to simple random sampling if stratified splitting is infeasible
    (e.g. a class has fewer than 2 samples).
    """
    if max_samples >= len(X):
        return X, y
    keep_frac = max_samples / len(X)
    try:
        _, X_sub, _, y_sub = train_test_split(
            X, y, test_size=keep_frac, random_state=seed, stratify=y
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    return X_sub, y_sub

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

# Classical baseline runner

def _run_baseline(
    baseline_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_components_auto: int,
) -> dict:
    """
    Run one classical baseline entry from the config ``baselines`` list.

    A baseline may optionally include a ``feature_map`` block (for RFF,
    polynomial features, or PCA) followed by a ``model`` block, or just a
    ``model`` block applied directly to raw features.

    The ``n_components_auto`` value is the mean output dimensionality of all
    QIE encodings for this run; it is used when a feature map specifies
    ``n_components: auto``, providing the matched-budget dimension.

    An optional ``max_samples`` key in the baseline block subsamples the
    training data before the model fit step, which is required for SVM on
    datasets with hundreds of thousands of rows.

    Returns
    -------
    dict with keys: name, feature_map, feature_map_params, model,
    model_params, input_dim, feature_dim, subsampled_train_n, metrics,
    timing_seconds, memory_bytes.
    """
    baseline_name = baseline_cfg["name"]
    feature_map_cfg = baseline_cfg.get("feature_map", None)
    model_cfg = baseline_cfg["model"]

    if feature_map_cfg is not None:
        fm_name = feature_map_cfg["name"]
        fm_params = {k: v for k, v in feature_map_cfg.items() if k != "name"}

        if fm_name not in FEATURE_MAP_REGISTRY:
            raise ValueError(
                f"Unknown feature map '{fm_name}'. "
                f"Available: {list(FEATURE_MAP_REGISTRY.keys())}"
            )

        # PCA: cap n_components so sklearn never receives an infeasible value.
        auto = n_components_auto
        if fm_name == "pca":
            auto = min(n_components_auto,
                       X_train.shape[1] - 1,
                       X_train.shape[0] - 1)

        feature_map = FEATURE_MAP_REGISTRY[fm_name](
            fm_params, auto, seed, n_features=int(X_train.shape[1])
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        feature_map.fit(X_train)
        X_train_mapped = feature_map.transform(X_train)
        X_test_mapped = feature_map.transform(X_test)
        fm_elapsed = time.perf_counter() - t0
        _, fm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        feature_dim = int(X_train_mapped.shape[1])
    else:
        X_train_mapped, X_test_mapped = X_train, X_test
        fm_elapsed, fm_peak = 0.0, 0
        feature_dim = int(X_train.shape[1])
        fm_name = None
        fm_params = {}

    max_samples = baseline_cfg.get("max_samples", None)
    X_tr, y_tr = X_train_mapped, y_train
    subsampled_n = None
    if max_samples is not None and len(X_tr) > int(max_samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), size=int(max_samples), replace=False)
        idx.sort()
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        subsampled_n = int(max_samples)

    # Special case: torch_mlp uses the PyTorch training path instead of
    # sklearn.  It handles its own scaling, subsampling, and curve logging.
    if model_cfg["name"] == "torch_mlp":
        from qie_research.models.torch_trainer import train_mlp
        torch_params = {k: v for k, v in model_cfg.items() if k != "name"}
        # Honour max_samples from the baseline block (already computed above).
        if subsampled_n is not None:
            torch_params.setdefault("max_samples", subsampled_n)
        # Pass mapped features (X_train_mapped) rather than raw X_train so that
        # any configured feature_map is actually applied.  When no feature_map is
        # configured, X_train_mapped is X_train (same object from the else branch
        # above), so this is a no-op for the common case.
        # Note: train_mlp applies StandardScaler internally.  If the feature_map
        # pipeline already includes a scaler, the features will be scaled twice.
        # Avoid pairing torch_mlp with a scaler-based feature_map in configs.
        result = train_mlp(
            X_train_mapped, y_train, X_test_mapped, y_test,
            seed=seed,
            hidden_layer_sizes=torch_params.pop("hidden_layer_sizes", [256, 128]),
            n_epochs=int(torch_params.pop("epochs", 100)),
            lr=float(torch_params.pop("lr", 1e-3)),
            weight_decay=float(torch_params.pop("weight_decay", 1e-4)),
            batch_size=int(torch_params.pop("batch_size", 256)),
            max_samples=torch_params.pop("max_samples", None),
        )
        return {
            "name": baseline_name,
            "feature_map": None,
            "feature_map_params": None,
            "model": "torch_mlp",
            "model_params": {k: v for k, v in model_cfg.items() if k != "name"},
            "input_dim": int(X_train.shape[1]),
            "feature_dim": int(X_train.shape[1]),
            "subsampled_train_n": result.pop("subsampled_train_n", None),
            **result,
        }

    if model_cfg["name"] not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_cfg['name']}'. "
            f"Available: {list(MODEL_REGISTRY.keys())} or 'torch_mlp'"
        )

    model_params = {k: v for k, v in model_cfg.items() if k != "name"}
    model_params["_seed"] = seed
    model = MODEL_REGISTRY[model_cfg["name"]](model_params)

    metrics, train_elapsed, train_peak = _train_and_evaluate(
        model, X_tr, y_tr, X_test_mapped, y_test
    )

    return {
        "name": baseline_name,
        "feature_map": fm_name,
        "feature_map_params": fm_params if fm_name else None,
        "model": model_cfg["name"],
        "model_params": {k: v for k, v in model_params.items() if k != "_seed"},
        "input_dim": int(X_train.shape[1]),
        "feature_dim": feature_dim,
        "subsampled_train_n": subsampled_n,
        "metrics": metrics,
        "training_curves": None,
        "timing_seconds": {
            "feature_map": round(fm_elapsed, 6),
            "training": round(train_elapsed, 6),
            "total": round(fm_elapsed + train_elapsed, 6),
        },
        "memory_bytes": {
            "feature_map_peak": fm_peak,
            "training_peak": train_peak,
        },
    }

# Main runner

def run(config_path: str | Path, torch_only: bool = False) -> dict:
    """
    Execute a full benchmark run from a YAML config file.

    Parameters
    ----------
    config_path : str or Path
    torch_only : bool
        If True, skip sklearn training paths.

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

        if not torch_only:
            # Build model, inject seed for reproducibility
            model_cfg = cfg["model"]
            model_params = {k: v for k, v in model_cfg.items() if k != "name"}
            model_params["_seed"] = seed
            model = MODEL_REGISTRY[model_cfg["name"]](model_params)

            # Train and evaluate (sklearn linear head)
            metrics, train_time, train_mem = _train_and_evaluate(
                model, X_train_enc, y_train, X_test_enc, y_test
            )
            sklearn_skipped = False
        else:
            # Skip sklearn training
            metrics = None
            train_time, train_mem = 0.0, 0
            sklearn_skipped = True

        # Activated when the config contains a top-level ``torch:`` block.
        # Trains nn.Linear on the frozen encoded features and records
        # per-epoch loss and gradient norm curves.
        torch_cfg = cfg["run"].get("torch", None)
        if torch_cfg is not None:
            from qie_research.models.torch_trainer import train_linear_head
            torch_result = train_linear_head(
                X_train_enc, y_train, X_test_enc, y_test,
                n_epochs=int(torch_cfg.get("epochs", 100)),
                lr=float(torch_cfg.get("lr", 1e-3)),
                weight_decay=float(torch_cfg.get("weight_decay", 1e-4)),
                batch_size=int(torch_cfg.get("batch_size", 256)),
                seed=seed,
                encoding_name=enc_name,
            )
        else:
            torch_result = None

        encoding_results.append({
            "encoding": enc_name,
            "encoding_params": enc_params,
            "input_dim": int(X_train.shape[1]),
            "output_dim": int(encoder.output_dim_),
            "metrics": metrics,
            "sklearn_skipped": sklearn_skipped,
            "timing_seconds": {
                "encoding": round(enc_time, 6),
                "training": round(train_time, 6),
                "total": round(enc_time + train_time, 6),
            },
            "memory_bytes": {
                "encoding_peak": enc_mem,
                "training_peak": train_mem,
            },
            "torch_linear_head": torch_result,
        })

    # n_components_auto is the median QIE output dimension for this run.
    # Median is used rather than mean because basis encoding's output dimension
    # (d * n_bits, e.g. 13*8=104 for wine) dominates an arithmetic mean and
    # inflates the matched budget far beyond amplitude encoding's output (16),
    # producing an unfair comparison for amplitude.  Median is more robust to
    # this skew across the three encoding scales.
    qie_d_outs = [r["output_dim"] for r in encoding_results]
    n_components_auto = int(np.median(qie_d_outs))

    baseline_results = []
    for bl_cfg in cfg.get("baselines", []):
        model_name = bl_cfg.get("model", {}).get("name") or ""
        if torch_only and model_name not in TORCH_BASELINE_MODELS:
            continue
        bl_result = _run_baseline(
            bl_cfg, X_train, y_train, X_test, y_test,
            seed=seed,
            n_components_auto=n_components_auto,
        )
        baseline_results.append(bl_result)

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
        "baselines": baseline_results,
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
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="Skip sklearn training paths and only run PyTorch differentiable paths.",
    )
    args = parser.parse_args()

    results = run(args.config, torch_only=args.torch_only)

    print("\nQIE Encodings")
    print("-" * 62)
    for r in results["results"]:
        metrics = r["metrics"]
        metric_source = "sklearn"
        if metrics is None and r["torch_linear_head"] is not None:
            metrics = r["torch_linear_head"].get("metrics")
            metric_source = "torch_linear_head"
        elif metrics is None:
            metric_source = "none"

        if metrics is None:
            acc_text = "n/a"
            f1_text = "n/a"
        else:
            acc_text = f"{metrics['accuracy']:.4f}"
            f1_text = f"{metrics['f1_macro']:.4f}"

        print(
            f"  {r['encoding']:<12}"
            f"  accuracy={acc_text}"
            f"  f1={f1_text}"
            f"  source={metric_source}"
            f"  enc={r['timing_seconds']['encoding']:.4f}s"
            f"  train={r['timing_seconds']['training']:.4f}s"
            f"  d_out={r['output_dim']}"
        )

    if results["baselines"]:
        print("\nClassical Baselines")
        print("-" * 62)
        for b in results["baselines"]:
            sub = f"  (subsample={b['subsampled_train_n']})" if b.get("subsampled_train_n") else ""
            t = b["timing_seconds"]
            total = t if isinstance(t, (int, float)) else t["total"]
            print(
                f"  {b['name']:<18}"
                f"  accuracy={b['metrics']['accuracy']:.4f}"
                f"  f1={b['metrics']['f1_macro']:.4f}"
                f"  total={total:.4f}s"
                f"  d_feat={b['feature_dim']}"
                f"{sub}"
            )

if __name__ == "__main__":
    main()
