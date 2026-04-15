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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from qie_research.encodings import ENCODING_REGISTRY


# Dataset registry

def _load_wine(params: dict) -> tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    return data.data, data.target


DATASET_REGISTRY: dict[str, callable] = {
    "wine": _load_wine,
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
