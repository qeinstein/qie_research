"""
Phase 2 Numerical Audit

Audits the numerical behaviour of all three encodings before any training
or performance claims are made.  Produces a single JSON file consumed by
the Phase 2 LaTeX report.

Checks performed
----------------
Per encoding:
  - Normalization / structural invariant verification
  - Output shape and dimensionality
  - Singular value distribution of the encoded feature matrix
  - Effective rank (erank)
  - Condition number (kappa)
  - Noise stability under epsilon ~ N(0, 0.01)

Formulae
--------
Effective rank:
    erank(X) = exp( -sum_i p_i * log(p_i) ),  p_i = sigma_i / sum_j sigma_j

Condition number:
    kappa = sigma_max / sigma_min   (over non-zero singular values)

Noise stability:
    delta = ||phi(X + eps) - phi(X)||_F / n_samples,  eps ~ N(0, 0.01)

Usage
-----
    python -m qie_research.analysis.numerical_audit
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from qie_research.encodings import ENCODING_REGISTRY

# Constants

SEED = 42
NOISE_STD = 0.01
OUTPUT_PATH = Path("results/metrics/phase_2_numerical_audit.json")

ENCODING_CONFIGS: list[dict] = [
    {"name": "amplitude", "pad_to_power_of_two": True},
    {"name": "angle", "scale": 3.141592653589793, "standardize": True},
    {"name": "basis", "n_bits": 8},
]


# Seed control

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# Spectral diagnostics

def _singular_values(X_enc: np.ndarray) -> np.ndarray:
    """Return singular values of X_enc in descending order."""
    _, sigma, _ = np.linalg.svd(X_enc, full_matrices=False)
    return sigma


def _effective_rank(sigma: np.ndarray) -> float:
    """
    Effective rank via the entropy of normalised singular values.

        erank = exp( -sum_i p_i * log(p_i) ),  p_i = sigma_i / sum(sigma)

    Returns 1.0 for a rank-1 matrix and len(sigma) for a flat spectrum.
    """
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def _condition_number(sigma: np.ndarray) -> float:
    """
    Condition number: ratio of largest to smallest non-zero singular value.

        kappa = sigma_max / sigma_min
    """
    sigma_nonzero = sigma[sigma > 0]
    if len(sigma_nonzero) < 2:
        return float("inf")
    return float(sigma_nonzero[0] / sigma_nonzero[-1])


# Noise stability

def _noise_stability(
    encoder,
    X: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> float:
    """
    Mean per-sample Frobenius distance between clean and noisy encodings.

        delta = ||phi(X + eps) - phi(X)||_F / n_samples,  eps ~ N(0, noise_std)
    """
    eps = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    X_enc_clean = encoder.transform(X)
    X_enc_noisy = encoder.transform(X + eps)
    diff = X_enc_clean - X_enc_noisy
    return float(np.linalg.norm(diff, "fro") / X.shape[0])


# Per-encoding invariant checks

def _verify_invariants(enc_name: str, encoder, X_enc: np.ndarray) -> dict:
    """
    Run the encoding-specific structural invariant check.

    Returns a dict with keys: passed (bool), details (str).
    """
    if enc_name == "amplitude":
        passed = encoder.verify_normalization(X_enc)
        details = "sum_i x_i^2 == 1 for all rows" if passed else "FAILED: rows are not unit norm"

    elif enc_name == "angle":
        passed = encoder.verify_unit_circle(X_enc)
        details = "cos^2 + sin^2 == 1 for all pairs" if passed else "FAILED: pairs not on unit circle"

    elif enc_name == "basis":
        passed = encoder.verify_binary(X_enc)
        details = "all values in {0, 1}" if passed else "FAILED: non-binary values found"

    else:
        passed = False
        details = f"No invariant check defined for '{enc_name}'"

    return {"passed": passed, "details": details}


# Main audit

def run_audit() -> dict:
    """
    Run the full numerical audit across all encodings.

    Returns
    -------
    report : dict
        Full audit results, also written to OUTPUT_PATH as JSON.
    """
    _set_seeds(SEED)
    rng = np.random.default_rng(SEED)

    # Load dataset
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    dataset_info = {
        "name": "wine",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "noise_std": NOISE_STD,
        "seed": SEED,
    }

    encoding_audits = []

    for enc_cfg in ENCODING_CONFIGS:
        enc_name = enc_cfg["name"]
        enc_params = {k: v for k, v in enc_cfg.items() if k != "name"}

        encoder = ENCODING_REGISTRY[enc_name](**enc_params)
        X_enc = encoder.fit_transform(X_train)

        # Shape
        n_samples, output_dim = X_enc.shape

        # Invariant check
        invariant = _verify_invariants(enc_name, encoder, X_enc)

        # Spectral diagnostics
        sigma = _singular_values(X_enc)
        erank = _effective_rank(sigma)
        kappa = _condition_number(sigma)

        # Noise stability
        stability = _noise_stability(encoder, X_train, NOISE_STD, rng)

        encoding_audits.append({
            "encoding": enc_name,
            "encoding_params": enc_params,
            "input_dim": int(X_train.shape[1]),
            "output_dim": int(output_dim),
            "n_samples": int(n_samples),
            "invariant_check": invariant,
            "singular_values": {
                "top_10": [round(float(s), 6) for s in sigma[:10]],
                "full_count": int(len(sigma)),
                "min": round(float(sigma[-1]), 6),
                "max": round(float(sigma[0]), 6),
            },
            "effective_rank": round(erank, 4),
            "condition_number": round(kappa, 4),
            "noise_stability": {
                "noise_std": NOISE_STD,
                "mean_frobenius_distance_per_sample": round(stability, 6),
            },
        })

    report = {
        "phase": 2,
        "title": "Numerical Behavior Audit",
        "dataset": dataset_info,
        "encodings": encoding_audits,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Audit written to {OUTPUT_PATH}")
    return report


# CLI entry point

def main() -> None:
    report = run_audit()

    print("\nNumerical Audit Summary")
    print("-" * 60)
    for enc in report["encodings"]:
        inv = enc["invariant_check"]
        status = "PASS" if inv["passed"] else "FAIL"
        print(
            f"  {enc['encoding']:<12}"
            f"  invariant={status}"
            f"  erank={enc['effective_rank']:.2f}"
            f"  kappa={enc['condition_number']:.2f}"
            f"  noise_delta={enc['noise_stability']['mean_frobenius_distance_per_sample']:.6f}"
            f"  d_out={enc['output_dim']}"
        )


if __name__ == "__main__":
    main()
