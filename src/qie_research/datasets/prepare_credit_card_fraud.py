"""
Prepare Credit Card Fraud Cache
================================
Downloads the Credit Card Fraud Detection dataset from OpenML (no credentials
required) and writes numpy cache files.

The dataset contains 284,807 transactions with 30 PCA-transformed features
and a binary label (0 = legitimate, 1 = fraud).  It is highly imbalanced
(0.17% fraud).  The PCA preprocessing makes it scientifically interesting
for this benchmark because it tests whether QIE adds representational value
on top of an existing linear projection.

Source
------
OpenML dataset 1597 — ULB Machine Learning Group:
https://www.openml.org/d/1597

Usage
-----
    python -m qie_research.datasets.prepare_credit_card_fraud

Output
------
    data/raw/credit_card_fraud_X.npy   float64, shape (284807, 29)
    data/raw/credit_card_fraud_y.npy   int32,   shape (284807,), values in {0, 1}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/credit_card_fraud_X.npy")
CACHE_Y = Path("data/raw/credit_card_fraud_y.npy")

# OpenML dataset ID for Credit Card Fraud Detection
_OPENML_ID = 1597


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
) -> None:
    if cache_x.exists() and cache_y.exists():
        print("Cache already exists — skipping download.")
        return

    from sklearn.datasets import fetch_openml

    print(f"Fetching Credit Card Fraud from OpenML (id={_OPENML_ID}) ...")
    dataset = fetch_openml(data_id=_OPENML_ID, as_frame=False, parser="auto")

    X = dataset.data.astype(float)

    # OpenML encodes the target as strings ('0' / '1')
    y = (dataset.target.astype(str) == "1").astype(np.int32)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    n_fraud = int(y.sum())
    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Class balance: {len(y) - n_fraud} legitimate, {n_fraud} fraud "
          f"({100 * n_fraud / len(y):.2f}% fraud)")
    print("Done.")


if __name__ == "__main__":
    prepare()
