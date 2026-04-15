"""
Prepare Credit Card Fraud Cache
================================
One-time script that reads the Credit Card Fraud Detection CSV and writes
numpy cache files.

The dataset contains 284,807 transactions with 30 PCA-transformed features
and a binary label (0 = legitimate, 1 = fraud).  It is highly imbalanced
(0.17% fraud).  The PCA preprocessing makes it scientifically interesting
for this benchmark because it tests whether QIE adds representational value
on top of an existing linear projection.

Source
------
Kaggle — ULB Machine Learning Group:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download instructions:
1. Log in to Kaggle
2. Download creditcard.csv from the link above
3. Place it at: data/raw/creditcard.csv

Usage
-----
    python -m qie_research.datasets.prepare_credit_card_fraud

Output
------
    data/raw/credit_card_fraud_X.npy   float64, shape (284807, 30)
    data/raw/credit_card_fraud_y.npy   int32,   shape (284807,), values in {0, 1}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RAW_CSV = Path("data/raw/creditcard.csv")
CACHE_X = Path("data/raw/credit_card_fraud_X.npy")
CACHE_Y = Path("data/raw/credit_card_fraud_y.npy")


def prepare(
    raw_csv: Path = RAW_CSV,
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to prepare Credit Card Fraud. "
            "Install with: pip install pandas"
        )

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {raw_csv}.\n"
            "Download creditcard.csv from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it at: {raw_csv}"
        )

    print(f"Reading {raw_csv} ...")
    df = pd.read_csv(raw_csv)

    # Features: V1-V28 (PCA components) + Time + Amount
    # Drop Time as it is not a meaningful feature for this benchmark
    feature_cols = [c for c in df.columns if c not in ("Time", "Class")]
    X = df[feature_cols].to_numpy().astype(float)
    y = df["Class"].to_numpy().astype(np.int32)

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
