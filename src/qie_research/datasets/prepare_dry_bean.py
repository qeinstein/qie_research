"""
Prepare Dry Bean Cache
======================
One-time script that downloads the UCI Dry Bean dataset and writes
numpy cache files.

The Dry Bean dataset contains 13,611 samples of 7 bean varieties
described by 16 morphological features derived from image processing.
It is a harder multiclass tabular benchmark than Wine or Breast Cancer,
testing QIE geometric properties under more classes and larger sample counts.

Source
------
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset is downloaded automatically via the ucimlrepo package.
Install it with: pip install ucimlrepo

Usage
-----
    python -m qie_research.datasets.prepare_dry_bean

Output
------
    data/raw/dry_bean_X.npy   float64, shape (13611, 16)
    data/raw/dry_bean_y.npy   int32,   shape (13611,), values in {0..6}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/dry_bean_X.npy")
CACHE_Y = Path("data/raw/dry_bean_y.npy")


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
) -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo is required to download Dry Bean. "
            "Install with: pip install ucimlrepo"
        )

    from sklearn.preprocessing import LabelEncoder

    print("Downloading Dry Bean from UCI ML Repository...")
    dataset = fetch_ucirepo(id=602)

    X = dataset.data.features.to_numpy().astype(float)
    y_raw = dataset.data.targets.to_numpy().ravel()
    y = LabelEncoder().fit_transform(y_raw).astype(np.int32)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Classes: {sorted(set(y.tolist()))}  ({len(set(y.tolist()))} total)")
    print("Done.")


if __name__ == "__main__":
    prepare()
