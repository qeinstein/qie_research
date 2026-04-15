"""
Prepare Covertype Cache
=======================
One-time script that downloads the Forest Cover Type dataset via scikit-learn
and writes numpy cache files.

The Covertype dataset contains 581,012 samples from 30m × 30m forest patches
in the Roosevelt National Forest of northern Colorado.  Each sample has 54
features (10 quantitative cartographic variables + 4 binary wilderness area
indicators + 40 binary soil type indicators) and a 7-class label indicating
the dominant tree species (Cover_Type 1–7).

This dataset tests QIE on a large-scale multiclass problem with mixed
continuous and binary features, which is scientifically distinct from the
purely continuous tabular benchmarks in the suite.

Source
------
Downloaded automatically via sklearn.datasets.fetch_covtype.  The dataset
is cached by scikit-learn in ~/scikit_learn_data/ after the first download.

Usage
-----
    python -m qie_research.datasets.prepare_covertype

Output
------
    data/raw/covertype_X.npy   float64, shape (581012, 54)
    data/raw/covertype_y.npy   int32,   shape (581012,), values in {0..6}

Notes
-----
scikit-learn returns labels in {1..7}.  We subtract 1 to shift to {0..6} so
that the label set is zero-indexed and consistent with the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/covertype_X.npy")
CACHE_Y = Path("data/raw/covertype_y.npy")


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
) -> None:
    from sklearn.datasets import fetch_covtype

    print("Downloading Covertype via scikit-learn (cached after first download)...")
    bunch = fetch_covtype()

    X = bunch.data.astype(float)
    # sklearn returns 1-indexed labels {1..7}; shift to {0..6}
    y = (bunch.target - 1).astype(np.int32)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    classes = sorted(set(y.tolist()))
    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Classes: {classes}  ({len(classes)} total)")
    print("Done.")


if __name__ == "__main__":
    prepare()
