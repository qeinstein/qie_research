"""
Prepare Fashion-MNIST Cache

One-time script that parses the Fashion-MNIST arff.gz file already
downloaded by fetch_openml and writes compact numpy cache files.

This is required because the liac-arff parser needs more memory than
is available on development machines with limited RAM.  The numpy
cache loads in under a second and uses ~430MB on disk.

Usage
-----
    python -m qie_research.datasets.prepare_fashion_mnist

Output
------
    data/raw/fashion_mnist_X.npy   float32, shape (70000, 784), values in [0, 1]
    data/raw/fashion_mnist_y.npy   int32,   shape (70000,),     values in {0..9}
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

ARFF_GZ_PATH = Path(
    "data/raw/openml/openml.org/data/v1/download/18238735/Fashion-MNIST.arff.gz"
)
CACHE_X = Path("data/raw/fashion_mnist_X.npy")
CACHE_Y = Path("data/raw/fashion_mnist_y.npy")

# Fashion-MNIST class name to integer index
CLASS_MAP = {
    "T-shirt/top": 0,
    "Trouser": 1,
    "Pullover": 2,
    "Dress": 3,
    "Coat": 4,
    "Sandal": 5,
    "Shirt": 6,
    "Sneaker": 7,
    "Bag": 8,
    "Ankle boot": 9,
}


def prepare(
    arff_gz_path: Path = ARFF_GZ_PATH,
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    chunk_size: int = 5000,
) -> None:
    """
    Parse the arff.gz file in streaming chunks and write numpy cache.

    Parameters
    ----------
    arff_gz_path : Path
        Path to the downloaded Fashion-MNIST.arff.gz file.
    cache_x : Path
        Output path for the feature matrix (.npy).
    cache_y : Path
        Output path for the label vector (.npy).
    chunk_size : int
        Number of data lines to hold in memory at once.
    """
    if not arff_gz_path.exists():
        raise FileNotFoundError(
            f"arff.gz not found at {arff_gz_path}. "
            "Run fetch_openml first to download Fashion-MNIST."
        )

    print(f"Parsing {arff_gz_path} ...")

    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    in_data = False
    n_parsed = 0

    with gzip.open(arff_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            if line.upper() == "@DATA":
                in_data = True
                continue

            if not in_data:
                continue

            parts = line.split(",")
            # Last column is the class label
            label_str = parts[-1].strip().strip("'\"")
            features = np.array(parts[:-1], dtype=np.float32) / 255.0

            # Map label string to integer
            if label_str in CLASS_MAP:
                label = CLASS_MAP[label_str]
            else:
                # Fallback: try to parse as integer directly
                try:
                    label = int(label_str)
                except ValueError:
                    raise ValueError(
                        f"Unknown class label: '{label_str}'. "
                        f"Expected one of {list(CLASS_MAP.keys())} or an integer."
                    )

            rows_X.append(features)
            rows_y.append(label)
            n_parsed += 1

            if n_parsed % chunk_size == 0:
                print(f"  Parsed {n_parsed} samples...", end="\r")

    print(f"\nTotal samples parsed: {n_parsed}")

    X = np.stack(rows_X, axis=0)   # (n, 784)
    y = np.array(rows_y, dtype=np.int32)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print("Done.")


if __name__ == "__main__":
    prepare()
