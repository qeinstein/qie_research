"""
Prepare Fashion-MNIST Cache
===========================
One-time script that downloads Fashion-MNIST via torchvision and writes
compact numpy cache files.

Fashion-MNIST contains 70,000 28x28 greyscale images across 10 clothing
classes (train: 60,000; test: 10,000).  Pixel values are normalised to
[0, 1].  Images are flattened to 784-dimensional vectors.

Usage
-----
    python -m qie_research.datasets.prepare_fashion_mnist

Output
------
    data/raw/fashion_mnist_X.npy   float32, shape (70000, 784), values in [0, 1]
    data/raw/fashion_mnist_y.npy   int32,   shape (70000,),     values in {0..9}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/fashion_mnist_X.npy")
CACHE_Y = Path("data/raw/fashion_mnist_y.npy")
DOWNLOAD_DIR = Path("data/raw/fashion_mnist_raw/")


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    download_dir: Path = DOWNLOAD_DIR,
) -> None:
    if cache_x.exists() and cache_y.exists():
        print("Cache already exists — skipping download and processing.")
        return

    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision

        print("Downloading Fashion-MNIST via torchvision...")
        train = torchvision.datasets.FashionMNIST(
            root=str(download_dir), train=True, download=True
        )
        test = torchvision.datasets.FashionMNIST(
            root=str(download_dir), train=False, download=True
        )

        X_train = train.data.numpy().astype(np.float32) / 255.0   # (60000, 28, 28)
        y_train = train.targets.numpy().astype(np.int32)
        X_test = test.data.numpy().astype(np.float32) / 255.0     # (10000, 28, 28)
        y_test = test.targets.numpy().astype(np.int32)

        X = np.concatenate([X_train, X_test], axis=0)             # (70000, 28, 28)
        y = np.concatenate([y_train, y_test], axis=0)
        X = X.reshape(len(X), -1)                                  # (70000, 784)

    except ImportError:
        print("torchvision not found. Fetching Fashion-MNIST from OpenML instead...")
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(data_id=40996, as_frame=False, parser="auto")
        X = dataset.data.astype(np.float32) / 255.0
        y = dataset.target.astype(np.int32)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Classes: {sorted(set(y.tolist()))}  ({len(set(y.tolist()))} total)")
    print("Done.")


if __name__ == "__main__":
    prepare()
