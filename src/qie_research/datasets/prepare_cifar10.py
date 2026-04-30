"""
Prepare CIFAR-10 Cache
======================
One-time script that downloads CIFAR-10 and writes flattened numpy cache files.

CIFAR-10 contains 60,000 32x32 colour images across 10 classes.  Images are
flattened to 3,072-dimensional vectors (32x32x3) and pixel values are
normalised to [0, 1].  It is a harder image benchmark than Fashion-MNIST due
to richer texture and colour structure, and is expected by reviewers at top
venues alongside Fashion-MNIST.

The dataset is downloaded automatically via torchvision or keras depending
on what is available.  torchvision is tried first.

Usage
-----
    python -m qie_research.datasets.prepare_cifar10

Output
------
    data/raw/cifar10_X.npy   float32, shape (60000, 3072), values in [0, 1]
    data/raw/cifar10_y.npy   int32,   shape (60000,), values in {0..9}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/cifar10_X.npy")
CACHE_Y = Path("data/raw/cifar10_y.npy")
DOWNLOAD_DIR = Path("data/raw/cifar10_raw/")


def _download_via_torchvision(download_dir: Path):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.ToTensor()
    train = torchvision.datasets.CIFAR10(
        root=str(download_dir), train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        root=str(download_dir), train=False, download=True, transform=transform
    )

    # Stack train + test
    X_train = np.stack([img.numpy() for img, _ in train])  # (50000, 3, 32, 32)
    y_train = np.array([lbl for _, lbl in train], dtype=np.int32)
    X_test = np.stack([img.numpy() for img, _ in test])    # (10000, 3, 32, 32)
    y_test = np.array([lbl for _, lbl in test], dtype=np.int32)

    X = np.concatenate([X_train, X_test], axis=0)          # (60000, 3, 32, 32)
    y = np.concatenate([y_train, y_test], axis=0)

    # Transpose to (N, H, W, C) then flatten to (N, 3072)
    X = X.transpose(0, 2, 3, 1).reshape(len(X), -1).astype(np.float32)

    return X, y


def _download_via_keras(download_dir: Path):
    import tensorflow as tf

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32) / 255.0
    y = np.concatenate([y_train.ravel(), y_test.ravel()]).astype(np.int32)
    X = X.reshape(len(X), -1)   # flatten to (60000, 3072)

    return X, y


def _download_via_urllib(download_dir: Path):
    import pickle
    import tarfile
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    archive = download_dir / "cifar-10-python.tar.gz"

    if not archive.exists():
        print(f"Downloading CIFAR-10 (~170 MB) from {url} ...")
        urllib.request.urlretrieve(url, str(archive))

    print("Extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(download_dir)

    batches_dir = download_dir / "cifar-10-batches-py"
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]

    X_parts, y_parts = [], []
    for fname in batch_files:
        with open(batches_dir / fname, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        X_parts.append(batch[b"data"])   # (10000, 3072) uint8
        y_parts.extend(batch[b"labels"])

    X = np.concatenate(X_parts, axis=0).astype(np.float32) / 255.0
    y = np.array(y_parts, dtype=np.int32)
    return X, y


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    download_dir: Path = DOWNLOAD_DIR,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)

    # Try torchvision → keras → pure urllib (no ML framework required)
    try:
        print("Downloading CIFAR-10 via torchvision...")
        X, y = _download_via_torchvision(download_dir)
    except ImportError:
        try:
            print("torchvision not found. Trying keras/tensorflow...")
            X, y = _download_via_keras(download_dir)
        except ImportError:
            print("torchvision/keras not found. Downloading via urllib (no ML framework needed)...")
            X, y = _download_via_urllib(download_dir)

    # Normalise to [0, 1] if not already done
    if X.max() > 1.0:
        X = X / 255.0

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Classes: {sorted(set(y.tolist()))}  ({len(set(y.tolist()))} total)")
    print("Done.")


if __name__ == "__main__":
    prepare()
