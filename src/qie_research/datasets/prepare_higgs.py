"""
Prepare HIGGS Cache
===================
One-time script that downloads the HIGGS dataset from UCI and writes a
500 000-sample numpy cache.

The full HIGGS dataset contains 11 million samples of 28 features (21 low-level
kinematic properties + 7 high-level derived features) from simulated particle
physics collisions.  The binary label indicates signal (Higgs boson production,
1) versus background (0).  We use a 500 000-sample stratified subset to keep
training tractable on CPU while preserving the class balance and feature
distribution of the full dataset.

Source
------
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/280/higgs

The dataset is a gzip-compressed CSV with no header.  Column 0 is the label;
columns 1-28 are the features.

Download instructions
---------------------
The file is ~2.6 GB compressed.  Either:

Option A — automatic download (requires ~2.6 GB disk, ~6 GB peak RAM):
    python -m qie_research.datasets.prepare_higgs

Option B — manual download to avoid network dependency:
    1. Download HIGGS.csv.gz from the UCI link above
    2. Place it at: data/raw/higgs_raw/HIGGS.csv.gz
    3. Run: python -m qie_research.datasets.prepare_higgs

Usage
-----
    python -m qie_research.datasets.prepare_higgs

Output
------
    data/raw/higgs_X.npy   float32, shape (500000, 28)
    data/raw/higgs_y.npy   int32,   shape (500000,), values in {0, 1}
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/higgs_X.npy")
CACHE_Y = Path("data/raw/higgs_y.npy")
DOWNLOAD_DIR = Path("data/raw/higgs_raw")
RAW_GZ = DOWNLOAD_DIR / "HIGGS.csv.gz"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

N_SUBSET = 500_000
RANDOM_SEED = 42

N_COLS = 29  # col 0 = label, cols 1-28 = features


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {url} ...")
    
    # Use a real User-Agent to avoid being blocked by academic servers
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        mb = downloaded / 1_048_576
        if total_size > 0:
            pct = min(100, 100 * downloaded / total_size)
            print(f"\r  Progress: {pct:.1f}% ({mb:.0f} MB / {total_size/1_048_576:.0f} MB)", end="", flush=True)
        else:
            # Fallback if server doesn't provide Content-Length
            print(f"\r  Downloaded: {mb:.0f} MB ...", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print("\nDownload complete.")


def _load_gz_subset(gz_path: Path, n_subset: int, seed: int):
    """Stream the gzip CSV and return a stratified subset without loading all 11M rows."""
    print(f"Reading {gz_path} ...")
    print("  Streaming all rows to build stratified subset — may take a few minutes...")

    # Two-pass strategy: first pass collects indices per class, second pass reads data.
    # To avoid loading 11M × 29 floats (~1.2 GB float32) we use a reservoir sample.
    # We use numpy's random choice on per-class line indices collected in one stream pass.

    rng = np.random.default_rng(seed)

    # --- Pass 1: count rows per class and collect all row indices per class ---
    class0_idx: list[int] = []
    class1_idx: list[int] = []

    print("  Pass 1: Identifying class indices...")
    with gzip.open(gz_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 500000 == 0 and i > 0:
                print(f"    Scanning row {i:,} ...")
            label = line[0]  # first character is '0' or '1'
            if label == "0":
                class0_idx.append(i)
            else:
                class1_idx.append(i)

    n0 = len(class0_idx)
    n1 = len(class1_idx)
    total = n0 + n1
    print(f"  Total rows: {total:,}  (class 0: {n0:,}, class 1: {n1:,})")

    # Stratified subset: preserve class ratio
    frac = n_subset / total
    n_keep0 = round(frac * n0)
    n_keep1 = n_subset - n_keep0

    chosen0 = set(rng.choice(class0_idx, size=n_keep0, replace=False).tolist())
    chosen1 = set(rng.choice(class1_idx, size=n_keep1, replace=False).tolist())
    chosen = chosen0 | chosen1

    # --- Pass 2: read selected rows ---
    print(f"  Pass 2: Extracting {n_subset:,} rows...")
    X = np.empty((n_subset, N_COLS - 1), dtype=np.float32)
    y = np.empty(n_subset, dtype=np.int32)

    out_idx = 0
    with gzip.open(gz_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 1000000 == 0 and i > 0:
                pct = (i / total) * 100
                print(f"    Extracting: {pct:.0f}% complete ({i:,} rows scanned) ...")
            if i not in chosen:
                continue
            vals = line.rstrip("\n").split(",")
            y[out_idx] = int(float(vals[0]))
            X[out_idx] = [float(v) for v in vals[1:]]
            out_idx += 1
            if out_idx >= n_subset:
                break

    # Shuffle so rows are not grouped by class
    perm = rng.permutation(n_subset)
    return X[perm], y[perm]


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    download_dir: Path = DOWNLOAD_DIR,
    raw_gz: Path = RAW_GZ,
    n_subset: int = N_SUBSET,
    seed: int = RANDOM_SEED,
) -> None:
    if cache_x.exists() and cache_y.exists():
        print("Cache already exists — skipping download and processing.")
        return

    download_dir.mkdir(parents=True, exist_ok=True)

    if not raw_gz.exists():
        _download(UCI_URL, raw_gz)

    X, y = _load_gz_subset(raw_gz, n_subset=n_subset, seed=seed)

    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    n_signal = int((y == 1).sum())
    print(f"Saved X: {cache_x}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y: {cache_y}  shape={y.shape}  dtype={y.dtype}")
    print(f"Class balance: {n_subset - n_signal} background, {n_signal} signal "
          f"({100 * n_signal / n_subset:.2f}% signal)")
    print("Done.")


if __name__ == "__main__":
    prepare()
