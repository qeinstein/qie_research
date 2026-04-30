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

import gzip
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/covertype_X.npy")
CACHE_Y = Path("data/raw/covertype_y.npy")
DOWNLOAD_DIR = Path("data/raw/covertype_raw")

# Stable figshare mirror (~11 MB gzip) — same source sklearn uses internally
_COVTYPE_URL = "https://ndownloader.figshare.com/files/5976039"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {url} ...")
    if shutil.which("wget"):
        print("  Using wget...")
        try:
            subprocess.run([
                "wget", "-c", "-t", "10", "-T", "15", "--waitretry", "5",
                url, "-O", str(dest), "--show-progress",
            ], check=True)
            print("  Download complete.")
            return
        except subprocess.CalledProcessError:
            print("  wget failed. Trying urllib fallback...")

    def _progress(block_num, block_size, total_size):
        mb = block_num * block_size / 1_048_576
        if total_size > 0:
            pct = min(100, 100 * block_num * block_size / total_size)
            print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB / {total_size/1_048_576:.1f} MB)",
                  end="", flush=True)
        else:
            print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print("\n  Download complete.")


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    download_dir: Path = DOWNLOAD_DIR,
) -> None:
    if cache_x.exists() and cache_y.exists():
        print("Cache already exists — skipping download.")
        return

    raw_gz = download_dir / "covtype.data.gz"
    if not raw_gz.exists():
        _download(_COVTYPE_URL, raw_gz)

    print("  Parsing covtype.data.gz ...")
    with gzip.open(raw_gz, "rt") as f:
        rows = [list(map(float, line.split(","))) for line in f if line.strip()]

    arr = np.array(rows, dtype=np.float64)
    X = arr[:, :-1]
    # sklearn convention: labels 1..7 → shift to 0..6
    y = (arr[:, -1].astype(np.int32) - 1)

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


if __name__ == "__main__":
    prepare()
