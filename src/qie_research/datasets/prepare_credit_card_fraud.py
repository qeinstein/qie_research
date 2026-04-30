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

import gzip
import shutil
import subprocess
import urllib.request
import warnings
from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/credit_card_fraud_X.npy")
CACHE_Y = Path("data/raw/credit_card_fraud_y.npy")
DOWNLOAD_DIR = Path("data/raw/creditcard_raw")

# Stable OpenML CSV export for dataset 1597 (creditcard.csv, ~66 MB)
_CSV_URL = "https://api.openml.org/data/v1/download/1597765"
_OPENML_ID = 1597


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
            print(f"\r  Progress: {pct:.1f}% ({mb:.0f} MB / {total_size/1_048_576:.0f} MB)",
                  end="", flush=True)
        else:
            print(f"\r  Downloaded: {mb:.0f} MB", end="", flush=True)

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

    raw_csv = download_dir / "creditcard.arff.gz"
    if not raw_csv.exists():
        _download(_CSV_URL, raw_csv)

    # Parse the downloaded ARFF (OpenML format) — fall back to fetch_openml on failure
    try:
        print("  Parsing ARFF...")
        with gzip.open(raw_csv, "rt", encoding="utf-8") as f:
            content = f.read()
        data_start = content.lower().index("@data") + len("@data")
        rows = [
            line.strip().split(",")
            for line in content[data_start:].strip().splitlines()
            if line.strip() and not line.startswith("%")
        ]
        arr = np.array(rows, dtype=np.float64)
        X = arr[:, :-1]
        y = arr[:, -1].astype(np.int32)
    except Exception as e:
        print(f"  ARFF parse failed ({e}), falling back to fetch_openml...")
        from sklearn.datasets import fetch_openml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = fetch_openml(data_id=_OPENML_ID, as_frame=False, parser="auto")
        X = dataset.data.astype(float)
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
