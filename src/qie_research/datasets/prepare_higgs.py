"""
Prepare HIGGS Cache
===================
One-time script that downloads the HIGGS dataset from UCI (ZIP format) and writes a
500 000-sample numpy cache.
"""

from __future__ import annotations

import zipfile
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np

CACHE_X = Path("data/raw/higgs_X.npy")
CACHE_Y = Path("data/raw/higgs_y.npy")
DOWNLOAD_DIR = Path("data/raw/higgs_raw")
RAW_ZIP = DOWNLOAD_DIR / "higgs.zip"
# This is the same file but the zip server is much more stable than the gz one
HIGGS_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"

N_SUBSET = 500_000
RANDOM_SEED = 42
N_COLS = 29  # col 0 = label, cols 1-28 = features


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {url} ...")

    if shutil.which("wget"):
        print("  Using wget with aggressive retries...")
        try:
            subprocess.run([
                "wget", "-c", "-t", "20", "-T", "15", "--waitretry", "5",
                url, "-O", str(dest), "--show-progress"
            ], check=True)
            print("Download complete via wget.")
            return
        except subprocess.CalledProcessError:
            print("  wget failed. Trying fallback...")

    # Fallback to urllib
    import urllib.request
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
            print(f"\r  Downloaded: {mb:.0f} MB (unknown total) ...", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print("\nDownload complete.")


def _open_inner_text(z: zipfile.ZipFile, name: str) -> io.TextIOWrapper:
    """Open a file inside a ZIP as a text stream, handling nested gzip transparently."""
    import gzip
    raw = z.open(name, "r")
    if name.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")
    return io.TextIOWrapper(raw, encoding="utf-8")


def _load_zip_subset(zip_path: Path, n_subset: int, seed: int):
    """Stream the CSV (or CSV.gz) inside the ZIP and return a stratified subset."""
    print(f"Reading {zip_path} ...")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        # Prefer plain CSV; fall back to .csv.gz; otherwise take the largest file
        inner = (
            next((n for n in names if n.lower().endswith(".csv")), None)
            or next((n for n in names if n.lower().endswith(".csv.gz")), None)
            or sorted(z.filelist, key=lambda x: x.file_size, reverse=True)[0].filename
        )
        is_gz = inner.lower().endswith(".gz")
        print(f"  Streaming {inner} ({'gzip inside ZIP' if is_gz else 'plain CSV'})...")

        rng = np.random.default_rng(seed)

        # --- Pass 1: identify row indices per class ---
        class0_idx: list[int] = []
        class1_idx: list[int] = []

        print("  Pass 1: Identifying class indices...")
        with z.open(inner, "r") as raw:
            fh = io.TextIOWrapper(
                __import__("gzip").GzipFile(fileobj=raw) if is_gz else raw,
                encoding="utf-8",
            )
            for i, line in enumerate(fh):
                if i % 500_000 == 0 and i > 0:
                    print(f"    Scanning row {i:,} ...")
                (class0_idx if line[0] == "0" else class1_idx).append(i)

        n0, n1 = len(class0_idx), len(class1_idx)
        total = n0 + n1
        print(f"  Total rows: {total:,}  (class 0: {n0:,}, class 1: {n1:,})")

        frac = n_subset / total
        n_keep0 = round(frac * n0)
        n_keep1 = n_subset - n_keep0
        chosen = (
            set(rng.choice(class0_idx, size=n_keep0, replace=False).tolist())
            | set(rng.choice(class1_idx, size=n_keep1, replace=False).tolist())
        )

        # --- Pass 2: extract selected rows ---
        print(f"  Pass 2: Extracting {n_subset:,} rows...")
        X = np.empty((n_subset, N_COLS - 1), dtype=np.float32)
        y = np.empty(n_subset, dtype=np.int32)
        out_idx = 0

        with z.open(inner, "r") as raw:
            fh = io.TextIOWrapper(
                __import__("gzip").GzipFile(fileobj=raw) if is_gz else raw,
                encoding="utf-8",
            )
            for i, line in enumerate(fh):
                if i % 1_000_000 == 0 and i > 0:
                    print(f"    Extracting: {i/total*100:.0f}% ({i:,} rows scanned) ...")
                if i not in chosen:
                    continue
                vals = line.rstrip("\n").split(",")
                y[out_idx] = int(float(vals[0]))
                X[out_idx] = [float(v) for v in vals[1:]]
                out_idx += 1
                if out_idx >= n_subset:
                    break

        return X[rng.permutation(n_subset)], y[rng.permutation(n_subset)]


def prepare(
    cache_x: Path = CACHE_X,
    cache_y: Path = CACHE_Y,
    download_dir: Path = DOWNLOAD_DIR,
    raw_zip: Path = RAW_ZIP,
    n_subset: int = N_SUBSET,
    seed: int = RANDOM_SEED,
) -> None:
    if cache_x.exists() and cache_y.exists():
        print("Cache already exists — skipping download and processing.")
        return

    download_dir.mkdir(parents=True, exist_ok=True)
    if not raw_zip.exists():
        _download(HIGGS_URL, raw_zip)

    X, y = _load_zip_subset(raw_zip, n_subset=n_subset, seed=seed)
    cache_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    print(f"Done. Saved to {cache_x}")


if __name__ == "__main__":
    prepare()
