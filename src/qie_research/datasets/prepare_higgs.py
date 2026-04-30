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


def _load_zip_subset(zip_path: Path, n_subset: int, seed: int):
    """Stream the CSV inside the ZIP and return a stratified subset."""
    print(f"Reading {zip_path} ...")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # The zip contains HIGGS.csv
        csv_name = "HIGGS.csv"
        if csv_name not in z.namelist():
            # Find the largest file if the name is different
            csv_name = sorted(z.filelist, key=lambda x: x.file_size, reverse=True)[0].filename
            
        print(f"  Streaming {csv_name} from ZIP to build stratified subset...")
        rng = np.random.default_rng(seed)

        # --- Pass 1: count rows per class ---
        class0_idx: list[int] = []
        class1_idx: list[int] = []

        print("  Pass 1: Identifying class indices...")
        with z.open(csv_name, "r") as fh:
            # We wrap in TextIOWrapper to read as strings
            import io
            text_fh = io.TextIOWrapper(fh)
            for i, line in enumerate(text_fh):
                if i % 500000 == 0 and i > 0:
                    print(f"    Scanning row {i:,} ...")
                label = line[0]
                if label == "0":
                    class0_idx.append(i)
                else:
                    class1_idx.append(i)

        n0 = len(class0_idx)
        n1 = len(class1_idx)
        total = n0 + n1
        print(f"  Total rows: {total:,}  (class 0: {n0:,}, class 1: {n1:,})")

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
        with z.open(csv_name, "r") as fh:
            text_fh = io.TextIOWrapper(fh)
            for i, line in enumerate(text_fh):
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

        perm = rng.permutation(n_subset)
        return X[perm], y[perm]


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

    try:
        X, y = _load_zip_subset(raw_zip, n_subset=n_subset, seed=seed)
        cache_x.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, X)
        np.save(cache_y, y)
        print(f"Done. Saved to {cache_x}")
    except Exception as e:
        print(f"Error processing ZIP: {e}")
        if raw_zip.exists():
            print("The ZIP file might be corrupted. Deleting and trying again next time.")
            raw_zip.unlink()
        raise e


if __name__ == "__main__":
    prepare()
