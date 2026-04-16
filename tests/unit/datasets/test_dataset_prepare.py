"""Tests for dataset preparation scripts — error paths and pure helpers."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# prepare_fashion_mnist — missing arff.gz raises FileNotFoundError

def test_prepare_fashion_mnist_missing_arff(tmp_path):
    from qie_research.datasets.prepare_fashion_mnist import prepare
    with pytest.raises(FileNotFoundError, match="arff.gz not found"):
        prepare(
            arff_gz_path=tmp_path / "missing.arff.gz",
            cache_x=tmp_path / "X.npy",
            cache_y=tmp_path / "y.npy",
        )


# prepare_fashion_mnist — parses a minimal valid arff.gz and writes npy

def test_prepare_fashion_mnist_writes_npy(tmp_path):
    from qie_research.datasets.prepare_fashion_mnist import prepare

    # build a tiny arff.gz with 2 samples, 3 features + class column
    arff_lines = [
        "@relation fashion-mnist\n",
        "@attribute px1 numeric\n",
        "@attribute px2 numeric\n",
        "@attribute px3 numeric\n",
        "@attribute class {T-shirt/top,Trouser}\n",
        "@data\n",
        "0,128,255,T-shirt/top\n",
        "10,20,30,Trouser\n",
    ]
    arff_gz = tmp_path / "test.arff.gz"
    with gzip.open(arff_gz, "wt", encoding="utf-8") as f:
        f.writelines(arff_lines)

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"
    prepare(arff_gz_path=arff_gz, cache_x=cache_x, cache_y=cache_y)

    X = np.load(cache_x)
    y = np.load(cache_y)
    assert X.shape == (2, 3)
    assert y.shape == (2,)
    assert set(y.tolist()).issubset({0, 1})
    # pixel values normalised to [0,1]
    assert X.max() <= 1.0
    assert X.min() >= 0.0


# prepare_fashion_mnist — integer label fallback path

def test_prepare_fashion_mnist_integer_label(tmp_path):
    from qie_research.datasets.prepare_fashion_mnist import prepare

    arff_lines = [
        "@relation test\n",
        "@attribute px1 numeric\n",
        "@attribute class numeric\n",
        "@data\n",
        "50,3\n",
    ]
    arff_gz = tmp_path / "int_label.arff.gz"
    with gzip.open(arff_gz, "wt", encoding="utf-8") as f:
        f.writelines(arff_lines)

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"
    prepare(arff_gz_path=arff_gz, cache_x=cache_x, cache_y=cache_y)
    y = np.load(cache_y)
    assert y[0] == 3


# prepare_fashion_mnist — unknown label raises ValueError

def test_prepare_fashion_mnist_unknown_label(tmp_path):
    from qie_research.datasets.prepare_fashion_mnist import prepare

    arff_lines = [
        "@relation test\n",
        "@attribute px1 numeric\n",
        "@attribute class string\n",
        "@data\n",
        "50,unknown_label_xyz\n",
    ]
    arff_gz = tmp_path / "bad_label.arff.gz"
    with gzip.open(arff_gz, "wt", encoding="utf-8") as f:
        f.writelines(arff_lines)

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"
    with pytest.raises(ValueError, match="Unknown class label"):
        prepare(arff_gz_path=arff_gz, cache_x=cache_x, cache_y=cache_y)


# prepare_credit_card_fraud — missing CSV raises FileNotFoundError

def test_prepare_credit_card_fraud_missing_csv(tmp_path):
    from qie_research.datasets.prepare_credit_card_fraud import prepare
    with pytest.raises(FileNotFoundError, match="Raw CSV not found"):
        prepare(
            raw_csv=tmp_path / "missing.csv",
            cache_x=tmp_path / "X.npy",
            cache_y=tmp_path / "y.npy",
        )


# prepare_credit_card_fraud — writes npy from a real CSV (mocked via pandas)

def test_prepare_credit_card_fraud_writes_npy(tmp_path):
    import pandas as pd
    from qie_research.datasets.prepare_credit_card_fraud import prepare

    # write a minimal CSV
    csv_path = tmp_path / "creditcard.csv"
    df = pd.DataFrame({
        "Time": [0, 1],
        "V1": [0.1, -0.2],
        "V2": [0.3, 0.4],
        "Amount": [100.0, 50.0],
        "Class": [0, 1],
    })
    df.to_csv(csv_path, index=False)

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"
    prepare(raw_csv=csv_path, cache_x=cache_x, cache_y=cache_y)

    X = np.load(cache_x)
    y = np.load(cache_y)
    assert X.shape == (2, 3)  # V1, V2, Amount (Time dropped)
    assert y.tolist() == [0, 1]


# prepare_covertype — writes npy from mocked fetch_covtype

def test_prepare_covertype_writes_npy(tmp_path):
    from qie_research.datasets.prepare_covertype import prepare

    rng = np.random.default_rng(0)
    fake_bunch = MagicMock()
    fake_bunch.data = rng.standard_normal((10, 54))
    fake_bunch.target = rng.integers(1, 8, size=10)

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"
    with patch("sklearn.datasets.fetch_covtype", return_value=fake_bunch):
        prepare(cache_x=cache_x, cache_y=cache_y)

    X = np.load(cache_x)
    y = np.load(cache_y)
    assert X.shape == (10, 54)
    assert y.min() >= 0  # shifted from 1-indexed to 0-indexed
    assert y.max() <= 6


# prepare_cifar10 — both torchvision and keras absent raises ImportError

def test_prepare_cifar10_no_backends_raises(tmp_path):
    from unittest.mock import patch
    with patch.dict("sys.modules", {"torchvision": None, "tensorflow": None}):
        from qie_research.datasets.prepare_cifar10 import prepare
        with pytest.raises((ImportError, TypeError)):
            prepare(cache_x=tmp_path / "X.npy", cache_y=tmp_path / "y.npy",
                    download_dir=tmp_path / "raw")


# prepare_cifar10 — torchvision present: writes npy from mocked datasets

def test_prepare_cifar10_via_torchvision_mock(tmp_path):
    import types
    from unittest.mock import MagicMock, patch

    rng = np.random.default_rng(0)

    # minimal mock torchvision module
    fake_tv = types.ModuleType("torchvision")
    fake_tv.transforms = types.ModuleType("torchvision.transforms")

    class FakeToTensor:
        pass

    fake_tv.transforms.ToTensor = FakeToTensor

    # fake dataset: 4 samples of 3×2×2 tensors
    import torch
    fake_items = [(torch.zeros(3, 2, 2), i % 10) for i in range(4)]

    class FakeCIFAR10:
        def __init__(self, **kwargs):
            pass
        def __iter__(self):
            return iter(fake_items)
        def __len__(self):
            return len(fake_items)

    fake_tv.datasets = types.ModuleType("torchvision.datasets")
    fake_tv.datasets.CIFAR10 = FakeCIFAR10

    cache_x = tmp_path / "cifar10_X.npy"
    cache_y = tmp_path / "cifar10_y.npy"

    with patch.dict("sys.modules", {"torchvision": fake_tv,
                                     "torchvision.transforms": fake_tv.transforms,
                                     "torchvision.datasets": fake_tv.datasets}):
        from qie_research.datasets.prepare_cifar10 import _download_via_torchvision, prepare
        X, y = _download_via_torchvision(tmp_path)

    assert X.shape[0] == 8  # 4 train + 4 test
    assert X.max() <= 1.0


# prepare_cifar10 — normalise path: X.max() > 1 gets divided

def test_prepare_cifar10_normalisation(tmp_path):
    from qie_research.datasets.prepare_cifar10 import prepare
    import types

    rng = np.random.default_rng(1)

    import torch
    # make items so X values are > 1 before normalisation is applied
    fake_items = [(torch.ones(3, 2, 2) * 200, i % 10) for i in range(2)]

    class FakeCIFAR10:
        def __init__(self, **kwargs): pass
        def __iter__(self): return iter(fake_items)
        def __len__(self): return len(fake_items)

    fake_tv = types.ModuleType("torchvision")
    fake_tv.transforms = types.ModuleType("torchvision.transforms")
    fake_tv.transforms.ToTensor = type("ToTensor", (), {})
    fake_tv.datasets = types.ModuleType("torchvision.datasets")
    fake_tv.datasets.CIFAR10 = FakeCIFAR10

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"

    with patch.dict("sys.modules", {"torchvision": fake_tv,
                                     "torchvision.transforms": fake_tv.transforms,
                                     "torchvision.datasets": fake_tv.datasets}):
        from importlib import reload
        import qie_research.datasets.prepare_cifar10 as mod
        reload(mod)
        mod.prepare(cache_x=cache_x, cache_y=cache_y, download_dir=tmp_path / "raw")

    X = np.load(cache_x)
    assert X.max() <= 1.0


# prepare_higgs — _load_gz_subset on a tiny gz file

def test_higgs_load_gz_subset(tmp_path):
    from qie_research.datasets.prepare_higgs import _load_gz_subset

    # write a minimal gzip CSV: 10 rows, 29 columns (col0=label, cols1-28=features)
    gz_path = tmp_path / "HIGGS.csv.gz"
    rows = []
    for i in range(10):
        label = i % 2
        features = [str(float(j)) for j in range(28)]
        rows.append(f"{label}," + ",".join(features))
    content = "\n".join(rows) + "\n"
    with gzip.open(gz_path, "wt") as f:
        f.write(content)

    X, y = _load_gz_subset(gz_path, n_subset=4, seed=0)
    assert X.shape == (4, 28)
    assert y.shape == (4,)
    assert set(y.tolist()).issubset({0, 1})


# prepare_higgs — prepare() downloads when gz absent (monkeypatched _download)

def test_prepare_higgs_calls_download_when_gz_missing(tmp_path):
    from unittest.mock import patch
    from qie_research.datasets.prepare_higgs import prepare, _load_gz_subset
    import gzip

    # write a tiny fake gz so _load_gz_subset works after "download"
    fake_gz = tmp_path / "raw" / "HIGGS.csv.gz"
    fake_gz.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(10):
        label = i % 2
        features = [str(float(j)) for j in range(28)]
        rows.append(f"{label}," + ",".join(features))
    with gzip.open(fake_gz, "wt") as f:
        f.write("\n".join(rows) + "\n")

    download_calls = []

    def fake_download(url, dest):
        download_calls.append(dest)
        # copy our fake gz to dest so the rest of prepare() can proceed
        import shutil
        shutil.copy(fake_gz, dest)

    with patch("qie_research.datasets.prepare_higgs._download", side_effect=fake_download):
        prepare(
            cache_x=tmp_path / "X.npy",
            cache_y=tmp_path / "y.npy",
            download_dir=tmp_path / "dl",
            raw_gz=tmp_path / "dl" / "HIGGS.csv.gz",
            n_subset=4,
            seed=0,
        )

    assert len(download_calls) == 1
    assert np.load(tmp_path / "X.npy").shape[0] == 4


# prepare_higgs — prepare() uses existing gz without downloading

def test_prepare_higgs_skips_download_when_gz_exists(tmp_path):
    from unittest.mock import patch
    from qie_research.datasets.prepare_higgs import prepare
    import gzip

    raw_gz = tmp_path / "HIGGS.csv.gz"
    rows = []
    for i in range(10):
        label = i % 2
        features = [str(float(j)) for j in range(28)]
        rows.append(f"{label}," + ",".join(features))
    with gzip.open(raw_gz, "wt") as f:
        f.write("\n".join(rows) + "\n")

    download_calls = []
    with patch("qie_research.datasets.prepare_higgs._download",
               side_effect=lambda *a, **kw: download_calls.append(a)):
        prepare(
            cache_x=tmp_path / "X.npy",
            cache_y=tmp_path / "y.npy",
            download_dir=tmp_path,
            raw_gz=raw_gz,
            n_subset=4,
            seed=0,
        )

    assert len(download_calls) == 0


# prepare_dry_bean — missing ucimlrepo raises ImportError

def test_prepare_dry_bean_missing_ucimlrepo(tmp_path):
    import sys
    ucimlrepo = sys.modules.pop("ucimlrepo", None)
    try:
        with patch.dict("sys.modules", {"ucimlrepo": None}):
            from qie_research.datasets.prepare_dry_bean import prepare
            with pytest.raises((ImportError, TypeError)):
                prepare(cache_x=tmp_path / "X.npy", cache_y=tmp_path / "y.npy")
    finally:
        if ucimlrepo is not None:
            sys.modules["ucimlrepo"] = ucimlrepo


# prepare_dry_bean — writes npy from mocked ucimlrepo

def test_prepare_dry_bean_writes_npy(tmp_path):
    import types
    rng = np.random.default_rng(0)

    # build a minimal mock ucimlrepo response
    import pandas as pd
    features_df = pd.DataFrame(rng.standard_normal((10, 16)),
                                columns=[f"f{i}" for i in range(16)])
    targets_df = pd.DataFrame({"Class": ["SEKER", "BARBUNYA", "BOMBAY", "CALI",
                                         "HOROZ", "SIRA", "DERMASON",
                                         "SEKER", "BOMBAY", "CALI"]})

    class FakeDataset:
        class data:
            features = features_df
            targets = targets_df

    fake_ucimlrepo = types.ModuleType("ucimlrepo")
    fake_ucimlrepo.fetch_ucirepo = lambda id: FakeDataset()

    cache_x = tmp_path / "X.npy"
    cache_y = tmp_path / "y.npy"

    with patch.dict("sys.modules", {"ucimlrepo": fake_ucimlrepo}):
        from importlib import reload
        import qie_research.datasets.prepare_dry_bean as mod
        reload(mod)
        mod.prepare(cache_x=cache_x, cache_y=cache_y)

    X = np.load(cache_x)
    y = np.load(cache_y)
    assert X.shape == (10, 16)
    assert y.shape == (10,)
    assert y.min() >= 0
