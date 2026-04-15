# [PHASE 3] Dataset Freeze and Stress Testing

- Record type: phase-outcome
- Phase: 3 — Dataset freeze and stress testing
- Status: Ready for review
- Owner: Fluxx
- Related branch: `phase-3-implementation`

## Objective

Freeze the full benchmark dataset roster at ten datasets, verify that all ten
load correctly through the runner, document provenance for externally sourced
datasets, and confirm that the suite includes scientifically non-trivial stress
tests beyond toy accuracy benchmarks.

## Deliverables

### 1. Dataset manifest

[`data/metadata/dataset_manifest.json`](../../data/metadata/dataset_manifest.json)

Machine-readable registry of all ten datasets.  Each entry documents source,
acquisition method, prep script path, expected shapes, label ranges, dtypes,
and scientific notes.

### 2. Preparation scripts

Scripts for datasets that cannot be loaded directly from sklearn or generated
at runtime.  Each script downloads or reads raw data once and writes numpy
cache files to `data/raw/`.

| Script | Dataset | Trigger |
|---|---|---|
| [`src/qie_research/datasets/prepare_fashion_mnist.py`](../../src/qie_research/datasets/prepare_fashion_mnist.py) | Fashion-MNIST | `python -m qie_research.datasets.prepare_fashion_mnist` |
| [`src/qie_research/datasets/prepare_dry_bean.py`](../../src/qie_research/datasets/prepare_dry_bean.py) | Dry Bean | `python -m qie_research.datasets.prepare_dry_bean` |
| [`src/qie_research/datasets/prepare_credit_card_fraud.py`](../../src/qie_research/datasets/prepare_credit_card_fraud.py) | Credit Card Fraud | `python -m qie_research.datasets.prepare_credit_card_fraud` |
| [`src/qie_research/datasets/prepare_cifar10.py`](../../src/qie_research/datasets/prepare_cifar10.py) | CIFAR-10 | `python -m qie_research.datasets.prepare_cifar10` |
| [`src/qie_research/datasets/prepare_higgs.py`](../../src/qie_research/datasets/prepare_higgs.py) | HIGGS | `python -m qie_research.datasets.prepare_higgs` |
| [`src/qie_research/datasets/prepare_covertype.py`](../../src/qie_research/datasets/prepare_covertype.py) | Covertype | `python -m qie_research.datasets.prepare_covertype` |

### 3. Per-dataset YAML configs

One config file per dataset.  All configs use matching encoding parameters and
the same logistic regression baseline so that inter-dataset comparisons are
fair.  Configs are stored in [`configs/`](../../configs/).

| Config | Dataset |
|---|---|
| [`configs/wine.yaml`](../../configs/wine.yaml) | Wine |
| [`configs/breast_cancer.yaml`](../../configs/breast_cancer.yaml) | Breast Cancer Wisconsin |
| [`configs/dry_bean.yaml`](../../configs/dry_bean.yaml) | Dry Bean |
| [`configs/credit_card_fraud.yaml`](../../configs/credit_card_fraud.yaml) | Credit Card Fraud |
| [`configs/fashion_mnist.yaml`](../../configs/fashion_mnist.yaml) | Fashion-MNIST |
| [`configs/cifar10.yaml`](../../configs/cifar10.yaml) | CIFAR-10 |
| [`configs/higgs.yaml`](../../configs/higgs.yaml) | HIGGS (500k) |
| [`configs/covertype.yaml`](../../configs/covertype.yaml) | Covertype |
| [`configs/high_dim_parity.yaml`](../../configs/high_dim_parity.yaml) | High-Dim Parity |
| [`configs/high_rank_noise.yaml`](../../configs/high_rank_noise.yaml) | High-Rank Noise |

## Frozen Dataset Roster

| # | Dataset | n_samples | n_features | n_classes | Task | Acquisition |
|---|---|---|---|---|---|---|
| 1 | Wine | 178 | 13 | 3 | multiclass | sklearn built-in |
| 2 | Breast Cancer Wisconsin | 569 | 30 | 2 | binary | sklearn built-in |
| 3 | Dry Bean | 13,611 | 16 | 7 | multiclass | ucimlrepo (auto) |
| 4 | Credit Card Fraud | 284,807 | 30 | 2 | binary (imbalanced) | Kaggle (manual) |
| 5 | Fashion-MNIST | 70,000 | 784 | 10 | multiclass | OpenML (auto) |
| 6 | CIFAR-10 | 60,000 | 3,072 | 10 | multiclass | torchvision/keras (auto) |
| 7 | HIGGS (500k) | 500,000 | 28 | 2 | binary | UCI (auto, 2-pass stream) |
| 8 | Covertype | 581,012 | 54 | 7 | multiclass | sklearn (auto) |
| 9 | High-Dim Parity | 10,000 | 20 | 2 | binary (synthetic) | generated at runtime |
| 10 | High-Rank Noise | 5,000 | 200 | 2 | binary (synthetic) | generated at runtime |

## Stress Test Justification

The suite is not a trivial accuracy benchmark.  Each dataset has a distinct
property that stresses a different aspect of encoding quality:

- **Wine / Breast Cancer** — small, clean tabular data.  Baseline sanity checks:
  any reasonable encoding should work.
- **Dry Bean** — 7-class morphological features from image processing.  Tests
  multiclass separability at moderate scale.
- **Credit Card Fraud** — highly imbalanced binary task (0.17% fraud) on top of
  PCA-transformed features.  Tests whether QIE adds representational value over
  an existing linear projection.
- **Fashion-MNIST** — 784-dimensional image flattening.  Tests high-dimensional
  image data; a harder replacement for MNIST as the standard low-compute image
  benchmark.
- **CIFAR-10** — 3,072-dimensional RGB image flattening.  Richer texture and
  colour structure than Fashion-MNIST; expected at top venues alongside it.
- **HIGGS** — 500k-sample particle physics benchmark canonical in the quantum ML
  literature.  28 features (21 low-level kinematic + 7 high-level derived);
  tests large-scale binary classification with physics-motivated features.
- **Covertype** — 581k-sample multiclass task with mixed continuous and binary
  features.  Tests whether encodings handle binary indicator subspaces without
  distortion.
- **High-Dim Parity** — synthetic XOR-of-signs task provably hard for linear
  models.  Directly tests representational capacity: the encoding must produce
  features that separate classes a linear classifier cannot separate in the
  raw space.
- **High-Rank Noise** — synthetic rank-5 signal embedded in 200-dimensional
  Gaussian noise at SNR=1.  Directly probes the effective-rank and conditioning
  behaviour measured in Phase 2; any encoding that collapses to low rank will
  fail to extract the signal.

## Load Verification

The four always-available datasets (Wine, Breast Cancer, High-Dim Parity,
High-Rank Noise) were verified end-to-end through the runner on this branch.
All six cache-dependent datasets raise `FileNotFoundError` with an actionable
prep command when the cache is absent — no silent failures.

Results from verified runs:

```
wine          amplitude acc=0.6667  angle acc=0.9722  basis acc=0.9167
breast_cancer amplitude acc=0.7895  angle acc=0.9561  basis acc=0.9474
high_dim_parity  all encodings ~0.50 (expected: linear classifier cannot solve XOR)
high_rank_noise  amplitude acc=0.9370  angle acc=0.8440  basis acc=0.8620
```

The near-chance accuracy on High-Dim Parity is the correct result — it
confirms the task is genuinely hard for linear models and will serve as a
discriminating benchmark in Phase 4.

## Manual Download Required

**Credit Card Fraud** requires a manual Kaggle download before the prep script
can run:

1. Log in to [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it at `data/raw/creditcard.csv`
4. Run `python -m qie_research.datasets.prepare_credit_card_fraud`

All other datasets are downloaded automatically by their prep scripts or by
sklearn/torchvision at runtime.

## Verification Method

A teammate should verify that:

- The dataset manifest exists at
  [`data/metadata/dataset_manifest.json`](../../data/metadata/dataset_manifest.json)
  and lists exactly ten datasets.
- All six prep scripts exist and are importable.
- `python -m qie_research.runner configs/wine.yaml` and
  `python -m qie_research.runner configs/breast_cancer.yaml` complete without
  error and write results to `results/metrics/`.
- Running any cache-dependent config (e.g. `configs/dry_bean.yaml`) without
  first running its prep script raises `FileNotFoundError` with the correct
  prep command — not a silent failure or import error.
- The stress test justification section above is scientifically coherent with
  the scope defined in
  [`records/phases/phase-0-scope-lock.md`](./phase-0-scope-lock.md).

After verification, the reviewer may change `Status` from `Ready for review`
to `Frozen`.

## Artifact Paths

- Manifest: [`data/metadata/dataset_manifest.json`](../../data/metadata/dataset_manifest.json)
- Prep scripts: [`src/qie_research/datasets/`](../../src/qie_research/datasets/)
- Configs: [`configs/`](../../configs/)
- Verified run outputs: [`results/metrics/wine.json`](../../results/metrics/wine.json), [`results/metrics/breast_cancer.json`](../../results/metrics/breast_cancer.json), [`results/metrics/high_dim_parity.json`](../../results/metrics/high_dim_parity.json), [`results/metrics/high_rank_noise.json`](../../results/metrics/high_rank_noise.json)

## Completion Checklist

- [x] All ten datasets registered in `DATASET_REGISTRY` in `runner.py`.
- [x] Dataset manifest written with provenance, shapes, and dtypes.
- [x] Prep scripts written for all six cache-dependent datasets.
- [x] One YAML config per dataset.
- [x] Four always-available datasets verified end-to-end through the runner.
- [x] Cache-absent datasets raise `FileNotFoundError` with actionable message.
- [x] Stress test justification written — no trivial accuracy benchmarks.
- [x] Manual download instructions documented for Credit Card Fraud.
- [x] Dataset roster frozen — no new datasets may be added without a scope amendment.

## Closure Statement

Phase 3 dataset freeze is complete.  The benchmark roster is fixed at ten
datasets spanning small tabular, large-scale tabular, image, physics, and
synthetic stress-test categories.  Phase 4 full experimental sweep may begin
once all cache prep scripts have been executed in the training environment.
