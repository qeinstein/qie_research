"""Phase 5b: CKA pairwise similarity sweep across all datasets and method pairs.

For each dataset, fits all QIE and classical feature maps on the training split
(seed 42, matching the Phase 4 runner), applies them to a fixed subsample of
up to MAX_CKA_N points, then computes linear CKA between every pair.

Usage:
    python3 -m qie_research.analysis.cka_sweep

Output:
    results/summary/cka_scores.csv  — one row per (dataset, rep_a, rep_b)
"""

import argparse
import csv
import importlib
import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from qie_research.analysis.cka_analysis import calculate_cka
from qie_research.encodings import ENCODING_REGISTRY
from qie_research.runner import DATASET_REGISTRY

MAX_CKA_N = 2000  # n² Gram matrix ≈ 32 MB at float64 — safe upper bound
SEED = 42

_DATASETS = [
    "wine", "breast_cancer", "dry_bean", "credit_card_fraud",
    "fashion_mnist", "cifar10", "higgs", "high_dim_parity",
    "high_rank_noise", "covertype",
]

# Map datasets to their preparation modules
_PREPARE_MAP = {
    "dry_bean": "qie_research.datasets.prepare_dry_bean",
    "credit_card_fraud": "qie_research.datasets.prepare_credit_card_fraud",
    "cifar10": "qie_research.datasets.prepare_cifar10",
    "fashion_mnist": "qie_research.datasets.prepare_fashion_mnist",
    "covertype": "qie_research.datasets.prepare_covertype",
    "higgs": "qie_research.datasets.prepare_higgs",
}

# Dense poly2 feature matrix exceeds RAM for these datasets (see benchmark_execution_plan.md)
_NO_POLY2 = {"cifar10", "fashion_mnist"}

_QIE = ["amplitude", "angle", "basis"]
_CLASSICAL = ["raw_linear", "rff", "pca", "poly2"]


def _n_components_auto(dataset: str, stats: list[dict]) -> int:
    """Mean QIE output dim — matches the runner's matched-budget n_components_auto."""
    dims = [
        float(r["feature_dim_mean"])
        for r in stats
        if r["dataset"] == dataset and r["method"] in _QIE and r["feature_dim_mean"] != ""
    ]
    return max(1, round(sum(dims) / len(dims))) if dims else 50


def _build_reps(
    X_cka: np.ndarray,
    X_train: np.ndarray,
    enc_configs: dict,
    n_auto: int,
    has_poly2: bool,
) -> dict[str, np.ndarray]:
    """Return {method_name: feature_matrix} for the CKA subset."""
    reps: dict[str, np.ndarray] = {}

    # Fit scaler on train — used by all classical maps and some encoders internally
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_cka_sc = scaler.transform(X_cka)

    # QIE encodings: fit on train (angle/basis record per-feature range), transform CKA subset
    for name in _QIE:
        cfg = enc_configs.get(name, {})
        encoder = ENCODING_REGISTRY[name](**cfg)
        encoder.fit(X_train)
        reps[name] = encoder.transform(X_cka)

    # raw_linear: StandardScaler output only
    reps["raw_linear"] = X_cka_sc

    # RFF: gamma = 1/d, matching runner's "auto" heuristic
    gamma = 1.0 / max(X_train.shape[1], 1)
    rff = RBFSampler(n_components=n_auto, gamma=gamma, random_state=SEED)
    rff.fit(X_train_sc)
    reps["rff"] = rff.transform(X_cka_sc)

    # PCA: n_components = n_auto, capped to avoid sklearn infeasibility
    n_pca = min(n_auto, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pca = PCA(n_components=max(1, n_pca), random_state=SEED)
    pca.fit(X_train_sc)
    reps["pca"] = pca.transform(X_cka_sc)

    if has_poly2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X_train_sc)
        reps["poly2"] = poly.transform(X_cka_sc)

    return reps


def _cka_matrix(reps: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Compute CKA for all off-diagonal pairs (symmetric, so compute upper triangle)."""
    scores: dict[tuple[str, str], float] = {}
    methods = list(reps.keys())
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            s = calculate_cka(reps[a], reps[b])
            scores[(a, b)] = s
            scores[(b, a)] = s
    return scores


def _print_qie_vs_classical(dataset: str, scores: dict, has_poly2: bool) -> None:
    cls = ["raw_linear", "rff", "pca"] + (["poly2"] if has_poly2 else [])
    header = f"  {'':12}" + "".join(f"{c:>12}" for c in cls)
    print(header)
    for qie in _QIE:
        row = f"  {qie:<12}"
        for c in cls:
            v = scores.get((qie, c))
            row += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        print(row)


def run_sweep(
    stats_path: Path,
    configs_dir: Path,
    out_path: Path,
    datasets: list[str] | None = None,
    existing: list[dict] | None = None,
) -> None:
    with open(stats_path) as f:
        stats = list(csv.DictReader(f))

    rows: list[dict] = list(existing or [])
    target = datasets if datasets is not None else _DATASETS

    for dataset in target:
        print(f"\n{dataset}", flush=True)

        with open(configs_dir / f"{dataset}.yaml") as f:
            config = yaml.safe_load(f)

        ds_params = config["dataset"]
        try:
            X, y = DATASET_REGISTRY[dataset](ds_params)
        except (FileNotFoundError, ValueError, KeyError):
            if dataset in _PREPARE_MAP:
                print(f"  Dataset '{dataset}' not found or incomplete. Attempting to prepare...")
                try:
                    prep_mod = importlib.import_module(_PREPARE_MAP[dataset])
                    prep_mod.prepare()
                    X, y = DATASET_REGISTRY[dataset](ds_params)
                except Exception as e:
                    print(f"  FAILED to prepare '{dataset}': {e}")
                    print(f"  Skipping '{dataset}' for CKA analysis.")
                    continue
            else:
                print(f"  Dataset '{dataset}' could not be loaded and has no preparation script.")
                continue

        test_size = ds_params.get("test_size", 0.2)
        try:
            X_train, _, _, _ = train_test_split(
                X, y, test_size=test_size, random_state=SEED, stratify=y
            )
        except ValueError:
            X_train, _, _, _ = train_test_split(
                X, y, test_size=test_size, random_state=SEED
            )

        # CKA subset: sample from full dataset (maximises n for small datasets)
        n_cka = min(len(X), MAX_CKA_N)
        idx = np.random.default_rng(SEED).choice(len(X), size=n_cka, replace=False)
        X_cka = X[np.sort(idx)]

        n_auto = _n_components_auto(dataset, stats)
        has_poly2 = dataset not in _NO_POLY2

        print(f"  n={n_cka}, n_auto={n_auto}, poly2={'yes' if has_poly2 else 'no'}", flush=True)

        enc_configs = {
            e["name"]: {k: v for k, v in e.items() if k != "name"}
            for e in config.get("encodings", [])
        }

        reps = _build_reps(X_cka, X_train, enc_configs, n_auto, has_poly2)
        scores = _cka_matrix(reps)

        for (a, b), s in scores.items():
            rows.append({"dataset": dataset, "rep_a": a, "rep_b": b, "cka_score": round(s, 6)})

        _print_qie_vs_classical(dataset, scores, has_poly2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "rep_a", "rep_b", "cka_score"])
        writer.writeheader()
        writer.writerows(rows)

    n_ds = len({r["dataset"] for r in rows})
    print(f"\nWrote {out_path} ({len(rows)} pairs across {n_ds} datasets)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/summary/stats.csv"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/cka_scores.csv"))
    parser.add_argument(
        "--datasets", nargs="+", metavar="DS",
        help="Only run these datasets; merges into existing CSV (replaces their rows).",
    )
    args = parser.parse_args()

    if not args.stats.exists():
        sys.exit(f"stats.csv not found at {args.stats} — run aggregate_results first")

    target = None
    existing = None
    if args.datasets:
        unknown = set(args.datasets) - set(_DATASETS)
        if unknown:
            sys.exit(f"Unknown datasets: {sorted(unknown)}. Valid: {_DATASETS}")
        target = [d for d in _DATASETS if d in set(args.datasets)]
        if args.out.exists():
            with open(args.out) as f:
                existing = [r for r in csv.DictReader(f) if r["dataset"] not in set(target)]

    run_sweep(args.stats, args.configs_dir, args.out, datasets=target, existing=existing)


if __name__ == "__main__":
    main()
