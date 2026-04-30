"""Phase 5c: Spectral attribution — correlate encoding geometry with accuracy.

For each (dataset, QIE encoding), fits the encoder on X_train (seed 42), then:
  - computes SVD of encoded X (≤ 2000-sample subsample)
      * large output dims (d > n): Gram-matrix trick — C = X @ X.T, eigvalsh, σ_i = √λ_i
        cost O(n²d) not O(nd²), so cifar10 basis (d=24576) takes ~3 s not ~17 s
  - derives effective rank erank, condition number κ, noise stability δ
  - joins with Phase 4 accuracy_mean from stats.csv
  - computes accuracy gap vs raw_linear and vs best available classical baseline

Output:
    results/summary/spectral_attribution.csv  — one row per (dataset, encoding)
    results/figures/spectral_kappa_vs_gap.png
    results/figures/spectral_erank_vs_acc.png
    results/figures/spectral_noise_vs_gap.png

Usage:
    python3 -m qie_research.analysis.spectral_attribution
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from qie_research.encodings import ENCODING_REGISTRY
from qie_research.runner import DATASET_REGISTRY

MAX_SVD_N = 2000   # subsample for SVD — same budget as CKA
NOISE_N = 200      # subsample for noise stability (stochastic estimate)
NOISE_STD = 0.01
SEED = 42

_QIE = ["amplitude", "angle", "basis"]
_QIE_ALL = {"amplitude", "angle", "basis", "amplitude_torch", "angle_torch", "basis_torch"}

_DATASETS = [
    "wine", "breast_cancer", "dry_bean", "credit_card_fraud",
    "fashion_mnist", "cifar10", "higgs", "high_dim_parity",
    "high_rank_noise", "covertype",
]

_PREPARE_MAP = {
    "dry_bean": "qie_research.datasets.prepare_dry_bean",
    "credit_card_fraud": "qie_research.datasets.prepare_credit_card_fraud",
    "cifar10": "qie_research.datasets.prepare_cifar10",
    "fashion_mnist": "qie_research.datasets.prepare_fashion_mnist",
    "covertype": "qie_research.datasets.prepare_covertype",
    "higgs": "qie_research.datasets.prepare_higgs",
}

_CSV_FIELDS = [
    "dataset", "encoding", "n_svd", "d_out",
    "erank", "erank_norm", "kappa", "log10_kappa", "noise_delta",
    "accuracy_mean", "accuracy_ci95",
    "best_classical_acc", "best_classical_method", "raw_linear_acc",
    "acc_gap_vs_raw", "acc_gap_vs_best",
]


# ---------- spectral primitives ----------

def _singular_values(X_enc: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (sigma, original_shape) in descending order.

    Uses Gram-matrix eigendecomposition when d > n to keep cost at O(n²d)
    rather than the O(nd²) path that numpy's thin-SVD dgesdd would take.
    """
    X = X_enc.astype(np.float64)
    n, d = X.shape
    original_shape = (n, d)
    if d > n:
        C = X @ X.T                           # (n×n) Gram matrix
        lam = np.linalg.eigvalsh(C)[::-1]    # descending eigenvalues
        sigma = np.sqrt(np.maximum(lam, 0.0))
    else:
        _, sigma, _ = np.linalg.svd(X, full_matrices=False)
    return sigma, original_shape


def _effective_rank(sigma: np.ndarray) -> float:
    s = sigma[sigma > 0]
    p = s / s.sum()
    return float(np.exp(-np.dot(p, np.log(p))))


def _condition_number(sigma: np.ndarray, shape: tuple[int, int]) -> float:
    if not len(sigma) or sigma[0] == 0:
        return float("inf")
    thr = np.finfo(np.float64).eps * sigma[0] * max(shape)
    sig = sigma[sigma > thr]
    return float(sig[0] / sig[-1]) if len(sig) >= 2 else float("inf")


def _noise_delta(encoder, X_sub: np.ndarray, rng: np.random.Generator) -> float:
    """Mean per-sample Frobenius distance between clean and ε-perturbed encodings."""
    noise = rng.normal(0.0, NOISE_STD, X_sub.shape).astype(X_sub.dtype)
    phi0 = encoder.transform(X_sub).astype(np.float64)
    phi1 = encoder.transform(X_sub + noise).astype(np.float64)
    return float(np.linalg.norm(phi0 - phi1, "fro") / X_sub.shape[0])


# ---------- stats.csv helpers ----------

def _load_stats(stats_path: Path) -> dict[tuple[str, str], dict]:
    table: dict[tuple[str, str], dict] = {}
    with open(stats_path) as f:
        for row in csv.DictReader(f):
            a = row["accuracy_mean"]
            ci = row["accuracy_ci95"]
            table[(row["dataset"], row["method"])] = {
                "acc": float(a) if a else None,
                "ci95": float(ci) if ci else None,
                "is_qie": row["method"] in _QIE_ALL,
            }
    return table


def _best_classical(dataset: str, table: dict) -> tuple[float, str]:
    candidates = [
        (v["acc"], k[1])
        for k, v in table.items()
        if k[0] == dataset and not v["is_qie"] and v["acc"] is not None
    ]
    return max(candidates, key=lambda x: x[0]) if candidates else (float("nan"), "")


# ---------- main sweep ----------

def run_attribution(
    stats_path: Path,
    configs_dir: Path,
    out_csv: Path,
    out_fig_dir: Path,
) -> None:
    acc_table = _load_stats(stats_path)
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for dataset in _DATASETS:
        print(f"\n{dataset}", flush=True)

        with open(configs_dir / f"{dataset}.yaml") as f:
            config = yaml.safe_load(f)

        ds_params = config["dataset"]
        try:
            X, y = DATASET_REGISTRY[dataset](ds_params)
        except (FileNotFoundError, ValueError, KeyError):
            if dataset in _PREPARE_MAP:
                print(f"  Attempting to prepare '{dataset}'...")
                try:
                    prep_mod = importlib.import_module(_PREPARE_MAP[dataset])
                    prep_mod.prepare()
                    X, y = DATASET_REGISTRY[dataset](ds_params)
                except Exception as e:
                    print(f"  FAILED: {e} — skipping")
                    continue
            else:
                print("  Cannot load — skipping")
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

        n_svd = min(len(X_train), MAX_SVD_N)
        idx = rng.choice(len(X_train), size=n_svd, replace=False)
        X_svd = X_train[np.sort(idx)]
        X_noise = X_svd[:NOISE_N]

        best_acc, best_method = _best_classical(dataset, acc_table)
        raw_acc = (acc_table.get((dataset, "raw_linear"), {}).get("acc")) or float("nan")

        enc_configs = {
            e["name"]: {k: v for k, v in e.items() if k != "name"}
            for e in config.get("encodings", [])
        }

        for enc_name in _QIE:
            cfg = enc_configs.get(enc_name, {})
            encoder = ENCODING_REGISTRY[enc_name](**cfg)
            encoder.fit(X_train)

            X_enc = encoder.transform(X_svd)
            print(f"  {enc_name}: d_out={X_enc.shape[1]}  SVD...", end="", flush=True)

            sigma, orig_shape = _singular_values(X_enc)
            er = _effective_rank(sigma)
            kappa = _condition_number(sigma, orig_shape)
            delta = _noise_delta(encoder, X_noise, rng)

            log_kappa = float(np.log10(kappa)) if np.isfinite(kappa) and kappa > 0 else float("nan")
            print(f"  erank={er:.1f}  κ={kappa:.2f}  δ={delta:.5f}")

            qie_row = acc_table.get((dataset, enc_name), {})
            qie_acc = qie_row.get("acc") or float("nan")
            qie_ci = qie_row.get("ci95") or float("nan")

            def _fmt(v: float) -> str | float:
                return round(v, 6) if np.isfinite(v) else ""

            rows.append({
                "dataset": dataset,
                "encoding": enc_name,
                "n_svd": n_svd,
                "d_out": X_enc.shape[1],
                "erank": round(er, 4),
                "erank_norm": round(er / X_enc.shape[1], 4),
                "kappa": _fmt(kappa),
                "log10_kappa": _fmt(log_kappa),
                "noise_delta": round(delta, 6),
                "accuracy_mean": _fmt(qie_acc),
                "accuracy_ci95": _fmt(qie_ci),
                "best_classical_acc": _fmt(best_acc),
                "best_classical_method": best_method,
                "raw_linear_acc": _fmt(raw_acc),
                "acc_gap_vs_raw": _fmt(qie_acc - raw_acc),
                "acc_gap_vs_best": _fmt(qie_acc - best_acc),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv} ({len(rows)} rows)")

    _plot(rows, out_fig_dir)


# ---------- scatter plots ----------

def _plot(rows: list[dict], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"amplitude": "#e05c5c", "angle": "#5c7ee0", "basis": "#5cb85c"}
    markers = {"amplitude": "s", "angle": "o", "basis": "^"}

    def _scatter(ax, key_x: str, key_y: str) -> None:
        for enc in _QIE:
            xs, ys, labs = [], [], []
            for r in rows:
                if r["encoding"] != enc:
                    continue
                x, y = r.get(key_x, ""), r.get(key_y, "")
                if x == "" or y == "":
                    continue
                xs.append(float(x))
                ys.append(float(y))
                labs.append(r["dataset"])
            if not xs:
                continue
            ax.scatter(xs, ys, c=colors[enc], marker=markers[enc], s=80,
                       label=enc, zorder=3, edgecolors="white", linewidths=0.4)
            for xi, yi, lab in zip(xs, ys, labs):
                ax.annotate(lab, (xi, yi), textcoords="offset points",
                            xytext=(4, 3), fontsize=6.5, alpha=0.8)

    # Figure 1: log₁₀(κ) vs accuracy gap vs best classical
    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, "log10_kappa", "acc_gap_vs_best")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("log₁₀(κ) — condition number of encoded feature matrix")
    ax.set_ylabel("Accuracy − best classical baseline")
    ax.set_title("Spectral conditioning vs. QIE accuracy advantage")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "spectral_kappa_vs_gap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")

    # Figure 2: effective rank vs accuracy_mean
    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, "erank", "accuracy_mean")
    ax.set_xlabel("Effective rank (erank) of encoded feature matrix")
    ax.set_ylabel("Test accuracy (mean, 5 seeds)")
    ax.set_title("Feature-space effective rank vs. classification accuracy")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "spectral_erank_vs_acc.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")

    # Figure 3: noise_delta vs accuracy gap (log x-scale)
    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, "noise_delta", "acc_gap_vs_best")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Noise stability δ (log scale) — Frobenius dist per sample, ε ~ N(0, 0.01)")
    ax.set_ylabel("Accuracy − best classical baseline")
    ax.set_title("Noise sensitivity vs. QIE accuracy advantage")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    p = fig_dir / "spectral_noise_vs_gap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/summary/stats.csv"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/spectral_attribution.csv"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    if not args.stats.exists():
        sys.exit(f"stats.csv not found at {args.stats} — run aggregate_results first")
    if not args.configs_dir.exists():
        sys.exit(f"configs dir not found: {args.configs_dir}")

    run_attribution(args.stats, args.configs_dir, args.out, args.fig_dir)


if __name__ == "__main__":
    main()
