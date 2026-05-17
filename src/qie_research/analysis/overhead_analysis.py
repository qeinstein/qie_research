"""Phase 5d: Overhead analysis — encoding time and memory vs. classical baselines.

Reads timing and memory columns already aggregated in stats.csv and produces:
  - results/summary/overhead_analysis.csv  (per dataset×method, sklearn methods only)
  - results/summary/overhead_summary.csv   (mean across datasets per method)
  - results/figures/overhead_ratio.png     (encoding overhead fraction, grouped bar)
  - results/figures/overhead_memory.png    (peak encoding memory, grouped bar)

"Overhead ratio" = encoding_time / total_time  (already in stats.csv).
"Encoding/training ratio" = encoding_time / training_time (computed here).

Only sklearn-based methods are included — they have a clean encoding/training split.
torch_mlp/amplitude_torch/etc. fold encoding into forward passes so have no separate
encoding_time, and are excluded.

Usage:
    python3 -m qie_research.analysis.overhead_analysis
"""

from __future__ import annotations

import argparse
import csv
import sys
import numpy as np
from pathlib import Path

_SKLEARN_METHODS = [
    "amplitude", "angle", "basis",
    "raw_linear", "rff", "pca", "poly2", "poly3",
]

_QIE = {"amplitude", "angle", "basis"}
_CLASSICAL = {"raw_linear", "rff", "pca", "poly2", "poly3"}

_DATASET_ORDER = [
    "wine", "breast_cancer", "dry_bean", "credit_card_fraud",
    "fashion_mnist", "cifar10", "higgs", "high_dim_parity",
    "high_rank_noise", "covertype",
]

_TIMING_COLS = [
    "encoding_s_mean", "encoding_s_std", "encoding_s_ci95",
    "training_s_mean", "training_s_std", "training_s_ci95",
    "total_s_mean", "total_s_std", "total_s_ci95",
    "overhead_ratio_mean", "overhead_ratio_std", "overhead_ratio_ci95",
]
_MEMORY_COLS = [
    "encoding_peak_mb_mean", "encoding_peak_mb_std", "encoding_peak_mb_ci95",
    "training_peak_mb_mean", "training_peak_mb_std", "training_peak_mb_ci95",
]

_OUT_FIELDS = [
    "dataset", "method", "method_type",
    "encoding_us_mean",        # encoding time in microseconds
    "training_ms_mean",        # training time in milliseconds
    "total_ms_mean",
    "overhead_ratio_mean",     # encoding / total  (already in stats.csv)
    "enc_train_ratio_mean",    # encoding / training  (computed here)
    "encoding_peak_mb_mean",
    "training_peak_mb_mean",
]


def _f(row: dict, col: str) -> float | None:
    v = row.get(col, "")
    return float(v) if v not in ("", None) else None


def load_overhead(stats_path: Path) -> list[dict]:
    rows = []
    with open(stats_path) as f:
        for r in csv.DictReader(f):
            if r["method"] not in _SKLEARN_METHODS:
                continue
            enc_s = _f(r, "encoding_s_mean")
            trn_s = _f(r, "training_s_mean")
            tot_s = _f(r, "total_s_mean")
            if enc_s is None or trn_s is None:
                continue  # skip rows with no timing breakdown

            enc_train_ratio = enc_s / trn_s if trn_s and trn_s > 0 else None
            rows.append({
                "dataset": r["dataset"],
                "method": r["method"],
                "method_type": "qie" if r["method"] in _QIE else "classical",
                "encoding_us_mean": round(enc_s * 1e6, 3),
                "training_ms_mean": round(trn_s * 1e3, 3),
                "total_ms_mean": round(tot_s * 1e3, 3) if tot_s else "",
                "overhead_ratio_mean": round(float(_f(r, "overhead_ratio_mean") or 0), 6),
                "enc_train_ratio_mean": round(enc_train_ratio, 6) if enc_train_ratio is not None else "",
                "encoding_peak_mb_mean": round(_f(r, "encoding_peak_mb_mean") or 0, 4),
                "training_peak_mb_mean": round(_f(r, "training_peak_mb_mean") or 0, 4),
            })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean across datasets for each method."""
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    mtype: dict[str, str] = {}
    for r in rows:
        buckets[r["method"]].append(r)
        mtype[r["method"]] = r["method_type"]

    summary = []
    for method in _SKLEARN_METHODS:
        group = buckets.get(method, [])
        if not group:
            continue

        def _mean(key):
            vals = [r[key] for r in group if r.get(key) not in ("", None)]
            return round(float(np.mean(vals)), 4) if vals else ""

        summary.append({
            "method": method,
            "method_type": mtype[method],
            "n_datasets": len(group),
            "encoding_us_mean": _mean("encoding_us_mean"),
            "training_ms_mean": _mean("training_ms_mean"),
            "total_ms_mean": _mean("total_ms_mean"),
            "overhead_ratio_mean": _mean("overhead_ratio_mean"),
            "enc_train_ratio_mean": _mean("enc_train_ratio_mean"),
            "encoding_peak_mb_mean": _mean("encoding_peak_mb_mean"),
            "training_peak_mb_mean": _mean("training_peak_mb_mean"),
        })
    return summary


def _plot(rows: list[dict], summary: list[dict], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_dir.mkdir(parents=True, exist_ok=True)

    method_labels = {
        "amplitude": "Amplitude", "angle": "Angle", "basis": "Basis",
        "raw_linear": "Raw linear", "rff": "RFF", "pca": "PCA",
        "poly2": "Poly-2", "poly3": "Poly-3",
    }

    methods = [r["method"] for r in summary if r.get("overhead_ratio_mean") != ""]
    ratios = [float(r["overhead_ratio_mean"]) for r in summary if r.get("overhead_ratio_mean") != ""]
    colors = ["#c0392b" if r["method_type"] == "qie" else "#2471a3"
              for r in summary if r.get("overhead_ratio_mean") != ""]
    method_display = [method_labels.get(m, m) for m in methods]

    # ---- Figure 1: overhead_ratio (encoding/total) per method ----
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(method_display, ratios, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.6)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.4, label="50% threshold")
    ax.set_ylabel("Encoding time / total pipeline time", labelpad=8)
    ax.set_title("Encoding overhead fraction (mean across datasets)", pad=10)
    ax.set_ylim(0, max(ratios) * 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#c0392b", label="QIE"),
        Patch(color="#2471a3", label="Classical baseline"),
        plt.Line2D([0], [0], color="black", ls="--", lw=0.8, label="50% threshold"),
    ], framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout(pad=1.5)
    p = fig_dir / "overhead_ratio.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ---- Figure 2: per-dataset overhead_ratio heatmap (methods × datasets) ----
    dataset_labels = {
        "wine": "Wine", "breast_cancer": "Breast\nCancer", "dry_bean": "Dry Bean",
        "credit_card_fraud": "Credit\nFraud", "fashion_mnist": "Fashion-\nMNIST",
        "cifar10": "CIFAR-10", "higgs": "HIGGS", "high_dim_parity": "Parity",
        "high_rank_noise": "High-Rank\nNoise", "covertype": "Covertype",
    }
    datasets = [d for d in _DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    present_methods = [m for m in _SKLEARN_METHODS if any(r["method"] == m for r in rows)]

    matrix = np.full((len(present_methods), len(datasets)), np.nan)
    for r in rows:
        if r.get("overhead_ratio_mean") in ("", None):
            continue
        mi = present_methods.index(r["method"]) if r["method"] in present_methods else -1
        di = datasets.index(r["dataset"]) if r["dataset"] in datasets else -1
        if mi >= 0 and di >= 0:
            matrix[mi, di] = float(r["overhead_ratio_mean"])

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([dataset_labels.get(d, d) for d in datasets], fontsize=9)
    ax.set_yticks(range(len(present_methods)))
    ax.set_yticklabels([method_labels.get(m, m) for m in present_methods], fontsize=10)
    for i in range(len(present_methods)):
        for j in range(len(datasets)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="black" if v < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Encoding / total time", fraction=0.02, pad=0.02)
    ax.set_title("Encoding overhead fraction per method and dataset", pad=10)
    n_qie = sum(1 for m in present_methods if m in _QIE)
    ax.axhline(n_qie - 0.5, color="white", lw=2)
    fig.tight_layout(pad=1.5)
    p = fig_dir / "overhead_heatmap.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ---- Figure 3: encoding peak memory per method ----
    mem_methods = [r["method"] for r in summary if r.get("encoding_peak_mb_mean") not in ("", None, 0)]
    mem_vals = [float(r["encoding_peak_mb_mean"]) for r in summary
                if r.get("encoding_peak_mb_mean") not in ("", None, 0)]
    mem_colors = ["#c0392b" if r["method_type"] == "qie" else "#2471a3"
                  for r in summary if r.get("encoding_peak_mb_mean") not in ("", None, 0)]

    if mem_methods:
        fig, ax = plt.subplots(figsize=(9, 5))
        mem_display = [method_labels.get(m, m) for m in mem_methods]
        bars = ax.bar(mem_display, mem_vals, color=mem_colors, edgecolor="white",
                      linewidth=0.5, width=0.6)
        ax.set_ylabel("Peak encoding memory (MB, mean across datasets)", labelpad=8)
        ax.set_title("Peak encoding memory per method", pad=10)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, mem_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        fig.tight_layout(pad=1.5)
        p = fig_dir / "overhead_memory.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {p}")


def run(stats_path: Path, out_dir: Path, fig_dir: Path) -> None:
    rows = load_overhead(stats_path)
    summary = summarise(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "overhead_analysis.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {detail_path} ({len(rows)} rows)")

    summary_path = out_dir / "overhead_summary.csv"
    summary_fields = ["method", "method_type", "n_datasets",
                      "encoding_us_mean", "training_ms_mean", "total_ms_mean",
                      "overhead_ratio_mean", "enc_train_ratio_mean",
                      "encoding_peak_mb_mean", "training_peak_mb_mean"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {summary_path} ({len(summary)} methods)")

    print("\n--- Overhead summary (mean across datasets) ---")
    print(f"{'method':<14} {'enc_µs':>8} {'train_ms':>10} {'overhead%':>10} {'enc_peak_MB':>12}")
    print("-" * 56)
    for r in summary:
        enc = r.get("encoding_us_mean", "")
        trn = r.get("training_ms_mean", "")
        ovr = r.get("overhead_ratio_mean", "")
        mem = r.get("encoding_peak_mb_mean", "")
        tag = " ← QIE" if r["method_type"] == "qie" else ""
        print(f"  {r['method']:<12} {enc if enc != '' else '-':>8} "
              f"{trn if trn != '' else '-':>10} "
              f"{f'{float(ovr)*100:.1f}%' if ovr != '' else '-':>10} "
              f"{mem if mem != '' else '-':>12}{tag}")

    _plot(rows, summary, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/summary/stats.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    if not args.stats.exists():
        sys.exit(f"stats.csv not found at {args.stats} — run aggregate_results first")

    run(args.stats, args.out_dir, args.fig_dir)


if __name__ == "__main__":
    main()
