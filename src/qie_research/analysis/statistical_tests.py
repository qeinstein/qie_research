"""Phase 5e: Paired statistical tests — QIE vs best classical baseline per dataset.

For each (dataset, QIE encoding) pair, uses the 5 per-seed accuracy values from
records.csv to run:

  Paired t-test  (scipy.stats.ttest_rel, df=4, t_crit=2.776 at α=0.05)
    - Parametric. Assumes normality of paired differences. Viable with n=5.
    - CAN reject at α=0.05 for large effect sizes.

  Wilcoxon signed-rank test  (scipy.stats.wilcoxon)
    - Non-parametric. Makes no normality assumption.
    - With n=5, minimum achievable two-tailed p = 0.0625 — CANNOT reject at
      α=0.05 regardless of effect size. Reported as a robustness check only.

  Cohen's d  (mean paired difference / std of paired differences, ddof=1)
    - First-class effect-size metric. Interpretable regardless of n.
    - |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large.

Comparisons: each QIE method vs (a) its dataset's best classical baseline by
mean accuracy, and (b) raw_linear as a fixed simple baseline.

Output:
    results/summary/statistical_tests.csv   — one row per (dataset, qie, baseline)
    results/figures/forest_plot.png         — Cohen's d forest plot (vs best classical)

Usage:
    python3 -m qie_research.analysis.statistical_tests
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_SEEDS = [7, 42, 99, 1337, 2026]
_QIE = ["amplitude", "angle", "basis"]
_QIE_ALL = {"amplitude", "angle", "basis", "amplitude_torch", "angle_torch", "basis_torch"}

_DATASETS = [
    "wine", "breast_cancer", "dry_bean", "credit_card_fraud",
    "fashion_mnist", "cifar10", "higgs", "high_dim_parity",
    "high_rank_noise", "covertype",
]

_OUT_FIELDS = [
    "dataset", "qie_method", "baseline_method", "comparison_type",
    "n_pairs",
    "mean_diff", "std_diff", "ci95_diff",
    "t_stat", "p_t", "significant_t05", "significant_t10",
    "W_stat", "p_W",
    "cohens_d", "effect_size_label",
]


def _cohens_d(diffs: np.ndarray) -> float:
    s = float(np.std(diffs, ddof=1))
    return float(np.mean(diffs) / s) if s > 0 else float("inf") * np.sign(np.mean(diffs))


def _effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def _wilcoxon_safe(diffs: np.ndarray) -> tuple[float, float]:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    if len(nonzero) == 1:
        # scipy wilcoxon requires n >= 2
        return float("nan"), float("nan")
    try:
        res = stats.wilcoxon(nonzero, zero_method="zsplit", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def load_records(records_path: Path) -> dict[tuple[str, str], dict[int, float]]:
    """Returns {(dataset, method): {seed: accuracy}}."""
    table: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with open(records_path) as f:
        for row in csv.DictReader(f):
            acc = row.get("accuracy", "")
            if acc == "":
                continue
            table[(row["dataset"], row["method"])][int(row["seed"])] = float(acc)
    return table


def best_classical_per_dataset(
    records: dict[tuple[str, str], dict[int, float]]
) -> dict[str, str]:
    """For each dataset, return the non-QIE method with the highest mean accuracy."""
    means: dict[tuple[str, str], float] = {}
    for (ds, method), seed_acc in records.items():
        if method in _QIE_ALL or not seed_acc:
            continue
        means[(ds, method)] = float(np.mean(list(seed_acc.values())))

    best: dict[str, str] = {}
    for (ds, method), mean_acc in means.items():
        if ds not in best or mean_acc > means[(ds, best[ds])]:
            best[ds] = method
    return best


def run_tests(
    records: dict[tuple[str, str], dict[int, float]],
    best_classical: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []

    for dataset in _DATASETS:
        best_cls = best_classical.get(dataset)

        for qie in _QIE:
            qie_seed_acc = records.get((dataset, qie), {})
            if not qie_seed_acc:
                continue

            for comp_type, cls_method in [
                ("vs_best_classical", best_cls),
                ("vs_raw_linear", "raw_linear"),
            ]:
                if cls_method is None:
                    continue
                cls_seed_acc = records.get((dataset, cls_method), {})
                if not cls_seed_acc:
                    continue

                # Pair by seed — only use seeds present in both
                common_seeds = sorted(set(qie_seed_acc) & set(cls_seed_acc))
                if len(common_seeds) < 2:
                    continue

                qie_vals = np.array([qie_seed_acc[s] for s in common_seeds])
                cls_vals = np.array([cls_seed_acc[s] for s in common_seeds])
                diffs = qie_vals - cls_vals

                n = len(diffs)
                mean_d = float(np.mean(diffs))
                std_d = float(np.std(diffs, ddof=1))
                ci95 = float(2.7764 * std_d / np.sqrt(n)) if n == 5 else float(
                    stats.t.ppf(0.975, df=n - 1) * std_d / np.sqrt(n)
                )

                t_res = stats.ttest_rel(qie_vals, cls_vals)
                t_stat = float(t_res.statistic)
                p_t = float(t_res.pvalue)

                W_stat, p_W = _wilcoxon_safe(diffs)
                cd = _cohens_d(diffs)

                rows.append({
                    "dataset": dataset,
                    "qie_method": qie,
                    "baseline_method": cls_method,
                    "comparison_type": comp_type,
                    "n_pairs": n,
                    "mean_diff": round(mean_d, 6),
                    "std_diff": round(std_d, 6),
                    "ci95_diff": round(ci95, 6),
                    "t_stat": round(t_stat, 4),
                    "p_t": round(p_t, 4),
                    "significant_t05": "yes" if p_t < 0.05 else "no",
                    "significant_t10": "yes" if p_t < 0.10 else "no",
                    "W_stat": round(W_stat, 4) if np.isfinite(W_stat) else "",
                    "p_W": round(p_W, 4) if np.isfinite(p_W) else "",
                    "cohens_d": round(cd, 4) if np.isfinite(cd) else "",
                    "effect_size_label": _effect_label(cd) if np.isfinite(cd) else "",
                })

    return rows


def _print_summary(rows: list[dict]) -> None:
    best = [r for r in rows if r["comparison_type"] == "vs_best_classical"]

    sig05 = sum(1 for r in best if r["significant_t05"] == "yes")
    sig10 = sum(1 for r in best if r["significant_t10"] == "yes")
    qie_wins = sum(1 for r in best if r["mean_diff"] > 0)
    qie_loses = sum(1 for r in best if r["mean_diff"] < 0)

    print(f"\n{'='*64}")
    print(f"  Statistical tests — QIE vs best classical (n=5 seeds)")
    print(f"  Comparisons: {len(best)}  (3 QIE × 10 datasets)")
    print(f"  QIE better : {qie_wins}   QIE worse: {qie_loses}")
    print(f"  Significant (paired t, α=0.05): {sig05}/{len(best)}")
    print(f"  Significant (paired t, α=0.10): {sig10}/{len(best)}")
    print(f"  Note: Wilcoxon p_min=0.0625 with n=5 — cannot reject at α=0.05")
    print(f"{'='*64}")

    print(f"\n{'dataset':<20} {'qie':<10} {'vs':<18} {'mean_diff':>10} {'t':>7} {'p_t':>7} {'d':>7} {'size'}")
    print("-" * 84)
    for r in sorted(best, key=lambda x: (x["dataset"], x["qie_method"])):
        sig = "*" if r["significant_t05"] == "yes" else ("~" if r["significant_t10"] == "yes" else " ")
        print(
            f"  {r['dataset']:<18} {r['qie_method']:<10} {r['baseline_method']:<18}"
            f" {r['mean_diff']:>+10.4f} {r['t_stat']:>7.3f} {r['p_t']:>7.4f}"
            f" {r['cohens_d'] if r['cohens_d'] != '' else '-':>7}  {r['effect_size_label']}"
            f"  {sig}"
        )
    print("\n  * p<0.05   ~ p<0.10")


def _plot(rows: list[dict], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    best = [r for r in rows if r["comparison_type"] == "vs_best_classical"
            and r["cohens_d"] != ""]

    # Sort by Cohen's d descending
    best_sorted = sorted(best, key=lambda r: float(r["cohens_d"]), reverse=True)

    labels = [f"{r['dataset']} / {r['qie_method']}" for r in best_sorted]
    ds = [float(r["cohens_d"]) for r in best_sorted]
    # 95% CI on Cohen's d ≈ d ± t_crit/sqrt(n) (approximate)
    errs = [abs(float(r["ci95_diff"])) / float(r["std_diff"])
            if r["std_diff"] not in ("", 0) else 0.0
            for r in best_sorted]

    colors_map = {"amplitude": "#e05c5c", "angle": "#5c7ee0", "basis": "#5cb85c"}
    colors = [colors_map[r["qie_method"]] for r in best_sorted]

    fig, ax = plt.subplots(figsize=(7, max(5, len(labels) * 0.35)))
    y = np.arange(len(labels))
    ax.barh(y, ds, color=colors, edgecolor="white", linewidth=0.4, height=0.6)
    ax.axvline(0, color="black", lw=0.9, ls="-")
    ax.axvline(-0.2, color="gray", lw=0.6, ls="--", alpha=0.5, label="|d|=0.2 (small)")
    ax.axvline(0.2, color="gray", lw=0.6, ls="--", alpha=0.5)

    # Mark significant results
    for i, r in enumerate(best_sorted):
        if r["significant_t05"] == "yes":
            ax.text(ds[i] + (0.05 if ds[i] >= 0 else -0.05), i, "*",
                    ha="left" if ds[i] >= 0 else "right", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cohen's d  (QIE − best classical, positive = QIE better)")
    ax.set_title("Effect sizes: QIE vs best classical baseline\n"
                 "Red=amplitude  Blue=angle  Green=basis   * p<0.05 paired t")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "forest_plot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")


def run(records_path: Path, out_path: Path, fig_dir: Path) -> None:
    records = load_records(records_path)
    best_classical = best_classical_per_dataset(records)
    rows = run_tests(records, best_classical)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")

    _print_summary(rows)
    _plot(rows, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("results/summary/records.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/statistical_tests.csv"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    if not args.records.exists():
        sys.exit(f"records.csv not found at {args.records} — run aggregate_results first")

    run(args.records, args.out, args.fig_dir)


if __name__ == "__main__":
    main()
