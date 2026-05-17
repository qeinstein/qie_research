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

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_dir.mkdir(parents=True, exist_ok=True)
    best = [r for r in rows if r["comparison_type"] == "vs_best_classical"
            and r["cohens_d"] != ""]

    dataset_labels = {
        "wine": "Wine", "breast_cancer": "Breast\nCancer", "dry_bean": "Dry\nBean",
        "credit_card_fraud": "Credit\nFraud", "fashion_mnist": "Fashion-\nMNIST",
        "cifar10": "CIFAR-10", "higgs": "HIGGS", "high_dim_parity": "Parity",
        "high_rank_noise": "High-Rank\nNoise", "covertype": "Covertype",
    }
    encs = ["amplitude", "angle", "basis"]
    enc_colors = {"amplitude": "#c0392b", "angle": "#2471a3", "basis": "#1e8449"}
    enc_labels = {"amplitude": "Amplitude", "angle": "Angle", "basis": "Basis"}

    datasets = [d for d in _DATASETS if any(r["dataset"] == d for r in best)]
    lookup = {(r["dataset"], r["qie_method"]): r for r in best}

    # Split into two rows of 5 datasets each
    split = len(datasets) // 2
    panels = [datasets[:split], datasets[split:]]

    n_enc = len(encs)
    group_w = 0.78
    bar_w = group_w / n_enc
    offsets = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2, n_enc)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=False)

    panel_labels = ["(a)", "(b)"]

    for panel_idx, (ax, panel_ds) in enumerate(zip(axes, panels)):
        x = np.arange(len(panel_ds))

        for ei, enc in enumerate(encs):
            ds_vals, ds_errs, ds_x = [], [], []
            sig_xs, sig_ds = [], []
            neg_xs, neg_ds = [], []
            for di, ds in enumerate(panel_ds):
                r = lookup.get((ds, enc))
                if r is None:
                    continue
                d = float(r["cohens_d"])
                std_d = float(r["std_diff"]) if r["std_diff"] not in ("", 0, "0") else 0.0
                ci_d = abs(float(r["ci95_diff"])) / std_d if std_d > 0 else 0.0
                ds_vals.append(d)
                ds_errs.append(ci_d)
                ds_x.append(x[di] + offsets[ei])
                if r["significant_t05"] == "yes":
                    sig_xs.append(x[di] + offsets[ei])
                    sig_ds.append((d, "**"))
                elif r.get("significant_t10") == "yes":
                    sig_xs.append(x[di] + offsets[ei])
                    sig_ds.append((d, "~"))
                if r["effect_size_label"] == "negligible":
                    neg_xs.append(x[di] + offsets[ei])
                    neg_ds.append(d)

            ax.bar(ds_x, ds_vals, width=bar_w * 0.9,
                   color=enc_colors[enc], label=enc_labels[enc],
                   edgecolor="white", linewidth=0.8)
            ax.errorbar(ds_x, ds_vals, yerr=ds_errs, fmt="none",
                        ecolor="black", elinewidth=1.0, capsize=3.0, zorder=4)

            y_pad = 0.18
            for bx, (d, marker) in zip(sig_xs, sig_ds):
                ypos = d + (y_pad if d >= 0 else -y_pad)
                va = "bottom" if d >= 0 else "top"
                ax.text(bx, ypos, marker, ha="center", va=va, fontsize=10, color="#222222")

            for bx, d in zip(neg_xs, neg_ds):
                # place text just inside the bar tip so it stays on the chart
                if d >= 0:
                    ypos = max(d - 0.02, 0.01)
                    va = "top"
                else:
                    ypos = min(d + 0.02, -0.01)
                    va = "bottom"
                ax.text(bx, ypos, "negl.", ha="center", va=va, fontsize=7.5,
                        color="black", fontstyle="italic", rotation=90, clip_on=True)

        ax.axhline(0, color="black", lw=1.2)
        ax.axhline(-0.2, color="gray", lw=0.9, ls="--", alpha=0.6)
        ax.axhline(0.2, color="gray", lw=0.9, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_labels.get(d, d) for d in panel_ds],
                           ha="center", multialignment="center")
        ax.set_ylabel(r"Cohen's $d$  (QIE $-$ best classical)", labelpad=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.text(0.01, 0.97, panel_labels[panel_idx], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend and note only on first panel
        if panel_idx == 0:
            legend_els = [Patch(color=enc_colors[e], label=enc_labels[e]) for e in encs]
            legend_els.append(
                Line2D([0], [0], color="gray", ls="--", lw=0.9, label=r"$|d|=0.2$ threshold")
            )
            ax.legend(handles=legend_els, loc="lower left", framealpha=0.9, fontsize=11)
            ax.set_title("Effect sizes: QIE vs. best classical baseline per dataset",
                         fontsize=14, pad=10)

        ax.text(0.99, 0.02, "$**$ $p<0.05$,  $\\sim$ $p<0.10$ (paired $t$-test)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#333333")

    fig.tight_layout(pad=2.0, h_pad=3.0)
    p = fig_dir / "forest_plot.pdf"
    fig.savefig(p, bbox_inches="tight")
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
