"""Aggregate Phase 4 sweep results into summary CSVs.

Usage:
    python3 -m qie_research.analysis.aggregate_results
    python3 -m qie_research.analysis.aggregate_results --results-dir path --out-dir path

Outputs:
    records.csv  — one row per (dataset, seed, method); all raw values
    stats.csv    — mean, std, 95% CI aggregated over seeds per (dataset, method)
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# t_{0.025, df=4}: two-tailed 95% CI for exactly 5 seeds
_T_CRIT = 2.7764

_DATASET_ORDER = [
    "wine", "breast_cancer", "dry_bean", "credit_card_fraud",
    "fashion_mnist", "cifar10", "higgs", "high_dim_parity",
    "high_rank_noise", "covertype",
]

_METHOD_ORDER = [
    "amplitude", "angle", "basis",
    "amplitude_torch", "angle_torch", "basis_torch",
    "raw_linear", "rff", "poly2", "poly3", "pca", "rbf_svm", "mlp", "torch_mlp",
]

_NUMERIC_COLS = [
    "accuracy", "f1_macro", "feature_dim",
    "encoding_s", "training_s", "total_s", "overhead_ratio",
    "encoding_peak_mb", "training_peak_mb",
]

_RECORD_FIELDS = [
    "dataset", "seed", "method", "method_type", "feature_dim",
    "accuracy", "f1_macro",
    "encoding_s", "training_s", "total_s", "overhead_ratio",
    "encoding_peak_mb", "training_peak_mb",
]

_STATS_FIELDS = (
    ["dataset", "method", "method_type", "n_seeds"]
    + [f"{col}_{stat}"
       for col in _NUMERIC_COLS
       for stat in ("mean", "std", "ci95")]
)


def _dataset_rank(name: str) -> int:
    return _DATASET_ORDER.index(name) if name in _DATASET_ORDER else len(_DATASET_ORDER)


def _method_rank(name: str) -> int:
    return _METHOD_ORDER.index(name) if name in _METHOD_ORDER else len(_METHOD_ORDER)


def _parse_file(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)

    dataset = d["run"]["name"]
    seed = d["run"]["seed"]
    rows: list[dict] = []

    for r in d["results"]:
        enc = r["encoding"]
        t = r["timing_seconds"]   # always a dict for the sklearn path
        m = r["memory_bytes"]     # always a dict for the sklearn path

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": enc,
            "method_type": "qie_sklearn",
            "feature_dim": r["output_dim"],
            "accuracy": r["metrics"]["accuracy"],
            "f1_macro": r["metrics"]["f1_macro"],
            "encoding_s": t["encoding"],
            "training_s": t["training"],
            "total_s": t["total"],
            "overhead_ratio": t["encoding"] / t["total"] if t["total"] > 0 else 0.0,
            "encoding_peak_mb": m["encoding_peak"] / 1e6,
            "training_peak_mb": m["training_peak"] / 1e6,
        })

        # Torch linear head: timing and memory are scalars (encoding + training bundled)
        th = r["torch_linear_head"]
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": f"{enc}_torch",
            "method_type": "qie_torch",
            "feature_dim": r["output_dim"],
            "accuracy": th["metrics"]["accuracy"],
            "f1_macro": th["metrics"]["f1_macro"],
            "encoding_s": None,
            "training_s": None,
            "total_s": th["timing_seconds"],
            "overhead_ratio": None,
            "encoding_peak_mb": None,
            "training_peak_mb": th["memory_bytes"] / 1e6,
        })

    for b in d["baselines"]:
        name = b["name"]
        t = b["timing_seconds"]
        m = b["memory_bytes"]

        # torch_mlp stores timing and memory as plain scalars (no feature-map stage)
        if isinstance(t, dict):
            enc_s = t["feature_map"]
            train_s = t["training"]
            total_s = t["total"]
            overhead = enc_s / total_s if total_s > 0 else 0.0
            enc_peak_mb = m["feature_map_peak"] / 1e6
            train_peak_mb = m["training_peak"] / 1e6
            method_type = "baseline_sklearn"
        else:
            enc_s = train_s = None
            total_s = t
            overhead = None
            enc_peak_mb = None
            train_peak_mb = m / 1e6
            method_type = "baseline_torch"

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": name,
            "method_type": method_type,
            "feature_dim": b["feature_dim"],
            "accuracy": b["metrics"]["accuracy"],
            "f1_macro": b["metrics"]["f1_macro"],
            "encoding_s": enc_s,
            "training_s": train_s,
            "total_s": total_s,
            "overhead_ratio": overhead,
            "encoding_peak_mb": enc_peak_mb,
            "training_peak_mb": train_peak_mb,
        })

    return rows


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["dataset"], r["method"])].append(r)

    stats_rows: list[dict] = []
    for (dataset, method), group in grouped.items():
        row: dict = {
            "dataset": dataset,
            "method": method,
            "method_type": group[0]["method_type"],
            "n_seeds": len(group),
        }
        for col in _NUMERIC_COLS:
            vals = [r[col] for r in group if r[col] is not None]
            if not vals:
                row[f"{col}_mean"] = ""
                row[f"{col}_std"] = ""
                row[f"{col}_ci95"] = ""
                continue
            mu = statistics.mean(vals)
            row[f"{col}_mean"] = round(mu, 8)
            if len(vals) < 2:
                row[f"{col}_std"] = ""
                row[f"{col}_ci95"] = ""
            else:
                sd = statistics.stdev(vals)
                row[f"{col}_std"] = round(sd, 8)
                row[f"{col}_ci95"] = round(_T_CRIT * sd / math.sqrt(len(vals)), 8)
        stats_rows.append(row)

    return stats_rows


def _write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_accuracy_table(stats: list[dict]) -> None:
    qie_methods = ["amplitude", "angle", "basis"]
    top_baselines = ["raw_linear", "rbf_svm", "mlp", "torch_mlp"]
    methods = qie_methods + top_baselines

    by_key = {(r["dataset"], r["method"]): r for r in stats}

    col_w = 22
    header = f"{'dataset':<20}" + "".join(f"{m:>{col_w}}" for m in methods)
    print()
    print(header)
    print("-" * len(header))
    for ds in _DATASET_ORDER:
        line = f"{ds:<20}"
        for m in methods:
            r = by_key.get((ds, m))
            if r is None or r.get("accuracy_mean") == "":
                line += f"{'—':>{col_w}}"
            else:
                mu = r["accuracy_mean"]
                ci = r["accuracy_ci95"]
                cell = f"{mu:.4f}±{ci:.4f}" if ci != "" else f"{mu:.4f}"
                line += f"{cell:>{col_w}}"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()

    json_files = sorted(
        p for p in args.results_dir.rglob("*.json")
        if "phase_2" not in p.name
    )
    if not json_files:
        raise SystemExit(f"No result JSONs found under {args.results_dir}")

    all_records: list[dict] = []
    for path in json_files:
        all_records.extend(_parse_file(path))

    all_records.sort(key=lambda r: (_dataset_rank(r["dataset"]), _method_rank(r["method"]), r["seed"]))

    stats = _aggregate(all_records)
    stats.sort(key=lambda r: (_dataset_rank(r["dataset"]), _method_rank(r["method"])))

    records_path = args.out_dir / "records.csv"
    stats_path = args.out_dir / "stats.csv"

    _write_csv(all_records, _RECORD_FIELDS, records_path)
    _write_csv(stats, _STATS_FIELDS, stats_path)

    n_datasets = len({r["dataset"] for r in all_records})
    n_methods = len({r["method"] for r in all_records})
    print(f"Parsed {len(json_files)} files → {len(all_records)} records "
          f"({n_datasets} datasets × {n_methods} methods × 5 seeds)")
    print(f"Wrote {records_path}")
    print(f"Wrote {stats_path}")

    _print_accuracy_table(stats)


if __name__ == "__main__":
    main()
