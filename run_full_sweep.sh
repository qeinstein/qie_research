#!/bin/bash
set -uo pipefail
# Note: -e is intentionally omitted so individual run failures don't abort the sweep.

# QIE Research: Full Experimental Sweep Master Script
# Usage: ./run_full_sweep.sh [--torch-only] [--skip-install]
#
# Runs all 10 datasets × 5 seeds = 500 total (dataset, method) evaluations.
# Results  -> results/metrics/<dataset>_seed<N>.json
# Logs     -> results/logs/<dataset>_seed<N>.txt
# Failures -> results/failed_runs.txt  (one "config seed" entry per failed run)

TORCH_FLAG=""
SKIP_INSTALL=0

for arg in "$@"; do
    if [ "$arg" == "--torch-only" ]; then
        TORCH_FLAG="--torch-only"
        echo "Running in TORCH-ONLY mode (skipping sklearn training)."
    elif [ "$arg" == "--skip-install" ]; then
        SKIP_INSTALL=1
        echo "Skipping dependency installation; expecting an activated Python environment."
    fi
done

# Seeds from configs/seed_registry.yaml — must match exactly.
SEEDS=(42 1337 2026 7 99)

CONFIGS=(
    "configs/wine.yaml"
    "configs/breast_cancer.yaml"
    "configs/dry_bean.yaml"
    "configs/fashion_mnist.yaml"
    "configs/cifar10.yaml"
    "configs/higgs.yaml"
    "configs/covertype.yaml"
    "configs/high_dim_parity.yaml"
    "configs/high_rank_noise.yaml"
    "configs/credit_card_fraud.yaml"
)

# 1. Environment Setup
echo "--- Step 1: Environment Setup ---"
if [ "$SKIP_INSTALL" -eq 0 ]; then
    pip install -r requirements-runpod.txt
    pip install -e .
fi
mkdir -p data/raw data/processed results/metrics results/logs

FAILED_LOG="results/failed_runs.txt"
# Start fresh — don't append to a previous failure log
> "$FAILED_LOG"

# 2. Data Preparation (idempotent — scripts skip if cache exists)
echo "--- Step 2: Data Preparation ---"
python3 -m qie_research.datasets.prepare_dry_bean
python3 -m qie_research.datasets.prepare_fashion_mnist
python3 -m qie_research.datasets.prepare_cifar10
python3 -m qie_research.datasets.prepare_higgs
python3 -m qie_research.datasets.prepare_covertype
python3 -m qie_research.datasets.prepare_credit_card_fraud

# 3. Execution Sweep — 10 datasets × 5 seeds
echo "--- Step 3: Executing Benchmark Sweep (${#CONFIGS[@]} datasets × ${#SEEDS[@]} seeds) ---"

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
DONE=0
N_FAILED=0

for CFG in "${CONFIGS[@]}"; do
    if [ ! -f "$CFG" ]; then
        echo "ERROR: Config $CFG not found — skipping."
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        DONE=$(( DONE + 1 ))
        echo "[$DONE/$TOTAL] $CFG  seed=$SEED"

        if python3 -m qie_research.runner "$CFG" --seed "$SEED" $TORCH_FLAG; then
            : # success — nothing to do
        else
            N_FAILED=$(( N_FAILED + 1 ))
            echo "  FAILED: $CFG seed=$SEED — logged to $FAILED_LOG"
            echo "$CFG $SEED" >> "$FAILED_LOG"
        fi
    done
done

echo ""
echo "--- Sweep complete ---"
echo "    Total : $TOTAL"
echo "    Failed: $N_FAILED"
if [ "$N_FAILED" -gt 0 ]; then
    echo "    Failed runs listed in: $FAILED_LOG"
    echo "    Rerun them with: ./retry_failed.sh"
fi
echo "    Results -> results/metrics/"
echo "    Logs    -> results/logs/"
