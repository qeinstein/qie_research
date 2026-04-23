#!/bin/bash
set -euo pipefail

# QIE Research: Full Experimental Sweep Master Script
# Usage: ./run_full_sweep.sh [--torch-only]

TORCH_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--torch-only" ]; then
        TORCH_FLAG="--torch-only"
        echo "Running in TORCH-ONLY mode (skipping sklearn training)."
        break
    fi
done

# 1. Environment Setup
echo "--- Step 1: Environment Setup ---"
pip install -r requirements.txt
mkdir -p data/raw
mkdir -p results/metrics

# 2. Data Preparation
echo "--- Step 2: Data Preparation ---"

# These are automated
python3 -m qie_research.datasets.prepare_dry_bean
python3 -m qie_research.datasets.prepare_fashion_mnist
python3 -m qie_research.datasets.prepare_cifar10
python3 -m qie_research.datasets.prepare_higgs
python3 -m qie_research.datasets.prepare_covertype

# Credit Card Fraud requires manual download check
if [ -f "data/raw/creditcard.csv" ]; then
    python3 -m qie_research.datasets.prepare_credit_card_fraud
else
    echo "WARNING: data/raw/creditcard.csv not found. Skipping Credit Card Fraud preparation."
    echo "To include this dataset, download it from Kaggle and place it in data/raw/ before running."
fi

# 3. Execution Sweep
echo "--- Step 3: Executing Benchmark Sweep ---"

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

for CFG in "${CONFIGS[@]}"; do
    if [ -f "$CFG" ]; then
        echo "Processing $CFG..."
        # Check if it's credit_card_fraud and if we should skip it
        if [[ "$CFG" == *"credit_card_fraud"* ]] && [ ! -f "data/raw/credit_card_fraud_X.npy" ]; then
            echo "Skipping $CFG (data not prepared)."
            continue
        fi
        
        python3 -m qie_research.runner "$CFG" $TORCH_FLAG
    else
        echo "Error: Config $CFG not found."
    fi
done

echo "--- Sweep Complete. Results are in results/metrics/ ---"
