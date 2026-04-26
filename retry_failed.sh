#!/bin/bash
set -uo pipefail

# QIE Research: Retry Failed Runs
# Reads results/failed_runs.txt and reruns only the failed (config, seed) pairs.
# Usage: ./retry_failed.sh [--torch-only]
#
# You can also rerun a single specific run directly:
#   python3 -m qie_research.runner configs/higgs.yaml --seed 1337

TORCH_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--torch-only" ]; then
        TORCH_FLAG="--torch-only"
    fi
done

FAILED_LOG="results/failed_runs.txt"

if [ ! -f "$FAILED_LOG" ]; then
    echo "No failed runs file found at $FAILED_LOG — nothing to retry."
    exit 0
fi

TOTAL=$(grep -c . "$FAILED_LOG" 2>/dev/null || echo 0)
if [ "$TOTAL" -eq 0 ]; then
    echo "No failed runs in $FAILED_LOG — nothing to retry."
    exit 0
fi

echo "Retrying $TOTAL failed run(s) from $FAILED_LOG ..."
echo ""

DONE=0
N_FAILED=0
STILL_FAILED="results/failed_runs_retry.txt"
> "$STILL_FAILED"

while IFS=" " read -r CFG SEED; do
    [ -z "$CFG" ] && continue
    DONE=$(( DONE + 1 ))
    echo "[$DONE/$TOTAL] $CFG  seed=$SEED"

    if python3 -m qie_research.runner "$CFG" --seed "$SEED" $TORCH_FLAG; then
        : # success
    else
        N_FAILED=$(( N_FAILED + 1 ))
        echo "  STILL FAILING: $CFG seed=$SEED"
        echo "$CFG $SEED" >> "$STILL_FAILED"
    fi
done < "$FAILED_LOG"

echo ""
echo "--- Retry complete ---"
echo "    Retried : $TOTAL"
echo "    Fixed   : $(( TOTAL - N_FAILED ))"
echo "    Still failing: $N_FAILED"

if [ "$N_FAILED" -gt 0 ]; then
    echo "    Persistent failures -> $STILL_FAILED"
    echo "    Check the corresponding log in results/logs/ for the error."
    # Replace the failed log with only the still-failing ones
    mv "$STILL_FAILED" "$FAILED_LOG"
else
    rm -f "$STILL_FAILED"
    > "$FAILED_LOG"
    echo "    All previously failed runs now succeeded."
fi
