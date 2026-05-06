#!/bin/bash
# Intervention 2: Run RR-MoA + AdaMix + best-fixed on Exchange and Solar datasets
# for the signal-ratio correlation extension (N=7 → N=9)
#
# Prerequisites:
#   1. Download exchange_rate.csv and solar_AL.csv to ~/neuralips26/data/
#      (from the Time-Series-Library Google Drive or Autoformer dataset collection)
#   2. rsync updated scripts to GPU VM
#
# Usage: bash scripts/run_new_datasets_vm.sh [Exchange|Solar|all]

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-all}"
SEEDS="42 43 44"

run_dataset() {
    local ds="$1"
    echo "=== Processing $ds ==="

    for seed in $SEEDS; do
        # RR-MoA (frozen, Top-2)
        OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}.json"
        if [ -f "$OUT" ]; then
            echo "  SKIP RR-MoA seed=$seed (exists)"
        else
            echo "  RR-MoA seed=$seed ..."
            python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --top-k 2 --unfreeze frozen 2>&1 | tail -1
        fi

        # AdaMix (frozen)
        OUT="results/adamix/${ds}_H96_K5_frozen_${seed}.json"
        if [ -f "$OUT" ]; then
            echo "  SKIP AdaMix frozen seed=$seed (exists)"
        else
            echo "  AdaMix frozen seed=$seed ..."
            python3 scripts/run_adamix.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --unfreeze frozen 2>&1 | tail -1
        fi

        # AdaMix (last-4 unfrozen, to measure collapse)
        OUT="results/adamix/${ds}_H96_K5_last4_${seed}.json"
        if [ -f "$OUT" ]; then
            echo "  SKIP AdaMix last4 seed=$seed (exists)"
        else
            echo "  AdaMix last4 seed=$seed ..."
            python3 scripts/run_adamix.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --unfreeze last4 2>&1 | tail -1
        fi
    done

    echo "  Done with $ds"
}

if [ "$DATASET" = "all" ]; then
    run_dataset "Exchange"
    run_dataset "Solar"
elif [ "$DATASET" = "Exchange" ] || [ "$DATASET" = "Solar" ]; then
    run_dataset "$DATASET"
else
    echo "Usage: $0 [Exchange|Solar|all]"
    exit 1
fi

echo ""
echo "=== All done. Now run locally: ==="
echo "  python3 scripts/analyze_routing_signal_ratio.py"
echo "  python3 scripts/bootstrap_correlation.py"
