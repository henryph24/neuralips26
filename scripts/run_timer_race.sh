#!/bin/bash
# Timer-XL backbone experiments: RR-MoA + AdaMix on 3 core datasets × 3 seeds
# Tests whether Timer-XL (no RevIN, LayerNorm only) is a negative control like Chronos
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="thuml/timer-base-84m"
DATASETS="ETTh1 ETTm1 Weather"
SEEDS="42 43 44"

for ds in $DATASETS; do
    for seed in $SEEDS; do
        # RR-MoA (frozen)
        OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_bb-timer-base-84m.json"
        if [ -f "$OUT" ]; then
            echo "SKIP RR-MoA $ds seed=$seed"
        else
            echo "RR-MoA $ds seed=$seed ..."
            python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --top-k 2 --unfreeze frozen --backbone "$BACKBONE" 2>&1 | tail -1
        fi

        # AdaMix frozen
        OUT="results/adamix/${ds}_H96_K5_frozen_${seed}_bb-timer-base-84m.json"
        if [ -f "$OUT" ]; then
            echo "SKIP AdaMix frozen $ds seed=$seed"
        else
            echo "AdaMix frozen $ds seed=$seed ..."
            python3 scripts/run_adamix.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --unfreeze frozen --backbone "$BACKBONE" 2>&1 | tail -1
        fi

        # AdaMix last-4 (to test for collapse)
        OUT="results/adamix/${ds}_H96_K5_last4_${seed}_bb-timer-base-84m.json"
        if [ -f "$OUT" ]; then
            echo "SKIP AdaMix last4 $ds seed=$seed"
        else
            echo "AdaMix last4 $ds seed=$seed ..."
            python3 scripts/run_adamix.py --dataset "$ds" --seed "$seed" --epochs 15 \
                --unfreeze last4 --backbone "$BACKBONE" 2>&1 | tail -1
        fi
    done
done

echo ""
echo "=== Timer-XL experiments complete ==="
echo "Expected: NO routing collapse (no RevIN), RR-MoA ~= AdaMix (negative control)"
