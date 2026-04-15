#!/bin/bash
# New Ablation Experiments: FFT-Router, SAIB Loss, IA-Gating
# Total: 90 runs (~4-5 hours on A10G)
#
# Usage:
#   bash scripts/run_new_ablations_race.sh              # run all sequentially
#   bash scripts/run_new_ablations_race.sh worker 1 3   # worker 1 of 3 (sharded)

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity"
SEEDS="42 43 44"
EPOCHS=15

# Worker sharding support
WORKER_ID=0
WORKER_TOTAL=1
if [ "${1:-}" = "worker" ]; then
    WORKER_ID=$(( ${2:?worker ID required} - 1 ))
    WORKER_TOTAL=${3:?total workers required}
    echo "Worker $((WORKER_ID+1))/$WORKER_TOTAL"
fi

RUN_IDX=0

should_run() {
    local idx=$1
    if [ $WORKER_TOTAL -eq 1 ]; then
        return 0
    fi
    [ $(( idx % WORKER_TOTAL )) -eq $WORKER_ID ]
}

# ============================================================
# Experiment 1: FFT-Router (18 runs)
# Spectral-Temporal routing: adds FFT amplitude features to router
# ============================================================
echo "=== FFT-Router ==="
for ds in $DATASETS; do
    for seed in $SEEDS; do
        OUT="results/rr_moa/${ds}_H96_K5_dense_frozen_${seed}_rarch-fft.json"
        if [ -f "$OUT" ]; then
            echo "SKIP (exists): $OUT"
            RUN_IDX=$((RUN_IDX + 1))
            continue
        fi
        if should_run $RUN_IDX; then
            echo "RUN: FFT-Router $ds seed=$seed"
            python3 scripts/run_rr_moa.py \
                --dataset "$ds" --seed "$seed" --epochs $EPOCHS \
                --router-arch fft --unfreeze frozen --no-baselines \
                2>&1 | tail -5
        fi
        RUN_IDX=$((RUN_IDX + 1))
    done
done

# ============================================================
# Experiment 2: SAIB Loss (36 runs)
# Statistic-Aware Information Bottleneck: auxiliary mu/sigma prediction
# ============================================================
echo "=== SAIB Loss ==="
for saib_coef in 0.1 1.0; do
    for ds in $DATASETS; do
        for seed in $SEEDS; do
            OUT="results/rr_moa/${ds}_H96_K5_dense_frozen_${seed}_saib-${saib_coef}.json"
            if [ -f "$OUT" ]; then
                echo "SKIP (exists): $OUT"
                RUN_IDX=$((RUN_IDX + 1))
                continue
            fi
            if should_run $RUN_IDX; then
                echo "RUN: SAIB(${saib_coef}) $ds seed=$seed"
                python3 scripts/run_rr_moa.py \
                    --dataset "$ds" --seed "$seed" --epochs $EPOCHS \
                    --saib-coef "$saib_coef" --unfreeze frozen --no-baselines \
                    2>&1 | tail -5
            fi
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
done

# ============================================================
# Experiment 3: IA-Gating (36 runs)
# Instance-Adaptive dual-stream: per-window backbone trust gate
# ============================================================
echo "=== IA-Gating ==="
for unfreeze in frozen last4; do
    for ds in $DATASETS; do
        for seed in $SEEDS; do
            OUT="results/gap_closing/ia-gating_${ds}_H96_${seed}.json"
            if [ "$unfreeze" != "frozen" ]; then
                OUT="results/gap_closing/ia-gating_${ds}_H96_${seed}_${unfreeze}.json"
            fi
            if [ -f "$OUT" ]; then
                echo "SKIP (exists): $OUT"
                RUN_IDX=$((RUN_IDX + 1))
                continue
            fi
            if should_run $RUN_IDX; then
                echo "RUN: IA-Gating($unfreeze) $ds seed=$seed"
                python3 scripts/run_gap_closing.py \
                    --variant ia-gating --dataset "$ds" --seed "$seed" \
                    --epochs $EPOCHS --unfreeze "$unfreeze" \
                    2>&1 | tail -10
            fi
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
done

echo ""
echo "=== All done. Total runs attempted: $RUN_IDX ==="
