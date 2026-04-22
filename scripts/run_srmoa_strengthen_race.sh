#!/bin/bash
# SR-MoA strengthening sweep — 87 new runs across 4 experiments.
# Addresses: W1 "obvious fix", W3 "single backbone", Frozen Paradox replication.
#
# Usage:
#   bash scripts/run_srmoa_strengthen_race.sh              # run all sequentially
#   bash scripts/run_srmoa_strengthen_race.sh worker 1 2   # sharded (worker K of N)

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="results/self_routed_moa"
mkdir -p "$RESULTS"

WORKER_K=0
WORKER_N=1
if [[ "${1:-}" == "worker" ]]; then
    WORKER_K=$(( ${2:?worker index required} - 1 ))
    WORKER_N=${3:?total workers required}
    echo "Worker $((WORKER_K+1)) of $WORKER_N"
fi

RUN_IDX=0

run_one() {
    local dataset="$1" horizon="$2" seed="$3" unfreeze="$4" routing_input="$5" backbone="$6"

    # Worker sharding
    if (( RUN_IDX % WORKER_N != WORKER_K )); then
        RUN_IDX=$((RUN_IDX + 1))
        return
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # Build expected output filename
    local suffix="gated_gh16"
    if [[ "$routing_input" != "raw" ]]; then
        suffix="${suffix}_ri-${routing_input}"
    fi

    local bb_suffix=""
    if [[ "$backbone" == *"timer"* ]]; then
        bb_suffix="_bb-timer-base-84m"
    elif [[ "$backbone" == *"moirai-moe"* || "$backbone" == *"moirai_moe"* ]]; then
        bb_suffix="_bb-moirai-moe"
    elif [[ "$backbone" == *"moirai"* ]]; then
        bb_suffix="_bb-moirai"
    elif [[ "$backbone" == *"chronos"* ]]; then
        bb_suffix="_bb-chronos"
    elif [[ "$backbone" == *"large"* ]]; then
        bb_suffix="_bb-moment-large"
    fi
    suffix="${suffix}${bb_suffix}"

    local out="${RESULTS}/${dataset}_H${horizon}_K5_${unfreeze}_${seed}_${suffix}.json"

    if [[ -f "$out" ]]; then
        echo "SKIP (exists): $out"
        return
    fi

    echo "RUN: $dataset H=$horizon seed=$seed unfreeze=$unfreeze input=$routing_input backbone=$(basename $backbone)"
    python3 scripts/run_self_routed_moa.py \
        --dataset "$dataset" \
        --horizon "$horizon" \
        --seed "$seed" \
        --unfreeze "$unfreeze" \
        --routing-input "$routing_input" \
        --routing-mode gated \
        --gate-hidden 16 \
        --backbone "$backbone" \
        --epochs 15 \
        --device cuda \
        --results-dir "$RESULTS"
}

MOMENT="AutonLab/MOMENT-1-small"
TIMER="thuml/timer-base-84m"
MOIRAI_MOE="Salesforce/moirai-moe-1.0-R-small"

echo "============================================"
echo "E1: Hidden-state control — all 6 datasets, 5 seeds"
echo "============================================"
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for seed in 42 43 44 45 46; do
        run_one "$ds" 96 "$seed" frozen hidden "$MOMENT"
    done
done

echo "============================================"
echo "E2: Timer-XL negative control — 3 datasets, 5 seeds, raw + hidden"
echo "============================================"
for ds in ETTh1 ETTm1 Weather; do
    for seed in 42 43 44 45 46; do
        run_one "$ds" 96 "$seed" frozen raw "$TIMER"
        run_one "$ds" 96 "$seed" frozen hidden "$TIMER"
    done
done

echo "============================================"
echo "E3: Freeze ablation — all 6 datasets, last2/last4, 3 seeds"
echo "============================================"
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for unfreeze in last2 last4; do
        for seed in 42 43 44; do
            run_one "$ds" 96 "$seed" "$unfreeze" raw "$MOMENT"
        done
    done
done

echo "============================================"
echo "E4: Cross-backbone — Moirai-MoE + Timer-XL, 3 datasets, 3 seeds"
echo "============================================"
for ds in ETTh1 ETTm1 Weather; do
    for seed in 42 43 44; do
        run_one "$ds" 96 "$seed" frozen raw "$MOIRAI_MOE"
        run_one "$ds" 96 "$seed" frozen raw "$TIMER"
    done
done

echo ""
echo "ALL DONE."
