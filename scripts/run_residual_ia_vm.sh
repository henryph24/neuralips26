#!/bin/bash
# Residual-IA gap-closing sweep — closes the DLinear gap.
#
# Phase 1: raw_hidden sweep {128, 192, 256} x 6 datasets x 3 seeds = 54 runs
# Phase 2: best raw_hidden x 6 datasets x 5 seeds = 30 runs (run after P1 review)
#
# Invocation:
#   tmux new-session -d -s ria_p1 'cd ~/neuralips26 && bash scripts/run_residual_ia_vm.sh phase1 2>&1 | tee results/ria_p1.log'
#   tmux new-session -d -s ria_p2 'cd ~/neuralips26 && bash scripts/run_residual_ia_vm.sh phase2 256 2>&1 | tee results/ria_p2.log'
#
# Worker sharding (optional): bash scripts/run_residual_ia_vm.sh phase1 worker 1 2

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)

PHASE="${1:-phase1}"

# Worker sharding support
WORKER_ID=0
NUM_WORKERS=1
if [ "${2:-}" = "worker" ]; then
    WORKER_ID="${3:-1}"
    NUM_WORKERS="${4:-1}"
    echo "Worker mode: $WORKER_ID / $NUM_WORKERS"
fi

echo "================================================================"
echo "Residual-IA gap-closing — Phase: $PHASE — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

run_one() {
    local ds="$1" seed="$2" raw_hidden="$3"
    local OUT="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh${raw_hidden}.json"

    if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    # Worker sharding check
    if [ $NUM_WORKERS -gt 1 ]; then
        local mod=$((RUN_IDX % NUM_WORKERS))
        local target=$((WORKER_ID - 1))
        if [ $mod -ne $target ]; then
            return
        fi
    fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds seed=$seed rh=$raw_hidden"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --variant residual-ia \
        --dataset "$ds" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --top-k 2 \
        --raw-hidden "$raw_hidden" \
        > /tmp/ria_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/ria_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|gate_values" /tmp/ria_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
    fi
}

if [ "$PHASE" = "phase1" ]; then
    # Phase 1: raw_hidden sweep
    RAW_HIDDENS=(128 192 256)
    SEEDS=(42 43 44)
    for rh in "${RAW_HIDDENS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_one "$ds" "$seed" "$rh"
                RUN_IDX=$((RUN_IDX + 1))
            done
        done
    done

elif [ "$PHASE" = "phase2" ]; then
    # Phase 2: best raw_hidden (passed as $2 or default 192) x 5 seeds
    BEST_RH="${2:-192}"
    SEEDS=(42 43 44 45 46)
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$ds" "$seed" "$BEST_RH"
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
fi

echo ""
echo "================================================================"
echo "Residual-IA $PHASE DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
