#!/bin/bash
# Router architecture grid: proves "any raw router works" (not architecture-dependent).
#
# 4 non-default routers x 6 datasets x 3 seeds = 72 runs.
# conv (default) already has 400+ runs — skip it here.
#
# Invocation:
#   tmux new-session -d -s rarch 'cd ~/neuralips26 && bash scripts/run_router_arch_race.sh 2>&1 | tee results/rarch.log'
#
# Worker sharding: bash scripts/run_router_arch_race.sh worker 1 2

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/rr_moa"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44)
ROUTERS=(stats ssr multiscale fft)

# Worker sharding
WORKER_ID=0
NUM_WORKERS=1
if [ "${1:-}" = "worker" ]; then
    WORKER_ID="${2:-1}"
    NUM_WORKERS="${3:-1}"
    echo "Worker mode: $WORKER_ID / $NUM_WORKERS"
fi

echo "================================================================"
echo "Router architecture grid — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for rarch in "${ROUTERS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            OUT="${RESULTS_DIR}/${ds}_H96_K5_top2_frozen_${seed}_rarch-${rarch}.json"
            # Also check without the rarch suffix (some may have been run differently)
            if [ -f "$OUT" ]; then
                SKIPPED=$((SKIPPED + 1))
                RUN_IDX=$((RUN_IDX + 1))
                continue
            fi

            # Worker sharding
            if [ $NUM_WORKERS -gt 1 ]; then
                mod=$((RUN_IDX % NUM_WORKERS))
                target=$((WORKER_ID - 1))
                if [ $mod -ne $target ]; then
                    RUN_IDX=$((RUN_IDX + 1))
                    continue
                fi
            fi

            echo "[$(date +%H:%M:%S)] [$RUN_IDX] $rarch $ds seed=$seed"
            set +e
            $PYTHON scripts/run_rr_moa.py \
                --dataset "$ds" \
                --seed "$seed" \
                --epochs "$EPOCHS" \
                --device "$DEVICE" \
                --top-k 2 \
                --unfreeze frozen \
                --router-arch "$rarch" \
                > /tmp/rarch_${RUN_IDX}.out 2>&1
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then
                FAILED=$((FAILED + 1))
                echo "  FAILED (rc=$RC):"
                tail -5 /tmp/rarch_${RUN_IDX}.out | sed 's/^/    /'
            else
                LAUNCHED=$((LAUNCHED + 1))
                grep -E "MSE=|Routing entropy" /tmp/rarch_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
            fi
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
done

echo ""
echo "================================================================"
echo "Router arch grid DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
