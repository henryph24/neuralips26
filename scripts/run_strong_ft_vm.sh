#!/bin/bash
# Strong FT sweep on GPU VM: 3 datasets x 3 seeds x 10 configs = 90 runs
# Supports worker sharding: bash scripts/run_strong_ft_vm.sh worker K N
#
# Usage:
#   bash scripts/run_strong_ft_vm.sh              # run all 90
#   bash scripts/run_strong_ft_vm.sh worker 1 3   # worker 1 of 3
set -e

DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)
NUM_CONFIGS=10  # 0..9

WORKER_ID=0
NUM_WORKERS=1
if [ "$1" = "worker" ]; then
    WORKER_ID=$(( $2 - 1 ))
    NUM_WORKERS=$3
    echo "Worker $2 of $NUM_WORKERS"
fi

RUN_IDX=0
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CFG_IDX in $(seq 0 $(( NUM_CONFIGS - 1 ))); do
            if [ $(( RUN_IDX % NUM_WORKERS )) -eq $WORKER_ID ]; then
                echo "=== Run $RUN_IDX: $DS seed=$SEED config=$CFG_IDX ==="
                python3 scripts/run_strong_ft.py \
                    --dataset "$DS" --seed "$SEED" --config-idx "$CFG_IDX" \
                    --device cuda || echo "FAILED: $DS $SEED $CFG_IDX"
            fi
            RUN_IDX=$(( RUN_IDX + 1 ))
        done
    done
done
echo "Done. Total attempted: $RUN_IDX runs."
