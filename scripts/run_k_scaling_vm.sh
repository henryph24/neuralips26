#!/bin/bash
# K-scaling sweep — tests K=15,20 for both RR-MoA and AdaMix.
# Confirms collapse is K-independent and tests if larger pools help.
#
# Grid: K in {15, 20} x 6 datasets x 3 seeds x {rr_moa, adamix} = 72 runs
# Expected: ~2 GPU-hours on A10G
#
# Invocation:
#   bash scripts/run_k_scaling_vm.sh                  # single worker
#   bash scripts/run_k_scaling_vm.sh worker K N       # worker K of N

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

MODE="${1:-single}"
WORKER_ID="${2:-1}"
NUM_WORKERS="${3:-1}"

echo "================================================================"
echo "K-scaling sweep — $(date)"
echo "Mode: $MODE  Worker: $WORKER_ID / $NUM_WORKERS"
echo "================================================================"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44)
K_VALUES=(15 20)

RUN_IDX=0
LAUNCHED=0
SKIPPED_EXISTING=0
SKIPPED_WORKER=0
FAILED=0

# --- Phase 1: RR-MoA with large K ---
for K in "${K_VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
        SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      OUTFILE="results/rr_moa/${ds}_H96_K${K}_dense_frozen_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA K=$K | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --seed "$seed" \
        --K "$K" \
        --unfreeze frozen \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --no-baselines \
        > /tmp/kscale_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, last 5 lines):"
        tail -5 /tmp/kscale_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

# --- Phase 2: AdaMix with large K (collapse confirmation) ---
for K in "${K_VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
        SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      OUTFILE="results/adamix/${ds}_H96_K${K}_last4_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] AdaMix K=$K | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_adamix.py \
        --dataset "$ds" \
        --seed "$seed" \
        --K "$K" \
        --unfreeze last4 \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --run-baselines no \
        > /tmp/kscale_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, last 5 lines):"
        tail -5 /tmp/kscale_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "K-scaling worker $WORKER_ID/$NUM_WORKERS DONE at $(date)"
echo "  Launched:          $LAUNCHED"
echo "  Skipped (existed): $SKIPPED_EXISTING"
echo "  Skipped (worker):  $SKIPPED_WORKER"
echo "  Failed:            $FAILED"
echo "  Total scanned:     $RUN_IDX"
echo "================================================================"
