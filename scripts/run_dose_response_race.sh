#!/bin/bash
# Dose-response + shuffle ablation sweep — normalization MI destruction curve.
#
# Grid:
#   Dose-response: alpha in {0.25, 0.50, 0.75, 1.00} x 6 datasets x 5 seeds = 120 runs
#     (alpha=0.0 already exists as standard raw-mode results in results/rr_moa/)
#   Shuffle: 6 datasets x 5 seeds = 30 runs
#   Total: 150 runs, ~2.5 GPU-hours on A10G
#
# Invocation:
#   bash scripts/run_dose_response_race.sh                  # single worker
#   bash scripts/run_dose_response_race.sh worker K N       # worker K of N
#
# Launch 4 workers in parallel via tmux:
#   tmux new-session -d -s dr_w1 'cd ~/neuralips26 && bash scripts/run_dose_response_race.sh worker 1 4 2>&1 | tee -a results/dr_w1.log'
#   tmux new-session -d -s dr_w2 'cd ~/neuralips26 && bash scripts/run_dose_response_race.sh worker 2 4 2>&1 | tee -a results/dr_w2.log'
#   tmux new-session -d -s dr_w3 'cd ~/neuralips26 && bash scripts/run_dose_response_race.sh worker 3 4 2>&1 | tee -a results/dr_w3.log'
#   tmux new-session -d -s dr_w4 'cd ~/neuralips26 && bash scripts/run_dose_response_race.sh worker 4 4 2>&1 | tee -a results/dr_w4.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/rr_moa"
mkdir -p "$RESULTS_DIR"

MODE="${1:-single}"
WORKER_ID="${2:-1}"
NUM_WORKERS="${3:-1}"

echo "================================================================"
echo "Dose-response + shuffle ablation sweep — $(date)"
echo "Mode: $MODE  Worker: $WORKER_ID / $NUM_WORKERS"
echo "================================================================"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)
# alpha=0.0 included for consistent baseline comparison through same code path
ALPHAS=(0.00 0.25 0.50 0.75 1.00)

RUN_IDX=0
LAUNCHED=0
SKIPPED_EXISTING=0
SKIPPED_WORKER=0
FAILED=0

# --- Phase 1: Dose-response (partial normalization) ---
for alpha in "${ALPHAS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
        SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_dense_frozen_${seed}_router-partial_alpha-${alpha}.json"
      if [ -f "$OUTFILE" ]; then
        SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] partial alpha=$alpha | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --seed "$seed" \
        --unfreeze frozen \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --router-input-mode partial \
        --alpha "$alpha" \
        --no-baselines \
        > /tmp/dose_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, last 5 lines):"
        tail -5 /tmp/dose_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

# --- Phase 2: Shuffle ablation ---
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
      RUN_IDX=$((RUN_IDX + 1))
      continue
    fi

    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_dense_frozen_${seed}_router-shuffled.json"
    if [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
      RUN_IDX=$((RUN_IDX + 1))
      continue
    fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] shuffled | $ds seed=$seed"
    set +e
    $PYTHON scripts/run_rr_moa.py \
      --dataset "$ds" \
      --seed "$seed" \
      --unfreeze frozen \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --router-input-mode shuffled \
      --no-baselines \
      > /tmp/dose_${WORKER_ID}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC, last 5 lines):"
      tail -5 /tmp/dose_${WORKER_ID}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "Dose-response worker $WORKER_ID/$NUM_WORKERS DONE at $(date)"
echo "  Launched:          $LAUNCHED"
echo "  Skipped (existed): $SKIPPED_EXISTING"
echo "  Skipped (worker):  $SKIPPED_WORKER"
echo "  Failed:            $FAILED"
echo "  Total scanned:     $RUN_IDX"
echo "================================================================"
