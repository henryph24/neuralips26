#!/bin/bash
# DLinear multi-horizon extension — fill the comparison grid at H=192/336/720.
#
# The appendix tab:horizon has RR-MoA multi-horizon data but DLinear
# comparison may be incomplete for extended horizons and datasets.
#
# Grid: 6 datasets × 3 horizons × 3 seeds = 54 runs.
# Runtime: ~30s per DLinear run (no backbone), total ~30 min.
#
# Invocation:
#   tmux new-session -d -s dlmh 'cd ~/neuralips26 && bash scripts/run_dlinear_multihor_race.sh 2>&1 | tee results/dlmh_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
HORIZONS=(192 336 720)
SEEDS=(42 43 44)

echo "================================================================"
echo "DLinear Multi-Horizon — $(date)"
echo "================================================================"

mkdir -p results/dlinear

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for H in "${HORIZONS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/dlinear/${ds}_H${H}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] H=$H | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_dlinear_baseline.py \
        --dataset "$ds" \
        --horizon "$H" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        > /tmp/dlmh_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/dlmh_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|dlinear_mse" /tmp/dlmh_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "DLinear Multi-Horizon DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
