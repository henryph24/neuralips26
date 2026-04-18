#!/bin/bash
# E4+E5: Multi-horizon 5-seed completion — RR-MoA + DLinear
#
# Goal: Upgrade multi-horizon table from 3 seeds (42-44) to 5 seeds (42-46).
# This enables Wilcoxon signed-rank on the horizon grid, which reviewers
# will expect for any table claiming statistical significance.
#
# Grid: 3 horizons × 6 datasets × 2 missing seeds × 2 methods = 72 runs
# (Seeds 42-44 already have results; only 45-46 need to run.)
#
# Invocation:
#   tmux new-session -d -s e4 'cd ~/neuralips26 && bash scripts/run_horizon_5seed_race.sh 2>&1 | tee results/e4_horizon.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
HORIZONS=(192 336 720)
SEEDS=(45 46)

mkdir -p results/rr_moa results/dlinear

echo "================================================================"
echo "E4+E5 Multi-horizon 5-seed completion — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for h in "${HORIZONS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      # -------- RR-MoA (Top-2, frozen, MOMENT-small) --------
      OUT="results/rr_moa/${ds}_H${h}_K5_top2_frozen_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
      else
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA H=${h} | $ds seed=$seed"
        set +e
        $PYTHON scripts/run_rr_moa.py \
          --dataset "$ds" --horizon "$h" --unfreeze frozen --top-k 2 \
          --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
          --no-baselines \
          > /tmp/e4_rrmoa_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, tail):"
          tail -5 /tmp/e4_rrmoa_${RUN_IDX}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
        fi
      fi
      RUN_IDX=$((RUN_IDX + 1))

      # -------- DLinear baseline --------
      OUT="results/dlinear/${ds}_H${h}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
      else
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] DLinear H=${h} | $ds seed=$seed"
        set +e
        $PYTHON scripts/run_dlinear_baseline.py \
          --dataset "$ds" --horizon "$h" \
          --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
          > /tmp/e4_dlinear_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, tail):"
          tail -5 /tmp/e4_dlinear_${RUN_IDX}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
        fi
      fi
      RUN_IDX=$((RUN_IDX + 1))

    done
  done
done

echo ""
echo "================================================================"
echo "E4+E5 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
