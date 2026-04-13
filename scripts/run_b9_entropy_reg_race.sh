#!/bin/bash
# B9: Entropy regularization on RR-MoA's raw router (symmetry test).
#
# Tests whether adding entropy reg to the raw router changes anything.
# Expected: no, because the raw router doesn't collapse. This confirms
# the collapse is the problem, not the lack of regularization.
#
# Grid: 3 λ × 3 datasets × 3 seeds = 27 runs.
# Runtime: ~30 min solo on A10G.
#
# Invocation:
#   tmux new-session -d -s b9 'cd ~/neuralips26 && bash scripts/run_b9_entropy_reg_race.sh 2>&1 | tee results/b9_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

LAMBDAS=(0.01 0.1 1.0)
DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)

echo "================================================================"
echo "B9 Entropy Reg on RR-MoA — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for lam in "${LAMBDAS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_entreg-${lam}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] λ=$lam | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --unfreeze frozen \
        --top-k 2 \
        --entropy-reg "$lam" \
        --no-baselines \
        > /tmp/b9_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/b9_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|mse|routing_entropy" /tmp/b9_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "B9 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
