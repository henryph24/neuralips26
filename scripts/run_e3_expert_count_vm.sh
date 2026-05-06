#!/bin/bash
# E3: Expert count scaling — how does RR-MoA scale with K={3,5,7,10}?
#
# K=5 is the default (existing results will be skipped). K=3 tests if
# fewer experts suffice. K=7 and K=10 test if more helps. Top-k stays
# at 2 for all (sparse routing).
#
# Grid: 4 K × 3 datasets × 3 seeds = 36 runs (9 skipped for K=5).
# Runtime: ~30 min solo on A10G.
#
# Invocation:
#   tmux new-session -d -s e3 'cd ~/neuralips26 && bash scripts/run_e3_expert_count_vm.sh 2>&1 | tee results/e3_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

K_VALUES=(3 5 7 10)
DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)

echo "================================================================"
echo "E3 Expert Count Scaling — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for K in "${K_VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K${K}_top2_frozen_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] K=$K | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --K "$K" \
        --top-k 2 \
        --unfreeze frozen \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --no-baselines \
        > /tmp/e3_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/e3_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|mse|routing_entropy" /tmp/e3_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "E3 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
