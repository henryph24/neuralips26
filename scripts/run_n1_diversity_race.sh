#!/bin/bash
# N1: Expert diversity ablation — do 5 identical experts match 5 diverse ones?
#
# Tests whether RR-MoA's improvement comes from the routing mechanism alone
# or requires architectural diversity in the expert pool. Three identical
# pools (5x mean, 5x conv1d, 5x attn) are compared against the canonical
# diverse pool (mean/last/max/attn/conv1d).
#
# Grid: 3 datasets × 3 seeds × 3 pools = 27 runs.
# Runtime: ~30s-3min per run. Total ~30 min solo on A10G.
#
# Invocation:
#   tmux new-session -d -s n1 'cd ~/neuralips26 && bash scripts/run_n1_diversity_race.sh 2>&1 | tee results/n1_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)
POOLS=(identical-mean identical-conv1d identical-attn)

echo "================================================================"
echo "N1 Expert Diversity Ablation — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for pool in "${POOLS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_pool-${pool}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] $pool | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --unfreeze frozen \
        --top-k 2 \
        --expert-pool "$pool" \
        > /tmp/n1_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/n1_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|mse|routing_entropy" /tmp/n1_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "N1 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
