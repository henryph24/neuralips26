#!/bin/bash
# P2: 10-seed extension completion — ETTh2, ETTm2, Electricity (seeds 47-51)
#
# Goal: Complete the 10-seed robustness check for all 6 datasets.
# Currently ETTh1/ETTm1/Weather have seeds 47-51; the other 3 are missing.
#
# Grid: 3 datasets × 5 seeds = 15 runs (RR-MoA frozen, MOMENT-small)
#
# Invocation:
#   tmux new-session -d -s p2 'cd ~/neuralips26 && bash scripts/run_10seed_completion_vm.sh 2>&1 | tee results/p2_10seed.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh2 ETTm2 Electricity)
SEEDS=(47 48 49 50 51)

mkdir -p results/rr_moa

echo "================================================================"
echo "P2 10-seed completion — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA 10-seed | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze frozen --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --no-baselines \
        > /tmp/p2_rrmoa_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/p2_rrmoa_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "P2 10-seed completion DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
