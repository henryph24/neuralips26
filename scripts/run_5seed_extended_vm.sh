#!/bin/bash
# 5-seed completion for extended datasets (ETTh2, ETTm2, Electricity).
#
# The main tab:rrmoa uses 3 seeds (42-44) for these datasets. Adding
# seeds 45, 46 upgrades to 5-seed reporting and tightens significance.
#
# Grid: 3 datasets × 3 freeze levels × 2 new seeds = 18 runs.
# Runtime: ~20 min solo on A10G.
#
# Invocation:
#   tmux new-session -d -s ext5 'cd ~/neuralips26 && bash scripts/run_5seed_extended_vm.sh 2>&1 | tee results/ext5_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh2 ETTm2 Electricity)
FREEZE_LEVELS=(frozen last2 last4)
SEEDS=(45 46)

echo "================================================================"
echo "5-Seed Extended Datasets — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for freeze in "${FREEZE_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        continue
      fi

      echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds $freeze seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" \
        --unfreeze "$freeze" \
        --top-k 2 \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        > /tmp/ext5_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -5 /tmp/ext5_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=" /tmp/ext5_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
      fi
      RUN_IDX=$((RUN_IDX + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "5-Seed Extended DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
