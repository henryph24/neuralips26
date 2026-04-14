#!/bin/bash
# B7: Cross-task classification — "one backbone, three tasks"
#
# Frozen MOMENT-small on 3 UEA classification datasets with
# multiple pooling strategies. Demonstrates the same backbone
# used for forecasting + imputation also supports classification.
#
# Grid: 3 datasets × 2 pooling × 2 heads × 3 seeds = 36 runs.
# Runtime: ~30 min (classification datasets are small).
#
# Invocation:
#   tmux new-session -d -s b7 'cd ~/neuralips26 && bash scripts/run_b7_classification_race.sh 2>&1 | tee results/b7_run.log'

set -e
DEVICE="cuda"
EPOCHS=20
PYTHON=python3

DATASETS=(EthanolConcentration JapaneseVowels BasicMotions)
POOLINGS=(mean max)
HEADS=(linear mlp1)
SEEDS=(42 43 44)

echo "================================================================"
echo "B7 Cross-Task Classification — $(date)"
echo "================================================================"

mkdir -p results/classification

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for pool in "${POOLINGS[@]}"; do
    for head in "${HEADS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        OUT="results/classification/${ds}_${pool}_${head}_${seed}.json"
        if [ -f "$OUT" ]; then
          SKIPPED=$((SKIPPED + 1))
          RUN_IDX=$((RUN_IDX + 1))
          continue
        fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds | $pool+$head seed=$seed"
        set +e
        $PYTHON scripts/run_classification.py \
          --dataset "$ds" \
          --pooling "$pool" \
          --head "$head" \
          --seed "$seed" \
          --epochs "$EPOCHS" \
          --device "$DEVICE" \
          > /tmp/b7_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC):"
          tail -5 /tmp/b7_${RUN_IDX}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
          grep "Accuracy:" /tmp/b7_${RUN_IDX}.out | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX + 1))
      done
    done
  done
done

echo ""
echo "================================================================"
echo "B7 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
