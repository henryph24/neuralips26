#!/bin/bash
# N2a: Per-window regime robustness analysis.
#
# Tests whether RR-MoA's raw router adapts to different temporal regimes
# by comparing per-quartile MSE coefficient of variation.
#
# Grid: 3 datasets × 3 seeds = 9 runs.
# Runtime: ~2-4 min per run. Total ~30 min on A10G.
#
# Invocation:
#   tmux new-session -d -s n2 'cd ~/neuralips26 && bash scripts/run_n2_regime_race.sh 2>&1 | tee results/n2_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)

echo "================================================================"
echo "N2a Regime Robustness Analysis — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="results/regime_robustness/${ds}_H96_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
      RUN_IDX=$((RUN_IDX + 1))
      continue
    fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds seed=$seed"
    set +e
    $PYTHON scripts/run_n2_regime_robustness.py \
      --dataset "$ds" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      > /tmp/n2_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC):"
      tail -5 /tmp/n2_${RUN_IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
      grep -E "CV by|MSE=" /tmp/n2_${RUN_IDX}.out | tail -4 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "N2a DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
