#!/bin/bash
# E2: 10-seed robustness extension on RR-MoA core trio.
#
# The current RR-MoA main table uses seeds {42,43,44,45,46} (5 seeds). To
# pre-empt a reviewer "is the 54/54 wins claim robust to seed choice?" we
# extend to 10 seeds by running {47,48,49,50,51} on the 3 canonical datasets
# under the frozen Top-2 setting that powers the headline claim.
#
# Grid: 3 datasets × 5 new seeds = 15 runs.
# Runtime: ~5 min per run solo, ~8-10 min under contention with B1/B2/E1
# = ~2-2.5 h wall-clock. Finishes before B1.
#
# Invocation:
#   tmux new-session -d -s e2 'cd ~/neuralips26 && bash scripts/run_10seed_vm.sh 2>&1 | tee results/e2_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(47 48 49 50 51)

echo "================================================================"
echo "E2 10-seed robustness (RR-MoA frozen Top-2) — $(date)"
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
      RUN_IDX=$((RUN_IDX + 1))
      continue
    fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds seed=$seed"
    set +e
    $PYTHON scripts/run_rr_moa.py \
      --dataset "$ds" \
      --unfreeze frozen \
      --top-k 2 \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      > /tmp/e2_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC, tail):"
      tail -5 /tmp/e2_${RUN_IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
      grep -E "MSE=|Delta" /tmp/e2_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "E2 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
