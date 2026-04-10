#!/bin/bash
# E1: Raw-MLP MoE ablation (no TSFM backbone at all) — Claim D defense.
#
# A critical reviewer asked for a baseline showing a Mixture of Adapters
# operating ONLY on raw MLPs, with no TSFM at all. If this matches or beats
# Dual-Stream MSE, the 35M-parameter TSFM is exposed as redundant. The most
# likely outcome is that Raw-MLP MoE is strictly worse than Dual-Stream (the
# TSFM contributes complementary signal, consistent with the learned blend
# alpha ~ 0.49 in tab:gap_closing), which defuses the attack cleanly.
#
# Grid: 6 datasets × 3 seeds = 18 runs.
# Runtime: ~2-3 min per run (no TSFM forward pass, MLPs only). Under 5-way
# contention with the running B1/B2 sweeps, expect ~6-8 min per run and a
# total wall-clock of ~60-90 min for the whole sweep.
#
# Invocation:
#   tmux new-session -d -s e1 'cd ~/neuralips26 && bash scripts/run_raw_mlp_moe_race.sh 2>&1 | tee results/e1_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/raw_mlp_moe"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44)

echo "================================================================"
echo "E1 raw-MLP MoE (no TSFM) — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="${RESULTS_DIR}/${ds}_H96_K5_top2_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
      RUN_IDX=$((RUN_IDX + 1))
      continue
    fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds seed=$seed"
    set +e
    $PYTHON scripts/run_raw_mlp_moe.py \
      --dataset "$ds" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --top-k 2 \
      > /tmp/e1_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC, tail):"
      tail -5 /tmp/e1_${RUN_IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
      # Echo the key result line for live monitoring
      grep -E "MSE=|Routing entropy" /tmp/e1_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "E1 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
