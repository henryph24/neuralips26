#!/bin/bash
# E6+E7: Completion sweep — Imputation (Exchange+Solar) + Raw-MLP MoE 5-seed
#
# E6: Imputation on Exchange+Solar (2 datasets × 5 seeds = 10 runs)
#     Completes the 8-dataset imputation grid (currently 6/8).
#
# E7: Raw-MLP MoE seeds 45-46 (6 datasets × 2 seeds = 12 runs)
#     Upgrades the "honest disclosure" ablation from 3 to 5 seeds.
#
# Total: 22 runs (~15 min on A10G)
#
# Invocation:
#   tmux new-session -d -s e6 'cd ~/neuralips26 && bash scripts/run_completion_race.sh 2>&1 | tee results/e6_completion.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

mkdir -p results/imputation results/raw_mlp_moe

echo "================================================================"
echo "E6+E7 Completion sweep — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

# --- E6: Imputation Exchange + Solar ---
for ds in Exchange Solar; do
  for seed in 42 43 44 45 46; do
    OUT="results/imputation/${ds}_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] Imputation | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_imputation.py \
        --dataset "$ds" --seed "$seed" --device "$DEVICE" \
        > /tmp/e6_imp_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/e6_imp_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

# --- E7: Raw-MLP MoE seeds 45-46 ---
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 45 46; do
    OUT="results/raw_mlp_moe/${ds}_H96_K5_top2_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] Raw-MLP-MoE | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_raw_mlp_moe.py \
        --dataset "$ds" --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        > /tmp/e7_rawmlp_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/e7_rawmlp_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "================================================================"
echo "E6+E7 Completion DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
