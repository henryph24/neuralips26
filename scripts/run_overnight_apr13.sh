#!/bin/bash
# Overnight batch — Apr 13-14 2026
# Three experiments in sequence, ~5h total on A10G.
#
# A. Moirai-MoE unfreezing control (36 runs, ~3h)
# B. Chronos extended to 6 datasets (27 runs, ~1.5h)
# C. Router temperature scaling (27 runs, ~30min)
#
# Invocation:
#   tmux new-session -d -s overnight 'cd ~/neuralips26 && bash scripts/run_overnight_apr13.sh 2>&1 | tee results/overnight_apr13.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

echo "================================================================"
echo "OVERNIGHT BATCH — $(date)"
echo "================================================================"

###############################################
# A. Moirai-MoE unfreezing control
# Tests: does unfreezing cause routing collapse
# on a non-RevIN backbone? Expected: NO.
###############################################
echo ""
echo "============================================================"
echo "=== A: Moirai-MoE Unfreezing Control ==="
echo "============================================================"

DATASETS_A=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
FREEZE_A=(last2 last4)
SEEDS_A=(42 43 44)
MOE_BB="Salesforce/moirai-moe-1.0-R-small"

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

# RR-MoA on Moirai-MoE with unfreezing
for freeze in "${FREEZE_A[@]}"; do
  for ds in "${DATASETS_A[@]}"; do
    for seed in "${SEEDS_A[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}_bb-moirai-moe.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [A-$IDX] RR-MoA $freeze | $ds seed=$seed (moirai-moe)"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze "$freeze" --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$MOE_BB" --no-baselines \
        > /tmp/overnight_a_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        tail -3 /tmp/overnight_a_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/overnight_a_${IDX}.out | tail -1 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done

echo "A done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# B. Chronos extended to 6 datasets
# Completes the negative control grid.
###############################################
echo ""
echo "============================================================"
echo "=== B: Chronos Extended (6 datasets) ==="
echo "============================================================"

DATASETS_B=(ETTh2 ETTm2 Electricity)
FREEZE_B=(frozen last2 last4)
SEEDS_B=(42 43 44)
CHRONOS_BB="amazon/chronos-t5-small"

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for freeze in "${FREEZE_B[@]}"; do
  for ds in "${DATASETS_B[@]}"; do
    for seed in "${SEEDS_B[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}_bb-chronos.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [B-$IDX] $freeze | $ds seed=$seed (chronos)"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze "$freeze" --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$CHRONOS_BB" \
        > /tmp/overnight_b_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        tail -3 /tmp/overnight_b_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/overnight_b_${IDX}.out | tail -1 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done

echo "B done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# C. Router temperature scaling
# τ < 1 = sharper routing, τ > 1 = softer.
###############################################
echo ""
echo "============================================================"
echo "=== C: Router Temperature Scaling ==="
echo "============================================================"

DATASETS_C=(ETTh1 ETTm1 Weather)
TEMPS=(0.5 2.0 5.0)
SEEDS_C=(42 43 44)

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for temp in "${TEMPS[@]}"; do
  for ds in "${DATASETS_C[@]}"; do
    for seed in "${SEEDS_C[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_temp-${temp}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [C-$IDX] τ=$temp | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze frozen --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --router-temp "$temp" --no-baselines \
        > /tmp/overnight_c_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        tail -3 /tmp/overnight_c_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/overnight_c_${IDX}.out | tail -1 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done

echo "C done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

echo ""
echo "================================================================"
echo "OVERNIGHT BATCH COMPLETE — $(date)"
echo "================================================================"
