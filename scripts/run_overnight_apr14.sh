#!/bin/bash
# Overnight batch #2 — Apr 14 2026 (after overnight_apr13 finishes)
#
# A. AdaMix unfreezing on Moirai-MoE (36 runs, ~3h)
#    — completes the converging control: does AdaMix collapse on
#      unfrozen Moirai-MoE (no RevIN)? Expected: NO.
#
# B. Extended top-k ablation on ETTm1+Weather (24 runs, ~30min)
#    — current tab:topk covers only ETTh1. Extending to 2 more datasets
#      shows top-k=2 is optimal broadly, not dataset-specific.
#
# C. AdaMix unfreezing on regular Moirai (18 runs, ~1h)
#    — another non-RevIN backbone unfreezing control.
#
# Total: ~78 runs, ~4.5h
#
# Invocation:
#   tmux new-session -d -s overnight2 'cd ~/neuralips26 && bash scripts/run_overnight_apr14.sh 2>&1 | tee results/overnight_apr14.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

echo "================================================================"
echo "OVERNIGHT BATCH #2 — $(date)"
echo "================================================================"

###############################################
# A. AdaMix unfreezing on Moirai-MoE
###############################################
echo ""
echo "============================================================"
echo "=== A: AdaMix Unfreezing on Moirai-MoE ==="
echo "============================================================"

DATASETS_A=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
FREEZE_A=(last2 last4)
SEEDS_A=(42 43 44)
MOE_BB="Salesforce/moirai-moe-1.0-R-small"
ADAMIX_MOE_DIR="results/adamix_moirai_moe"
mkdir -p "$ADAMIX_MOE_DIR"

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for freeze in "${FREEZE_A[@]}"; do
  for ds in "${DATASETS_A[@]}"; do
    for seed in "${SEEDS_A[@]}"; do
      OUT="${ADAMIX_MOE_DIR}/${ds}_H96_K5_${freeze}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [A-$IDX] AdaMix $freeze | $ds seed=$seed (moirai-moe)"
      set +e
      $PYTHON scripts/run_adamix.py \
        --dataset "$ds" --unfreeze "$freeze" \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$MOE_BB" \
        --run-baselines no --results-dir "$ADAMIX_MOE_DIR" \
        > /tmp/overnight2_a_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/overnight2_a_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|entropy" /tmp/overnight2_a_${IDX}.out | tail -2 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done
echo "A done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# B. Extended top-k ablation (ETTm1, Weather)
###############################################
echo ""
echo "============================================================"
echo "=== B: Extended Top-K Ablation ==="
echo "============================================================"

DATASETS_B=(ETTm1 Weather)
TOPK_VALUES=(1 3 4 5)
SEEDS_B=(42 43 44)

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for topk in "${TOPK_VALUES[@]}"; do
  for ds in "${DATASETS_B[@]}"; do
    for seed in "${SEEDS_B[@]}"; do
      if [ "$topk" -eq 5 ]; then
        TK_LABEL="dense"
      else
        TK_LABEL="top${topk}"
      fi
      OUT="results/rr_moa/${ds}_H96_K5_${TK_LABEL}_frozen_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [B-$IDX] top-k=$topk | $ds seed=$seed"
      set +e
      TOPK_ARG=""
      if [ "$topk" -lt 5 ]; then
        TOPK_ARG="--top-k $topk"
      fi
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze frozen \
        $TOPK_ARG \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --no-baselines \
        > /tmp/overnight2_b_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/overnight2_b_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/overnight2_b_${IDX}.out | tail -1 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done
echo "B done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# C. AdaMix unfreezing on regular Moirai
###############################################
echo ""
echo "============================================================"
echo "=== C: AdaMix Unfreezing on Moirai ==="
echo "============================================================"

DATASETS_C=(ETTh1 ETTm1 Weather)
FREEZE_C=(last2 last4)
SEEDS_C=(42 43 44)
MOIRAI_BB="Salesforce/moirai-1.1-R-small"
ADAMIX_MOIRAI_DIR="results/adamix_moirai"
mkdir -p "$ADAMIX_MOIRAI_DIR"

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for freeze in "${FREEZE_C[@]}"; do
  for ds in "${DATASETS_C[@]}"; do
    for seed in "${SEEDS_C[@]}"; do
      OUT="${ADAMIX_MOIRAI_DIR}/${ds}_H96_K5_${freeze}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] [C-$IDX] AdaMix $freeze | $ds seed=$seed (moirai)"
      set +e
      $PYTHON scripts/run_adamix.py \
        --dataset "$ds" --unfreeze "$freeze" \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$MOIRAI_BB" \
        --run-baselines no --results-dir "$ADAMIX_MOIRAI_DIR" \
        > /tmp/overnight2_c_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/overnight2_c_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep -E "MSE=|entropy" /tmp/overnight2_c_${IDX}.out | tail -2 | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done
echo "C done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

echo ""
echo "================================================================"
echo "OVERNIGHT BATCH #2 COMPLETE — $(date)"
echo "================================================================"
