#!/bin/bash
# E3: Timer-XL (decoder-only) backbone — RR-MoA + AdaMix
#
# Goal: Add 6th backbone family to cross-backbone grid. Timer-XL is
# decoder-only with LayerNorm only (no RevIN), providing a structural
# control distinct from MOMENT (encoder-only + RevIN) and Moirai
# (encoder-only + LayerNorm). If RR-MoA still outperforms AdaMix here,
# it strengthens the "routing collapse is backbone-general" claim.
#
# Grid: 6 datasets × 5 seeds × 2 methods = 60 runs
#   1. RR-MoA (Top-2, frozen, raw router)
#   2. AdaMix (frozen, hidden-state router)
#
# Invocation:
#   tmux new-session -d -s e3 'cd ~/neuralips26 && bash scripts/run_timer_xl_vm.sh 2>&1 | tee results/e3_timer.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
BACKBONE="thuml/timer-base-84m"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)

RR_MOA_DIR="results/rr_moa"
ADAMIX_DIR="results/adamix_timer"
mkdir -p "$RR_MOA_DIR" "$ADAMIX_DIR"

echo "================================================================"
echo "E3 Timer-XL cross-backbone — $(date)"
echo "Backbone: $BACKBONE"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    # -------- RR-MoA (Top-2, frozen) --------
    OUT="${RR_MOA_DIR}/${ds}_H96_K5_top2_frozen_${seed}_bb-timer-base-84m.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA  | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze frozen --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$BACKBONE" --no-baselines \
        --batch-size 64 \
        > /tmp/e3_rrmoa_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/e3_rrmoa_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # -------- AdaMix frozen --------
    OUT="${ADAMIX_DIR}/${ds}_H96_K5_frozen_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] AdaMix  | $ds seed=$seed (timer-xl)"
      set +e
      $PYTHON scripts/run_adamix.py \
        --dataset "$ds" --unfreeze frozen \
        --seed "$seed" --backbone "$BACKBONE" \
        --device "$DEVICE" \
        --results-dir "$ADAMIX_DIR" \
        --run-baselines no \
        > /tmp/e3_adamix_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/e3_adamix_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

  done
done

echo ""
echo "================================================================"
echo "E3 Timer-XL DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
