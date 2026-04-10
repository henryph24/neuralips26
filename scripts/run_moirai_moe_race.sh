#!/bin/bash
# B2: Native MoE backbone evaluation — Moirai-MoE + RR-MoA / AdaMix / fixed
#
# Goal: evaluate whether RR-MoA's adapter-level routing is still beneficial on
# a backbone that was pre-trained with internal expert routing.  If so, this
# directly addresses Limitation (2) in the paper.
#
# Grid: 6 datasets × 5 seeds × 3 methods = 90 runs
# Methods:
#   1. RR-MoA (Top-2, frozen, raw router)  — the paper's headline method
#   2. Best fixed adapter (conv)             — baseline for comparison
#   3. AdaMix (frozen)                       — secondary check: does
#      hidden-state routing also work on a native MoE backbone?
#
# Invocation (runs next to B1 in its own tmux session):
#   tmux new-session -d -s b2 'cd ~/neuralips26 && bash scripts/run_moirai_moe_race.sh 2>&1 | tee results/b2_run.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
BACKBONE="Salesforce/moirai-moe-1.0-R-small"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)

RR_MOA_DIR="results/rr_moa"
ADAMIX_DIR="results/adamix"
mkdir -p "$RR_MOA_DIR" "$ADAMIX_DIR"

echo "================================================================"
echo "B2 Moirai-MoE cross-backbone — $(date)"
echo "Backbone: $BACKBONE"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    # -------- RR-MoA (Top-2, frozen) --------
    OUT="${RR_MOA_DIR}/${ds}_H96_K5_top2_frozen_${seed}_bb-moirai-moe.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA  | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze frozen --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$BACKBONE" \
        > /tmp/b2_rrmoa_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/b2_rrmoa_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # -------- AdaMix frozen (cross-check) --------
    OUT="${ADAMIX_DIR}/${ds}_H96_K5_frozen_${seed}_bb-moirai-moe.json"
    # run_adamix.py doesn't append a backbone suffix — check for it via JSON
    # content.  Since we want distinct filenames per backbone, we launch the
    # run with a custom results-dir that pins the backbone identity.
    OUT_DIR="results/adamix_moirai_moe"
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/${ds}_H96_K5_frozen_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] [$RUN_IDX] AdaMix  | $ds seed=$seed (moirai-moe)"
      set +e
      $PYTHON scripts/run_adamix.py \
        --dataset "$ds" --unfreeze frozen \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        --backbone "$BACKBONE" \
        --results-dir "$OUT_DIR" \
        --run-baselines no \
        > /tmp/b2_adamix_${RUN_IDX}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC, tail):"
        tail -5 /tmp/b2_adamix_${RUN_IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # -------- Fixed conv adapter (baseline) --------
    # The RR-MoA script already runs the fixed baselines as part of its main
    # output, so a separate "fixed conv" run is redundant.  The RR-MoA JSON
    # contains a "baselines" field with all three fixed heads' results.
    # We skip step 3 to halve the runtime.
  done
done

echo ""
echo "================================================================"
echo "B2 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
