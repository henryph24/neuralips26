#!/bin/bash
# P1: Timer-XL unfreezing control — AdaMix + RR-MoA under last-2/last-4
#
# Goal: DEFINITIVE CAUSAL PROOF. On MOMENT (RevIN), unfreezing causes
# entropy→0.000 (collapse). On Timer-XL (no RevIN), if unfreezing keeps
# entropy high, that proves RevIN+unfreezing is the collapse mechanism.
#
# Grid: 6 datasets × 5 seeds × 2 freeze levels × 2 methods = 120 runs
# (frozen already done in E3; this adds last-2 and last-4)
#
# Invocation:
#   tmux new-session -d -s p1 'cd ~/neuralips26 && bash scripts/run_timer_unfreeze_vm.sh 2>&1 | tee results/p1_timer_unfreeze.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
BACKBONE="thuml/timer-base-84m"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)
FREEZE_LEVELS=(last2 last4)

RR_MOA_DIR="results/rr_moa"
ADAMIX_DIR="results/adamix_timer"
mkdir -p "$RR_MOA_DIR" "$ADAMIX_DIR"

echo "================================================================"
echo "P1 Timer-XL unfreezing control — $(date)"
echo "Backbone: $BACKBONE"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for freeze in "${FREEZE_LEVELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      # -------- RR-MoA --------
      OUT="${RR_MOA_DIR}/${ds}_H96_K5_top2_${freeze}_${seed}_bb-timer-base-84m.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
      else
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] RR-MoA ${freeze} | $ds seed=$seed"
        set +e
        $PYTHON scripts/run_rr_moa.py \
          --dataset "$ds" --unfreeze "$freeze" --top-k 2 \
          --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
          --backbone "$BACKBONE" --no-baselines \
          --batch-size 64 \
          > /tmp/p1_rrmoa_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, tail):"
          tail -5 /tmp/p1_rrmoa_${RUN_IDX}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
        fi
      fi
      RUN_IDX=$((RUN_IDX + 1))

      # -------- AdaMix --------
      OUT="${ADAMIX_DIR}/${ds}_H96_K5_${freeze}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
      else
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] AdaMix ${freeze} | $ds seed=$seed (timer-xl)"
        set +e
        $PYTHON scripts/run_adamix.py \
          --dataset "$ds" --unfreeze "$freeze" \
          --seed "$seed" --backbone "$BACKBONE" \
          --device "$DEVICE" \
          --results-dir "$ADAMIX_DIR" \
          --run-baselines no \
          > /tmp/p1_adamix_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, tail):"
          tail -5 /tmp/p1_adamix_${RUN_IDX}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
        fi
      fi
      RUN_IDX=$((RUN_IDX + 1))

    done
  done
done

echo ""
echo "================================================================"
echo "P1 Timer-XL unfreezing DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
