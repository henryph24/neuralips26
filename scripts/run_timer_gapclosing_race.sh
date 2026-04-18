#!/bin/bash
# P3: Residual-IA+ on Timer-XL — cross-backbone gap-closing extension
#
# Goal: Add Timer-XL to the cross-backbone Residual-IA+ table.
# Currently covers MOMENT-large, Moirai, Chronos (30/36 cells).
# Timer-XL would extend to 40/48.
#
# Config matches existing cross-backbone runs:
#   variant=residual-ia, raw-hidden=256, raw-depth=1, wd=0.0001,
#   cosine-schedule, warmup-epochs=5, shared-raw, early-stop patience=5,
#   raw-type=nlinear, gate-init=-2, K=1
#
# Grid: 3 datasets × 5 seeds × 4 horizons = 60 runs
# (H=96 with 5 seeds, H=192/336/720 with 5 seeds since we now have them)
#
# Invocation:
#   tmux new-session -d -s p3 'cd ~/neuralips26 && bash scripts/run_timer_gapclosing_race.sh 2>&1 | tee results/p3_timer_gc.log'

set -e
DEVICE="cuda"
PYTHON=python3
BACKBONE="thuml/timer-base-84m"

DATASETS=(ETTh1 ETTm1 Weather)
HORIZONS=(96 192 336 720)
SEEDS=(42 43 44 45 46)

mkdir -p results/gap_closing

echo "================================================================"
echo "P3 Residual-IA+ on Timer-XL — $(date)"
echo "Backbone: $BACKBONE"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
  for h in "${HORIZONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="results/gap_closing/residual-ia_${ds}_H${h}_${seed}_bb-timer-base-84m_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1_K1.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
      else
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] Residual-IA+ H=$h | $ds seed=$seed"
        set +e
        $PYTHON scripts/run_gap_closing.py \
          --variant residual-ia \
          --dataset "$ds" --horizon "$h" --seed "$seed" \
          --backbone "$BACKBONE" \
          --unfreeze frozen \
          --raw-hidden 256 --raw-depth 1 \
          --weight-decay 0.0001 --cosine \
          --warmup-epochs 5 --raw-branch-shared \
          --val-early-stop --val-patience 5 \
          --raw-arch nlinear --gate-init -2 \
          --K 1 --grad-clip 1 \
          --batch-size 64 \
          --device "$DEVICE" \
          > /tmp/p3_gc_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, tail):"
          tail -5 /tmp/p3_gc_${RUN_IDX}.out | sed 's/^/    /'
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
echo "P3 Timer-XL gap-closing DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
