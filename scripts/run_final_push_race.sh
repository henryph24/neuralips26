#!/bin/bash
# Q1-Q3: Final push experiments
#
# Q1: Multi-dataset trajectories (ETTm1+Weather × MOMENT+Timer-XL, last-4)
#     Proves the ETTh1 collapse pattern generalizes. 4 runs, ~5 min.
#
# Q2: Chronos trajectory (ETTh1 frozen + last-4)
#     Third "no collapse" backbone. 2 runs, ~5 min.
#
# Q3: RR-MoA classification (3 datasets × 3 seeds)
#     Proves routing generalizes beyond forecasting. 9 runs, ~15 min.
#
# Invocation:
#   tmux new-session -d -s final 'cd ~/neuralips26 && bash scripts/run_final_push_race.sh 2>&1 | tee results/final_push.log'

set -e
DEVICE="cuda"
PYTHON=python3

mkdir -p results/adamix results/adamix_timer results/classification

echo "================================================================"
echo "Q1-Q3 Final push — $(date)"
echo "================================================================"

LAUNCHED=0
SKIPPED=0
FAILED=0

# ================================================================
# Q1: Multi-dataset trajectories (MOMENT + Timer-XL, last-4)
# ================================================================
echo ""
echo "=== Q1: Multi-dataset trajectories ==="

for ds in ETTm1 Weather; do
  # MOMENT last-4 (expect collapse)
  OUT="results/adamix/trajectory_${ds}_last4_42.jsonl"
  if [ -f "$OUT" ]; then
    SKIPPED=$((SKIPPED + 1))
  else
    echo "[$(date +%H:%M:%S)] Q1: MOMENT last-4 trajectory $ds"
    set +e
    $PYTHON scripts/run_adamix.py \
      --dataset "$ds" --unfreeze last4 --seed 42 \
      --run-baselines no \
      --trajectory "$OUT" --trajectory-max-steps 400 \
      --device "$DEVICE" \
      > /tmp/q1_moment_${ds}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/q1_moment_${ds}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
  fi

  # Timer-XL last-4 (expect NO collapse)
  OUT="results/adamix_timer/trajectory_${ds}_last4_42_timer.jsonl"
  if [ -f "$OUT" ]; then
    SKIPPED=$((SKIPPED + 1))
  else
    echo "[$(date +%H:%M:%S)] Q1: Timer-XL last-4 trajectory $ds"
    set +e
    $PYTHON scripts/run_adamix.py \
      --dataset "$ds" --unfreeze last4 --seed 42 \
      --backbone thuml/timer-base-84m \
      --results-dir results/adamix_timer \
      --run-baselines no \
      --trajectory "$OUT" --trajectory-max-steps 400 \
      --device "$DEVICE" \
      > /tmp/q1_timer_${ds}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/q1_timer_${ds}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
  fi
done

echo "Q1 done"

# ================================================================
# Q2: Chronos trajectory (ETTh1, frozen + last-4)
# ================================================================
echo ""
echo "=== Q2: Chronos trajectories ==="

for freeze in frozen last4; do
  OUT="results/adamix/trajectory_ETTh1_${freeze}_42_chronos.jsonl"
  if [ -f "$OUT" ]; then
    SKIPPED=$((SKIPPED + 1))
  else
    echo "[$(date +%H:%M:%S)] Q2: Chronos ${freeze} trajectory ETTh1"
    set +e
    # Chronos needs a different results dir to avoid filename collision
    $PYTHON scripts/run_adamix.py \
      --dataset ETTh1 --unfreeze "$freeze" --seed 42 \
      --backbone amazon/chronos-t5-small \
      --run-baselines no \
      --trajectory "$OUT" --trajectory-max-steps 400 \
      --device "$DEVICE" \
      > /tmp/q2_chronos_${freeze}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/q2_chronos_${freeze}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
  fi
done

echo "Q2 done"

# ================================================================
# Q3: RR-MoA Classification (3 datasets × 3 seeds)
# ================================================================
echo ""
echo "=== Q3: RR-MoA Classification ==="

for ds in BasicMotions EthanolConcentration JapaneseVowels; do
  for seed in 42 43 44; do
    OUT="results/classification/${ds}_rrmoa_K5_top2_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] Q3: RR-MoA classification | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_rrmoa_classification.py \
        --dataset "$ds" --seed "$seed" \
        --K 5 --top-k 2 \
        --device "$DEVICE" \
        > /tmp/q3_cls_${ds}_${seed}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/q3_cls_${ds}_${seed}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
    fi
  done
done

echo ""
echo "================================================================"
echo "Q1-Q3 DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "================================================================"
