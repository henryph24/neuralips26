#!/bin/bash
# P4-P7: Paper-strengthening experiments (sequential, single GPU)
#
# P4: Timer-XL per-step trajectory (4 runs, ~5 min)
#     Generates JSONL files for side-by-side figure: MOMENT entropy DOWN vs Timer-XL entropy UP
#
# P5: TRACE baseline 5-seed frozen completion (12 runs, ~15 min)
#     Upgrades TRACE comparison from 3 to 5 seeds
#
# P6: Independent ensemble on Timer-XL (30 runs, ~30 min)
#     Proves routing (not just diversity) matters on Timer-XL
#
# Invocation:
#   tmux new-session -d -s strengthen 'cd ~/neuralips26 && bash scripts/run_strengthening_vm.sh 2>&1 | tee results/strengthen.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

mkdir -p results/adamix results/adamix_timer results/trace_baseline results/independent_ensemble

echo "================================================================"
echo "P4-P7 Paper strengthening — $(date)"
echo "================================================================"

LAUNCHED=0
SKIPPED=0
FAILED=0

# ================================================================
# P4: Timer-XL per-step trajectory logging (4 runs)
# ================================================================
echo ""
echo "=== P4: Timer-XL trajectory logging ==="

# Timer-XL frozen (control — expect entropy to stay high)
OUT="results/adamix_timer/trajectory_ETTh1_frozen_42_timer.jsonl"
if [ -f "$OUT" ]; then
  SKIPPED=$((SKIPPED + 1))
else
  echo "[$(date +%H:%M:%S)] P4: AdaMix Timer-XL frozen trajectory"
  set +e
  $PYTHON scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze frozen --seed 42 \
    --backbone thuml/timer-base-84m \
    --results-dir results/adamix_timer \
    --run-baselines no \
    --trajectory "$OUT" --trajectory-max-steps 400 \
    --device "$DEVICE" \
    > /tmp/p4_timer_frozen.out 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p4_timer_frozen.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
fi

# Timer-XL last-4 (expect entropy to RISE — opposite of MOMENT)
OUT="results/adamix_timer/trajectory_ETTh1_last4_42_timer.jsonl"
if [ -f "$OUT" ]; then
  SKIPPED=$((SKIPPED + 1))
else
  echo "[$(date +%H:%M:%S)] P4: AdaMix Timer-XL last-4 trajectory"
  set +e
  $PYTHON scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze last4 --seed 42 \
    --backbone thuml/timer-base-84m \
    --results-dir results/adamix_timer \
    --run-baselines no \
    --trajectory "$OUT" --trajectory-max-steps 400 \
    --device "$DEVICE" \
    > /tmp/p4_timer_last4.out 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p4_timer_last4.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
fi

# MOMENT frozen (control — for comparison panel)
OUT="results/adamix/trajectory_ETTh1_frozen_42_refresh.jsonl"
if [ -f "results/adamix/trajectory_ETTh1_frozen_42.jsonl" ]; then
  SKIPPED=$((SKIPPED + 1))
  echo "  (MOMENT frozen trajectory already exists)"
elif [ -f "$OUT" ]; then
  SKIPPED=$((SKIPPED + 1))
else
  echo "[$(date +%H:%M:%S)] P4: AdaMix MOMENT frozen trajectory (refresh)"
  set +e
  $PYTHON scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze frozen --seed 42 \
    --run-baselines no \
    --trajectory "$OUT" --trajectory-max-steps 400 \
    --device "$DEVICE" \
    > /tmp/p4_moment_frozen.out 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p4_moment_frozen.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
fi

# MOMENT last-4 (expect entropy collapse — the known result)
OUT="results/adamix/trajectory_ETTh1_last4_42_refresh.jsonl"
if [ -f "results/adamix/trajectory_ETTh1_last4_42.jsonl" ]; then
  SKIPPED=$((SKIPPED + 1))
  echo "  (MOMENT last-4 trajectory already exists)"
elif [ -f "$OUT" ]; then
  SKIPPED=$((SKIPPED + 1))
else
  echo "[$(date +%H:%M:%S)] P4: AdaMix MOMENT last-4 trajectory (refresh)"
  set +e
  $PYTHON scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze last4 --seed 42 \
    --run-baselines no \
    --trajectory "$OUT" --trajectory-max-steps 400 \
    --device "$DEVICE" \
    > /tmp/p4_moment_last4.out 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p4_moment_last4.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
fi

echo "P4 done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ================================================================
# P5: TRACE baseline 5-seed frozen completion (12 runs)
# ================================================================
echo ""
echo "=== P5: TRACE baseline frozen 5-seed ==="

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="results/trace_baseline/${ds}_H96_frozen_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] P5: TRACE frozen | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_trace_baseline.py \
        --dataset "$ds" --seed "$seed" --unfreeze frozen \
        --device "$DEVICE" \
        > /tmp/p5_trace_${ds}_${seed}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p5_trace_${ds}_${seed}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
    fi
  done
done

echo "P5 done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ================================================================
# P6: Independent ensemble on Timer-XL (30 runs)
# ================================================================
echo ""
echo "=== P6: Independent ensemble on Timer-XL ==="

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="results/independent_ensemble/${ds}_H96_K5_frozen_${seed}_bb-timer-base-84m.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1))
    else
      echo "[$(date +%H:%M:%S)] P6: Ind.Ensemble Timer | $ds seed=$seed"
      set +e
      $PYTHON scripts/run_independent_ensemble.py \
        --dataset "$ds" --seed "$seed" \
        --backbone thuml/timer-base-84m \
        --batch-size 64 \
        --device "$DEVICE" \
        > /tmp/p6_ens_${ds}_${seed}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); echo "  FAILED"; tail -3 /tmp/p6_ens_${ds}_${seed}.out | sed 's/^/    /'; else LAUNCHED=$((LAUNCHED+1)); fi
    fi
  done
done

echo ""
echo "================================================================"
echo "P4-P7 ALL DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "================================================================"
