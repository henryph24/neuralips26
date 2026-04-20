#!/bin/bash
# Broader scope experiments: learnable alpha + AdaMix imputation collapse + SSR fill
#
# Phase 1: Learnable alpha (30 runs, ~23 min)
# Phase 2: AdaMix imputation collapse (18 runs, ~14 min with model reload overhead)
# Phase 3: SSR fill missing seeds (≤12 runs, ~9 min)
#
# Total: ~60 runs, ~46 min
#
# Invocation:
#   bash scripts/run_broader_scope_race.sh

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

echo "================================================================"
echo "Broader scope experiments — $(date)"
echo "================================================================"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS_5=(42 43 44 45 46)
SEEDS_3=(42 43 44)

LAUNCHED=0
FAILED=0

# --- Phase 1: Learnable Alpha ---
echo ""
echo "=== PHASE 1: Learnable Alpha (6 datasets × 5 seeds) ==="
mkdir -p results/learnable_alpha

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS_5[@]}"; do
    OUTFILE="results/learnable_alpha/${ds}_H96_${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "  SKIP (exists): $OUTFILE"
      continue
    fi
    echo "[$(date +%H:%M:%S)] learnable_alpha | $ds seed=$seed"
    set +e
    $PYTHON scripts/run_learnable_alpha.py \
      --dataset "$ds" --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
      > /tmp/lalpha.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC):"
      tail -3 /tmp/lalpha.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
      # Print final alpha
      grep "final_alpha" /tmp/lalpha.out | tail -1 | sed 's/^/    /'
    fi
  done
done

# --- Phase 2: AdaMix Imputation Collapse ---
echo ""
echo "=== PHASE 2: AdaMix Imputation Collapse (6 datasets × 3 seeds) ==="

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS_3[@]}"; do
    # Check if already has adamix data
    OUTFILE="results/imputation/${ds}_${seed}.json"
    if [ -f "$OUTFILE" ]; then
      HAS_ADAMIX=$(python3 -c "import json; d=json.load(open('$OUTFILE')); print('yes' if 'adamix_mse' in d else 'no')" 2>/dev/null)
      if [ "$HAS_ADAMIX" = "yes" ]; then
        echo "  SKIP (adamix exists): $OUTFILE"
        continue
      fi
    fi
    echo "[$(date +%H:%M:%S)] adamix_imputation | $ds seed=$seed"
    set +e
    $PYTHON scripts/run_imputation.py \
      --dataset "$ds" --seed "$seed" --device "$DEVICE" \
      > /tmp/imp_adamix.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC):"
      tail -3 /tmp/imp_adamix.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
      grep "AdaMix\|Collapse" /tmp/imp_adamix.out | sed 's/^/    /'
    fi
  done
done

# --- Phase 3: SSR Fill Missing Seeds ---
echo ""
echo "=== PHASE 3: SSR Router Fill (missing seeds 45,46) ==="

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS_5[@]}"; do
    OUTFILE="results/rr_moa/${ds}_H96_K5_dense_frozen_${seed}_rarch-ssr.json"
    if [ -f "$OUTFILE" ]; then
      continue
    fi
    echo "[$(date +%H:%M:%S)] SSR fill | $ds seed=$seed"
    set +e
    $PYTHON scripts/run_rr_moa.py \
      --dataset "$ds" --seed "$seed" --unfreeze frozen \
      --router-arch ssr --epochs "$EPOCHS" --device "$DEVICE" \
      --no-baselines \
      > /tmp/ssr.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo "  FAILED (rc=$RC):"
      tail -3 /tmp/ssr.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
  done
done

echo ""
echo "================================================================"
echo "Broader scope DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Failed:   $FAILED"
echo "================================================================"
