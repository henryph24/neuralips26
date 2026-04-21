#!/usr/bin/env bash
# Deep SR-MoA experiments: freeze sweep + multi-horizon + cross-backbone
# Targets the 3 remaining paper vulnerabilities directly.
#
# Batch A: Freeze-level sweep (frozen/last2/last4) — tests frozen paradox in SR-MoA
# Batch B: Multi-horizon (H=192,336,720) — extends coverage to match paper grid
# Batch C: Cross-backbone (Moirai, Chronos, MOMENT-large) — generality
# Batch D: SR-MoA vs DLinear head-to-head on all 6 datasets × 4 horizons
#
# Usage:
#   bash scripts/run_srmoa_deep_race.sh                  # run all
#   bash scripts/run_srmoa_deep_race.sh worker 1 4       # worker 1 of 4
set -e

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="results/self_routed_moa"
mkdir -p "$RESULTS_DIR"

MODE="${1:-single}"
WORKER_ID="${2:-1}"
NUM_WORKERS="${3:-1}"

LAUNCHED=0
SKIPPED_EXISTING=0
SKIPPED_WORKER=0
FAILED=0
RUN_IDX=0

echo "============================================="
echo "SR-MoA Deep Sweep — $(date)"
echo "Mode: $MODE  Worker: $WORKER_ID / $NUM_WORKERS"
echo "============================================="

run_one() {
    local LABEL="$1"; shift
    local OUTFILE="$1"; shift
    # remaining args are the python command args

    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
        SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
        RUN_IDX=$((RUN_IDX + 1))
        return
    fi
    if [ -f "$OUTFILE" ]; then
        SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
        RUN_IDX=$((RUN_IDX + 1))
        return
    fi

    echo "[$LABEL] $(basename $OUTFILE .json)"
    set +e
    $PYTHON scripts/run_self_routed_moa.py "$@" > /tmp/srmoa_deep_${WORKER_ID}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_deep_${WORKER_ID}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_deep_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
}

# ================================================================
# BATCH A: Freeze-level sweep (frozen paradox test)
# Does frozen > unfrozen hold for SR-MoA? Critical for the paradox claim.
# 3 datasets × 2 extra freeze levels × 3 seeds = 18 runs (~20 min)
# ================================================================
echo ""
echo "=== BATCH A: Freeze-level sweep ==="
for ds in ETTh1 ETTm1 Weather; do
  for unf in last2 last4; do
    for seed in 42 43 44; do
      OUT="${RESULTS_DIR}/${ds}_H96_K5_${unf}_${seed}_gated_gh16.json"
      run_one "A" "$OUT" \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze "$unf"
    done
  done
done

# ================================================================
# BATCH B: Multi-horizon extension
# SR-MoA at H=192, 336, 720 to match the paper's horizon grid.
# 3 datasets × 3 horizons × 3 seeds = 27 runs (~35 min)
# ================================================================
echo ""
echo "=== BATCH B: Multi-horizon ==="
for ds in ETTh1 ETTm1 Weather; do
  for H in 192 336 720; do
    for seed in 42 43 44; do
      OUT="${RESULTS_DIR}/${ds}_H${H}_K5_frozen_${seed}_gated_gh16.json"
      run_one "B" "$OUT" \
        --dataset "$ds" --horizon "$H" --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen
    done
  done
done

# ================================================================
# BATCH C: Cross-backbone generality
# Test SR-MoA on Moirai, Chronos, MOMENT-large
# 3 backbones × 3 datasets × 3 seeds = 27 runs (~60 min for larger backbones)
# ================================================================
echo ""
echo "=== BATCH C: Cross-backbone ==="
for bb in "Salesforce/moirai-1.1-R-small" "amazon/chronos-t5-small" "AutonLab/MOMENT-1-large"; do
  # Derive bb suffix for filename
  bb_lower=$(echo "$bb" | tr '[:upper:]' '[:lower:]')
  if echo "$bb_lower" | grep -q "moirai"; then
    BB_SUFFIX="bb-moirai"
  elif echo "$bb_lower" | grep -q "chronos"; then
    BB_SUFFIX="bb-chronos"
  elif echo "$bb_lower" | grep -q "moment-1-large"; then
    BB_SUFFIX="bb-moment-large"
  else
    BB_SUFFIX="bb-other"
  fi

  for ds in ETTh1 ETTm1 Weather; do
    for seed in 42 43 44; do
      OUT="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16_${BB_SUFFIX}.json"
      run_one "C" "$OUT" \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen \
        --backbone "$bb"
    done
  done
done

# ================================================================
# BATCH D: Full 6-dataset × multi-horizon for DLinear comparison
# Remaining datasets (ETTh2, ETTm2, Electricity) at H=192,336,720
# 3 datasets × 3 horizons × 3 seeds = 27 runs (~35 min)
# ================================================================
echo ""
echo "=== BATCH D: Full dataset × horizon grid ==="
for ds in ETTh2 ETTm2 Electricity; do
  for H in 192 336 720; do
    for seed in 42 43 44; do
      OUT="${RESULTS_DIR}/${ds}_H${H}_K5_frozen_${seed}_gated_gh16.json"
      run_one "D" "$OUT" \
        --dataset "$ds" --horizon "$H" --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen
    done
  done
done

# Also fill in remaining H=96 seeds for ETTh2/ETTm2/Electricity (seeds 44,45,46)
for ds in ETTh2 ETTm2 Electricity; do
  for seed in 45 46; do
    OUT="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16.json"
    run_one "D" "$OUT" \
      --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
      --routing-mode gated --gate-hidden 16 --unfreeze frozen
  done
done

echo ""
echo "============================================="
echo "SR-MoA Deep Sweep DONE at $(date)"
echo "  Launched:          $LAUNCHED"
echo "  Skipped (existed): $SKIPPED_EXISTING"
echo "  Skipped (worker):  $SKIPPED_WORKER"
echo "  Failed:            $FAILED"
echo "  Total scanned:     $RUN_IDX"
echo "============================================="
