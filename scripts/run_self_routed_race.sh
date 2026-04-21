#!/usr/bin/env bash
# Self-Routed MoA (SR-MoA) experiment sweep on RACE VM.
#
# Three priorities:
#   P1: Core comparison — gated(gh16), gated(gh1), eigenbasis on 3 datasets × 3 seeds
#   P2: Ablations — gate init bias, all 6 datasets, hidden-state control
#   P3: Extended — 5-seed, hybrid mode
#
# Usage:
#   bash scripts/run_self_routed_race.sh                  # run all
#   bash scripts/run_self_routed_race.sh worker 1 4       # worker 1 of 4
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
echo "SR-MoA Sweep — $(date)"
echo "Mode: $MODE  Worker: $WORKER_ID / $NUM_WORKERS"
echo "============================================="

# ---- P1: Core comparison (27 runs) ----
echo ""
echo "=== P1: Core comparison ==="
for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do

    # P1a: Gated, gate_hidden=16
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P1a] $ds seed=$seed gated gh=16"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # P1b: Gated, gate_hidden=1 (minimal/linear gate)
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh1.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P1b] $ds seed=$seed gated gh=1"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 1 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

    # P1c: Eigenbasis, basis_dim=64
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_eigenbasis_bd64_t0.10.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P1c] $ds seed=$seed eigenbasis bd=64"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode eigenbasis --basis-dim 64 --temperature 0.1 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))

  done
done

# ---- P2: Ablations (36 runs) ----
echo ""
echo "=== P2: Ablations ==="

# P2a: Gate init bias = -2 (gated gh=16)
for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16_gib-2.0.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P2a] $ds seed=$seed gated gh=16 gib=-2"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --gate-init-bias -2.0 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

# P2b: Extend best (gated gh=16) to remaining 3 datasets
for ds in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P2b] $ds seed=$seed gated gh=16"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

# P2c: Hidden-state control (should collapse — validates diagnosis)
for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16_ri-hidden.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P2c] $ds seed=$seed gated gh=16 hidden-state control"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --routing-input hidden --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

# ---- P3: Extended (24 runs) ----
echo ""
echo "=== P3: Extended ==="

# P3a: 5-seed on core trio (seeds 44,45,46 — 42,43 already in P1)
for ds in ETTh1 ETTm1 Weather; do
  for seed in 44 45 46; do
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_gated_gh16.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P3a] $ds seed=$seed gated gh=16 (5-seed extension)"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode gated --gate-hidden 16 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

# P3b: Hybrid mode on core trio
for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_frozen_${seed}_hybrid_bd64_t0.10.json"
    if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
      SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
    elif [ -f "$OUTFILE" ]; then
      SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    else
      echo "[P3b] $ds seed=$seed hybrid bd=64"
      set +e
      $PYTHON scripts/run_self_routed_moa.py \
        --dataset "$ds" --horizon 96 --seed "$seed" --epochs 15 \
        --routing-mode hybrid --basis-dim 64 --temperature 0.1 --unfreeze frozen \
        > /tmp/srmoa_${WORKER_ID}.out 2>&1
      RC=$?
      set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/srmoa_${WORKER_ID}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "SR-MoA: MSE" /tmp/srmoa_${WORKER_ID}.out | tail -1 | sed 's/^/    /'
      fi
    fi
    RUN_IDX=$((RUN_IDX + 1))
  done
done

echo ""
echo "============================================="
echo "SR-MoA Sweep DONE at $(date)"
echo "  Launched:          $LAUNCHED"
echo "  Skipped (existed): $SKIPPED_EXISTING"
echo "  Skipped (worker):  $SKIPPED_WORKER"
echo "  Failed:            $FAILED"
echo "  Total scanned:     $RUN_IDX"
echo "============================================="
