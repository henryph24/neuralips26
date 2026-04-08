#!/bin/bash
# Checkmate Package: Few-shot curve + 50-epoch adapters
# Total: ~5 GPU-hours on A10G
# Run: bash scripts/run_checkmate_race.sh 2>&1 | tee results/checkmate_run.log

set -e
PYTHON=python3
DEVICE="cuda"

echo "================================================================"
echo "CHECKMATE PACKAGE — $(date)"
echo "================================================================"

# ===================================================================
# EXP 1: Few-Shot Learning Curve (3 datasets × 3 seeds)
# ~3 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 1: Few-Shot Learning Curve ==="

for ds in ETTh1 Weather Electricity; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Few-shot: $ds seed=$seed"
    $PYTHON scripts/run_fewshot_curve.py \
      --dataset $ds --seed $seed --epochs 15 --device $DEVICE \
      || echo "FAILED: fewshot $ds $seed"
  done
done

# ===================================================================
# EXP 2: Extended Adapter Training (50 epochs)
# ~2 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 2: Extended Adapter Training (50 epochs) ==="

for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] 50-epoch: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --seed $seed --epochs 50 --device $DEVICE --no-baselines \
      || echo "FAILED: 50ep $ds $seed"
  done
done

echo ""
echo "================================================================"
echo "CHECKMATE PACKAGE COMPLETE — $(date)"
echo "================================================================"
