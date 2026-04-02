#!/bin/bash
# Overnight batch: address all paper limitations on RACE VM
# Usage: nohup bash scripts/run_overnight_batch.sh > overnight.log 2>&1 &
set -e
cd ~/neuralips26
mkdir -p results/standard_evolution

echo "============================================"
echo "Overnight Batch — Address Paper Limitations"
echo "Started: $(date)"
echo "============================================"

# === EXP 1: Multi-seed (8 runs) ===
echo ""
echo "========== EXP 1: Multi-seed =========="
for ds in ETTm1 ETTh2 ETTm2 Weather; do
  for seed in 43 44; do
    echo ">>> $ds seed=$seed"
    python3 -u scripts/run_standard_evolution.py --dataset $ds --seed $seed 2>&1 | tail -3
  done
done

# === EXP 2: Full unfreezing (2 runs) ===
echo ""
echo "========== EXP 2: Full Unfreeze =========="
for ds in ETTh1 Weather; do
  echo ">>> $ds --unfreeze all"
  python3 -u scripts/run_standard_evolution.py --dataset $ds --unfreeze all 2>&1 | tail -3
done

# === EXP 3: Multi-horizon (4 runs) ===
echo ""
echo "========== EXP 3: Multi-horizon =========="
for ds in ETTh1 ETTm1; do
  for h in 192 336; do
    echo ">>> $ds H=$h"
    python3 -u scripts/run_standard_evolution.py --dataset $ds --horizon $h 2>&1 | tail -3
  done
done

echo ""
echo "============================================"
echo "All experiments complete: $(date)"
echo "Results in results/standard_evolution/"
echo "============================================"
