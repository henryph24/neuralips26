#!/bin/bash
# Tier 4: Close DLinear gap + broaden impact
# Total: ~5 GPU-hours on A10G
# Run: bash scripts/run_tier4_race.sh 2>&1 | tee results/tier4_run.log

set -e
PYTHON=python3
DEVICE="cuda"
EPOCHS=15

echo "================================================================"
echo "TIER 4 EXPERIMENT BATCH — $(date)"
echo "================================================================"

# ===================================================================
# EXP A: Moirai-Base RR-MoA (6 datasets × 3 seeds)
# ~4.5 GPU-hours — the key experiment to close the DLinear gap
# ===================================================================
echo ""
echo "=== EXP A: Moirai-Base RR-MoA ==="

for ds in ETTh1 ETTm1 Weather ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Moirai-base: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --backbone Salesforce/moirai-1.1-R-base \
      --seed $seed --epochs $EPOCHS --batch-size 32 --device $DEVICE \
      || echo "FAILED: moirai_base $ds $seed"
  done
done

# ===================================================================
# EXP B: BatchNorm/GroupNorm Routing Collapse (generalization test)
# ~20 min — proves the finding extends beyond RevIN
# ===================================================================
echo ""
echo "=== EXP B: Normalization Generalization ==="

echo "[$(date +%H:%M:%S)] BatchNorm trajectory"
$PYTHON scripts/run_adamix.py --dataset ETTh1 --unfreeze last4 --seed 42 \
  --norm-type batchnorm \
  --trajectory results/adamix/trajectory_ETTh1_last4_42_batchnorm.jsonl \
  --epochs $EPOCHS --device $DEVICE \
  || echo "FAILED: batchnorm_trajectory"

echo "[$(date +%H:%M:%S)] GroupNorm trajectory"
$PYTHON scripts/run_adamix.py --dataset ETTh1 --unfreeze last4 --seed 42 \
  --norm-type groupnorm \
  --trajectory results/adamix/trajectory_ETTh1_last4_42_groupnorm.jsonl \
  --epochs $EPOCHS --device $DEVICE \
  || echo "FAILED: groupnorm_trajectory"

echo ""
echo "================================================================"
echo "ALL TIER 4 EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
