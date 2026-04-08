#!/bin/bash
# Tier 5: Re-run Moirai with fixed in_proj (proper pre-trained projection)
# Total: ~7.5 GPU-hours on A10G
# Run: bash scripts/run_tier5_race.sh 2>&1 | tee results/tier5_run.log

set -e
PYTHON=python3
DEVICE="cuda"
EPOCHS=15

echo "================================================================"
echo "TIER 5: MOIRAI WITH FIXED INPUT PROJECTION — $(date)"
echo "================================================================"

# ===================================================================
# Moirai-small re-run (fixed in_proj, 6 datasets × 3 seeds)
# ===================================================================
echo ""
echo "=== Moirai-small (fixed projection) ==="

for ds in ETTh1 ETTm1 Weather ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Moirai-small: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --backbone Salesforce/moirai-1.1-R-small \
      --seed $seed --epochs $EPOCHS --batch-size 64 --device $DEVICE \
      || echo "FAILED: moirai_small $ds $seed"
  done
done

# ===================================================================
# Moirai-base re-run (fixed in_proj, 6 datasets × 3 seeds)
# ===================================================================
echo ""
echo "=== Moirai-base (fixed projection) ==="

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

echo ""
echo "================================================================"
echo "TIER 5 COMPLETE — $(date)"
echo "================================================================"
