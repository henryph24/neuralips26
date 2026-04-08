#!/bin/bash
# Reviewer response experiments — addresses the #1 concern:
# "Frozen Paradox may reflect optimization budget, not fundamental limitation"
#
# Run AFTER tier3 completes: bash scripts/run_reviewer_response.sh 2>&1 | tee results/reviewer_response.log

set -e
PYTHON=python3
DEVICE="cuda"

echo "================================================================"
echo "REVIEWER RESPONSE EXPERIMENTS — $(date)"
echo "================================================================"

# ===================================================================
# EXP R1: Extended Full-FT (50 epochs, cosine, layer-wise LR)
# Tests 5 configurations per dataset to rule out optimization confound
# ~1.5 hours per dataset (5 configs × ~15-20 min each)
# ===================================================================
echo ""
echo "=== EXP R1: Extended Full Fine-Tuning ==="

for ds in ETTh1 ETTm1 Weather; do
  echo "[$(date +%H:%M:%S)] Extended FT: $ds seed=42"
  $PYTHON scripts/run_extended_ft.py \
    --dataset $ds --seed 42 --device $DEVICE \
    || echo "FAILED: extended_ft $ds"
done

# Also run seed 43 on ETTh1 for variance estimate
echo "[$(date +%H:%M:%S)] Extended FT: ETTh1 seed=43"
$PYTHON scripts/run_extended_ft.py \
  --dataset ETTh1 --seed 43 --device $DEVICE \
  || echo "FAILED: extended_ft ETTh1 43"

echo ""
echo "================================================================"
echo "REVIEWER RESPONSE COMPLETE — $(date)"
echo "================================================================"
echo ""
echo "Check results/extended_ft/ for extended training results"
echo "Compare best extended-FT MSE against frozen RR-MoA:"
echo "  ETTh1 RR-MoA frozen: 0.680"
echo "  ETTm1 RR-MoA frozen: 0.564"
echo "  Weather RR-MoA frozen: 0.276"
