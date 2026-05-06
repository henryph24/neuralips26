#!/bin/bash
# Strengthening experiments: D (AdaMix-Raw), F (K=1,2), A (Noise robustness)
# Total: ~180 runs, ~3 hours on A10G
#
# Usage:
#   bash scripts/run_strengthening_v2_vm.sh           # run all
#   bash scripts/run_strengthening_v2_vm.sh expD      # AdaMix-Raw only
#   bash scripts/run_strengthening_v2_vm.sh expF      # K=1,2 only
#   bash scripts/run_strengthening_v2_vm.sh expA      # Noise robustness only

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
DEVICE=cuda

# ========================================================================
# EXP D: AdaMix with Raw Router (proves diagnosis is architecture-agnostic)
# 6 datasets x 2 freeze levels x 3 seeds = 36 runs (~30 min)
# ========================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "expD" ]; then
echo ""
echo "================================================================"
echo "EXP D: AdaMix with Raw Router"
echo "================================================================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for unfreeze in frozen last4; do
    for seed in 42 43 44; do
      OUT="results/adamix/${ds}_H96_K5_${unfreeze}_${seed}_rawrouter.json"
      if [ ! -f "$OUT" ]; then
        echo "=== AdaMix-Raw: $ds $unfreeze seed=$seed ==="
        python3 scripts/run_adamix.py --dataset "$ds" --seed "$seed" \
          --unfreeze "$unfreeze" --router-input raw --run-baselines no \
          --device $DEVICE
      else
        echo "SKIP: $OUT"
      fi
    done
  done
done

echo "=== Exp D complete ==="
fi

# ========================================================================
# EXP F: K=1,2 ablation (quantifies routing contribution)
# 2 K values x 6 datasets x 3 seeds = 36 runs (~30 min)
# ========================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "expF" ]; then
echo ""
echo "================================================================"
echo "EXP F: K=1 and K=2 ablation"
echo "================================================================"

for K in 1 2; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for seed in 42 43 44; do
      if [ "$K" -eq 1 ]; then
        TOPK_LABEL="dense"
        TOPK_FLAG=""
      else
        TOPK_LABEL="top2"
        TOPK_FLAG="--top-k 2"
      fi
      OUT="results/rr_moa/${ds}_H96_K${K}_${TOPK_LABEL}_frozen_${seed}.json"
      if [ ! -f "$OUT" ]; then
        echo "=== K=$K: $ds seed=$seed ==="
        python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" \
          --K "$K" $TOPK_FLAG --unfreeze frozen --no-baselines --device $DEVICE
      else
        echo "SKIP: $OUT"
      fi
    done
  done
done

echo "=== Exp F complete ==="
fi

# ========================================================================
# EXP A: Noise Robustness (demonstrates TSFM value under perturbation)
# 3 noise levels x 6 datasets x 3 seeds = 54 runs for Residual-IA+
# DLinear comparison: same grid = 54 runs
# Total: 108 runs (~90 min)
# ========================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "expA" ]; then
echo ""
echo "================================================================"
echo "EXP A: Noise Robustness"
echo "================================================================"

for sigma in 0.1 0.5 1.0; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for seed in 42 43 44; do
      # Residual-IA+
      OUT="results/gap_closing/residual-ia_${ds}_H96_frozen_${seed}_noise-${sigma}.json"
      if [ ! -f "$OUT" ]; then
        echo "=== Residual-IA+ noise=$sigma: $ds seed=$seed ==="
        python3 scripts/run_gap_closing.py --variant residual-ia \
          --dataset "$ds" --seed "$seed" --unfreeze frozen \
          --raw-branch-shared --raw-arch nlinear --val-early-stop \
          --gate-init -2.0 --noise-sigma "$sigma" --device $DEVICE 2>/dev/null || \
          echo "WARN: residual-ia noise=$sigma $ds $seed failed"
      else
        echo "SKIP: $OUT"
      fi

      # DLinear
      OUT="results/dlinear/${ds}_H96_${seed}_noise-${sigma}.json"
      if [ ! -f "$OUT" ]; then
        echo "=== DLinear noise=$sigma: $ds seed=$seed ==="
        python3 scripts/run_dlinear_baseline.py --dataset "$ds" --seed "$seed" \
          --noise-sigma "$sigma" --device $DEVICE 2>/dev/null || \
          echo "WARN: dlinear noise=$sigma $ds $seed failed"
      else
        echo "SKIP: $OUT"
      fi
    done
  done
done

echo "=== Exp A complete ==="
fi

echo ""
echo "================================================================"
echo "ALL STRENGTHENING EXPERIMENTS COMPLETE"
echo "================================================================"
