#!/bin/bash
# NeurIPS 8+ experiments: RevIN ablation + Traffic full grid
# Run on RACE VM (A10G GPU)
# Estimated: ~28 hours total (run overnight)

set -e
cd ~/neuralips26

LOG_DIR="results/8plus_logs"
mkdir -p "$LOG_DIR" results/adamix results/rr_moa

echo "=========================================="
echo "PHASE 1: RevIN Ablation (Kill Shot)"
echo "AdaMix with RevIN disabled on MOMENT"
echo "9 runs: 3 datasets × 3 seeds"
echo "=========================================="

for DATASET in ETTh1 ETTm1 Weather; do
  for SEED in 42 43 44; do
    echo ""
    echo ">>> AdaMix NO-REVIN: $DATASET seed=$SEED unfreeze=last4"
    python3 scripts/run_adamix.py \
      --dataset "$DATASET" \
      --unfreeze last4 \
      --seed "$SEED" \
      --disable-revin \
      --device cuda \
      2>&1 | tee "$LOG_DIR/adamix_norevin_${DATASET}_last4_${SEED}.log"
  done
done

echo ""
echo "=========================================="
echo "PHASE 2: Traffic RR-MoA Full Grid"
echo "9 runs: 3 freeze levels × 3 seeds"
echo "=========================================="

for UNFREEZE in frozen last2 last4; do
  for SEED in 42 43 44; do
    echo ""
    echo ">>> RR-MoA Traffic: unfreeze=$UNFREEZE seed=$SEED"
    python3 scripts/run_rr_moa.py \
      --dataset Traffic \
      --top-k 2 \
      --unfreeze "$UNFREEZE" \
      --seed "$SEED" \
      --device cuda \
      2>&1 | tee "$LOG_DIR/rrmoa_Traffic_${UNFREEZE}_${SEED}.log"
  done
done

echo ""
echo "=========================================="
echo "PHASE 3: Traffic AdaMix Baselines"
echo "9 runs: 3 freeze levels × 3 seeds"
echo "=========================================="

for UNFREEZE in frozen last2 last4; do
  for SEED in 42 43 44; do
    echo ""
    echo ">>> AdaMix Traffic: unfreeze=$UNFREEZE seed=$SEED"
    python3 scripts/run_adamix.py \
      --dataset Traffic \
      --unfreeze "$UNFREEZE" \
      --seed "$SEED" \
      --device cuda \
      2>&1 | tee "$LOG_DIR/adamix_Traffic_${UNFREEZE}_${SEED}.log"
  done
done

echo ""
echo "=========================================="
echo "PHASE 4: Traffic DLinear Calibration"
echo "3 runs: 3 seeds"
echo "=========================================="

for SEED in 42 43 44; do
  echo ""
  echo ">>> DLinear Traffic: seed=$SEED"
  python3 scripts/run_dlinear_baseline.py \
    --dataset Traffic \
    --seed "$SEED" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/dlinear_Traffic_${SEED}.log"
done

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Total experiments: 30"
echo "=========================================="
echo "Results saved to:"
echo "  results/adamix/*_no_revin.json (RevIN ablation)"
echo "  results/rr_moa/Traffic_*.json (Traffic RR-MoA)"
echo "  results/adamix/Traffic_*.json (Traffic AdaMix)"
