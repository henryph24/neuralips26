#!/bin/bash
# Tier 2 RACE VM batch runner — new experiments for NeurIPS acceptance
# Total: ~8 GPU-hours on A10G
# Run from repo root: bash scripts/run_tier2_race.sh 2>&1 | tee results/tier2_run.log

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

echo "================================================================"
echo "TIER 2 EXPERIMENT BATCH — $(date)"
echo "================================================================"

# ===================================================================
# EXP 1: Full Fine-Tuning Baseline (highest priority)
# Unfreeze ALL encoder blocks + single head. ~2.4 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 1: Full Fine-Tuning Baseline ==="

for ds in ETTh1 ETTm1 Weather ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Full-FT: $ds seed=$seed"
    $PYTHON scripts/run_full_finetune.py \
      --dataset $ds --seed $seed --epochs $EPOCHS \
      --device $DEVICE || echo "FAILED: full_finetune $ds $seed"
  done
done

# ===================================================================
# EXP 2: 5 Seeds for Core Table (seeds 45, 46)
# RR-MoA frozen Top-2 on 6 datasets. ~1.6 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 2: 5 Seeds (45, 46) ==="

for ds in ETTh1 ETTm1 Weather ETTh2 ETTm2 Electricity; do
  for seed in 45 46; do
    echo "[$(date +%H:%M:%S)] RR-MoA 5-seed: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --seed $seed --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: rrmoa_5seed $ds $seed"
  done
done

# ===================================================================
# EXP 3: Chronos Cross-Backbone (decoder-only)
# RR-MoA + AdaMix control on Chronos-T5-small. ~1.7 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 3: Chronos Cross-Backbone ==="

for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Chronos RR-MoA: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --backbone amazon/chronos-t5-small \
      --seed $seed --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: chronos_rrmoa $ds $seed"
  done
done

# AdaMix on Chronos (should NOT collapse — no RevIN)
echo "[$(date +%H:%M:%S)] Chronos AdaMix control (should not collapse)"
$PYTHON scripts/run_adamix.py \
  --dataset ETTh1 --backbone amazon/chronos-t5-small \
  --unfreeze last4 --seed 42 --epochs $EPOCHS --device $DEVICE \
  || echo "FAILED: chronos_adamix"

# ===================================================================
# EXP 4: Moirai Extended Grid (3 more datasets)
# ~1.5 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 4: Moirai Extended Grid ==="

for ds in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] Moirai RR-MoA: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --top-k 2 \
      --backbone Salesforce/moirai-1.1-R-small \
      --seed $seed --epochs $EPOCHS --batch-size 64 --device $DEVICE \
      || echo "FAILED: moirai_extended $ds $seed"
  done
done

# ===================================================================
# EXP 5: LoRA with Unfreezing (explains why LoRA fails)
# ~0.5 GPU-hours
# ===================================================================
echo ""
echo "=== EXP 5: LoRA with Last-4 Unfreezing ==="

for ds in ETTh1 ETTm1 Weather; do
  for seed in 42 43 44; do
    echo "[$(date +%H:%M:%S)] LoRA unfrozen: $ds seed=$seed"
    $PYTHON scripts/run_lora_baseline.py \
      --dataset $ds --rank 16 --target-modules qkvo --head mlp2 \
      --unfreeze last4 --seed $seed --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: lora_unfrozen $ds $seed"
  done
done

# ===================================================================
# EXP 6: Inference Benchmarks (no training)
# ~15 minutes
# ===================================================================
echo ""
echo "=== EXP 6: Inference Benchmarks ==="

echo "[$(date +%H:%M:%S)] Benchmark: MOMENT-small"
$PYTHON scripts/benchmark_inference.py \
  --backbone AutonLab/MOMENT-1-small --device $DEVICE \
  || echo "FAILED: benchmark_moment"

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Check results/full_finetune/ for full-FT baseline numbers"
echo "  2. Check results/rr_moa/ for new seed 45/46 and Chronos/Moirai files"
echo "  3. Check results/lora_baseline/ for LoRA unfrozen results"
echo "  4. Check results/benchmark/ for inference latency numbers"
echo "  5. Run: python evidence_vm/verify.py (verify existing claims still hold)"
