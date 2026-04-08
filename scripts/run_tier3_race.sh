#!/bin/bash
# Tier 3 RACE VM batch runner — push to 8+ experiments
# Total: ~6 GPU-hours on A10G
# Run: bash scripts/run_tier3_race.sh 2>&1 | tee results/tier3_run.log

set -e
PYTHON=python3
DEVICE="cuda"
EPOCHS=15

echo "================================================================"
echo "TIER 3 EXPERIMENT BATCH — $(date)"
echo "================================================================"

# ===================================================================
# EXP A: Multi-Horizon Extension (4 datasets × 3 horizons × 3 seeds)
# ~4.8 GPU-hours
# ===================================================================
echo ""
echo "=== EXP A: Multi-Horizon Extension ==="

for ds in Weather ETTh2 ETTm2 Electricity; do
  for h in 192 336 720; do
    for seed in 42 43 44; do
      echo "[$(date +%H:%M:%S)] Multi-H: $ds H=$h seed=$seed"
      $PYTHON scripts/run_rr_moa.py \
        --dataset $ds --horizon $h --unfreeze frozen --top-k 2 \
        --seed $seed --epochs $EPOCHS --device $DEVICE --no-baselines \
        || echo "FAILED: multih $ds $h $seed"
    done
  done
done

# ===================================================================
# EXP B: RevIN-Disabled Trajectory (1 run)
# ~10 min
# ===================================================================
echo ""
echo "=== EXP B: RevIN-Disabled Trajectory ==="

echo "[$(date +%H:%M:%S)] Trajectory: ETTh1 last4 no-revin"
$PYTHON scripts/run_adamix.py \
  --dataset ETTh1 --unfreeze last4 --seed 42 --disable-revin \
  --trajectory results/adamix/trajectory_ETTh1_last4_42_no_revin.jsonl \
  --epochs $EPOCHS --device $DEVICE \
  || echo "FAILED: trajectory_no_revin"

# ===================================================================
# EXP C: Full-FT with RevIN Disabled (3 runs)
# ~30 min
# ===================================================================
echo ""
echo "=== EXP C: Full-FT No-RevIN ==="

for ds in ETTh1 ETTm1 Weather; do
  echo "[$(date +%H:%M:%S)] Full-FT no-revin: $ds seed=42"
  $PYTHON scripts/run_full_finetune.py \
    --dataset $ds --seed 42 --epochs $EPOCHS \
    --device $DEVICE --disable-revin \
    || echo "FAILED: fullft_norevin $ds"
done

# ===================================================================
# EXP D: Multi-Tenant Benchmark (no training)
# ~15 min
# ===================================================================
echo ""
echo "=== EXP D: Multi-Tenant Benchmark ==="

echo "[$(date +%H:%M:%S)] Multi-tenant benchmark"
$PYTHON scripts/benchmark_multitenant.py \
  --backbone AutonLab/MOMENT-1-small --device $DEVICE \
  || echo "FAILED: multitenant"

echo ""
echo "================================================================"
echo "ALL TIER 3 EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
