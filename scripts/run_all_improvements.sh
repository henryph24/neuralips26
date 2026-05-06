#!/bin/bash
# Master orchestrator for NeurIPS revision experiments.
# Run on GPU VM (A10G GPU). Total: ~2 GPU hours.
# Each experiment checks for existing results and skips if found.

set -e
cd "$(dirname "$0")/.."

DATASETS="ETTh1 ETTm1 Weather"
SEEDS="42 43 44"

echo "=============================================="
echo "NeurIPS 2026 Revision — Experiment Orchestrator"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

# ============================================================
# EXP 1A: TRACE baseline under FROZEN protocol (W1)
# ============================================================
echo "=== EXP 1A: TRACE (frozen) ==="
for SEED in $SEEDS; do
  for DS in $DATASETS; do
    OUT="results/trace_baseline/${DS}_H96_${SEED}_frozen.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN TRACE frozen: $DS seed=$SEED"
      python scripts/run_trace_baseline.py --dataset $DS --seed $SEED --unfreeze frozen
    fi
  done
done

# ============================================================
# EXP 1B: Independent ensemble baseline (W1, W4)
# ============================================================
echo ""
echo "=== EXP 1B: Independent Ensemble ==="
for SEED in $SEEDS; do
  for DS in $DATASETS; do
    OUT="results/independent_ensemble/${DS}_H96_frozen_${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN Ensemble: $DS seed=$SEED"
      python scripts/run_independent_ensemble.py --dataset $DS --seed $SEED --unfreeze frozen
    fi
  done
done

# ============================================================
# EXP 1C: RR-MoA with unfreeze=all (full fine-tuning ceiling)
# ============================================================
echo ""
echo "=== EXP 1C: RR-MoA (unfreeze=all) ==="
for SEED in $SEEDS; do
  for DS in $DATASETS; do
    OUT="results/rr_moa/${DS}_H96_K5_top2_all_${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN RR-MoA all: $DS seed=$SEED"
      python scripts/run_rr_moa.py --dataset $DS --seed $SEED --unfreeze all --top-k 2 --no-baselines
    fi
  done
done

# ============================================================
# EXP 2: MOMENT-large RR-MoA (W5)
# ============================================================
echo ""
echo "=== EXP 2: MOMENT-large RR-MoA ==="
for SEED in $SEEDS; do
  for DS in $DATASETS; do
    OUT="results/rr_moa/${DS}_H96_K5_top2_frozen_${SEED}_bb-moment-large.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN RR-MoA MOMENT-large: $DS seed=$SEED"
      python scripts/run_rr_moa.py \
        --dataset $DS --seed $SEED \
        --backbone AutonLab/MOMENT-1-large \
        --unfreeze frozen --top-k 2 --batch-size 64 --no-baselines
    fi
  done
done
# Baselines on MOMENT-large (seed 42 only)
for DS in $DATASETS; do
  OUT="results/rr_moa/${DS}_H96_K5_top2_frozen_42_bb-moment-large.json"
  # Re-run with baselines for seed 42
  echo "  RUN RR-MoA MOMENT-large + baselines: $DS seed=42"
  python scripts/run_rr_moa.py \
    --dataset $DS --seed 42 \
    --backbone AutonLab/MOMENT-1-large \
    --unfreeze frozen --top-k 2 --batch-size 64
done

# ============================================================
# EXP 3: Moirai RR-MoA (W5 — 3rd backbone)
# ============================================================
echo ""
echo "=== EXP 3: Moirai RR-MoA ==="
for SEED in $SEEDS; do
  for DS in $DATASETS; do
    OUT="results/rr_moa/${DS}_H96_K5_top2_frozen_${SEED}_bb-moirai.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN RR-MoA Moirai: $DS seed=$SEED"
      python scripts/run_rr_moa.py \
        --dataset $DS --seed $SEED \
        --backbone Salesforce/moirai-1.1-R-small \
        --unfreeze frozen --top-k 2 --batch-size 64 --no-baselines || \
        echo "  FAILED Moirai: $DS seed=$SEED (continuing...)"
    fi
  done
done

# ============================================================
# EXP 4: High-capacity Frozen Paradox (W3)
# ============================================================
echo ""
echo "=== EXP 4: Macro Experts + Unfreezing ==="
for DS in $DATASETS; do
  for UF in last2 last4; do
    for SEED in $SEEDS; do
      OUT="results/rr_moa/${DS}_H96_K5_top2_${UF}_${SEED}_pool-macro.json"
      if [ -f "$OUT" ]; then
        echo "  SKIP $OUT (exists)"
      else
        echo "  RUN Macro $UF: $DS seed=$SEED"
        python scripts/run_rr_moa.py \
          --dataset $DS --seed $SEED \
          --unfreeze $UF --top-k 2 --expert-pool macro --no-baselines
      fi
    done
  done
done

# ============================================================
# EXP 5: Multi-seed imputation
# ============================================================
echo ""
echo "=== EXP 5: Multi-seed Imputation ==="
for SEED in 43 44; do
  for DS in $DATASETS; do
    OUT="results/imputation/${DS}_${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "  SKIP $OUT (exists)"
    else
      echo "  RUN Imputation: $DS seed=$SEED"
      python scripts/run_imputation.py --dataset $DS --seed $SEED
    fi
  done
done

# ============================================================
# EXP 6: Routing visualization
# ============================================================
echo ""
echo "=== EXP 6: Routing Analysis ==="
for DS in ETTh1 Weather; do
  OUT="results/routing_analysis/${DS}_42.json"
  if [ -f "$OUT" ]; then
    echo "  SKIP $OUT (exists)"
  else
    echo "  RUN Routing analysis: $DS"
    python scripts/run_patchwise_analysis.py --dataset $DS --seed 42
  fi
done

# ============================================================
# EXP 7: AdaMix collapse on Moirai
# ============================================================
echo ""
echo "=== EXP 7: AdaMix on Moirai ==="
for DS in $DATASETS; do
  echo "  RUN AdaMix Moirai: $DS"
  python scripts/run_adamix.py \
    --dataset $DS --seed 42 \
    --backbone Salesforce/moirai-1.1-R-small \
    --unfreeze last4 || echo "  FAILED AdaMix Moirai: $DS (continuing...)"
done

# ============================================================
# SIGNIFICANCE TESTS (no GPU needed)
# ============================================================
echo ""
echo "=== Statistical Significance Tests ==="
python scripts/compute_significance.py

echo ""
echo "=============================================="
echo "All experiments completed: $(date)"
echo "=============================================="
