#!/bin/bash
# Phase 2: High-impact experiments to run overnight after Phase 1
# Estimated: ~4-5 GPU-hours on A10G
# Run: nohup bash scripts/run_phase2_race.sh 2>&1 | tee results/phase2_run.log &

set -e
PYTHON=python3
DEVICE="cuda"

echo "================================================================"
echo "PHASE 2: HIGH-IMPACT OVERNIGHT EXPERIMENTS — $(date)"
echo "================================================================"

# ──────────────────────────────────────────────────────────────────
# B3: MACRO EXPERT POOL on remaining datasets
# Already have: ETTh1/ETTm1/Weather × 3 freeze × 3 seeds (27 files)
# Need: ETTh2/ETTm2/Electricity × 3 freeze × 3 seeds (27 runs)
# Answers: "hand-designed expert pool" critique
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== B3: MACRO EXPERT POOL — REMAINING DATASETS ==="
echo "============================================================"

mkdir -p results/rr_moa

for unfreeze in frozen last2 last4; do
  for ds in ETTh2 ETTm2 Electricity; do
    for seed in 42 43 44; do
      OUTFILE="results/rr_moa/${ds}_H96_K5_top2_${unfreeze}_${seed}_pool-macro.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): macro $ds $unfreeze seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Macro pool: $ds unfreeze=$unfreeze seed=$seed"
      $PYTHON scripts/run_rr_moa.py \
        --dataset $ds --unfreeze $unfreeze --seed $seed \
        --epochs 15 --device $DEVICE \
        --expert-pool macro --top-k 2 --no-baselines \
        || echo "FAILED: macro $ds $unfreeze $seed"
    done
  done
done

echo "[$(date +%H:%M:%S)] B3 COMPLETE"

# ──────────────────────────────────────────────────────────────────
# B3b: MACRO EXPERT POOL on MOIRAI backbone (all 6 datasets, frozen)
# Shows pool-agnostic + backbone-agnostic
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== B3b: MACRO EXPERT POOL — MOIRAI BACKBONE ==="
echo "============================================================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_bb-moirai_pool-macro.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): macro Moirai $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Macro pool (Moirai): $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --seed $seed \
      --epochs 15 --device $DEVICE \
      --backbone "Salesforce/moirai-1.1-R-small" \
      --expert-pool macro --top-k 2 --no-baselines \
      || echo "FAILED: macro Moirai $ds $seed"
  done
done

echo "[$(date +%H:%M:%S)] B3b COMPLETE"

# ──────────────────────────────────────────────────────────────────
# GAP-CLOSING: All 4 variants on remaining datasets (ETTh2, ETTm2, Electricity)
# Already have: ETTh1/ETTm1/Weather × 4 variants × 3 seeds
# Need: ETTh2/ETTm2/Electricity × 3 remaining variants × 3 seeds
# (dual-stream already covered by Phase 1 D3)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== GAP-CLOSING: REMAINING DATASETS × VARIANTS ==="
echo "============================================================"

mkdir -p results/gap_closing

for variant in film raw-expert multi-res; do
  for ds in ETTh2 ETTm2 Electricity; do
    for seed in 42 43 44; do
      OUTFILE="results/gap_closing/${variant}_${ds}_H96_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): $variant $ds seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Gap-closing $variant: $ds seed=$seed"
      $PYTHON scripts/run_gap_closing.py \
        --variant $variant --dataset $ds \
        --seed $seed --epochs 15 --device $DEVICE \
        || echo "FAILED: $variant $ds $seed"
    done
  done
done

echo "[$(date +%H:%M:%S)] GAP-CLOSING REMAINING COMPLETE"

# ──────────────────────────────────────────────────────────────────
# D5a: EXTENDED TRAINING (50 epochs) — DUAL-STREAM
# The DLinear gap is the #1 reviewer concern. Extended training may
# narrow it further. Run on all 6 datasets × 3 seeds.
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D5a: EXTENDED DUAL-STREAM (50 EPOCHS) ==="
echo "============================================================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}_ep50.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): dual-stream-50ep $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Dual-stream 50ep: $ds seed=$seed"
    # We save with a different name by temporarily renaming after run
    $PYTHON scripts/run_gap_closing.py \
      --variant dual-stream --dataset $ds \
      --seed $seed --epochs 50 --device $DEVICE \
      || { echo "FAILED: dual-stream-50ep $ds $seed"; continue; }
    # Rename the output to include _ep50 suffix
    SRCFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}.json"
    if [ -f "$SRCFILE" ]; then
      mv "$SRCFILE" "$OUTFILE"
      echo "  Renamed → $OUTFILE"
    fi
  done
done

echo "[$(date +%H:%M:%S)] D5a COMPLETE"

# ──────────────────────────────────────────────────────────────────
# D5b: EXTENDED TRAINING (50 epochs) — RR-MoA (frozen, canonical)
# Strengthens main results table.
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D5b: EXTENDED RR-MoA (50 EPOCHS) ==="
echo "============================================================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_ep50.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): RR-MoA-50ep $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] RR-MoA 50ep: $ds seed=$seed"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $ds --unfreeze frozen --seed $seed \
      --epochs 50 --device $DEVICE \
      --top-k 2 --no-baselines \
      || { echo "FAILED: RR-MoA-50ep $ds $seed"; continue; }
    # Rename to include _ep50
    SRCFILE="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}.json"
    if [ -f "$SRCFILE" ]; then
      mv "$SRCFILE" "$OUTFILE"
      echo "  Renamed → $OUTFILE"
    fi
  done
done

echo "[$(date +%H:%M:%S)] D5b COMPLETE"

# ──────────────────────────────────────────────────────────────────
# MULTI-HORIZON DUAL-STREAM (H=192, 336, 720)
# Shows DLinear gap narrows at longer horizons with dual-stream too
# Run on ETTh1 + Weather (datasets with clear gap narrative)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== MULTI-HORIZON DUAL-STREAM ==="
echo "============================================================"

for horizon in 192 336 720; do
  for ds in ETTh1 ETTm1 Weather; do
    for seed in 42 43 44; do
      OUTFILE="results/gap_closing/dual-stream_${ds}_H${horizon}_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): dual-stream $ds H=$horizon seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Dual-stream: $ds H=$horizon seed=$seed"
      $PYTHON scripts/run_gap_closing.py \
        --variant dual-stream --dataset $ds --horizon $horizon \
        --seed $seed --epochs 15 --device $DEVICE \
        || echo "FAILED: dual-stream $ds H$horizon $seed"
    done
  done
done

echo "[$(date +%H:%M:%S)] MULTI-HORIZON DUAL-STREAM COMPLETE"

# ──────────────────────────────────────────────────────────────────
# D4: 5-SEED CROSS-BACKBONE (seeds 45, 46)
# MOMENT-large + Moirai + Chronos, frozen, all 3 core datasets
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D4: 5-SEED CROSS-BACKBONE ==="
echo "============================================================"

for bb in "AutonLab/MOMENT-1-large" "Salesforce/moirai-1.1-R-small" "amazon/chronos-t5-small"; do
  # Determine suffix
  if echo "$bb" | grep -qi "moment.*large"; then
    BB_SUFFIX="bb-moment-large"
  elif echo "$bb" | grep -qi "moirai"; then
    BB_SUFFIX="bb-moirai"
  elif echo "$bb" | grep -qi "chronos"; then
    BB_SUFFIX="bb-chronos"
  fi

  for ds in ETTh1 ETTm1 Weather; do
    for seed in 45 46; do
      OUTFILE="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_${BB_SUFFIX}.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): $BB_SUFFIX $ds seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Cross-backbone $BB_SUFFIX: $ds seed=$seed"
      $PYTHON scripts/run_rr_moa.py \
        --dataset $ds --unfreeze frozen --seed $seed \
        --epochs 15 --device $DEVICE \
        --backbone "$bb" --top-k 2 --no-baselines \
        || echo "FAILED: $BB_SUFFIX $ds $seed"
    done
  done
done

echo "[$(date +%H:%M:%S)] D4 COMPLETE"

# ──────────────────────────────────────────────────────────────────
# DUAL-STREAM on MOIRAI with EXTENDED TRAINING (50 epochs)
# Highest chance of beating DLinear on multiple datasets
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== DUAL-STREAM + MOIRAI + 50 EPOCHS ==="
echo "============================================================"

for ds in ETTh1 ETTm1 Weather ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}_bb-moirai_ep50.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): dual-stream Moirai 50ep $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Dual-stream Moirai 50ep: $ds seed=$seed"
    $PYTHON scripts/run_gap_closing.py \
      --variant dual-stream --dataset $ds \
      --seed $seed --epochs 50 --device $DEVICE \
      --backbone "Salesforce/moirai-1.1-R-small" \
      || { echo "FAILED: dual-stream Moirai 50ep $ds $seed"; continue; }
    # Rename to include _ep50
    SRCFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}_bb-moirai.json"
    if [ -f "$SRCFILE" ]; then
      mv "$SRCFILE" "$OUTFILE"
      echo "  Renamed → $OUTFILE"
    fi
  done
done

echo "[$(date +%H:%M:%S)] DUAL-STREAM MOIRAI 50EP COMPLETE"

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "PHASE 2 COMPLETE — $(date)"
echo "================================================================"

echo ""
echo "=== FILE COUNTS ==="
echo "Macro pool (new datasets): $(ls results/rr_moa/*pool-macro* 2>/dev/null | grep -E 'ETTh2|ETTm2|Electricity' | wc -l) new"
echo "Macro pool (Moirai): $(ls results/rr_moa/*bb-moirai*pool-macro* 2>/dev/null | wc -l) files"
echo "Gap-closing (new datasets): $(ls results/gap_closing/*_ETTh2_* results/gap_closing/*_ETTm2_* results/gap_closing/*_Electricity_* 2>/dev/null | wc -l) files"
echo "Extended dual-stream (50ep): $(ls results/gap_closing/dual-stream_*_ep50.json 2>/dev/null | wc -l) files"
echo "Extended RR-MoA (50ep): $(ls results/rr_moa/*_ep50.json 2>/dev/null | wc -l) files"
echo "Multi-horizon dual-stream: $(ls results/gap_closing/dual-stream_*_H192_* results/gap_closing/dual-stream_*_H336_* results/gap_closing/dual-stream_*_H720_* 2>/dev/null | wc -l) files"
echo "5-seed cross-backbone: $(ls results/rr_moa/*_4[56]_bb-*.json 2>/dev/null | wc -l) files"
echo "Dual-stream Moirai 50ep: $(ls results/gap_closing/dual-stream_*_bb-moirai_ep50.json 2>/dev/null | wc -l) files"
