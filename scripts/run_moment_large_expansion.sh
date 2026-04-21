#!/bin/bash
# Fill MOMENT-large cross-backbone gaps: ETTh2, ETTm2, Electricity
# Plus AdaMix baselines for comparison
# Each run: ~90 sec on A10G for MOMENT-large
set -e
PYTHON=python3
DEVICE="cuda"
BACKBONE="AutonLab/MOMENT-1-large"

mkdir -p results/rr_moa results/adamix

# RR-MoA: 3 missing datasets × 5 seeds × frozen = 15 runs
for dataset in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44 45 46; do
    OUT="results/rr_moa/${dataset}_H96_K5_top2_frozen_${seed}_bb-moment-large.json"
    if [ -f "$OUT" ]; then
      echo "SKIP (exists): $OUT"
      continue
    fi
    echo "RUN: $dataset seed=$seed backbone=moment-large frozen"
    $PYTHON scripts/run_rr_moa.py \
      --dataset $dataset --horizon 96 --K 5 --top-k 2 \
      --unfreeze frozen --seed $seed --epochs 15 \
      --backbone "$BACKBONE" --device $DEVICE --no-baselines \
      || echo "FAILED: $dataset seed=$seed"
  done
done

# AdaMix baselines for the 3 missing datasets (needed for comparison)
for dataset in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44 45 46; do
    OUT="results/adamix/${dataset}_H96_K5_frozen_${seed}_bb-moment-large.json"
    if [ -f "$OUT" ]; then
      echo "SKIP (exists): $OUT"
      continue
    fi
    echo "RUN AdaMix: $dataset seed=$seed backbone=moment-large"
    $PYTHON scripts/run_adamix.py \
      --dataset $dataset --horizon 96 --K 5 \
      --unfreeze frozen --seed $seed --epochs 15 \
      --backbone "$BACKBONE" --device $DEVICE \
      || echo "FAILED AdaMix: $dataset seed=$seed"
  done
done

echo "MOMENT-large expansion DONE"
