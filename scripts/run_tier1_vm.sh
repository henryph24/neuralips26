#!/usr/bin/env bash
# Tier-1 + T2.A experiment batch for the GPU VM (A10G).
#
#   T1.A : already complete offline via denormalize_existing_results.py;
#          new runs below will emit exact mse_denorm via compute_denorm_mse.
#   T1.B : RR-MoA uniform-router control on {ETTh1, ETTm1, Weather} x 3 seeds,
#          strictly frozen.
#   T1.C : RR-MoA + AdaMix freeze grid on {ETTh2, ETTm2, Electricity}
#          x {frozen, last2, last4} x 3 seeds.
#   T2.A : AdaMix trajectory logging on ETTh1 last-4 seed 42 (collapse case)
#          and ETTh1 frozen seed 42 (control). Used to plot the per-step
#          router entropy vs. per-expert gradient norm figure that upgrades
#          the gradient co-adaptation claim from correlational to mechanistic.
#   T2.B : Weather x 3 freeze seed 45 -- reconciliation of the Frozen Paradox
#          exception on Weather.
#
# Cost estimate: ~2.5 GPU-hours total on A10G. Run from repo root:
#   bash scripts/run_tier1_vm.sh 2>&1 | tee results/tier1_race.log

set -eu

SEEDS="42 43 44"
EXTRA_DATASETS="ETTh2,ETTm2,Electricity"
MAIN_DATASETS="ETTh1,ETTm1,Weather"
EPOCHS=15

mkdir -p results/adamix results/rr_moa results/dlinear

echo "=========================================="
echo "TIER 1.C: extended freeze grid (new datasets)"
echo "=========================================="
for s in $SEEDS; do
    python scripts/run_freeze_ablation.py \
        --experiment freeze \
        --datasets "$EXTRA_DATASETS" \
        --seed "$s" --epochs "$EPOCHS"
    python scripts/run_freeze_ablation.py \
        --experiment adamix \
        --datasets "$EXTRA_DATASETS" \
        --seed "$s" --epochs "$EPOCHS"
done

echo "=========================================="
echo "TIER 1.B: uniform-router control"
echo "=========================================="
for s in $SEEDS; do
    python scripts/run_freeze_ablation.py \
        --experiment uniform \
        --datasets "$MAIN_DATASETS" \
        --seed "$s" --epochs "$EPOCHS"
done

echo "=========================================="
echo "TIER 1.A: DLinear reference (re-run to capture exact denorm MSE)"
echo "=========================================="
for s in $SEEDS; do
    python scripts/run_freeze_ablation.py \
        --experiment dlinear \
        --datasets "$MAIN_DATASETS,$EXTRA_DATASETS" \
        --seed "$s" --epochs "$EPOCHS"
done

echo "=========================================="
echo "TIER 2.A: AdaMix trajectory logging"
echo "=========================================="
# Collapse case: ETTh1 last-4 seed 42
python scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze last4 --seed 42 --epochs "$EPOCHS" \
    --trajectory results/adamix/trajectory_ETTh1_last4_42.jsonl \
    --trajectory-max-steps 400

# Frozen control: ETTh1 frozen seed 42
python scripts/run_adamix.py \
    --dataset ETTh1 --unfreeze frozen --seed 42 --epochs "$EPOCHS" \
    --trajectory results/adamix/trajectory_ETTh1_frozen_42.jsonl \
    --trajectory-max-steps 400

echo "=========================================="
echo "TIER 2.B: Weather Frozen Paradox reconciliation (seed 45)"
echo "=========================================="
python scripts/run_freeze_ablation.py \
    --experiment freeze \
    --datasets "Weather" \
    --seed 45 --epochs "$EPOCHS"

echo "=========================================="
echo "TIER 3.A: macro-expert pool"
echo "=========================================="
for s in $SEEDS; do
    python scripts/run_freeze_ablation.py \
        --experiment macro \
        --datasets "$MAIN_DATASETS" \
        --seed "$s" --epochs "$EPOCHS"
done

echo "=========================================="
echo "TIER 3.B: LoRA sweep (rank x targets x head x seeds x datasets)"
echo "=========================================="
python scripts/run_lora_sweep.py \
    --datasets "$MAIN_DATASETS" \
    --epochs "$EPOCHS" --device cuda

echo "=========================================="
echo "ALL TIER 1 + T2 + T3 EXPERIMENTS COMPLETE"
echo "=========================================="
python scripts/run_lora_sweep.py --summarize

# Re-run the denormalized post-processor to back-fill approximate denorm
# columns on any historical JSONs that were written before this refactor.
python scripts/denormalize_existing_results.py
