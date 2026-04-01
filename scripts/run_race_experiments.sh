#!/bin/bash
# Run all NeurIPS experiments on RACE VM
# Usage: bash scripts/run_race_experiments.sh
#
# Runs sequentially on single A10G GPU. Total: ~5 hours.

set -e
cd ~/neuralips26
mkdir -p results/local_evolution

DATASETS="etth1 etth2 ettm1"
SEED=42
GENS=12
POP=20

echo "============================================================"
echo "NeurIPS 2026 Experiment Suite — RACE VM"
echo "Datasets: $DATASETS | Seed: $SEED | Gens: $GENS | Pop: $POP"
echo "============================================================"

# Phase 1: Random Code + Evolution baselines
echo ""
echo "========== PHASE 1: Random Code + Evolution =========="
for ds in $DATASETS; do
    echo ""
    echo ">>> Random Code+Evo: $ds (seed=$SEED)"
    python3 -u scripts/run_local_evolution.py \
        --dataset $ds --mode random \
        --n-generations $GENS --pop-size $POP --seed $SEED
done

# Phase 2: Self-improving iteration 0 (fine-tuned on Modal winners)
echo ""
echo "========== PHASE 2: Self-Improving Iter 0 =========="
for ds in $DATASETS; do
    echo ""
    echo ">>> Self-Improving iter0: $ds (seed=$SEED)"
    python3 -u scripts/run_local_evolution.py \
        --dataset $ds --mode local_llm \
        --adapter-path models/qwen-adapter-coder \
        --n-generations $GENS --pop-size $POP --seed $SEED
done

# Phase 3: Self-improvement iteration 1
echo ""
echo "========== PHASE 3: Self-Improvement Iteration 1 =========="
python3 -u scripts/self_improve_iterate.py --iteration 1

# Phase 4: Self-improving iteration 1 evolution
echo ""
echo "========== PHASE 4: Self-Improving Iter 1 =========="
for ds in $DATASETS; do
    echo ""
    echo ">>> Self-Improving iter1: $ds (seed=$SEED)"
    python3 -u scripts/run_local_evolution.py \
        --dataset $ds --mode local_llm \
        --adapter-path models/qwen-adapter-coder-iter1 \
        --n-generations $GENS --pop-size $POP --seed $SEED
done

# Phase 5: Compare results
echo ""
echo "========== PHASE 5: Results Comparison =========="
python3 scripts/compare_results.py

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Results in: results/local_evolution/"
echo "============================================================"
