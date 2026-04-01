#!/bin/bash
# Run all standard benchmarks on RACE VM
# Usage: bash scripts/run_all_benchmarks.sh
set -e
cd ~/neuralips26
mkdir -p results/standard_benchmark

echo "============================================"
echo "Standard LTSF Benchmark Suite"
echo "============================================"

# ETTh1 — primary comparison dataset
echo ">>> ETTh1 all horizons"
python3 -u scripts/run_standard_benchmark.py --dataset ETTh1 --horizon 96,192,336,720 --n-epochs 15

# ETTm1
echo ">>> ETTm1 all horizons"
python3 -u scripts/run_standard_benchmark.py --dataset ETTm1 --horizon 96,192,336,720 --n-epochs 15

# ETTh2
echo ">>> ETTh2 all horizons"
python3 -u scripts/run_standard_benchmark.py --dataset ETTh2 --horizon 96,192,336,720 --n-epochs 15

# ETTm2
echo ">>> ETTm2 all horizons"
python3 -u scripts/run_standard_benchmark.py --dataset ETTm2 --horizon 96,192,336,720 --n-epochs 15

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "Results in results/standard_benchmark/"
echo "============================================"
