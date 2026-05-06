#!/bin/bash
# V1 ablation: Naive raw routing variants
# Tests whether the fix requires careful design beyond "route on raw input"
#
# 3 variants x 3 datasets x 3 seeds = 27 runs (~20 min on A10G)
#
# Variants:
#   (a) linear router — no temporal conv, just Linear(512, K)
#   (b) frozen-random router — Conv1d router frozen at random init
#   (c) identical-mean experts — 5 identical MeanPool heads

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="ETTh1 ETTm1 Weather"
SEEDS="42 43 44"
EPOCHS=15
DEVICE=cuda
TOPK=2  # match paper's main results

for ds in $DATASETS; do
  for seed in $SEEDS; do
    # (a) Linear router
    OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-linear.json"
    if [ ! -f "$OUT" ]; then
      echo "=== Linear router: $ds seed=$seed ==="
      python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" --epochs $EPOCHS \
        --router-arch linear --unfreeze frozen --no-baselines --device $DEVICE --top-k $TOPK
    else
      echo "SKIP (exists): $OUT"
    fi

    # (b) Frozen-random router (Conv1d at random init, frozen)
    OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_frozenrouter.json"
    if [ ! -f "$OUT" ]; then
      echo "=== Frozen-random router: $ds seed=$seed ==="
      python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" --epochs $EPOCHS \
        --freeze-router --unfreeze frozen --no-baselines --device $DEVICE --top-k $TOPK
    else
      echo "SKIP (exists): $OUT"
    fi

    # (c) Identical-mean experts
    OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_pool-identical-mean.json"
    if [ ! -f "$OUT" ]; then
      echo "=== Identical-mean experts: $ds seed=$seed ==="
      python3 scripts/run_rr_moa.py --dataset "$ds" --seed "$seed" --epochs $EPOCHS \
        --expert-pool identical-mean --unfreeze frozen --no-baselines --device $DEVICE --top-k $TOPK
    else
      echo "SKIP (exists): $OUT"
    fi
  done
done

echo ""
echo "=== All naive routing ablation runs complete ==="
