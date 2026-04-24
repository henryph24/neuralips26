#!/bin/bash
# SR-RIA+: Self-Routed Residual-IA+ sweep
# 6 datasets x 5 seeds = 30 runs (~45 min on A10G)
set -e

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44 45 46)

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT="results/sr_ria/${DS}_H96_K5_frozen_${SEED}.json"
        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT"
            continue
        fi
        echo "=== $DS seed=$SEED ==="
        python3 scripts/run_sr_ria.py \
            --dataset "$DS" --seed "$SEED" --K 5 --epochs 15 \
            --unfreeze frozen --warmup-epochs 5 --raw-arch nlinear \
            --device cuda || echo "FAILED: $DS $SEED"
    done
done
echo "Done."
