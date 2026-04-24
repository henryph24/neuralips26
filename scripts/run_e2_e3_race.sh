#!/bin/bash
# E2: Hidden-reinjected ablation (18 runs) + E3: 50-epoch frozen RR-MoA (9 runs)
# Total: 27 runs, ~75 min on A10G
set -e

echo "=== E2: Statistics re-injection ablation ==="
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS=(42 43 44)
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT="results/rr_moa/${DS}_H96_K5_top2_frozen_${SEED}_router-hidden_reinjected.json"
        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT"
            continue
        fi
        echo "--- E2: $DS seed=$SEED ---"
        python scripts/run_rr_moa.py \
            --dataset "$DS" --K 5 --top-k 2 --unfreeze frozen --seed "$SEED" \
            --router-input-mode hidden_reinjected --epochs 15 \
            --no-baselines --device cuda || echo "FAILED: $DS $SEED"
    done
done

echo ""
echo "=== E3: 50-epoch frozen RR-MoA ==="
CORE_DS=(ETTh1 ETTm1 Weather)
for DS in "${CORE_DS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT="results/rr_moa/${DS}_H96_K5_top2_frozen_${SEED}_ep50.json"
        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT"
            continue
        fi
        echo "--- E3: $DS seed=$SEED ---"
        python scripts/run_rr_moa.py \
            --dataset "$DS" --K 5 --top-k 2 --unfreeze frozen --seed "$SEED" \
            --epochs 50 --no-baselines --device cuda || echo "FAILED: $DS $SEED"
    done
done

echo "Done."
