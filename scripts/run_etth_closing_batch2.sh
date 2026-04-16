#!/bin/bash
# Batch 2: Validation-based early stopping.
#
# 6 datasets × 5 seeds × 1 config (val-early-stop + grad-clip) = 30 runs (~75 min)
#
# Uses best-known base config PLUS:
#   --val-early-stop --val-patience 5 --grad-clip 1.0
#   --epochs 25  (longer, but early-stop will prune)
#
# Diagnostic output: best_epoch + val_mse_curve in each JSON.
#
# Invocation:
#   tmux new-session -d -s b2 'cd ~/neuralips26 && bash scripts/run_etth_closing_batch2.sh 2>&1 | tee results/etth_b2.log'

set -e
DEVICE="cuda"
EPOCHS=25  # longer than default; early-stop will prune
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS5=(42 43 44 45 46)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  BATCH 2 — $1 — $(date)\n================================================================"; }

banner "Val early-stop × 6 datasets × 5 seeds (30 runs)"
for ds in "${ALL_DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256_d1_wd0.0001_cos_wu5_es5_gc1.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] ds=$ds s=$seed ES+GC"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
            --dataset "$ds" --seed "$seed" \
            $BASE_FLAGS > /tmp/b2_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED+1))
            echo "  FAILED (rc=$RC):"; tail -3 /tmp/b2_${RUN_IDX}.out | sed 's/^/    /'
        else
            LAUNCHED=$((LAUNCHED+1))
            grep -E "MSE=|Early stop|Restored" /tmp/b2_${RUN_IDX}.out | tail -3 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX+1))
    done
done

echo -e "\n================================================================"
echo "BATCH 2 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
