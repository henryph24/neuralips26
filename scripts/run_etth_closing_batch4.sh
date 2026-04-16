#!/bin/bash
# Batch 4: NLinear raw branch (shift-linear-unshift, standard LTSF trick).
#
# 6 datasets × 3 seeds = 18 new runs (linear baseline already in results).
# Uses best-known base config + --raw-arch nlinear.
#
# Invocation:
#   tmux new-session -d -s b4 'cd ~/neuralips26 && bash scripts/run_etth_closing_batch4.sh 2>&1 | tee results/etth_b4.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS3=(42 43 44)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --raw-arch nlinear --grad-clip 1.0"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  BATCH 4 — $1 — $(date)\n================================================================"; }

banner "NLinear raw branch × 6 datasets × 3 seeds (18 runs)"
for ds in "${ALL_DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256_d1_wd0.0001_cos_wu5_nlinear_gc1.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] ds=$ds s=$seed NLINEAR"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
            --dataset "$ds" --seed "$seed" \
            $BASE_FLAGS > /tmp/b4_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED+1))
            echo "  FAILED (rc=$RC):"; tail -3 /tmp/b4_${RUN_IDX}.out | sed 's/^/    /'
        else
            LAUNCHED=$((LAUNCHED+1))
            grep "MSE=" /tmp/b4_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX+1))
    done
done

echo -e "\n================================================================"
echo "BATCH 4 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
