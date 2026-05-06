#!/bin/bash
# 10-seed confirmation for Residual-IA+ (extends n=5 → n=10).
#
# Adds seeds 47-51 on all 6 datasets. Tightens statistical power on:
#   - ETTh1 (current +1.8%, p=0.023) — could flip to marginal at n=10
#   - ETTh2 (current +2.3% parity) — confirms parity with tighter CI
#   - ETTm1, ETTm2, Weather, Electricity — confirm wins/parity
#
# 6 datasets × 5 new seeds = 30 runs.
#
# Invocation:
#   tmux new-session -d -s t10 'cd ~/neuralips26 && bash scripts/run_ria_plus_10seed_vm.sh 2>&1 | tee results/ria_10seed.log'

set -e
DEVICE="cuda"
EPOCHS=25
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
NEW_SEEDS=(47 48 49 50 51)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  10SEED — $1 — $(date)\n================================================================"; }

banner "Residual-IA+ seeds 47-51 × 6 datasets (30 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
            --dataset "$ds" --seed "$seed" \
            $BASE_FLAGS > /tmp/t10_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED+1))
            echo "  FAILED (rc=$RC):"; tail -3 /tmp/t10_${RUN_IDX}.out | sed 's/^/    /'
        else
            LAUNCHED=$((LAUNCHED+1))
            grep "MSE=" /tmp/t10_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX+1))
    done
done

echo -e "\n================================================================"
echo "10SEED DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
