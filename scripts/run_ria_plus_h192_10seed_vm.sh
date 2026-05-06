#!/bin/bash
# H=192 10-seed extension for Residual-IA+ (locks in 6/6 claim).
#
# 6 datasets × seeds {47,48,49,50,51} = 30 runs
#
# Invocation:
#   tmux new-session -d -s h192 'cd ~/neuralips26 && bash scripts/run_ria_plus_h192_10seed_vm.sh 2>&1 | tee results/ria_h192_10seed.log'

set -e
DEVICE="cuda"
EPOCHS=25
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
NEW_SEEDS=(47 48 49 50 51)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared --horizon 192"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  H192-10seed — $1 — $(date)\n================================================================"; }

banner "Residual-IA+ H=192 seeds 47-51 × 6 datasets (30 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H192_${seed}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed H=192"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
            --dataset "$ds" --seed "$seed" \
            $BASE_FLAGS > /tmp/h192_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED+1))
            echo "  FAILED (rc=$RC):"; tail -3 /tmp/h192_${RUN_IDX}.out | sed 's/^/    /'
        else
            LAUNCHED=$((LAUNCHED+1))
            grep "MSE=" /tmp/h192_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX+1))
    done
done

echo -e "\n================================================================"
echo "H192-10seed DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
