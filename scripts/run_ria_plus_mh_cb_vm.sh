#!/bin/bash
# Multi-horizon cross-backbone Residual-IA+ sweep.
#
# 3 backbones × 3 datasets × 3 horizons × 3 seeds = 81 runs
# Creates a dense 3×3×3 generalization matrix extending H=96 cross-backbone
# result into longer horizons.
#
# Invocation:
#   tmux new-session -d -s mhcb 'cd ~/neuralips26 && bash scripts/run_ria_plus_mh_cb_vm.sh 2>&1 | tee results/ria_mhcb.log'

set -e
DEVICE="cuda"
EPOCHS=20
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTm1 Weather)
HORIZONS=(192 336 720)
SEEDS3=(42 43 44)

declare -a BACKBONES=(
    "Salesforce/moirai-1.1-R-small|bb-moirai"
    "amazon/chronos-t5-small|bb-chronos"
    "AutonLab/MOMENT-1-large|bb-moment-large"
)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  $1 — $(date)\n================================================================"; }

banner "Multi-Horizon Cross-Backbone Residual-IA+ (81 runs)"
for bb_entry in "${BACKBONES[@]}"; do
    backbone="${bb_entry%|*}"
    bbsuf="${bb_entry#*|}"
    for horizon in "${HORIZONS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for seed in "${SEEDS3[@]}"; do
                out="${RESULTS_DIR}/residual-ia_${ds}_H${horizon}_${seed}_${bbsuf}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
                if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

                echo "[$(date +%H:%M:%S)] [$RUN_IDX] bb=$bbsuf $ds H=$horizon s=$seed"
                set +e
                $PYTHON scripts/run_gap_closing.py \
                    --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
                    --dataset "$ds" --seed "$seed" --horizon "$horizon" \
                    --backbone "$backbone" \
                    $BASE_FLAGS > /tmp/mhcb_${RUN_IDX}.out 2>&1
                RC=$?
                set -e
                if [ $RC -ne 0 ]; then
                    FAILED=$((FAILED+1))
                    echo "  FAILED (rc=$RC):"; tail -3 /tmp/mhcb_${RUN_IDX}.out | sed 's/^/    /'
                else
                    LAUNCHED=$((LAUNCHED+1))
                    grep "MSE=" /tmp/mhcb_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
                fi
                RUN_IDX=$((RUN_IDX+1))
            done
        done
    done
done

echo -e "\n================================================================"
echo "MH-CB DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
