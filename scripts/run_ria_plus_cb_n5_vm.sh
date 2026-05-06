#!/bin/bash
# Cross-backbone H=96 n=5 extension (was n=3).
#
# 3 backbones × 3 datasets × 2 new seeds (45, 46) = 18 runs
#
# Invocation:
#   tmux new-session -d -s cbn5 'cd ~/neuralips26 && bash scripts/run_ria_plus_cb_n5_vm.sh 2>&1 | tee results/ria_cb_n5.log'

set -e
DEVICE="cuda"
EPOCHS=20
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTm1 Weather)
NEW_SEEDS=(45 46)

declare -a BACKBONES=(
    "Salesforce/moirai-1.1-R-small|bb-moirai"
    "amazon/chronos-t5-small|bb-chronos"
    "AutonLab/MOMENT-1-large|bb-moment-large"
)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  CB-N5 — $1 — $(date)\n================================================================"; }

banner "Cross-backbone n=5 extension × 3 bb × 3 ds × 2 seeds (18 runs)"
for bb_entry in "${BACKBONES[@]}"; do
    backbone="${bb_entry%|*}"
    bbsuf="${bb_entry#*|}"
    for ds in "${DATASETS[@]}"; do
        for seed in "${NEW_SEEDS[@]}"; do
            out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_${bbsuf}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
            if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

            echo "[$(date +%H:%M:%S)] [$RUN_IDX] bb=$bbsuf $ds s=$seed"
            set +e
            $PYTHON scripts/run_gap_closing.py \
                --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
                --dataset "$ds" --seed "$seed" --backbone "$backbone" \
                $BASE_FLAGS > /tmp/cbn5_${RUN_IDX}.out 2>&1
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then
                FAILED=$((FAILED+1))
                echo "  FAILED (rc=$RC):"; tail -3 /tmp/cbn5_${RUN_IDX}.out | sed 's/^/    /'
            else
                LAUNCHED=$((LAUNCHED+1))
                grep "MSE=" /tmp/cbn5_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
            fi
            RUN_IDX=$((RUN_IDX+1))
        done
    done
done

echo -e "\n================================================================"
echo "CB-N5 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
