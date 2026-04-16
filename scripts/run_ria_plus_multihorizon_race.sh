#!/bin/bash
# Multi-horizon sweep for Residual-IA+ (the strongest single paper addition).
#
# Standard LTSF horizons H ∈ {96, 192, 336, 720}. H=96 already done in
# final-beat/B3b (30 runs). Add H ∈ {192, 336, 720} × 6 datasets × 5 seeds = 90 runs.
#
# Rationale: non-stationarity is MORE pronounced at longer horizons, so NLinear's
# shift-subtract-unshift trick should help MORE. This makes the paper's 5/6
# result at H=96 look like a lower bound.
#
# Config: Residual-IA+ (shared raw, NLinear, val-ES, grad-clip, gate-init=-2,
# wu=5, cos+wd=1e-4).
#
# Invocation:
#   tmux new-session -d -s mh 'cd ~/neuralips26 && bash scripts/run_ria_plus_multihorizon_race.sh 2>&1 | tee results/ria_mh.log'

set -e
DEVICE="cuda"
EPOCHS=25
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
HORIZONS=(192 336 720)
SEEDS5=(42 43 44 45 46)

BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  MULTIHORIZON — $1 — $(date)\n================================================================"; }

banner "Residual-IA+ × H∈{192,336,720} × 6 datasets × 5 seeds (90 runs)"
for horizon in "${HORIZONS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS5[@]}"; do
            out="${RESULTS_DIR}/residual-ia_${ds}_H${horizon}_${seed}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
            if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi

            echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds H=$horizon s=$seed"
            set +e
            $PYTHON scripts/run_gap_closing.py \
                --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
                --dataset "$ds" --seed "$seed" --horizon "$horizon" \
                $BASE_FLAGS > /tmp/mh_${RUN_IDX}.out 2>&1
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then
                FAILED=$((FAILED+1))
                echo "  FAILED (rc=$RC):"; tail -3 /tmp/mh_${RUN_IDX}.out | sed 's/^/    /'
            else
                LAUNCHED=$((LAUNCHED+1))
                grep -E "MSE=|Restored" /tmp/mh_${RUN_IDX}.out | tail -2 | sed 's/^/    /'
            fi
            RUN_IDX=$((RUN_IDX+1))
        done
    done
done

echo -e "\n================================================================"
echo "MULTIHORIZON DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
