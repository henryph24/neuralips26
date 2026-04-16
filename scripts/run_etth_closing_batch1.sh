#!/bin/bash
# Batch 1: Decompose data-scarcity vs parameter-scarcity on ETTh1/ETTh2.
#
# 1a. Cap removal:       ETTh1/h2 × 5 seeds × max_samples∈{5000, 100000}  = 20 runs
# 1b. Shared raw branch: 6 datasets × 5 seeds                             = 30 runs
# 1c. K ablation:        ETTh1/h2 × 3 seeds × K∈{1, 5}                    = 12 runs
#                                                                          Total: 62
#
# All runs use the best-known base config (no grad_clip so we stay consistent
# with existing result filenames — grad_clip test moved to Batch 2).
#
# Invocation:
#   tmux new-session -d -s b1 'cd ~/neuralips26 && bash scripts/run_etth_closing_batch1.sh 2>&1 | tee results/etth_b1.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

HARD_DATASETS=(ETTh1 ETTh2)
ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS5=(42 43 44 45 46)
SEEDS3=(42 43 44)

# Best-known base config (matches existing wu5 runs)
BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

run_cfg() {
    local out="$1"; shift
    local label="$1"; shift
    if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); return; fi
    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $label"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
        $BASE_FLAGS "$@" > /tmp/b1_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED+1))
        echo "  FAILED (rc=$RC):"; tail -3 /tmp/b1_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED+1))
        grep "MSE=" /tmp/b1_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX+1))
}

banner() { echo -e "\n================================================================\n  BATCH 1 — $1 — $(date)\n================================================================"; }

# Base config suffix (appears on all output filenames): _d1_wd0.0001_cos_wu5
BASE_SUFFIX="_d1_wd0.0001_cos_wu5"

# ===== 1a: Cap removal on hard datasets =====
banner "1a: Cap removal ETTh1/h2 × {5K, 100K} (20 runs)"
for ds in "${HARD_DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        # maxN=5000 is the default — already exists from final-beat sweep
        # maxN=100000 is the new test: does more data close the gap?
        for maxn in 5000 100000; do
            if [ "$maxn" = "5000" ]; then
                out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${BASE_SUFFIX}.json"
                flags="--dataset $ds --seed $seed"
            else
                out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${BASE_SUFFIX}_maxN${maxn}.json"
                flags="--dataset $ds --seed $seed --max-samples $maxn"
            fi
            run_cfg "$out" "1a ds=$ds s=$seed maxN=$maxn" $flags
        done
    done
done

# ===== 1b: Shared raw branch on all 6 datasets =====
banner "1b: Shared raw branch × 6 datasets (30 runs)"
for ds in "${ALL_DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${BASE_SUFFIX}_shared.json"
        flags="--dataset $ds --seed $seed --raw-branch-shared"
        run_cfg "$out" "1b ds=$ds s=$seed SHARED" $flags
    done
done

# ===== 1c: K ablation on hard datasets =====
banner "1c: K∈{1,5} on ETTh1/h2 (12 runs)"
for ds in "${HARD_DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        for K in 1 5; do
            if [ "$K" = "5" ]; then
                # K=5 is default — already exists from final-beat
                out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${BASE_SUFFIX}.json"
                flags="--dataset $ds --seed $seed"
            else
                # K=1 with top_k=1 (can't take top-2 of 1 expert)
                out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${BASE_SUFFIX}_K${K}.json"
                flags="--dataset $ds --seed $seed --K $K --top-k 1"
            fi
            run_cfg "$out" "1c ds=$ds s=$seed K=$K" $flags
        done
    done
done

echo -e "\n================================================================"
echo "BATCH 1 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
