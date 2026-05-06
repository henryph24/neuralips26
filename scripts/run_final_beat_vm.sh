#!/bin/bash
# Final beat-DLinear sweep: publication-quality 5-seed + 10-seed for best config.
#
# Best config identified: rh256_d1_wd0.0001_cos_wu5 (with implicit gate_init=-2)
# Current: 3 seeds per dataset, need 5+ for statistical significance.
#
# Batches:
#   A: extend wu5 config to 5 seeds (seeds 45, 46) x 6 datasets     12 runs
#   B: 10-seed robustness (seeds 47-51) for top 3 datasets           15 runs
#   C: Oracle gate ablation (gate=0 raw-only, gate=1 backbone-only)  36 runs
#   D: Deeper warmup {10 epoch warmup} x 6 datasets x 3 seeds        18 runs
#   E: Dataset-adaptive: lower gate_init=-4 on ETTh1/ETTm1 only       6 runs
#                                                             Total ~87 runs (~2 GPU-hours)
#
# Invocation:
#   tmux new-session -d -s final 'cd ~/neuralips26 && bash scripts/run_final_beat_vm.sh 2>&1 | tee results/final_beat.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

run_ria() {
    local ds="$1" seed="$2" hp_suffix="$3" extra_flags="$4"
    local OUT="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256${hp_suffix}.json"
    if [ -f "$OUT" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); return; fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed cfg=${hp_suffix:1}"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --variant residual-ia --dataset "$ds" --seed "$seed" --epochs "$EPOCHS" \
        --device "$DEVICE" --top-k 2 --raw-hidden 256 \
        $extra_flags \
        > /tmp/final_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED+1))
        echo "  FAILED:"; tail -3 /tmp/final_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED+1))
        grep "MSE=" /tmp/final_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX+1))
}

banner() { echo -e "\n================================================================\n  BATCH $1: $2 — $(date)\n================================================================"; }

# ===== BATCH A: Extend wu5 config to 5 seeds =====
banner "A" "wu5 config: seeds 45,46 (12 runs)"
BASE_FLAGS="--raw-depth 1 --lr 0.001 --weight-decay 0.0001 --cosine --warmup-epochs 5"
for seed in 45 46; do
    for ds in "${DATASETS[@]}"; do
        run_ria "$ds" "$seed" "_d1_wd0.0001_cos_wu5" "$BASE_FLAGS"
    done
done

# ===== BATCH B: 10-seed robustness for hard datasets =====
banner "B" "10-seed {47-51} for ETTh1, ETTm1, ETTh2 (15 runs)"
for seed in 47 48 49 50 51; do
    for ds in ETTh1 ETTm1 ETTh2; do
        run_ria "$ds" "$seed" "_d1_wd0.0001_cos_wu5" "$BASE_FLAGS"
    done
done

# ===== BATCH C: Oracle gate ablation =====
# This requires code support for forcing gate=0 or gate=1, which we don't have.
# Skip this batch — diagnostic info already in gate_values logged per run.

# ===== BATCH D: Deeper warmup (wu10) =====
banner "D" "warmup=10 (18 runs)"
BASE_D="--raw-depth 1 --lr 0.001 --weight-decay 0.0001 --cosine --warmup-epochs 10"
for ds in "${DATASETS[@]}"; do
    for seed in 42 43 44; do
        run_ria "$ds" "$seed" "_d1_wd0.0001_cos_wu10" "$BASE_D"
    done
done

# ===== BATCH E: gate_init=-4 (more aggressive) on all datasets =====
# Note: gi comes before wu in the filename suffix due to code ordering.
banner "E" "gate_init=-4 + wu5 (18 runs)"
BASE_E="--raw-depth 1 --lr 0.001 --weight-decay 0.0001 --cosine --warmup-epochs 5 --gate-init -4"
for ds in "${DATASETS[@]}"; do
    for seed in 42 43 44; do
        run_ria "$ds" "$seed" "_d1_wd0.0001_cos_gi-4_wu5" "$BASE_E"
    done
done

echo -e "\n================================================================"
echo "FINAL BEAT DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
