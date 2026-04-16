#!/bin/bash
# Batch 3: Stacked winner candidates for 6/6 match-or-beat DLinear.
#
# Based on B2 + B4 early signals:
#   - NLinear raw branch beats linear on ETTh1 (+3.3% vs +5.1%) and ETTh2 (+6.9% vs +10.6%)
#   - Val early-stop nails Electricity parity (p=0.99)
#   - Weather hits p=0.055 with ES alone
#
# Two stacked variants:
#   3a. NLinear + ES + grad_clip:              6 × 5 = 30 runs
#   3b. NLinear + shared raw + ES + grad_clip: 6 × 5 = 30 runs
#                                              Total: 60 runs (~2 GPU-hours)
#
# Invocation:
#   tmux new-session -d -s b3 'cd ~/neuralips26 && bash scripts/run_etth_closing_batch3.sh 2>&1 | tee results/etth_b3.log'

set -e
DEVICE="cuda"
EPOCHS=25
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS5=(42 43 44 45 46)

BASE_FLAGS_3A="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear"
BASE_FLAGS_3B="$BASE_FLAGS_3A --raw-branch-shared"

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

run_cfg() {
    local out="$1"; shift
    local label="$1"; shift
    if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); return; fi
    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $label"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --device "$DEVICE" --epochs "$EPOCHS" --top-k 2 \
        "$@" > /tmp/b3_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED+1))
        echo "  FAILED (rc=$RC):"; tail -3 /tmp/b3_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED+1))
        grep -E "MSE=|Early stop|best_val|Restored" /tmp/b3_${RUN_IDX}.out | tail -3 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX+1))
}

banner() { echo -e "\n================================================================\n  BATCH 3 — $1 — $(date)\n================================================================"; }

# ===== 3a: NLinear + ES + GC =====
banner "3a: NLinear + ES + GC × 6 datasets × 5 seeds (30 runs)"
for ds in "${ALL_DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        # Suffix ordering from hp_parts: d1, wd, cos, wu, (shared), es, nlinear, gc
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256_d1_wd0.0001_cos_wu5_es5_nlinear_gc1.json"
        flags="--dataset $ds --seed $seed $BASE_FLAGS_3A"
        run_cfg "$out" "3a ds=$ds s=$seed NLin+ES+GC" $flags
    done
done

# ===== 3b: NLinear + Shared + ES + GC =====
banner "3b: NLinear + Shared raw + ES + GC × 6 datasets × 5 seeds (30 runs)"
for ds in "${ALL_DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
        flags="--dataset $ds --seed $seed $BASE_FLAGS_3B"
        run_cfg "$out" "3b ds=$ds s=$seed SHR+NLin+ES+GC" $flags
    done
done

echo -e "\n================================================================"
echo "BATCH 3 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
