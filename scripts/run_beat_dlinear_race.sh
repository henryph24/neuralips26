#!/bin/bash
# Beat-DLinear sweep: targeted experiments to close the remaining gap.
#
# Diagnosis: gate is stuck at ~0.47-0.50, backbone adds noise on easy datasets.
# Three surgical fixes:
#   1. gate_init=-2 (sigmoid=0.12): start raw-dominant, backbone earns contribution
#   2. warmup: train raw branch alone first, then unfreeze backbone+gate
#   3. combo: d1 (linear raw) + gate_init=-2 + cosine+wd + warmup
#
# Plus: adapter_hidden=128 (bigger backbone expert) and unfreezing last2.
#
# Batches:
#   A: gate_init effects {-2, -3, -5} x d1 + cos+wd           54 runs
#   B: warmup {3, 5, 7} epochs x d1 + gate_init=-2 + cos+wd   54 runs
#   C: adapter_hidden {128, 256} x d1 + gate_init=-2 + cos+wd  36 runs
#   D: best combo x unfreeze last2                              18 runs
#   E: best combo x 5 seeds (publication-quality)               30 runs
#   F: best combo x multi-horizon {192, 336}                    36 runs
#                                                        Total ~228 runs (~4-5 GPU-hours)
#
# Invocation:
#   tmux new-session -d -s beat 'cd ~/neuralips26 && bash scripts/run_beat_dlinear_race.sh 2>&1 | tee results/beat_dlinear.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/gap_closing"
mkdir -p "$RESULTS_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
SEEDS3=(42 43 44)
SEEDS5=(42 43 44 45 46)

LAUNCHED=0
SKIPPED=0
FAILED=0
RUN_IDX=0

run_ria() {
    local ds="$1" seed="$2" rh="$3" depth="$4" lr="$5" wd="$6" epochs="$7" \
          cosine="$8" gate_init="$9" warmup="${10}" ahidden="${11}" unfreeze="${12:-frozen}"

    # Build filename suffix
    local hp=""
    [ "$depth" != "2" ] && hp="${hp}_d${depth}"
    [ "$lr" != "0.001" ] && hp="${hp}_lr${lr}"
    [ "$wd" != "0" ] && hp="${hp}_wd${wd}"
    [ "$cosine" = "1" ] && hp="${hp}_cos"
    [ "$gate_init" != "-2" ] && hp="${hp}_gi${gate_init}"
    [ "$warmup" != "0" ] && hp="${hp}_wu${warmup}"
    [ "$ahidden" != "0" ] && hp="${hp}_ah${ahidden}"
    local uf=""
    [ "$unfreeze" != "frozen" ] && uf="_${unfreeze}"

    local OUT="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}${uf}_rh${rh}${hp}.json"
    if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        return
    fi

    local flags=""
    [ "$cosine" = "1" ] && flags="$flags --cosine"
    [ "$ahidden" != "0" ] && flags="$flags --adapter-hidden $ahidden"

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed d=$depth gi=$gate_init wu=$warmup ah=$ahidden uf=$unfreeze"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --variant residual-ia \
        --dataset "$ds" --seed "$seed" --epochs "$epochs" --device "$DEVICE" \
        --top-k 2 --raw-hidden "$rh" --raw-depth "$depth" \
        --lr "$lr" --weight-decay "$wd" --gate-init "$gate_init" \
        --warmup-epochs "$warmup" --unfreeze "$unfreeze" \
        $flags \
        > /tmp/beat_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/beat_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/beat_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
}

banner() { echo -e "\n================================================================\n  BATCH $1: $2 — $(date)\n================================================================"; }

# ===== BATCH A: gate_init sweep with linear raw branch + cos+wd =====
banner "A" "gate_init {-2,-3,-5} x d1 + cos+wd (54 runs)"
for gi in -2 -3 -5; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            #          ds   seed rh  depth lr    wd     ep cosine gi wu ahidden
            run_ria "$ds" "$seed" 256 1 0.001 0.0001 15 1 "$gi" 0 0
        done
    done
done

# ===== BATCH B: warmup sweep with d1 + gate_init=-2 + cos+wd =====
banner "B" "warmup {3,5,7} + d1 + gi=-2 + cos+wd (54 runs)"
for wu in 3 5 7; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 1 0.001 0.0001 15 1 -2 "$wu" 0
        done
    done
done

# ===== BATCH C: adapter_hidden {128,256} + d1 + gi=-2 + cos+wd =====
banner "C" "adapter_hidden {128,256} (36 runs)"
for ah in 128 256; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 1 0.001 0.0001 15 1 -2 0 "$ah"
        done
    done
done

# ===== BATCH D: unfreeze last2 with d1 + gi=-2 + cos+wd =====
banner "D" "unfreeze last2 + d1 + gi=-2 + cos+wd (18 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        run_ria "$ds" "$seed" 256 1 0.001 0.0001 15 1 -2 0 0 last2
    done
done

# ===== BATCH E: 5-seed grid of gi=-2 + d1 + cos+wd (best base config) =====
banner "E" "5-seed gi=-2 + d1 + cos+wd (30 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        run_ria "$ds" "$seed" 256 1 0.001 0.0001 15 1 -2 0 0
    done
done

# ===== BATCH F: multi-horizon =====
banner "F" "multi-horizon H={192,336} (36 runs)"
for horizon in 192 336; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            local_out="${RESULTS_DIR}/residual-ia_${ds}_H${horizon}_${seed}_rh256_d1_wd0.0001_cos.json"
            if [ -f "$local_out" ]; then
                SKIPPED=$((SKIPPED + 1))
                RUN_IDX=$((RUN_IDX + 1))
                continue
            fi
            echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed H=$horizon d1+gi=-2+cos+wd"
            set +e
            $PYTHON scripts/run_gap_closing.py \
                --variant residual-ia \
                --dataset "$ds" --seed "$seed" --horizon "$horizon" \
                --epochs 15 --device "$DEVICE" --top-k 2 --raw-hidden 256 \
                --raw-depth 1 --lr 0.001 --weight-decay 0.0001 --cosine \
                --gate-init -2 \
                > /tmp/beat_${RUN_IDX}.out 2>&1
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); else LAUNCHED=$((LAUNCHED+1)); grep "MSE=" /tmp/beat_${RUN_IDX}.out | tail -1 | sed 's/^/    /'; fi
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
done

echo -e "\n================================================================"
echo "BEAT-DLINEAR DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
