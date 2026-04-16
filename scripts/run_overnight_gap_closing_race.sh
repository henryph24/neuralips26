#!/bin/bash
# Overnight DLinear-gap-closing sweep — 8 hours of GPU time.
#
# Strategy: systematically explore every lever that could close the gap.
# All experiments use residual-ia variant (raw branch primary, backbone residual).
#
# Batches (in priority order):
#   A: raw_hidden sweep extension {320, 384, 512}         54 runs
#   B: raw_depth {1=linear, 3=deep} at best rh=256        36 runs
#   C: LR sweep {5e-4, 2e-3, 5e-3} at rh=256             54 runs
#   D: cosine schedule + weight decay at rh=256            36 runs
#   E: combo: best LR + cosine + wd at rh=256             18 runs
#   F: extended epochs {25, 30} + cosine + wd at rh=256   36 runs
#   G: 5-seed grid of best config (filled after review)   30 runs
#   H: multi-horizon {192, 336} of best config            36 runs
#   I: cross-backbone Moirai at best config                18 runs
#                                                   Total ~318 runs
#
# At ~90s/run solo (2 concurrent = ~120s effective): ~5.3-6.4 GPU-hours.
# Adding the already-running P1 sweep (~38 remaining): ~7-8 hours total.
#
# Invocation:
#   tmux new-session -d -s overnight 'cd ~/neuralips26 && bash scripts/run_overnight_gap_closing_race.sh 2>&1 | tee results/overnight.log'

set -e
DEVICE="cuda"
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
    local ds="$1" seed="$2" rh="$3" depth="${4:-2}" lr="${5:-0.001}" wd="${6:-0}" epochs="${7:-15}" cosine="${8:-0}"

    # Build output filename matching Python's convention
    local hp_parts=""
    if [ "$depth" != "2" ]; then hp_parts="${hp_parts}_d${depth}"; fi
    if [ "$lr" != "0.001" ]; then hp_parts="${hp_parts}_lr${lr}"; fi
    if [ "$wd" != "0" ]; then hp_parts="${hp_parts}_wd${wd}"; fi
    if [ "$cosine" = "1" ]; then hp_parts="${hp_parts}_cos"; fi

    local OUT="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_rh${rh}${hp_parts}.json"
    if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1))
        RUN_IDX=$((RUN_IDX + 1))
        return
    fi

    local cosine_flag=""
    if [ "$cosine" = "1" ]; then cosine_flag="--cosine"; fi

    echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed rh=$rh d=$depth lr=$lr wd=$wd ep=$epochs cos=$cosine"
    set +e
    $PYTHON scripts/run_gap_closing.py \
        --variant residual-ia \
        --dataset "$ds" \
        --seed "$seed" \
        --epochs "$epochs" \
        --device "$DEVICE" \
        --top-k 2 \
        --raw-hidden "$rh" \
        --raw-depth "$depth" \
        --lr "$lr" \
        --weight-decay "$wd" \
        $cosine_flag \
        > /tmp/overnight_${RUN_IDX}.out 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=$RC):"
        tail -3 /tmp/overnight_${RUN_IDX}.out | sed 's/^/    /'
    else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/overnight_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
    fi
    RUN_IDX=$((RUN_IDX + 1))
}

banner() {
    echo ""
    echo "================================================================"
    echo "  BATCH $1: $2"
    echo "  $(date)"
    echo "================================================================"
}

# ===== BATCH A: Extended raw_hidden {320, 384, 512} =====
banner "A" "raw_hidden extension (54 runs)"
for rh in 320 384 512; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" "$rh"
        done
    done
done

# ===== BATCH B: raw_depth sweep {1=linear, 3=deep} at rh=256 =====
banner "B" "raw_depth sweep (36 runs)"
for depth in 1 3; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 "$depth"
        done
    done
done

# ===== BATCH C: LR sweep at rh=256 =====
banner "C" "LR sweep (54 runs)"
for lr in 0.0005 0.002 0.005; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 2 "$lr"
        done
    done
done

# ===== BATCH D: cosine + weight decay at rh=256 =====
banner "D" "cosine + weight decay (36 runs)"
for wd in 0.0001 0.001; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 2 0.001 "$wd" 15 1
        done
    done
done

# ===== BATCH E: combo best LR candidates + cosine + wd =====
banner "E" "combo: lr=2e-3 + cosine + wd (18 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        run_ria "$ds" "$seed" 256 2 0.002 0.0001 15 1
    done
done

# ===== BATCH F: extended epochs + cosine + wd =====
banner "F" "extended epochs 25+30 (36 runs)"
for epochs in 25 30; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            run_ria "$ds" "$seed" 256 2 0.001 0.0001 "$epochs" 1
        done
    done
done

# ===== BATCH G: 5-seed of rh=256 default (expand from 3→5 seeds) =====
banner "G" "5-seed rh=256 (up to 30 runs, skips existing 3-seed)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS5[@]}"; do
        run_ria "$ds" "$seed" 256
    done
done

# ===== BATCH H: multi-horizon =====
banner "H" "multi-horizon H=192,336 at rh=256 (36 runs)"
for horizon in 192 336; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            local_out="${RESULTS_DIR}/residual-ia_${ds}_H${horizon}_${seed}_rh256.json"
            if [ -f "$local_out" ]; then
                SKIPPED=$((SKIPPED + 1))
                RUN_IDX=$((RUN_IDX + 1))
                continue
            fi
            echo "[$(date +%H:%M:%S)] [$RUN_IDX] $ds s=$seed H=$horizon rh=256"
            set +e
            $PYTHON scripts/run_gap_closing.py \
                --variant residual-ia \
                --dataset "$ds" --seed "$seed" --horizon "$horizon" \
                --epochs 15 --device "$DEVICE" --top-k 2 --raw-hidden 256 \
                > /tmp/overnight_${RUN_IDX}.out 2>&1
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then
                FAILED=$((FAILED + 1))
            else
                LAUNCHED=$((LAUNCHED + 1))
                grep "MSE=" /tmp/overnight_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
            fi
            RUN_IDX=$((RUN_IDX + 1))
        done
    done
done

# ===== BATCH I: cross-backbone Moirai =====
banner "I" "Moirai cross-backbone at rh=256 (18 runs)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        local_out="${RESULTS_DIR}/residual-ia_${ds}_H96_${seed}_bb-moirai_rh256.json"
        if [ -f "$local_out" ]; then
            SKIPPED=$((SKIPPED + 1))
            RUN_IDX=$((RUN_IDX + 1))
            continue
        fi
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] Moirai $ds s=$seed rh=256"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --variant residual-ia \
            --dataset "$ds" --seed "$seed" --epochs 15 --device "$DEVICE" \
            --top-k 2 --raw-hidden 256 \
            --backbone Salesforce/moirai-1.1-R-small \
            > /tmp/overnight_${RUN_IDX}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED + 1))
        else
            LAUNCHED=$((LAUNCHED + 1))
            grep "MSE=" /tmp/overnight_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX + 1))
    done
done

echo ""
echo "================================================================"
echo "OVERNIGHT SWEEP DONE at $(date)"
echo "  Launched: $LAUNCHED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"
echo "  Total:    $RUN_IDX"
echo "================================================================"
