#!/bin/bash
# Strengthen Round 2: DLinear n=10 + Exchange/Solar expansion + Moirai-MoE
#
# Part A: DLinear n=10 extension (seeds 45-51 × 6 datasets × 4 horizons)
#         168 runs at ~1s each = ~3 min. Transforms ALL p-values in the paper.
#         Currently DLinear has n=3 everywhere; n=10 gives fairer Welch tests.
#
# Part B: Exchange + Solar with Residual-IA+ (2 datasets × 4 horizons × 3 seeds)
#         24 runs at ~60s = 24 min. Expands R(D) correlation from N=9 to N=11.
#
# Part C: DLinear on Exchange + Solar (2 datasets × 4 horizons × 3 seeds)
#         24 runs at ~1s = ~30s. Needed for RIA+ vs DLinear comparison.
#
# Part D: Moirai-MoE + Residual-IA+ (6 datasets × 3 seeds)
#         18 runs at ~90s = ~27 min. Tests recipe on native sparse MoE backbone.
#
# Part E: DLinear n=10 at H=96 only (seed 45-51 × 6 datasets = 42 runs)
#         Already have n=3 (seeds 42-44). Need n=10 for fair comparison.
#         42 runs at ~1s = ~42s.
#
# Total: ~276 runs, ~1 GPU-hour (DLinear is <1s/run, RIA ~60-90s/run)
#
# Invocation:
#   tmux new-session -d -s r2 'cd ~/neuralips26 && bash scripts/run_strengthen_round2_race.sh 2>&1 | tee results/strengthen_r2.log'

set -e
DEVICE="cuda"
PYTHON=python3

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
HORIZONS=(96 192 336 720)
NEW_SEEDS=(45 46 47 48 49 50 51)
SEEDS3=(42 43 44)

LAUNCHED=0; SKIPPED=0; FAILED=0; RUN_IDX=0

banner() { echo -e "\n================================================================\n  $1 — $(date)\n================================================================"; }

# ===== PART A: DLinear n=10 extension (seeds 45-51 × H∈{192,336,720}) =====
banner "A: DLinear seeds 45-51 × 6 datasets × H∈{192,336,720} (126 runs, ~2 min)"
mkdir -p results/dlinear
for horizon in 192 336 720; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${NEW_SEEDS[@]}"; do
            out="results/dlinear/${ds}_H${horizon}_${seed}.json"
            if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi
            set +e
            $PYTHON scripts/run_dlinear_baseline.py --dataset "$ds" --horizon "$horizon" --seed "$seed" --device "$DEVICE" > /tmp/r2_${RUN_IDX}.out 2>&1
            RC=$?; set -e
            if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); else LAUNCHED=$((LAUNCHED+1)); fi
            RUN_IDX=$((RUN_IDX+1))
        done
    done
done
echo "  Part A done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ===== PART E: DLinear n=10 at H=96 (seeds 45-51) =====
banner "E: DLinear seeds 45-51 × 6 datasets × H=96 (42 runs, ~42s)"
for ds in "${DATASETS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        out="results/dlinear/${ds}_H96_${seed}.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi
        set +e
        $PYTHON scripts/run_dlinear_baseline.py --dataset "$ds" --horizon 96 --seed "$seed" --device "$DEVICE" > /tmp/r2_${RUN_IDX}.out 2>&1
        RC=$?; set -e
        if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); else LAUNCHED=$((LAUNCHED+1)); fi
        RUN_IDX=$((RUN_IDX+1))
    done
done
echo "  Part E done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ===== PART C: DLinear on Exchange + Solar =====
banner "C: DLinear × Exchange/Solar × 4 horizons × 3 seeds (24 runs, ~30s)"
for ds in Exchange Solar; do
    for horizon in "${HORIZONS[@]}"; do
        for seed in "${SEEDS3[@]}"; do
            out="results/dlinear/${ds}_H${horizon}_${seed}.json"
            if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi
            set +e
            $PYTHON scripts/run_dlinear_baseline.py --dataset "$ds" --horizon "$horizon" --seed "$seed" --device "$DEVICE" > /tmp/r2_${RUN_IDX}.out 2>&1
            RC=$?; set -e
            if [ $RC -ne 0 ]; then FAILED=$((FAILED+1)); else LAUNCHED=$((LAUNCHED+1)); fi
            RUN_IDX=$((RUN_IDX+1))
        done
    done
done
echo "  Part C done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ===== PART B: Residual-IA+ on Exchange + Solar =====
banner "B: Residual-IA+ × Exchange/Solar × {96,192} × 3 seeds (12 runs, ~12 min)"
BASE_FLAGS="--variant residual-ia --raw-hidden 256 --raw-depth 1 --cosine --weight-decay 1e-4 --gate-init -2 --warmup-epochs 5 --val-early-stop --val-patience 5 --grad-clip 1.0 --raw-arch nlinear --raw-branch-shared"
mkdir -p results/gap_closing
for ds in Exchange Solar; do
    for horizon in 96 192; do
        for seed in "${SEEDS3[@]}"; do
            out="results/gap_closing/residual-ia_${ds}_H${horizon}_${seed}_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
            if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi
            echo "[$(date +%H:%M:%S)] [$RUN_IDX] RIA+ $ds H=$horizon s=$seed"
            set +e
            $PYTHON scripts/run_gap_closing.py \
                --device "$DEVICE" --epochs 25 --top-k 2 \
                --dataset "$ds" --seed "$seed" --horizon "$horizon" \
                $BASE_FLAGS > /tmp/r2_${RUN_IDX}.out 2>&1
            RC=$?; set -e
            if [ $RC -ne 0 ]; then
                FAILED=$((FAILED+1))
                echo "  FAILED:"; tail -3 /tmp/r2_${RUN_IDX}.out | sed 's/^/    /'
            else
                LAUNCHED=$((LAUNCHED+1))
                grep "MSE=" /tmp/r2_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
            fi
            RUN_IDX=$((RUN_IDX+1))
        done
    done
done
echo "  Part B done: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

# ===== PART D: Moirai-MoE + Residual-IA+ =====
banner "D: Moirai-MoE + Residual-IA+ × 6 datasets × 3 seeds (18 runs, ~27 min)"
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS3[@]}"; do
        out="results/gap_closing/residual-ia_${ds}_H96_${seed}_bb-moirai-moe_rh256_d1_wd0.0001_cos_wu5_shared_es5_nlinear_gc1.json"
        if [ -f "$out" ]; then SKIPPED=$((SKIPPED+1)); RUN_IDX=$((RUN_IDX+1)); continue; fi
        echo "[$(date +%H:%M:%S)] [$RUN_IDX] MoiraiMoE $ds s=$seed"
        set +e
        $PYTHON scripts/run_gap_closing.py \
            --device "$DEVICE" --epochs 20 --top-k 2 \
            --dataset "$ds" --seed "$seed" \
            --backbone "Salesforce/moirai-moe-1.0-R-small" \
            $BASE_FLAGS > /tmp/r2_${RUN_IDX}.out 2>&1
        RC=$?; set -e
        if [ $RC -ne 0 ]; then
            FAILED=$((FAILED+1))
            echo "  FAILED:"; tail -3 /tmp/r2_${RUN_IDX}.out | sed 's/^/    /'
        else
            LAUNCHED=$((LAUNCHED+1))
            grep "MSE=" /tmp/r2_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
        fi
        RUN_IDX=$((RUN_IDX+1))
    done
done

echo -e "\n================================================================"
echo "ROUND 2 DONE at $(date)"
echo "  Launched: $LAUNCHED | Skipped: $SKIPPED | Failed: $FAILED | Total: $RUN_IDX"
echo "================================================================"
