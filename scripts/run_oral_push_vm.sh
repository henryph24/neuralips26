#!/bin/bash
# Oral-push experiments — 4 high-impact architectural tweaks.
#
# 1. SSR: Sufficient Statistic Router [μ,σ] only — Prop 2 empirical proof
# 2. RDGF: Router-Detached Gradient Flow — cure the Frozen Paradox
# 3. RDGF + SSR combined — the theoretically optimal configuration
# 4. SSR on extended datasets — if SSR matches conv, prove it broadly
#
# Grid: ~72 runs total, ~1.5h on A10G.
#
# Invocation:
#   tmux new-session -d -s oral 'cd ~/neuralips26 && bash scripts/run_oral_push_vm.sh 2>&1 | tee results/oral_push.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)

echo "================================================================"
echo "ORAL PUSH — $(date)"
echo "================================================================"

RUN_IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

run_one() {
  local label="$1"; shift
  local outfile="$1"; shift
  if [ -f "$outfile" ]; then
    SKIPPED=$((SKIPPED + 1)); RUN_IDX=$((RUN_IDX + 1)); return
  fi
  echo "[$(date +%H:%M:%S)] [$RUN_IDX] $label"
  set +e
  $PYTHON scripts/run_rr_moa.py "$@" > /tmp/oral_${RUN_IDX}.out 2>&1
  RC=$?; set -e
  if [ $RC -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "  FAILED (rc=$RC):"; tail -3 /tmp/oral_${RUN_IDX}.out | sed 's/^/    /'
  else
    LAUNCHED=$((LAUNCHED + 1))
    grep "MSE=" /tmp/oral_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
  fi
  RUN_IDX=$((RUN_IDX + 1))
}

###############################################
# 1. SSR [μ,σ] — the mic-drop ablation
###############################################
echo ""
echo "=== 1. Sufficient Statistic Router (SSR) [μ,σ] ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "SSR | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-ssr.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch ssr --no-baselines
  done
done

###############################################
# 2. RDGF — cure the Frozen Paradox
#    Run with last2 and last4 unfreezing
###############################################
echo ""
echo "=== 2. RDGF (Router-Detached Gradient Flow) ==="
for freeze in last2 last4; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "RDGF $freeze | $ds s=$seed" \
        "results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}_rdgf.json" \
        --dataset "$ds" --unfreeze "$freeze" --top-k 2 --seed "$seed" \
        --epochs "$EPOCHS" --device "$DEVICE" --rdgf --no-baselines
    done
  done
done

###############################################
# 3. RDGF + SSR combined
###############################################
echo ""
echo "=== 3. RDGF + SSR Combined ==="
for freeze in last2 last4; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "RDGF+SSR $freeze | $ds s=$seed" \
        "results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}_rarch-ssr_rdgf.json" \
        --dataset "$ds" --unfreeze "$freeze" --top-k 2 --seed "$seed" \
        --epochs "$EPOCHS" --device "$DEVICE" --router-arch ssr --rdgf --no-baselines
    done
  done
done

###############################################
# 4. SSR on extended datasets
###############################################
echo ""
echo "=== 4. SSR Extended (6 datasets) ==="
for ds in ETTh2 ETTm2 Electricity; do
  for seed in "${SEEDS[@]}"; do
    run_one "SSR ext | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-ssr.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch ssr --no-baselines
  done
done

echo ""
echo "================================================================"
echo "ORAL PUSH DONE at $(date)"
echo "  Launched: $LAUNCHED  Skipped: $SKIPPED  Failed: $FAILED  Total: $RUN_IDX"
echo "================================================================"
