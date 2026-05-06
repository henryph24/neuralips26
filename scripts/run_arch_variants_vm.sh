#!/bin/bash
# Architectural variants sweep — novel algorithm tweaks.
#
# A. Stats router: route on [mean, std, range, slope, autocorr] — tests theory directly
# B. Multi-scale router: parallel Conv1d at k=4/16/64
# C. Expert dropout (p=0.3): regularization that forces routing diversity
# D. Cosine LR: CosineAnnealingLR (implemented via shorter-epoch + restart)
# E. Stats router + expert dropout combined
#
# Grid: 5 variants × 3 datasets × 3 seeds = 45 runs.
# Estimated: ~25 min solo on A10G.
#
# Invocation:
#   tmux new-session -d -s arch 'cd ~/neuralips26 && bash scripts/run_arch_variants_vm.sh 2>&1 | tee results/arch_variants.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

DATASETS=(ETTh1 ETTm1 Weather)
SEEDS=(42 43 44)

echo "================================================================"
echo "ARCHITECTURAL VARIANTS — $(date)"
echo "================================================================"

RUN_IDX=0
LAUNCHED=0
SKIPPED=0
FAILED=0

run_variant() {
  local label="$1"; shift
  local outfile="$1"; shift
  # remaining args are passed to run_rr_moa.py

  if [ -f "$outfile" ]; then
    SKIPPED=$((SKIPPED + 1)); RUN_IDX=$((RUN_IDX + 1)); return
  fi
  echo "[$(date +%H:%M:%S)] [$RUN_IDX] $label"
  set +e
  $PYTHON scripts/run_rr_moa.py "$@" > /tmp/arch_${RUN_IDX}.out 2>&1
  RC=$?; set -e
  if [ $RC -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "  FAILED (rc=$RC):"
    tail -3 /tmp/arch_${RUN_IDX}.out | sed 's/^/    /'
  else
    LAUNCHED=$((LAUNCHED + 1))
    grep "MSE=" /tmp/arch_${RUN_IDX}.out | tail -1 | sed 's/^/    /'
  fi
  RUN_IDX=$((RUN_IDX + 1))
}

# A. Stats router
echo ""
echo "=== A: Stats Router ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_variant "stats | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-stats.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch stats --no-baselines
  done
done

# B. Multi-scale router
echo ""
echo "=== B: Multi-Scale Router ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_variant "multiscale | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-multiscale.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch multiscale --no-baselines
  done
done

# C. Expert dropout p=0.3
echo ""
echo "=== C: Expert Dropout (p=0.3) ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_variant "edrop=0.3 | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_edrop-0.3.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --expert-dropout 0.3 --no-baselines
  done
done

# D. Stats router + expert dropout combined
echo ""
echo "=== D: Stats Router + Expert Dropout ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_variant "stats+edrop | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-stats_edrop-0.3.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch stats --expert-dropout 0.3 --no-baselines
  done
done

# E. Multi-scale + sharper temperature
echo ""
echo "=== E: Multi-Scale + Sharp Temp (τ=0.5) ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_variant "multiscale+τ=0.5 | $ds s=$seed" \
      "results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_rarch-multiscale_temp-0.5.json" \
      --dataset "$ds" --unfreeze frozen --top-k 2 --seed "$seed" \
      --epochs "$EPOCHS" --device "$DEVICE" --router-arch multiscale --router-temp 0.5 --no-baselines
  done
done

echo ""
echo "================================================================"
echo "ARCH VARIANTS DONE at $(date)"
echo "  Launched: $LAUNCHED  Skipped: $SKIPPED  Failed: $FAILED  Total: $RUN_IDX"
echo "================================================================"
