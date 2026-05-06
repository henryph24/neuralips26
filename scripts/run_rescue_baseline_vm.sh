#!/bin/bash
# B1: AdaMix rescue-baseline sweep — demonstrates that no standard MoE rescue
# mechanism recovers routing entropy on TSFM hidden states under RevIN.
#
# Grid: 6 datasets × 2 freeze levels × 5 seeds × 11 rescue configurations = 660 runs
# Expected: all 11 configurations collapse to entropy < 0.3 on ≥80% of cells
#           under last-4 unfreezing (baseline already collapses; rescue fails).
#
# Invocation:
#   bash scripts/run_rescue_baseline_vm.sh                  # single worker
#   bash scripts/run_rescue_baseline_vm.sh worker K N       # worker K of N (K∈[1,N])
#
# Worker sharding: the i-th run (0-indexed) runs only if (i % N) == (K-1).
# Launch 4 workers in parallel via tmux:
#   tmux new-session -d -s b1_w1 'cd ~/neuralips26 && bash scripts/run_rescue_baseline_vm.sh worker 1 4 2>&1 | tee -a results/b1_w1.log'
#   tmux new-session -d -s b1_w2 'cd ~/neuralips26 && bash scripts/run_rescue_baseline_vm.sh worker 2 4 2>&1 | tee -a results/b1_w2.log'
#   tmux new-session -d -s b1_w3 'cd ~/neuralips26 && bash scripts/run_rescue_baseline_vm.sh worker 3 4 2>&1 | tee -a results/b1_w3.log'
#   tmux new-session -d -s b1_w4 'cd ~/neuralips26 && bash scripts/run_rescue_baseline_vm.sh worker 4 4 2>&1 | tee -a results/b1_w4.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3
RESULTS_DIR="results/adamix_rescue"
mkdir -p "$RESULTS_DIR"

# Worker sharding: each run is indexed by global RUN_IDX; we only execute if
# (RUN_IDX % NUM_WORKERS) == (WORKER_ID - 1).
MODE="${1:-single}"
WORKER_ID="${2:-1}"
NUM_WORKERS="${3:-1}"

echo "================================================================"
echo "B1 rescue-baseline sweep — $(date)"
echo "Mode: $MODE  Worker: $WORKER_ID / $NUM_WORKERS"
echo "================================================================"

# 12 rescue configurations from the plan.
# The "baseline_legacy" (softmax, lb=0.01, mean-prob, no aux losses) config is
# intentionally NOT included — it's already covered by existing paper Table 2
# results under results/adamix/ with the exact same default CLI, so re-running
# it would just duplicate (and overwrite) those files.
declare -a RESCUE_CONFIGS=(
  # tag,  router,       lb_coef, lb_variant, entropy_reg, z_loss, relu_l1, capacity
  "baseline_correct,softmax,0.01,argmax,0.0,0.0,0.0,2.0"
  "ent_light,softmax,0.01,argmax,0.01,0.0,0.0,2.0"
  "ent_medium,softmax,0.01,argmax,0.1,0.0,0.0,2.0"
  "ent_strong,softmax,0.01,argmax,1.0,0.0,0.0,2.0"
  "zloss_light,softmax,0.01,argmax,0.0,0.001,0.0,2.0"
  "zloss_medium,softmax,0.01,argmax,0.0,0.01,0.0,2.0"
  "zloss_strong,softmax,0.01,argmax,0.0,0.1,0.0,2.0"
  "lb_strong,softmax,0.1,argmax,0.0,0.0,0.0,2.0"
  "lb_extreme,softmax,1.0,argmax,0.0,0.0,0.0,2.0"
  "lb_supermax,softmax,10.0,argmax,0.0,0.0,0.0,2.0"
  "remoe_relu,relu,0.0,argmax,0.0,0.0,0.01,2.0"
  "expert_choice,expert-choice,0.0,argmax,0.0,0.0,0.0,2.0"
)

# 12 rescue configs × 6 datasets × 2 freeze × 5 seeds = 720 runs.
# Every config has at least one non-default flag, so rescue_active=True in
# run_adamix.py and the filename always carries the full rescue_tag.

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
FREEZES=(last4 last2)
SEEDS=(42 43 44 45 46)

RUN_IDX=0
LAUNCHED=0
SKIPPED_EXISTING=0
SKIPPED_WORKER=0
FAILED=0

for cfg in "${RESCUE_CONFIGS[@]}"; do
  IFS=',' read -r tag router lbc lbv ent zl l1 cf <<< "$cfg"
  for ds in "${DATASETS[@]}"; do
    for freeze in "${FREEZES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        # Worker sharding check
        if [ "$MODE" = "worker" ] && [ $((RUN_IDX % NUM_WORKERS)) -ne $((WORKER_ID - 1)) ]; then
          SKIPPED_WORKER=$((SKIPPED_WORKER + 1))
          RUN_IDX=$((RUN_IDX + 1))
          continue
        fi

        # Expected output file (must match run_adamix.py's path convention)
        OUTFILE="${RESULTS_DIR}/${ds}_H96_K5_${freeze}_${seed}_rtr${router}_lb${lbc}_lv${lbv}_ent${ent}_z${zl}_l1${l1}_cf${cf}.json"
        # Handle printf of 0.0 vs 0 in filename (bash and Python %g differ)
        OUTFILE_ALT="${RESULTS_DIR}/${ds}_H96_K5_${freeze}_${seed}_rtr${router}_lb$(printf %g $lbc)_lv${lbv}_ent$(printf %g $ent)_z$(printf %g $zl)_l1$(printf %g $l1)_cf$(printf %g $cf).json"

        if [ -f "$OUTFILE" ] || [ -f "$OUTFILE_ALT" ]; then
          SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
          RUN_IDX=$((RUN_IDX + 1))
          continue
        fi

        echo "[$(date +%H:%M:%S)] [$RUN_IDX] $tag | $ds $freeze seed=$seed"
        set +e
        $PYTHON scripts/run_adamix.py \
          --dataset "$ds" \
          --seed "$seed" \
          --unfreeze "$freeze" \
          --epochs "$EPOCHS" \
          --device "$DEVICE" \
          --router-type "$router" \
          --load-balance-coef "$lbc" \
          --load-balance-variant "$lbv" \
          --entropy-reg-coef "$ent" \
          --z-loss-coef "$zl" \
          --relu-l1-coef "$l1" \
          --capacity-factor "$cf" \
          --run-baselines no \
          --results-dir "$RESULTS_DIR" \
          > /tmp/rescue_${WORKER_ID}.out 2>&1
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "  FAILED (rc=$RC, last 5 lines):"
          tail -5 /tmp/rescue_${WORKER_ID}.out | sed 's/^/    /'
        else
          LAUNCHED=$((LAUNCHED + 1))
        fi
        RUN_IDX=$((RUN_IDX + 1))
      done
    done
  done
done

echo ""
echo "================================================================"
echo "B1 worker $WORKER_ID/$NUM_WORKERS DONE at $(date)"
echo "  Launched:          $LAUNCHED"
echo "  Skipped (existed): $SKIPPED_EXISTING"
echo "  Skipped (worker):  $SKIPPED_WORKER"
echo "  Failed:            $FAILED"
echo "  Total scanned:     $RUN_IDX"
echo "================================================================"
