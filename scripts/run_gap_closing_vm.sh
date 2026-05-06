#!/bin/bash
# Gap-Closing RR-MoA Variants — 4 directions × 3 datasets × 3 seeds = 36 runs
# Estimated: ~2-3 GPU-hours on A10G
# Run: bash scripts/run_gap_closing_vm.sh 2>&1 | tee results/gap_closing_run.log

set -e
PYTHON=python3
DEVICE="cuda"
EPOCHS=15

echo "================================================================"
echo "GAP-CLOSING RR-MoA VARIANTS — $(date)"
echo "================================================================"

mkdir -p results/gap_closing

for variant in dual-stream film raw-expert multi-res; do
  echo ""
  echo "============================================================"
  echo "=== Variant: $variant ==="
  echo "============================================================"
  for ds in ETTh1 ETTm1 Weather; do
    for seed in 42 43 44; do
      OUTFILE="results/gap_closing/${variant}_${ds}_H96_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): $variant $ds seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] $variant: $ds seed=$seed"
      $PYTHON scripts/run_gap_closing.py \
        --variant $variant --dataset $ds \
        --seed $seed --epochs $EPOCHS --device $DEVICE \
        || echo "FAILED: $variant $ds $seed"
    done
  done
done

echo ""
echo "================================================================"
echo "GAP-CLOSING COMPLETE — $(date)"
echo "================================================================"

# Summary table
echo ""
echo "=== RESULTS SUMMARY ==="
$PYTHON -c "
import json, glob, numpy as np

# Load DLinear baselines
dlinear = {}
for ds in ['ETTh1', 'ETTm1', 'Weather']:
    mses = []
    for s in [42, 43, 44]:
        try:
            d = json.load(open(f'results/dlinear/{ds}_H96_{s}.json'))
            mses.append(d['dlinear_mse'])
        except: pass
    if mses:
        dlinear[ds] = np.mean(mses)

# Load RR-MoA baselines
rrmoa = {}
for ds in ['ETTh1', 'ETTm1', 'Weather']:
    mses = []
    for s in [42, 43, 44]:
        try:
            d = json.load(open(f'results/rr_moa/{ds}_H96_K5_top2_frozen_{s}.json'))
            mses.append(d['rr_moa']['mse'])
        except:
            try:
                files = glob.glob(f'evidence_vm/rr_moa/{ds}_H96_K5_top2_frozen_{s}.json')
                if files:
                    d = json.load(open(files[0]))
                    mses.append(d['rr_moa']['mse'])
            except: pass
    if mses:
        rrmoa[ds] = np.mean(mses)

print(f\"{'Variant':15} {'Dataset':10} {'MSE':8} {'vs DLinear':12} {'vs RR-MoA':12} {'Entropy':8}\")
print('-' * 70)

for variant in ['dual-stream', 'film', 'raw-expert', 'multi-res']:
    for ds in ['ETTh1', 'ETTm1', 'Weather']:
        mses = []
        ents = []
        for s in [42, 43, 44]:
            try:
                d = json.load(open(f'results/gap_closing/{variant}_{ds}_H96_{s}.json'))
                mses.append(d['mse'])
                ents.append(d['routing_entropy'])
            except: pass
        if mses:
            m = np.mean(mses)
            e = np.mean(ents)
            dl_gap = f\"+{(m/dlinear[ds]-1)*100:.1f}%\" if ds in dlinear else 'N/A'
            rr_gap = f\"{(m/rrmoa[ds]-1)*100:+.1f}%\" if ds in rrmoa else 'N/A'
            print(f\"{variant:15} {ds:10} {m:.4f}   {dl_gap:12} {rr_gap:12} {e:.3f}\")
" 2>/dev/null || echo "(summary script failed — check individual JSONs)"
