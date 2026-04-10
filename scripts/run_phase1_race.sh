#!/bin/bash
# Phase 1: Close open critiques — D1 + D2 + D3
# D1: Moirai extended grid (last2/last4 freeze levels, all 6 datasets × 3 seeds)
# D2: LoRA + unfreezing (last2 on core 3 datasets, last4 on remaining datasets)
# D3: Dual-stream on remaining datasets + Moirai backbone
#
# Estimated: ~6-8 GPU-hours on A10G
# Run: nohup bash scripts/run_phase1_race.sh 2>&1 | tee results/phase1_run.log &

set -e
PYTHON=python3
DEVICE="cuda"
EPOCHS=15

echo "================================================================"
echo "PHASE 1: CLOSE OPEN CRITIQUES — $(date)"
echo "================================================================"

# ──────────────────────────────────────────────────────────────────
# D1: Moirai Extended Grid — last2 + last4 freeze levels
# Already have: frozen × 6 datasets × 3 seeds (18 files)
# Need: last2 + last4 × 6 datasets × 3 seeds (36 runs)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D1: MOIRAI EXTENDED GRID ==="
echo "============================================================"

mkdir -p results/rr_moa

for unfreeze in last2 last4; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for seed in 42 43 44; do
      OUTFILE="results/rr_moa/${ds}_H96_K5_top2_${unfreeze}_${seed}_bb-moirai.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): Moirai $ds $unfreeze seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Moirai: $ds unfreeze=$unfreeze seed=$seed"
      $PYTHON scripts/run_rr_moa.py \
        --dataset $ds --unfreeze $unfreeze --seed $seed \
        --epochs $EPOCHS --device $DEVICE \
        --backbone "Salesforce/moirai-1.1-R-small" \
        --top-k 2 --no-baselines \
        || echo "FAILED: Moirai $ds $unfreeze $seed"
    done
  done
done

echo ""
echo "[$(date +%H:%M:%S)] D1 COMPLETE"

# ──────────────────────────────────────────────────────────────────
# D2: LoRA + Unfreezing Ablation
# Already have: qkvo_mlp2_last4 for ETTh1/ETTm1/Weather
# Need: last2 for core 3, last4 for ETTh2/ETTm2/Electricity
# Also: qv (default targets) with last2/last4 for all
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D2: LoRA + UNFREEZING ABLATION ==="
echo "============================================================"

mkdir -p results/lora_baseline

# LoRA qv (default) with last2/last4 on all 6 datasets
for unfreeze in last2 last4; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
    for seed in 42 43 44; do
      OUTFILE="results/lora_baseline/${ds}_H96_r16_${unfreeze}_${seed}.json"
      if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (exists): LoRA-qv $ds $unfreeze seed=$seed"
        continue
      fi
      echo "[$(date +%H:%M:%S)] LoRA-qv: $ds unfreeze=$unfreeze seed=$seed"
      $PYTHON scripts/run_lora_baseline.py \
        --dataset $ds --unfreeze $unfreeze --seed $seed \
        --rank 16 --target-modules qv --head linear \
        --epochs $EPOCHS --device $DEVICE \
        || echo "FAILED: LoRA-qv $ds $unfreeze $seed"
    done
  done
done

# LoRA qkvo_mlp2 with last4 for ETTh2/ETTm2/Electricity (extending existing grid)
for ds in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/lora_baseline/${ds}_H96_r16_qkvo_mlp2_last4_${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): LoRA-qkvo-mlp2 $ds last4 seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] LoRA-qkvo-mlp2: $ds last4 seed=$seed"
    $PYTHON scripts/run_lora_baseline.py \
      --dataset $ds --unfreeze last4 --seed $seed \
      --rank 16 --target-modules qkvo --head mlp2 \
      --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: LoRA-qkvo-mlp2 $ds last4 $seed"
  done
done

# LoRA qkvo_mlp2 with last2 for all 6 datasets
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/lora_baseline/${ds}_H96_r16_qkvo_mlp2_last2_${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): LoRA-qkvo-mlp2 $ds last2 seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] LoRA-qkvo-mlp2: $ds last2 seed=$seed"
    $PYTHON scripts/run_lora_baseline.py \
      --dataset $ds --unfreeze last2 --seed $seed \
      --rank 16 --target-modules qkvo --head mlp2 \
      --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: LoRA-qkvo-mlp2 $ds last2 $seed"
  done
done

echo ""
echo "[$(date +%H:%M:%S)] D2 COMPLETE"

# ──────────────────────────────────────────────────────────────────
# D3: Dual-Stream Scaling
# Already have: dual-stream on ETTh1/ETTm1/Weather × 3 seeds
# Need: ETTh2/ETTm2/Electricity × 3 seeds (MOMENT backbone)
# Need: all 6 datasets × 3 seeds (Moirai backbone)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== D3: DUAL-STREAM SCALING ==="
echo "============================================================"

mkdir -p results/gap_closing

# D3a: Dual-stream on remaining MOMENT datasets
for ds in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): dual-stream $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Dual-stream (MOMENT): $ds seed=$seed"
    $PYTHON scripts/run_gap_closing.py \
      --variant dual-stream --dataset $ds \
      --seed $seed --epochs $EPOCHS --device $DEVICE \
      || echo "FAILED: dual-stream $ds $seed"
  done
done

# D3b: Dual-stream on ALL datasets with Moirai backbone
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity; do
  for seed in 42 43 44; do
    OUTFILE="results/gap_closing/dual-stream_${ds}_H96_${seed}_bb-moirai.json"
    if [ -f "$OUTFILE" ]; then
      echo "[$(date +%H:%M:%S)] SKIP (exists): dual-stream Moirai $ds seed=$seed"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Dual-stream (Moirai): $ds seed=$seed"
    $PYTHON scripts/run_gap_closing.py \
      --variant dual-stream --dataset $ds \
      --seed $seed --epochs $EPOCHS --device $DEVICE \
      --backbone "Salesforce/moirai-1.1-R-small" \
      || echo "FAILED: dual-stream Moirai $ds $seed"
  done
done

echo ""
echo "[$(date +%H:%M:%S)] D3 COMPLETE"

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "PHASE 1 COMPLETE — $(date)"
echo "================================================================"

echo ""
echo "=== FILE COUNTS ==="
echo "Moirai RR-MoA (last2+last4): $(ls results/rr_moa/*bb-moirai* 2>/dev/null | grep -E 'last2|last4' | wc -l) / 36 expected"
echo "LoRA unfreezing (qv): $(ls results/lora_baseline/*_r16_last*_*.json 2>/dev/null | wc -l) files"
echo "LoRA unfreezing (qkvo_mlp2): $(ls results/lora_baseline/*qkvo_mlp2_last*_*.json 2>/dev/null | wc -l) files"
echo "Gap-closing dual-stream (MOMENT): $(ls results/gap_closing/dual-stream_*_H96_*.json 2>/dev/null | grep -v moirai | wc -l) / 18 expected"
echo "Gap-closing dual-stream (Moirai): $(ls results/gap_closing/dual-stream_*_bb-moirai.json 2>/dev/null | wc -l) / 18 expected"

echo ""
echo "=== D3 RESULTS PREVIEW ==="
$PYTHON -c "
import json, glob, numpy as np

# Load DLinear baselines
dlinear = {}
for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity']:
    mses = []
    for s in [42, 43, 44]:
        try:
            d = json.load(open(f'results/dlinear/{ds}_H96_{s}.json'))
            mses.append(d['dlinear_mse'])
        except: pass
    if mses:
        dlinear[ds] = np.mean(mses)

print(f\"{'Backbone':12} {'Dataset':12} {'MSE':8} {'vs DLinear':12}\")
print('-' * 50)

# MOMENT dual-stream
for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity']:
    mses = []
    for s in [42, 43, 44]:
        try:
            d = json.load(open(f'results/gap_closing/dual-stream_{ds}_H96_{s}.json'))
            mses.append(d['mse'])
        except: pass
    if mses:
        m = np.mean(mses)
        dl_gap = f\"+{(m/dlinear[ds]-1)*100:.1f}%\" if ds in dlinear else 'N/A'
        print(f\"{'MOMENT':12} {ds:12} {m:.4f}   {dl_gap:12}\")

# Moirai dual-stream
for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity']:
    mses = []
    for s in [42, 43, 44]:
        try:
            d = json.load(open(f'results/gap_closing/dual-stream_{ds}_H96_{s}_bb-moirai.json'))
            mses.append(d['mse'])
        except: pass
    if mses:
        m = np.mean(mses)
        dl_gap = f\"+{(m/dlinear[ds]-1)*100:.1f}%\" if ds in dlinear else 'N/A'
        print(f\"{'Moirai':12} {ds:12} {m:.4f}   {dl_gap:12}\")
" 2>/dev/null || echo "(summary script failed — check individual JSONs)"
