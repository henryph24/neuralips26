#!/bin/bash
# Completeness sweep — fill every gap in the experimental grid.
#
# Runs after overnight batches finish. Ensures every table has
# 5-seed coverage on all 6 datasets where applicable.
#
# Estimated: ~300 runs, ~3h solo on A10G.
#
# Invocation:
#   tmux new-session -d -s sweep 'cd ~/neuralips26 && bash scripts/run_completeness_sweep.sh 2>&1 | tee results/completeness_sweep.log'

set -e
DEVICE="cuda"
EPOCHS=15
PYTHON=python3

echo "================================================================"
echo "COMPLETENESS SWEEP — $(date)"
echo "================================================================"

###############################################
# 1. B7 re-run (9 failed classification runs)
###############################################
echo ""
echo "=== 1. B7 Classification re-run ==="
bash scripts/run_b7_classification_vm.sh 2>&1 | tail -5

###############################################
# 2. TRACE baseline on extended datasets
#    + 5-seed upgrade on core datasets
###############################################
echo ""
echo "=== 2. TRACE Baseline Extension ==="

TRACE_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
TRACE_SEEDS=(42 43 44 45 46)
IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for ds in "${TRACE_DATASETS[@]}"; do
  for seed in "${TRACE_SEEDS[@]}"; do
    OUT="results/trace_baseline/${ds}_H96_last4_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
    fi
    echo "[$(date +%H:%M:%S)] TRACE $ds seed=$seed"
    set +e
    $PYTHON scripts/run_trace_baseline.py \
      --dataset "$ds" --seed "$seed" --device "$DEVICE" \
      > /tmp/trace_${IDX}.out 2>&1
    RC=$?; set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      tail -2 /tmp/trace_${IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
    IDX=$((IDX + 1))
  done
done
echo "TRACE: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# 3. Independent Ensemble on extended datasets
#    + 5-seed upgrade
###############################################
echo ""
echo "=== 3. Independent Ensemble Extension ==="

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0
for ds in "${TRACE_DATASETS[@]}"; do
  for seed in "${TRACE_SEEDS[@]}"; do
    OUT="results/independent_ensemble/${ds}_H96_frozen_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
    fi
    echo "[$(date +%H:%M:%S)] IndEns $ds seed=$seed"
    set +e
    $PYTHON scripts/run_independent_ensemble.py \
      --dataset "$ds" --seed "$seed" --unfreeze frozen --device "$DEVICE" \
      > /tmp/indens_${IDX}.out 2>&1
    RC=$?; set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      tail -2 /tmp/indens_${IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
    IDX=$((IDX + 1))
  done
done
echo "IndEns: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# 4. Imputation extended to 6 datasets × 5 seeds
###############################################
echo ""
echo "=== 4. Imputation Extension ==="

IMPUT_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
IMPUT_SEEDS=(42 43 44 45 46)
IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for ds in "${IMPUT_DATASETS[@]}"; do
  for seed in "${IMPUT_SEEDS[@]}"; do
    OUT="results/imputation/${ds}_mask0.2_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
    fi
    echo "[$(date +%H:%M:%S)] Imputation $ds seed=$seed"
    set +e
    $PYTHON scripts/run_imputation.py \
      --dataset "$ds" --seed "$seed" --device "$DEVICE" \
      > /tmp/imput_${IDX}.out 2>&1
    RC=$?; set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      tail -2 /tmp/imput_${IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
    IDX=$((IDX + 1))
  done
done
echo "Imputation: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# 5. RR-MoA 5-seed on extended datasets
#    (seeds 45-46 at all freeze levels)
###############################################
echo ""
echo "=== 5. RR-MoA 5-seed Extension ==="

EXT_DATASETS=(ETTh2 ETTm2 Electricity)
EXT_SEEDS=(45 46)
FREEZE_LEVELS=(frozen last2 last4)
IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for ds in "${EXT_DATASETS[@]}"; do
  for freeze in "${FREEZE_LEVELS[@]}"; do
    for seed in "${EXT_SEEDS[@]}"; do
      OUT="results/rr_moa/${ds}_H96_K5_top2_${freeze}_${seed}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] RR-MoA $ds $freeze seed=$seed"
      set +e
      $PYTHON scripts/run_rr_moa.py \
        --dataset "$ds" --unfreeze "$freeze" --top-k 2 \
        --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" \
        > /tmp/ext5b_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        tail -2 /tmp/ext5b_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
      fi
      IDX=$((IDX + 1))
    done
  done
done
echo "RR-MoA ext: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# 6. Full fine-tuning on extended datasets
###############################################
echo ""
echo "=== 6. Full Fine-Tuning Extension ==="

IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0
for ds in ETTh2 ETTm2 Electricity; do
  for seed in 42 43 44; do
    OUT="results/full_finetune/${ds}_H96_${seed}.json"
    if [ -f "$OUT" ]; then
      SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
    fi
    echo "[$(date +%H:%M:%S)] FullFT $ds seed=$seed"
    set +e
    $PYTHON scripts/run_full_finetune.py \
      --dataset "$ds" --seed "$seed" --device "$DEVICE" \
      > /tmp/fullft_${IDX}.out 2>&1
    RC=$?; set -e
    if [ $RC -ne 0 ]; then
      FAILED=$((FAILED + 1))
      tail -2 /tmp/fullft_${IDX}.out | sed 's/^/    /'
    else
      LAUNCHED=$((LAUNCHED + 1))
    fi
    IDX=$((IDX + 1))
  done
done
echo "FullFT: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

###############################################
# 7. Learning rate sensitivity
###############################################
echo ""
echo "=== 7. Learning Rate Sensitivity ==="

LR_VALUES=(0.0005 0.002 0.005)
LR_DATASETS=(ETTh1 ETTm1 Weather)
IDX=0; LAUNCHED=0; SKIPPED=0; FAILED=0

for lr in "${LR_VALUES[@]}"; do
  for ds in "${LR_DATASETS[@]}"; do
    for seed in 42 43 44; do
      # Use a distinct output name via a wrapper approach
      OUT="results/rr_moa/${ds}_H96_K5_top2_frozen_${seed}_lr-${lr}.json"
      if [ -f "$OUT" ]; then
        SKIPPED=$((SKIPPED + 1)); IDX=$((IDX + 1)); continue
      fi
      echo "[$(date +%H:%M:%S)] LR=$lr | $ds seed=$seed"
      set +e
      # Train with custom LR, then rename the output
      # Since run_rr_moa.py doesn't have a --lr flag, we use a temp approach:
      # Run the standard script and check if lr flag exists
      $PYTHON -c "
import sys, os, json, time, torch, numpy as np
sys.path.insert(0, '.')
from scripts.run_rr_moa import RawRoutedMoA, HEAD_CLASSES, HEAD_NAMES, EXPERT_POOLS
from feasibility.model import load_backbone, _get_encoder_blocks, _get_hidden_dim, _disable_gradient_checkpointing
from feasibility.finetune import _extract_features_batch
from feasibility.standard_data import load_standard_data, _detect_backbone_type
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed($seed); np.random.seed($seed)
splits, _ = load_standard_data('$ds', 96)
X_tr, Y_tr = splits['train']; X_te, Y_te = splits['test']
bb_type = _detect_backbone_type('AutonLab/MOMENT-1-small')
model = load_backbone('AutonLab/MOMENT-1-small', '$DEVICE')
_disable_gradient_checkpointing(model)
blocks = _get_encoder_blocks(model)
for p in model.parameters(): p.requires_grad = False
hdim = _get_hidden_dim(model)

adapter = RawRoutedMoA(hdim, 96, input_len=512, K=5, hidden=64, top_k=2, router_input_mode='raw').to('$DEVICE')
optimizer = torch.optim.Adam(adapter.parameters(), lr=$lr)
mse_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float()), batch_size=128, shuffle=True)

for ep in range(15):
    model.train(); adapter.train()
    for bx, by in loader:
        bx_raw = bx.to('$DEVICE'); bx_enc = bx.to('$DEVICE').unsqueeze(1); by = by.to('$DEVICE')
        mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device='$DEVICE')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb_type)
            pred = adapter(feat, bx_raw)
            loss = mse_fn(pred, by)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

model.eval(); adapter.eval()
preds, tgts = [], []
eval_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).float()), batch_size=128)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for bx, by in eval_loader:
        bx_raw = bx.to('$DEVICE'); bx_enc = bx.to('$DEVICE').unsqueeze(1); by = by.to('$DEVICE')
        mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device='$DEVICE')
        feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb_type)
        preds.append(adapter(feat, bx_raw).float().cpu()); tgts.append(by.cpu())
preds = torch.cat(preds); tgts = torch.cat(tgts)
mse = nn.MSELoss()(preds, tgts).item()
print(f'MSE={mse:.4f} lr=$lr')
json.dump({'mse': mse, 'lr': $lr, 'dataset': '$ds', 'seed': $seed, 'K': 5, 'top_k': 2}, open('$OUT', 'w'))
" > /tmp/lr_${IDX}.out 2>&1
      RC=$?; set -e
      if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        tail -3 /tmp/lr_${IDX}.out | sed 's/^/    /'
      else
        LAUNCHED=$((LAUNCHED + 1))
        grep "MSE=" /tmp/lr_${IDX}.out | sed 's/^/    /'
      fi
      IDX=$((IDX + 1))
    done
  done
done
echo "LR sweep: launched=$LAUNCHED skipped=$SKIPPED failed=$FAILED"

echo ""
echo "================================================================"
echo "COMPLETENESS SWEEP DONE — $(date)"
echo "================================================================"
