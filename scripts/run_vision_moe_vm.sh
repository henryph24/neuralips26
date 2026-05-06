#!/bin/bash
# Vision MoE Routing Collapse — GPU VM runner
# Proves normalization-induced routing collapse is cross-modal
# Expected runtime: ~15 min on A10G (3 seeds × 4 conditions × 20 epochs)
set -e

PYTHON=python3
DEVICE="cuda"

echo "=== Vision MoE Routing Collapse — $(date) ==="

# Ensure torchvision is available
$PYTHON -c "import torchvision" 2>/dev/null || pip install torchvision

mkdir -p results/vision_moe

for seed in 42 43 44; do
  echo ""
  echo "[$(date +%H:%M:%S)] Vision MoE collapse: seed=$seed"
  $PYTHON scripts/run_vision_moe_collapse.py \
    --seed $seed --device $DEVICE --epochs 20 \
    || echo "FAILED: vision_moe seed=$seed"
done

echo ""
echo "=== Vision MoE COMPLETE — $(date) ==="
echo "Results in results/vision_moe/"
ls -la results/vision_moe/
