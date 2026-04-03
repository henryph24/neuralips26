"""TRACE-style baseline for AAS comparison.

Implements the two key ideas from TRACE (Li & Zhu, 2025):
1. Importance-based LoRA module selection (Gated DSIC simplified)
2. Reconstructed prediction head (factorized for parameter efficiency)

This is NOT an official TRACE implementation (no public code available).
We implement the core ideas to create a strong, hand-designed PEFT baseline.

Usage:
    python scripts/run_trace_baseline.py --dataset ETTh1
    python scripts/run_trace_baseline.py --dataset ETTh1 --horizon 336
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


# --- TRACE-inspired adapter architectures ---

TRACE_RECONSTRUCTED_HEAD = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """TRACE-style reconstructed prediction head.

    Factorizes the large (d_model -> H) projection into:
    1. Pool temporal dim via learned attention weights
    2. Down-project: d_model -> bottleneck (64)
    3. Up-project: bottleneck -> H

    This reduces params from d_model*H to d_model*64 + 64*H.
    Inspired by TRACE (Li & Zhu, 2025) reconstructed heads.
    """
    def __init__(self, d_model, output_dim):
        super().__init__()
        bottleneck = 64
        # Learned temporal attention (like TRACE importance weighting)
        self.attn = nn.Linear(d_model, 1)
        # Factorized projection
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, output_dim)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # hidden_states: (B, T, d_model)
        h = self.norm(hidden_states)
        # Learned temporal attention pooling
        w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        pooled = (h * w).sum(dim=1)  # (B, d_model)
        # Factorized head
        out = self.down(pooled)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up(out)
        return out
'''

TRACE_FULL = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """TRACE-style adapter with importance-weighted multi-layer pooling
    and reconstructed prediction head.

    Key ideas from TRACE (Li & Zhu, 2025):
    1. Multi-scale temporal modeling via Conv1d at multiple kernel sizes
    2. Gated importance weighting of features
    3. Factorized prediction head for parameter efficiency
    """
    def __init__(self, d_model, output_dim):
        super().__init__()
        bottleneck = 64

        # Multi-scale temporal convolutions (captures different frequencies)
        self.conv3 = nn.Conv1d(d_model, bottleneck, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(d_model, bottleneck, kernel_size=7, padding=3)

        # Gated feature importance (simplified Gated DSIC idea)
        self.gate = nn.Sequential(
            nn.Linear(bottleneck * 2, bottleneck),
            nn.Sigmoid()
        )

        # Reconstructed head
        self.norm = nn.LayerNorm(bottleneck)
        self.head = nn.Linear(bottleneck, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # hidden_states: (B, T, d_model)
        x = hidden_states.permute(0, 2, 1)  # (B, d, T)

        # Multi-scale conv features
        f3 = F.gelu(self.conv3(x)).mean(dim=2)  # (B, bottleneck)
        f7 = F.gelu(self.conv7(x)).mean(dim=2)  # (B, bottleneck)

        # Gated combination
        combined = torch.cat([f3, f7], dim=-1)  # (B, 2*bottleneck)
        gate = self.gate(combined)  # (B, bottleneck)
        fused = gate * f3 + (1 - gate) * f7  # (B, bottleneck)

        # Prediction
        out = self.norm(fused)
        out = self.dropout(out)
        out = self.head(out)
        return out
'''

# Standard LoRA-style baseline (best from our discrete config space)
LORA_BEST = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """Best hand-designed adapter from discrete config space.

    Mean pooling + 2-layer MLP with GELU, representing the
    best LoRA-style configuration typically used in practice.
    """
    def __init__(self, d_model, output_dim):
        super().__init__()
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid // 2, output_dim),
        )

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)
'''


BASELINES = {
    "trace_reconstructed": TRACE_RECONSTRUCTED_HEAD,
    "trace_full": TRACE_FULL,
    "mlp2_mean": LORA_BEST,
    "linear": SEED_ADAPTERS[0],
    "attention": SEED_ADAPTERS[3],
    "conv": SEED_ADAPTERS[4],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/trace_baseline", exist_ok=True)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    print("Unfreezing last 4 encoder blocks, d_model=%d" % hdim)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, val=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_val), len(X_test)))

    # Validate all baselines
    print("\nValidating adapter architectures...")
    valid_baselines = {}
    for name, code in BASELINES.items():
        val = validate_adapter_code(code, d_model=hdim, output_dim=args.horizon)
        if val["valid"]:
            print("  %-25s VALID  params=%d" % (name, val["param_count"]))
            valid_baselines[name] = code
        else:
            print("  %-25s INVALID: %s" % (name, val["error"]))

    # Evaluate all on test set at 15 epochs
    print("\nEvaluating on TEST set (15 epochs)...")
    results = {}
    for name, code in valid_baselines.items():
        try:
            start = time.time()
            tr = train_adapter(
                code, model, blocks, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            elapsed = time.time() - start
            results[name] = {
                "test_mse": tr["mse"],
                "test_mae": tr["mae"],
                "param_count": tr["param_count"],
                "elapsed": elapsed,
            }
            print("  %-25s MSE=%.4f  MAE=%.4f  params=%d  time=%.0fs" % (
                name, tr["mse"], tr["mae"], tr["param_count"], elapsed))
        except Exception as e:
            print("  %-25s ERROR: %s" % (name, e))
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRACE BASELINE COMPARISON: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    valid_results = {k: v for k, v in results.items() if "test_mse" in v}
    for name, r in sorted(valid_results.items(), key=lambda x: x[1]["test_mse"]):
        print("  %-25s MSE=%.4f  params=%d" % (name, r["test_mse"], r["param_count"]))

    best_name = min(valid_results, key=lambda k: valid_results[k]["test_mse"])
    print("\nBest: %s (MSE=%.4f)" % (best_name, valid_results[best_name]["test_mse"]))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "results": results,
    }
    path = "results/trace_baseline/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
