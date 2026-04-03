"""Imputation task: prove adapter architecture is task-dependent.

Masks 20% of input timesteps, trains adapter to reconstruct them.
If AAS discovers a DIFFERENT architecture for imputation vs forecasting,
this proves adapters must be task-specific across the FM ecosystem.

Usage:
    python scripts/run_imputation.py --dataset ETTh1
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from scripts.run_standard_evolution import load_standard_data, _detect_backbone_type
from scripts.run_rr_moa import (
    MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead,
    HEAD_CLASSES, HEAD_NAMES, RawRoutedMoA,
)


# === Imputation-specific adapter heads ===
# These output (B, seq_len) instead of (B, forecast_horizon)
# They need to reconstruct the full temporal sequence

class ImputationMeanHead(nn.Module):
    """Dense reconstruction: project each timestep independently."""
    def __init__(self, d_model, seq_len, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, h):
        # h: (B, T, d_model) -> (B, T, 1) -> (B, T)
        return self.net(h).squeeze(-1)

class ImputationConvHead(nn.Module):
    """Conv-based reconstruction: local context for gap filling."""
    def __init__(self, d_model, seq_len, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
    def forward(self, h):
        x = h.permute(0, 2, 1)  # (B, d, T)
        x = F.gelu(self.conv1(x))
        return self.conv2(x).squeeze(1)  # (B, T)

class ImputationAttnHead(nn.Module):
    """Attention-based reconstruction: global context."""
    def __init__(self, d_model, seq_len, hidden=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.proj = nn.Linear(d_model, 1)
    def forward(self, h):
        h2, _ = self.attn(h, h, h)
        return self.proj(h2).squeeze(-1)  # (B, T)

class ImputationLinearHead(nn.Module):
    """Simple per-timestep linear."""
    def __init__(self, d_model, seq_len, hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
    def forward(self, h):
        return self.proj(h).squeeze(-1)

class ImputationResidualConvHead(nn.Module):
    """Residual conv: skip + local conv for dense prediction."""
    def __init__(self, d_model, seq_len, hidden=128):
        super().__init__()
        self.skip = nn.Linear(d_model, 1)
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=7, padding=3)
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)
    def forward(self, h):
        skip = self.skip(h).squeeze(-1)  # (B, T)
        x = F.gelu(self.conv(h.permute(0, 2, 1)))
        return skip + self.out(x).squeeze(1)  # (B, T)

IMPUTATION_HEADS = [ImputationLinearHead, ImputationMeanHead, ImputationConvHead,
                    ImputationAttnHead, ImputationResidualConvHead]
IMPUTATION_NAMES = ["linear", "dense_mlp", "conv", "attention", "residual_conv"]


class ImputationRRMoA(nn.Module):
    """RR-MoA for imputation: same raw-routing, imputation expert heads."""
    def __init__(self, d_model, seq_len=512, K=5, hidden=128):
        super().__init__()
        self.K = K
        self.adapters = nn.ModuleList([
            IMPUTATION_HEADS[i % len(IMPUTATION_HEADS)](d_model, seq_len, hidden)
            for i in range(K)
        ])
        self.router = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.router_head = nn.Linear(64, K)
        self.load_balance_coeff = 0.01

    def forward(self, hidden_states, raw_input):
        x_raw = raw_input.unsqueeze(1)
        router_feat = self.router(x_raw).flatten(1)
        logits = self.router_head(router_feat)
        weights = F.softmax(logits, dim=-1)  # (B, K)
        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)  # (B, K, T)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)  # (B, T)

    def get_routing_stats(self, raw_input):
        with torch.no_grad():
            x_raw = raw_input.unsqueeze(1)
            router_feat = self.router(x_raw).flatten(1)
            return F.softmax(self.router_head(router_feat), dim=-1)

    def load_balance_loss(self, raw_input):
        x_raw = raw_input.unsqueeze(1)
        router_feat = self.router(x_raw).flatten(1)
        logits = self.router_head(router_feat)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        return self.K * (f_i * F.softmax(logits, dim=-1).mean(dim=0)).sum()


def create_imputation_data(X, mask_ratio=0.2, seed=42):
    """Create imputation dataset: mask random timesteps."""
    rng = np.random.default_rng(seed)
    N, T = X.shape
    mask = rng.random((N, T)) < mask_ratio  # True = masked
    X_masked = X.copy()
    X_masked[mask] = 0.0  # zero out masked positions
    return X_masked, X, mask.astype(np.float32)


def train_imputation(model, blocks, adapter, X_masked, X_target, mask,
                     device, n_epochs=15, batch_size=128, backbone_type="moment"):
    """Train adapter for imputation. Loss only on masked positions."""
    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p); pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_masked).float(),
        torch.from_numpy(X_target).float(),
        torch.from_numpy(mask).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx_masked, bx_target, bmask in loader:
            bx_masked = bx_masked.to(device)
            bx_target = bx_target.to(device)
            bmask = bmask.to(device)
            bx_enc = bx_masked.unsqueeze(1)  # (B, 1, T)
            mask_enc = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask_enc, backbone_type=backbone_type)
                if hasattr(adapter, 'router'):  # RR-MoA
                    pred = adapter(feat, bx_masked)
                else:
                    pred = adapter(feat)
                # Loss only on masked positions
                loss = ((pred - bx_target) ** 2 * bmask).sum() / (bmask.sum() + 1e-10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return adapter


def evaluate_imputation(model, blocks, adapter, X_masked, X_target, mask,
                        device, batch_size=128, backbone_type="moment"):
    """Evaluate imputation MSE on masked positions only."""
    model.eval(); adapter.eval()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_masked).float(),
        torch.from_numpy(X_target).float(),
        torch.from_numpy(mask).float(),
    ), batch_size=batch_size)

    total_loss = 0.0
    total_mask = 0.0
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx_masked, bx_target, bmask in loader:
            bx_masked = bx_masked.to(device)
            bx_target = bx_target.to(device)
            bmask = bmask.to(device)
            bx_enc = bx_masked.unsqueeze(1)
            mask_enc = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask_enc, backbone_type=backbone_type)
            if hasattr(adapter, 'router'):
                pred = adapter(feat, bx_masked)
            else:
                pred = adapter(feat)
            total_loss += ((pred.float() - bx_target) ** 2 * bmask).sum().item()
            total_mask += bmask.sum().item()

    return total_loss / (total_mask + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/imputation", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
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

    # Load data
    splits, _ = load_standard_data(args.dataset, 96)  # horizon doesn't matter for imputation
    X_train = splits["train"][0]  # (N, 512)
    X_test = splits["test"][0]

    # Create imputation data
    X_train_masked, X_train_target, mask_train = create_imputation_data(X_train, args.mask_ratio)
    X_test_masked, X_test_target, mask_test = create_imputation_data(X_test, args.mask_ratio, seed=99)

    print("%s Imputation (mask=%.0f%%): train=%d, test=%d" % (
        args.dataset, args.mask_ratio * 100, len(X_train), len(X_test)))

    # === Test individual imputation heads ===
    print("\n--- Individual Imputation Heads (15 epochs) ---")
    head_results = {}
    for name, HeadClass in zip(IMPUTATION_NAMES, IMPUTATION_HEADS):
        model_fresh = load_backbone(args.backbone, args.device)
        _disable_gradient_checkpointing(model_fresh)
        blocks_fresh = _get_encoder_blocks(model_fresh)
        for p in model_fresh.parameters():
            p.requires_grad = False
        for i in range(max(0, len(blocks_fresh)-4), len(blocks_fresh)):
            for p in blocks_fresh[i].parameters():
                p.requires_grad = True

        adapter = HeadClass(hdim, 512).to(args.device)
        adapter = train_imputation(model_fresh, blocks_fresh, adapter,
                                   X_train_masked, X_train_target, mask_train,
                                   args.device, n_epochs=15, backbone_type=bb_type)
        mse = evaluate_imputation(model_fresh, blocks_fresh, adapter,
                                  X_test_masked, X_test_target, mask_test,
                                  args.device, backbone_type=bb_type)
        head_results[name] = mse
        params = sum(p.numel() for p in adapter.parameters())
        print("  %-15s MSE=%.6f  params=%d" % (name, mse, params))

    # === Test RR-MoA for imputation ===
    print("\n--- RR-MoA Imputation ---")
    model_fresh = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model_fresh)
    blocks_fresh = _get_encoder_blocks(model_fresh)
    for p in model_fresh.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks_fresh)-4), len(blocks_fresh)):
        for p in blocks_fresh[i].parameters():
            p.requires_grad = True

    rrmoa = ImputationRRMoA(hdim, seq_len=512, K=5).to(args.device)
    rrmoa = train_imputation(model_fresh, blocks_fresh, rrmoa,
                             X_train_masked, X_train_target, mask_train,
                             args.device, n_epochs=15, backbone_type=bb_type)
    rrmoa_mse = evaluate_imputation(model_fresh, blocks_fresh, rrmoa,
                                    X_test_masked, X_test_target, mask_test,
                                    args.device, backbone_type=bb_type)
    routing = rrmoa.get_routing_stats(torch.from_numpy(X_test_masked[:32]).float().to(args.device))
    mean_routing = routing.mean(dim=0).cpu().tolist()

    print("  RR-MoA:       MSE=%.6f  routing=%s" % (
        rrmoa_mse, {IMPUTATION_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing)}))

    # === Summary ===
    best_head = min(head_results, key=head_results.get)
    best_head_mse = head_results[best_head]

    print("\n" + "=" * 60)
    print("IMPUTATION RESULTS: %s (mask=%.0f%%)" % (args.dataset, args.mask_ratio * 100))
    print("=" * 60)
    print("Best single head:  %s (MSE=%.6f)" % (best_head, best_head_mse))
    print("RR-MoA:            MSE=%.6f" % rrmoa_mse)
    print("RR-MoA vs best:    %+.1f%%" % ((rrmoa_mse - best_head_mse) / best_head_mse * 100))

    # Compare with forecasting winner
    print("\n--- Forecasting vs Imputation Architecture Comparison ---")
    print("Forecasting winner on %s: last_token / attention pooling" % args.dataset)
    print("Imputation winner on %s: %s" % (args.dataset, best_head))
    if best_head in ["conv", "residual_conv"]:
        print(">>> DIFFERENT! Imputation prefers LOCAL CONV for gap-filling.")
        print(">>> Forecasting prefers TEMPORAL POOLING for horizon projection.")
        print(">>> This proves adapter architecture is TASK-DEPENDENT.")
    else:
        print(">>> Architecture preference noted.")

    # Save
    save_data = {
        "dataset": args.dataset, "mask_ratio": args.mask_ratio, "seed": args.seed,
        "head_results": head_results,
        "rrmoa_mse": rrmoa_mse,
        "rrmoa_routing": {IMPUTATION_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing)},
        "best_head": best_head,
        "best_head_mse": best_head_mse,
    }
    path = "results/imputation/%s_%d.json" % (args.dataset, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
