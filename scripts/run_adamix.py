"""AdaMix: Learned Adapter Mixture for TSFM Adaptation.

Instead of searching for one adapter architecture, trains a fixed mixture
of K canonical adapter heads with a learned per-sample router.

Key insight from DARTS experiments: the soft mixture (undiscretized supernet)
beats any single discretized architecture. AdaMix makes this permanent —
instance-level routing lets different time series windows get different
adapter combinations.

Usage:
    python scripts/run_adamix.py --dataset ETTh1
    python scripts/run_adamix.py --dataset ETTh1 --K 3
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
from feasibility.code_evolution import SEED_ADAPTERS, validate_adapter_code
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


# --- Individual adapter heads (simple, canonical designs) ---

class MeanPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.mean(dim=1)
        return self.out(F.gelu(self.fc(x)))


class LastTokenHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h[:, -1, :]
        return self.out(F.gelu(self.fc(x)))


class MaxPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.max(dim=1).values
        return self.out(F.gelu(self.fc(x)))


class AttentionPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        x = (h * w).sum(dim=1)
        return self.out(F.gelu(self.fc(x)))


class Conv1dPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=8, stride=4, padding=2)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.permute(0, 2, 1)  # (B, d, T)
        x = F.gelu(self.conv(x))  # (B, hidden, T')
        x = x.mean(dim=2)  # (B, hidden)
        return self.out(x)


# --- AdaMix: Learned Adapter Mixture ---

HEAD_CLASSES = [MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead]
HEAD_NAMES = ["mean", "last", "max", "attention", "conv1d"]


class AdaMix(nn.Module):
    """Learned mixture of K canonical adapter heads with per-sample routing.

    The router produces instance-level weights over K adapter heads.
    Different time series windows get different adapter combinations.
    """
    def __init__(self, d_model, output_dim, K=5, hidden=64, router_hidden=32):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # K adapter heads
        self.adapters = nn.ModuleList([
            HEAD_CLASSES[i % len(HEAD_CLASSES)](d_model, output_dim, hidden)
            for i in range(K)
        ])

        # Instance-level router
        self.router = nn.Sequential(
            nn.Linear(d_model, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, K),
        )

        # Optional: load balancing loss coefficient
        self.load_balance_coeff = 0.01

    def forward(self, hidden_states):
        # Router: produce per-sample weights based on mean-pooled representation
        h_summary = hidden_states.mean(dim=1)  # (B, d_model)
        logits = self.router(h_summary)  # (B, K)
        weights = F.softmax(logits, dim=-1)  # (B, K)

        # Compute all adapter outputs
        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)  # (B, K, output_dim)

        # Weighted combination
        mixed = (weights.unsqueeze(-1) * outputs).sum(dim=1)  # (B, output_dim)

        return mixed

    def get_routing_stats(self, hidden_states):
        """Get routing weights for analysis (no grad)."""
        with torch.no_grad():
            h_summary = hidden_states.mean(dim=1)
            logits = self.router(h_summary)
            weights = F.softmax(logits, dim=-1)
        return weights

    def load_balance_loss(self, hidden_states):
        """Auxiliary loss to prevent router collapse (standard MoE technique)."""
        h_summary = hidden_states.mean(dim=1)
        logits = self.router(h_summary)
        weights = F.softmax(logits, dim=-1)

        # Fraction of samples routed to each expert
        f_i = weights.mean(dim=0)  # (K,)
        # Probability assigned to each expert
        p_i = F.softmax(logits, dim=-1).mean(dim=0)  # (K,)

        return self.K * (f_i * p_i).sum()

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def train_adamix(model, blocks, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", K=5, hidden=64):
    """Train AdaMix adapter."""
    hdim = _get_hidden_dim(model)
    adapter = AdaMix(hdim, forecast_horizon, K=K, hidden=hidden).to(device)

    # Collect trainable params (adapter + unfrozen backbone)
    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float(),
    ), batch_size=batch_size)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                pred = adapter(feat)
                loss = mse_fn(pred, by) + adapter.load_balance_coeff * adapter.load_balance_loss(feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on test
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    all_routing_weights = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
            preds.append(adapter(feat).float().cpu())
            tgts.append(by.cpu())
            all_routing_weights.append(adapter.get_routing_stats(feat).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing_weights)

    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    # Routing analysis
    mean_routing = routing.mean(dim=0).tolist()
    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()

    return {
        "mse": mse,
        "mae": mae,
        "param_count": adapter.param_count(),
        "mean_routing_weights": {HEAD_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(HEAD_NAMES)])},
        "routing_entropy": routing_entropy,
    }


def _apply_unfreeze(blocks, unfreeze):
    """Selectively unfreeze encoder blocks."""
    n = len(blocks)
    if unfreeze == "frozen":
        return
    elif unfreeze == "last2":
        start = max(0, n - 2)
    elif unfreeze == "last4":
        start = max(0, n - 4)
    elif unfreeze == "all":
        start = 0
    else:
        raise ValueError("Unknown unfreeze: %s" % unfreeze)
    for i in range(start, n):
        for p in blocks[i].parameters():
            p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--unfreeze", default="last4", choices=["frozen", "last2", "last4", "all"],
                        help="Backbone unfreezing strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/adamix", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, args.unfreeze)
    backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("d_model=%d, K=%d, hidden=%d, unfreeze=%s, backbone_trainable=%d" % (
        hdim, args.K, args.hidden, args.unfreeze, backbone_trainable))

    # Load data
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, val=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_val), len(X_test)))

    # === Run AdaMix ===
    print("\n" + "=" * 60)
    print("AdaMix: %s H=%d K=%d seed=%d" % (args.dataset, args.horizon, args.K, args.seed))
    print("=" * 60)

    start = time.time()
    result = train_adamix(
        model, blocks, X_train, Y_train, X_val, Y_val, X_test, Y_test,
        device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
        backbone_type=bb_type, K=args.K, hidden=args.hidden,
    )
    elapsed = time.time() - start

    print("AdaMix: MSE=%.4f  MAE=%.4f  params=%d  time=%.0fs" % (
        result["mse"], result["mae"], result["param_count"], elapsed))
    print("Routing: %s" % result["mean_routing_weights"])
    print("Routing entropy: %.3f (max=%.3f for K=%d)" % (
        result["routing_entropy"], np.log(args.K), args.K))

    # === Run fixed baselines ===
    print("\nFixed baselines (%d epochs, unfreeze=%s):" % (args.epochs, args.unfreeze))
    # Reload model for fair baseline comparison
    model2 = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model2)
    blocks2 = _get_encoder_blocks(model2)
    for p in model2.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks2, args.unfreeze)

    baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
    baseline_results = {}
    for name, code in baselines.items():
        try:
            tr = train_adapter(code, model2, blocks2, X_train, Y_train, X_test, Y_test,
                               device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                               backbone_type=bb_type)
            baseline_results[name] = tr
            print("  %-15s MSE=%.4f  params=%d" % (name, tr["mse"], tr["param_count"]))
        except Exception as e:
            print("  %-15s ERROR: %s" % (name, e))

    # Summary
    best_bl_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"])
    best_bl_mse = baseline_results[best_bl_name]["mse"]
    delta = (result["mse"] - best_bl_mse) / best_bl_mse * 100
    winner = "AdaMix" if result["mse"] < best_bl_mse else "BASELINE"

    print("\n" + "=" * 60)
    print("SUMMARY: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    print("AdaMix (K=%d):  MSE=%.4f  params=%d  time=%.0fs" % (
        args.K, result["mse"], result["param_count"], elapsed))
    print("Best baseline: MSE=%.4f  (%s)" % (best_bl_mse, best_bl_name))
    print("Delta: %+.1f%% -> Winner: %s" % (delta, winner))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "K": args.K,
        "hidden": args.hidden,
        "unfreeze": args.unfreeze,
        "backbone_trainable_params": backbone_trainable,
        "adamix": result,
        "elapsed": elapsed,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner,
        "delta_pct": delta,
    }
    path = "results/adamix/%s_H%d_K%d_%s_%d.json" % (args.dataset, args.horizon, args.K, args.unfreeze, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
