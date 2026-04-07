"""Per-sample routing analysis: what temporal characteristics drive expert selection?

Trains RR-MoA, then for each test sample records:
1. Which expert the router selects (Top-1 and Top-2)
2. Temporal statistics of the raw input window (trend, amplitude, volatility, stationarity)

Produces a scatter plot showing routing decisions correlate with signal characteristics.
This is the "smoking gun" that the router learns meaningful signal→expert mapping.

Usage:
    python scripts/run_patchwise_analysis.py --dataset ETTh1 --seed 42
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
from scripts.run_standard_evolution import (
    load_standard_data, _detect_backbone_type,
)
from scripts.run_rr_moa import (
    RawRoutedMoA, HEAD_NAMES, _apply_unfreeze, train_rr_moa,
)


def compute_temporal_stats(raw_windows):
    """Compute temporal statistics for each window.

    Args:
        raw_windows: (N, L) numpy array of raw time series windows

    Returns:
        dict of (N,) arrays: trend_slope, amplitude, volatility, mean_level
    """
    N, L = raw_windows.shape
    t = np.arange(L, dtype=np.float64)

    # Trend slope: linear regression slope
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    x_mean = raw_windows.mean(axis=1, keepdims=True)
    slopes = ((raw_windows - x_mean) * (t - t_mean)).sum(axis=1) / t_var

    # Amplitude: max - min range
    amplitude = raw_windows.max(axis=1) - raw_windows.min(axis=1)

    # Volatility: std of first differences
    diffs = np.diff(raw_windows, axis=1)
    volatility = diffs.std(axis=1)

    # Mean level
    mean_level = raw_windows.mean(axis=1)

    return {
        "trend_slope": slopes,
        "amplitude": amplitude,
        "volatility": volatility,
        "mean_level": mean_level,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/routing_analysis", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    bb_type = _detect_backbone_type(args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    hdim = _get_hidden_dim(model)

    print("Training RR-MoA for routing analysis...")
    result = train_rr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                          device=args.device, forecast_horizon=args.horizon,
                          backbone_type=bb_type, K=5, top_k=2,
                          n_epochs=15, router_input_mode="raw")
    print("RR-MoA MSE=%.4f" % result["mse"])

    # Re-instantiate and load the trained adapter to get routing decisions
    # The trained adapter is inside train_rr_moa — we need the routing weights
    # Instead, we extract them from the result's per-sample routing data
    # But train_rr_moa returns aggregated stats. We need per-sample routing.
    # Re-train with access to the adapter object.

    print("Re-training to capture per-sample routing decisions...")
    torch.manual_seed(args.seed)
    adapter = RawRoutedMoA(hdim, args.horizon, input_len=512, K=5, hidden=64,
                           top_k=2, router_input_mode="raw").to(args.device)

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = args.device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=128, shuffle=True)

    for epoch in range(15):
        model.train(); adapter.train()
        for bx, by in loader:
            bx_raw = bx.to(args.device)
            bx_enc = bx.to(args.device).unsqueeze(1)
            by = by.to(args.device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=args.device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Collect per-sample routing decisions on test set
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
    ), batch_size=128)

    all_logits = []
    all_weights = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx_raw = bx.to(args.device)
            logits = adapter._compute_logits(bx_raw)
            weights = F.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
            all_weights.append(weights.cpu())

    all_weights = torch.cat(all_weights).numpy()  # (N_test, K)
    top1_expert = all_weights.argmax(axis=1)  # (N_test,)

    # Compute temporal stats
    stats = compute_temporal_stats(X_test)

    print("\nPer-expert temporal stats (Top-1 assignment):")
    for k in range(5):
        mask = (top1_expert == k)
        n = mask.sum()
        if n > 0:
            print("  Expert %d (%s): n=%d, trend=%.4f±%.4f, amp=%.4f±%.4f, vol=%.4f±%.4f" % (
                k, HEAD_NAMES[k], n,
                stats["trend_slope"][mask].mean(), stats["trend_slope"][mask].std(),
                stats["amplitude"][mask].mean(), stats["amplitude"][mask].std(),
                stats["volatility"][mask].mean(), stats["volatility"][mask].std(),
            ))

    # Save for plotting
    save_data = {
        "dataset": args.dataset, "seed": args.seed,
        "routing_weights": all_weights.tolist(),
        "top1_expert": top1_expert.tolist(),
        "expert_names": HEAD_NAMES,
        "trend_slope": stats["trend_slope"].tolist(),
        "amplitude": stats["amplitude"].tolist(),
        "volatility": stats["volatility"].tolist(),
        "mean_level": stats["mean_level"].tolist(),
        "n_test": len(X_test),
        "mse": result["mse"],
    }
    path = "results/routing_analysis/%s_%d.json" % (args.dataset, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved to %s" % path)

    # Generate matplotlib figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        expert_labels = [HEAD_NAMES[k] for k in range(5)]

        # Plot 1: trend vs amplitude, colored by top-1 expert
        ax = axes[0]
        for k in range(5):
            mask = (top1_expert == k)
            if mask.sum() > 0:
                ax.scatter(stats["trend_slope"][mask], stats["amplitude"][mask],
                           c=colors[k], alpha=0.3, s=8, label=expert_labels[k])
        ax.set_xlabel("Trend Slope")
        ax.set_ylabel("Amplitude (max - min)")
        ax.set_title("%s: Expert Selection by Signal Shape" % args.dataset)
        ax.legend(fontsize=8, markerscale=3)

        # Plot 2: volatility vs mean_level, colored by top-1 expert
        ax = axes[1]
        for k in range(5):
            mask = (top1_expert == k)
            if mask.sum() > 0:
                ax.scatter(stats["volatility"][mask], stats["mean_level"][mask],
                           c=colors[k], alpha=0.3, s=8, label=expert_labels[k])
        ax.set_xlabel("Volatility (diff std)")
        ax.set_ylabel("Mean Level")
        ax.set_title("%s: Expert Selection by Signal Statistics" % args.dataset)
        ax.legend(fontsize=8, markerscale=3)

        plt.tight_layout()
        fig_path = "figures/routing_analysis_%s_%d.pdf" % (args.dataset, args.seed)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print("Figure saved to %s" % fig_path)
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping figure generation")


if __name__ == "__main__":
    main()
