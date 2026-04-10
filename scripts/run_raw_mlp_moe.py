"""Raw-MLP Mixture of Experts — Claim D ablation (no TSFM backbone at all).

A critical reviewer asked: "if the TSFM destroys the non-linear structure in
raw data (as the paper's own tab:diagnostic shows), isn't the raw-branch in
Dual-Stream doing 99% of the heavy lifting? The paper needs an ablation
showing a mixture of adapters operating ONLY on raw MLPs (deleting the 35M-
parameter TSFM entirely)."

This script answers that question directly:
  - 5 experts, each a size-diverse MLP operating directly on X_raw (512-dim)
  - Same raw-signal router architecture as RR-MoA (Conv1d + pool + linear)
  - Top-2 sparse routing (matches RR-MoA default)
  - NO TSFM, NO backbone forward, NO hidden states at all
  - Parameter budget ~446K (matches RR-MoA's 426K within ~5%)

Expected outcomes and interpretations:
  1. Raw-MLP MoE >> Dual-Stream  -> TSFM is actively hurting (catastrophic)
  2. Raw-MLP MoE ~  Dual-Stream  -> TSFM is dead weight (damaging)
  3. Raw-MLP MoE <  Dual-Stream  -> TSFM adds complementary signal (positive
                                    for the paper, consistent with alpha~0.49)
  4. Raw-MLP MoE ~  DLinear      -> Raw-MLP captures linear structure (baseline)

The most likely outcome is (3). The runtime is ~2-3 min per run (no backbone
forward), so 6 datasets * 3 seeds = 18 runs finishes in ~60 min wall-clock
even under contention with the B1/B2 sweeps.

Usage:
    python3 scripts/run_raw_mlp_moe.py --dataset ETTh1 --seed 42
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

from scripts.run_standard_evolution import load_standard_data, compute_denorm_mse


# --- 5 size-diverse raw-input MLP experts ------------------------------- #
#
# Hidden sizes chosen so that (a) each expert has a different capacity to
# force architectural diversity (rather than relying purely on init noise)
# and (b) the total parameter count lands close to RR-MoA's 426K.
#
# Param count per expert: 512*h + h + h*96 + 96 = 609h + 96
#   h=64  -> 39K
#   h=96  -> 58K
#   h=128 -> 78K
#   h=192 -> 117K
#   h=256 -> 156K
# Total: 448K (matches RR-MoA 426K within 5%).

EXPERT_HIDDEN_SIZES = [64, 96, 128, 192, 256]


class RawMLPExpert(nn.Module):
    """Two-layer MLP operating directly on the raw time-series window."""

    def __init__(self, input_len: int, hidden: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_len, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B, input_len)
        h = F.gelu(self.fc1(x_raw))
        return self.fc2(h)


class RawMLPMoE(nn.Module):
    """Mixture of 5 raw-input MLPs with a raw-signal router and Top-K routing.

    Crucially, this module takes NO hidden states, NO TSFM embeddings, and no
    backbone forward pass ever runs. Every computation is performed directly
    on the 512-dim raw time-series window.
    """

    def __init__(self, input_len: int = 512, output_dim: int = 96, K: int = 5, top_k: int = 2):
        super().__init__()
        self.K = K
        self.top_k = top_k
        self.output_dim = output_dim

        assert len(EXPERT_HIDDEN_SIZES) == K, \
            "EXPERT_HIDDEN_SIZES must have length K=%d" % K
        self.experts = nn.ModuleList([
            RawMLPExpert(input_len, h, output_dim) for h in EXPERT_HIDDEN_SIZES
        ])

        # Raw-signal router: identical architecture to RR-MoA's router in
        # scripts/run_rr_moa.py so the ONLY thing that changes vs RR-MoA is
        # whether the experts see hidden states or raw input. This ensures
        # the ablation isolates the "experts-on-raw" axis cleanly.
        self.router = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),  # -> (B, 16, 4)
        )
        self.router_head = nn.Linear(64, K)  # 16 * 4 = 64

        self.load_balance_coeff = 0.01

    def _compute_logits(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B, input_len). Add a singleton channel for Conv1d.
        x = x_raw.unsqueeze(1)  # (B, 1, input_len)
        router_feat = self.router(x).flatten(1)  # (B, 64)
        return self.router_head(router_feat)  # (B, K)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        logits = self._compute_logits(x_raw)  # (B, K)

        if self.top_k >= self.K:
            # Dense: all experts every sample
            weights = F.softmax(logits, dim=-1)  # (B, K)
            outputs = torch.stack([e(x_raw) for e in self.experts], dim=1)  # (B, K, H)
            return (weights.unsqueeze(-1) * outputs).sum(dim=1)  # (B, H)

        # Top-K sparse routing identical to RR-MoA
        B = x_raw.shape[0]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, top_k)
        weights = F.softmax(topk_vals, dim=-1)                 # (B, top_k)

        result = x_raw.new_zeros(B, self.output_dim)
        for i in range(self.top_k):
            expert_ids = topk_idx[:, i]  # (B,)
            w = weights[:, i].unsqueeze(-1)  # (B, 1)
            for k in range(self.K):
                mask = (expert_ids == k)
                if mask.any():
                    result[mask] += w[mask] * self.experts[k](x_raw[mask])
        return result

    def get_routing_stats(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Full dense softmax routing weights, for analysis only."""
        with torch.no_grad():
            logits = self._compute_logits(x_raw)
            return F.softmax(logits, dim=-1)

    def load_balance_loss(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Standard Switch-style load-balance auxiliary with alpha=0.01."""
        logits = self._compute_logits(x_raw)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        p_i = F.softmax(logits, dim=-1).mean(dim=0)
        return self.K * (f_i * p_i).sum()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Training loop ------------------------------------------------------- #


def train_raw_mlp_moe(X_train, Y_train, X_test, Y_test,
                      device="cuda", n_epochs=15, forecast_horizon=96,
                      batch_size=128, K=5, top_k=2,
                      test_ch=None, scaler=None):
    """Train a RawMLPMoE and return the test metrics."""
    input_len = X_train.shape[1]
    adapter = RawMLPMoE(
        input_len=input_len, output_dim=forecast_horizon, K=K, top_k=top_k,
    ).to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float(),
        ),
        batch_size=batch_size, shuffle=True,
    )

    for _ in range(n_epochs):
        adapter.train()
        for bx, by in train_loader:
            bx_raw = bx.to(device)  # (B, input_len)
            by = by.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                pred = adapter(bx_raw)
                loss = mse_fn(pred, by) \
                     + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    adapter.eval()
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(Y_test).float(),
        ),
        batch_size=batch_size,
    )

    preds, tgts, all_routing = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(device)
            preds.append(adapter(bx_raw).float().cpu())
            tgts.append(by)
            all_routing.append(adapter.get_routing_stats(bx_raw).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing)

    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    mean_routing = routing.mean(dim=0).tolist()
    entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    max_w = routing.max(dim=-1).values.mean().item()

    out = {
        "mse": mse,
        "mae": mae,
        "param_count": adapter.param_count(),
        "top_k": top_k,
        "expert_hidden_sizes": EXPERT_HIDDEN_SIZES,
        "mean_routing_weights": {
            "h%d" % EXPERT_HIDDEN_SIZES[i]: round(w, 3)
            for i, w in enumerate(mean_routing[:K])
        },
        "routing_entropy": entropy,
        "routing_max_weight": max_w,
    }

    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d

    return out


# --- CLI ----------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2,
                        help="Sparse routing top-k (default 2 to match RR-MoA).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results/raw_mlp_moe")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the same normalized data splits used by every other RR-MoA run
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_test)))

    print("\n" + "=" * 60)
    print("Raw-MLP MoE (NO TSFM): %s H=%d K=%d top%d seed=%d" % (
        args.dataset, args.horizon, args.K, args.top_k, args.seed))
    print("=" * 60)

    start = time.time()
    result = train_raw_mlp_moe(
        X_train, Y_train, X_test, Y_test,
        device=args.device, n_epochs=args.epochs,
        forecast_horizon=args.horizon, K=args.K, top_k=args.top_k,
        test_ch=test_ch, scaler=scaler,
    )
    elapsed = time.time() - start

    print("Raw-MLP MoE: MSE=%.4f  MAE=%.4f  params=%d  time=%.0fs" % (
        result["mse"], result["mae"], result["param_count"], elapsed))
    if "mse_denorm" in result:
        print("Raw-MLP MoE: MSE_denorm=%.4f (original units)" % result["mse_denorm"])
    print("Routing: %s" % result["mean_routing_weights"])
    print("Routing entropy: %.3f (max=%.3f for K=%d)" % (
        result["routing_entropy"], float(np.log(args.K)), args.K))

    scaler_info = None
    if scaler is not None:
        scaler_info = {
            "scale_": [float(x) for x in scaler.scale_],
            "mean_": [float(x) for x in scaler.mean_],
            "mean_scale_sq": float(np.mean(scaler.scale_ ** 2)),
        }
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "K": args.K,
        "top_k": args.top_k,
        "expert_hidden_sizes": EXPERT_HIDDEN_SIZES,
        "raw_mlp_moe": result,
        "elapsed": elapsed,
        "scaler": scaler_info,
        "has_tsfm_backbone": False,  # key ablation flag
    }
    top_k_label = "top%d" % args.top_k if args.top_k < args.K else "dense"
    path = "%s/%s_H%d_K%d_%s_%d.json" % (
        args.results_dir, args.dataset, args.horizon, args.K, top_k_label, args.seed,
    )
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
