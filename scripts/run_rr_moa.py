"""Raw-Routed Mixture of Adapters (RR-MoA).

Fixes the AdaMix/MoE failure: standard routers collapse because the TSFM
normalization cascade (RevIN + LayerNorm) destroys hidden-state heterogeneity.
RR-MoA routes on the RAW input instead, preserving the spectral and
statistical diversity needed for per-sample adapter selection.

Key insight: the backbone processes deep semantics, while a lightweight
router reads the raw signal to select the adapter topology per-sample.

Usage:
    python scripts/run_rr_moa.py --dataset ETTh1
    python scripts/run_rr_moa.py --dataset ETTh1 --top-k 2 --unfreeze frozen
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


# --- Expert adapter heads (same as AdaMix) ---

class MeanPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.net(h.mean(dim=1))

class LastTokenHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.net(h[:, -1, :])

class MaxPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.net(h.max(dim=1).values)

class AttentionPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return self.net((h * w).sum(dim=1))

class Conv1dPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=8, stride=4, padding=2)
        self.out = nn.Linear(hidden, output_dim)
    def forward(self, h):
        x = F.gelu(self.conv(h.permute(0, 2, 1)))
        return self.out(x.mean(dim=2))

HEAD_CLASSES = [MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead]
HEAD_NAMES = ["mean", "last", "max", "attention", "conv1d"]


class RawRoutedMoA(nn.Module):
    """Raw-Routed Mixture of Adapters.

    Routes on RAW input (not hidden states) to avoid normalization cascade
    collapse (RevIN + LayerNorm). The router sees the original time series
    signal; the adapters see the backbone's hidden states.

    Supports Top-K sparse routing: only the top_k experts with highest
    routing probability execute per sample (top_k=K gives dense mode).
    """
    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64, top_k=None):
        super().__init__()
        self.K = K
        self.top_k = top_k if top_k is not None else K  # default: dense
        self.output_dim = output_dim

        # K expert adapter heads (operate on hidden states)
        self.adapters = nn.ModuleList([
            HEAD_CLASSES[i % len(HEAD_CLASSES)](d_model, output_dim, hidden)
            for i in range(K)
        ])

        # RAW-INPUT router (operates on unnormalized time series)
        # Lightweight: small conv + global pool + linear
        self.router = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),  # (B, 16, 4)
        )
        self.router_head = nn.Linear(64, K)  # 16*4 = 64

        self.load_balance_coeff = 0.01

    def _compute_logits(self, raw_input):
        x_raw = raw_input.unsqueeze(1)  # (B, 1, input_len)
        router_feat = self.router(x_raw).flatten(1)  # (B, 64)
        return self.router_head(router_feat)  # (B, K)

    def forward(self, hidden_states, raw_input):
        """
        hidden_states: (B, T, d_model) from backbone
        raw_input: (B, input_len) raw time series
        """
        logits = self._compute_logits(raw_input)  # (B, K)

        if self.top_k >= self.K:
            # Dense mode: all experts
            weights = F.softmax(logits, dim=-1)  # (B, K)
            outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)
            return (weights.unsqueeze(-1) * outputs).sum(dim=1)

        # Sparse Top-K routing
        B = hidden_states.shape[0]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, top_k)
        weights = F.softmax(topk_vals, dim=-1)  # (B, top_k) normalized over selected

        result = torch.zeros(B, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(self.top_k):
            expert_ids = topk_idx[:, i]  # (B,)
            w = weights[:, i].unsqueeze(-1)  # (B, 1)
            for k in range(self.K):
                mask = (expert_ids == k)
                if mask.any():
                    result[mask] += w[mask] * self.adapters[k](hidden_states[mask])

        return result

    def get_routing_stats(self, raw_input):
        """Full softmax routing weights for analysis (always dense)."""
        with torch.no_grad():
            logits = self._compute_logits(raw_input)
            return F.softmax(logits, dim=-1)

    def load_balance_loss(self, raw_input):
        logits = self._compute_logits(raw_input)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        p_i = F.softmax(logits, dim=-1).mean(dim=0)
        return self.K * (f_i * p_i).sum()

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def train_rr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", K=5, hidden=64, top_k=None):
    """Train RR-MoA: raw-routed mixture of adapters."""
    hdim = _get_hidden_dim(model)
    adapter = RawRoutedMoA(hdim, forecast_horizon, input_len=512, K=K, hidden=hidden, top_k=top_k).to(device)

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

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            bx_raw = bx.to(device)  # (B, 512) raw input
            bx_enc = bx.to(device).unsqueeze(1)  # (B, 1, 512) for backbone
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by) + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    all_routing = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(adapter(feat, bx_raw).float().cpu())
            tgts.append(by.cpu())
            all_routing.append(adapter.get_routing_stats(bx_raw).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    mean_routing = routing.mean(dim=0).tolist()
    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    routing_max = routing.max(dim=-1).values.mean().item()

    return {
        "mse": mse, "mae": mae, "param_count": adapter.param_count(),
        "top_k": adapter.top_k,
        "routing": {HEAD_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(HEAD_NAMES)])},
        "routing_entropy": routing_entropy,
        "routing_max_weight": routing_max,
    }


def _apply_unfreeze(blocks, unfreeze):
    """Selectively unfreeze encoder blocks."""
    n = len(blocks)
    if unfreeze == "frozen":
        return  # all frozen
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
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-K sparse routing (default: dense, all K experts)")
    parser.add_argument("--unfreeze", default="last4", choices=["frozen", "last2", "last4", "all"],
                        help="Backbone unfreezing strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline evaluation (faster for ablation sweeps)")
    args = parser.parse_args()

    os.makedirs("results/rr_moa", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, args.unfreeze)

    n_unfrozen = sum(1 for b in blocks for p in b.parameters() if p.requires_grad) > 0
    backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Unfreeze=%s, backbone trainable params=%d" % (args.unfreeze, backbone_trainable))

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    # === RR-MoA ===
    top_k_label = "top%d" % args.top_k if args.top_k else "dense"
    print("\nRR-MoA: %s K=%d %s unfreeze=%s seed=%d" % (
        args.dataset, args.K, top_k_label, args.unfreeze, args.seed))
    start = time.time()
    result = train_rr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                          device=args.device, forecast_horizon=args.horizon,
                          backbone_type=bb_type, K=args.K, top_k=args.top_k,
                          n_epochs=args.epochs)
    elapsed = time.time() - start

    print("RR-MoA: MSE=%.4f  params=%d  time=%.0fs" % (result["mse"], result["param_count"], elapsed))
    print("Routing: %s" % result["routing"])
    print("Routing entropy: %.3f / %.3f (max)" % (result["routing_entropy"], np.log(args.K)))
    print("Routing max weight: %.3f (1.0 = collapsed)" % result["routing_max_weight"])

    # === Baselines ===
    baseline_results = {}
    if not args.no_baselines:
        print("\nBaselines (%d epochs, unfreeze=%s):" % (args.epochs, args.unfreeze))
        model2 = load_backbone(args.backbone, args.device)
        _disable_gradient_checkpointing(model2)
        blocks2 = _get_encoder_blocks(model2)
        for p in model2.parameters():
            p.requires_grad = False
        _apply_unfreeze(blocks2, args.unfreeze)

        baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
        for name, code in baselines.items():
            try:
                tr = train_adapter(code, model2, blocks2, X_train, Y_train, X_test, Y_test,
                                   device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
                                   backbone_type=bb_type)
                baseline_results[name] = tr
                print("  %-15s MSE=%.4f" % (name, tr["mse"]))
            except Exception as e:
                print("  %-15s ERROR: %s" % (name, e))

    if baseline_results:
        best_bl = min(baseline_results.values(), key=lambda x: x["mse"])["mse"]
        best_bl_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"])
        delta = (result["mse"] - best_bl) / best_bl * 100
        winner = "RR-MoA" if result["mse"] < best_bl else "BASELINE"
        print("\n>>> %s wins: RR-MoA=%.4f vs %s=%.4f  delta=%+.1f%%" % (
            winner, result["mse"], best_bl_name, best_bl, delta))
    else:
        winner = "N/A"
        delta = 0.0

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "K": args.K, "top_k": args.top_k or args.K, "unfreeze": args.unfreeze,
        "backbone_trainable_params": backbone_trainable,
        "rr_moa": result, "elapsed": elapsed,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner, "delta_pct": delta,
    }
    path = "results/rr_moa/%s_H%d_K%d_%s_%s_%d.json" % (
        args.dataset, args.horizon, args.K, top_k_label, args.unfreeze, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
