"""Gap-Closing RR-MoA Variants.

Four architectural directions to close the DLinear gap by giving experts
access to raw input information (not just backbone hidden states):

  1. dual-stream  — each expert blends backbone + raw branches (learnable alpha)
  2. film          — FiLM conditioning re-injects (mu, sigma) into hidden states
  3. raw-expert    — 6th expert operates on raw input, bypassing backbone
  4. multi-res     — each expert concatenates projected raw + pooled hidden

Usage:
    python scripts/run_gap_closing.py --variant dual-stream --dataset ETTh1 --seed 42
    python scripts/run_gap_closing.py --variant film --dataset Weather --seed 43
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
    load_standard_data, _detect_backbone_type, compute_denorm_mse,
)
from scripts.run_rr_moa import (
    MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead,
    HEAD_CLASSES, HEAD_NAMES,
)


# ---------------------------------------------------------------------------
# Wrappers to unify expert signatures: forward(hidden_states, raw_input)
# ---------------------------------------------------------------------------

class BackboneOnlyWrapper(nn.Module):
    """Wraps an existing head to accept (h, x) but ignore x."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden_states, raw_input):
        return self.head(hidden_states)


# ---------------------------------------------------------------------------
# Direction 1: Dual-Stream Experts
# ---------------------------------------------------------------------------

class DualStreamWrapper(nn.Module):
    """Each expert blends a backbone branch with a raw-input branch."""
    def __init__(self, backbone_head, input_len, output_dim):
        super().__init__()
        self.backbone_branch = backbone_head
        self.raw_branch = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, hidden_states, raw_input):
        h_out = self.backbone_branch(hidden_states)
        r_out = self.raw_branch(raw_input)
        a = torch.sigmoid(self.alpha)
        return a * h_out + (1 - a) * r_out


# ---------------------------------------------------------------------------
# Direction 2: FiLM Statistics Re-Injection
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """Re-inject (mu, sigma) into hidden states via FiLM modulation."""
    def __init__(self, d_model):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 2 * d_model),
        )

    def forward(self, hidden_states, raw_input):
        mu = raw_input.mean(dim=-1)       # (B,)
        sigma = raw_input.std(dim=-1)     # (B,)
        stats = torch.stack([mu, sigma], dim=-1)  # (B, 2)
        gamma_beta = self.film_net(stats)         # (B, 2*d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # each (B, d_model)
        # Modulate: H' = H * (1 + gamma) + beta
        return hidden_states * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Direction 3: Raw-Input Expert
# ---------------------------------------------------------------------------

class RawInputExpert(nn.Module):
    """Expert that bypasses the backbone, operating directly on raw input."""
    def __init__(self, input_len, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, hidden_states, raw_input):
        return self.net(raw_input)


# ---------------------------------------------------------------------------
# Direction 4: Multi-Resolution Feature Tapping
# ---------------------------------------------------------------------------

class MeanPoolModule(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
    def forward(self, h):
        return self.proj(h.mean(dim=1))

class LastTokenModule(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
    def forward(self, h):
        return self.proj(h[:, -1, :])

class MaxPoolModule(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
    def forward(self, h):
        return self.proj(h.max(dim=1).values)

class AttnPoolModule(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, hidden)
    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return self.proj((h * w).sum(dim=1))

class ConvPoolModule(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=8, stride=4, padding=2)
    def forward(self, h):
        x = F.gelu(self.conv(h.permute(0, 2, 1)))
        return x.mean(dim=2)

POOL_MODULES = [MeanPoolModule, LastTokenModule, MaxPoolModule, AttnPoolModule, ConvPoolModule]


class MultiResHead(nn.Module):
    """Expert that concatenates projected raw input + pooled hidden states."""
    def __init__(self, pool_cls, d_model, output_dim, input_len=512, hidden=64):
        super().__init__()
        self.raw_proj = nn.Linear(input_len, hidden)
        self.h_pool = pool_cls(d_model, hidden)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden * 2, output_dim),
        )

    def forward(self, hidden_states, raw_input):
        raw_feat = F.gelu(self.raw_proj(raw_input))       # (B, hidden)
        h_feat = self.h_pool(hidden_states)                # (B, hidden)
        return self.output(torch.cat([raw_feat, h_feat], dim=-1))


# ---------------------------------------------------------------------------
# Unified GapClosingMoA
# ---------------------------------------------------------------------------

class GapClosingMoA(nn.Module):
    """Gap-closing RR-MoA with selectable variant."""

    VARIANTS = ("dual-stream", "film", "raw-expert", "multi-res")

    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64,
                 top_k=2, variant="dual-stream"):
        super().__init__()
        assert variant in self.VARIANTS, f"Unknown variant: {variant}"
        self.variant = variant
        self.output_dim = output_dim

        # Build experts based on variant
        if variant == "dual-stream":
            self.adapters = nn.ModuleList([
                DualStreamWrapper(
                    HEAD_CLASSES[i](d_model, output_dim, hidden),
                    input_len, output_dim,
                ) for i in range(K)
            ])
            self._expert_names = [f"dual_{n}" for n in HEAD_NAMES]
            self.K = K

        elif variant == "film":
            self.film_conditioner = FiLMConditioner(d_model)
            self.adapters = nn.ModuleList([
                HEAD_CLASSES[i](d_model, output_dim, hidden)
                for i in range(K)
            ])
            self._expert_names = list(HEAD_NAMES)
            self.K = K

        elif variant == "raw-expert":
            # 5 backbone experts (wrapped) + 1 raw expert
            backbone_experts = [
                BackboneOnlyWrapper(HEAD_CLASSES[i](d_model, output_dim, hidden))
                for i in range(K)
            ]
            raw_exp = RawInputExpert(input_len, output_dim, hidden=128)
            self.adapters = nn.ModuleList(backbone_experts + [raw_exp])
            self._expert_names = list(HEAD_NAMES) + ["raw_bypass"]
            self.K = K + 1  # 6 experts

        elif variant == "multi-res":
            self.adapters = nn.ModuleList([
                MultiResHead(POOL_MODULES[i], d_model, output_dim, input_len, hidden)
                for i in range(K)
            ])
            self._expert_names = [f"mres_{n}" for n in HEAD_NAMES]
            self.K = K

        self.top_k = min(top_k, self.K)

        # Router: identical to RawRoutedMoA (Conv1d + pool + linear)
        self.router = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.router_head = nn.Linear(64, self.K)
        self.load_balance_coeff = 0.01

    def _compute_logits(self, raw_input):
        x = raw_input.unsqueeze(1)  # (B, 1, input_len)
        router_feat = self.router(x).flatten(1)  # (B, 64)
        return self.router_head(router_feat)      # (B, K)

    def forward(self, hidden_states, raw_input):
        logits = self._compute_logits(raw_input)

        if self.variant == "film":
            # Condition hidden states BEFORE expert dispatch
            hidden_states = self.film_conditioner(hidden_states, raw_input)

        if self.top_k >= self.K:
            # Dense mode
            weights = F.softmax(logits, dim=-1)
            if self.variant == "film":
                outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)
            else:
                outputs = torch.stack([a(hidden_states, raw_input) for a in self.adapters], dim=1)
            return (weights.unsqueeze(-1) * outputs).sum(dim=1)

        # Sparse Top-K routing
        B = hidden_states.shape[0]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        result = torch.zeros(B, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(self.top_k):
            expert_ids = topk_idx[:, i]
            w = weights[:, i].unsqueeze(-1)
            for k in range(self.K):
                mask = (expert_ids == k)
                if mask.any():
                    if self.variant == "film":
                        result[mask] += w[mask] * self.adapters[k](hidden_states[mask])
                    else:
                        result[mask] += w[mask] * self.adapters[k](hidden_states[mask], raw_input[mask])
        return result

    def get_routing_stats(self, raw_input):
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

    def get_alpha_values(self):
        """For dual-stream: return per-expert alpha (sigmoid) values."""
        if self.variant != "dual-stream":
            return None
        return {
            self._expert_names[i]: torch.sigmoid(self.adapters[i].alpha).item()
            for i in range(self.K)
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gap_closing(model, blocks, X_train, Y_train, X_test, Y_test,
                      device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                      backbone_type="moment", K=5, hidden=64, top_k=2,
                      variant="dual-stream", test_ch=None, scaler=None):
    """Train gap-closing MoA variant."""
    hdim = _get_hidden_dim(model)
    adapter = GapClosingMoA(
        hdim, forecast_horizon, input_len=512, K=K, hidden=hidden,
        top_k=top_k, variant=variant,
    ).to(device)

    print(f"  Variant: {variant}, K={adapter.K}, top_k={adapter.top_k}, "
          f"params={adapter.param_count():,}")

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
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by) + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/len(train_loader):.4f}")

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
    routing_np = routing.float().numpy()
    per_sample_std = float(np.mean(np.std(routing_np, axis=1)))
    cross_sample_var = float(np.mean(np.var(routing_np, axis=0)))

    names = adapter._expert_names
    out = {
        "mse": mse, "mae": mae, "param_count": adapter.param_count(),
        "variant": variant,
        "top_k": adapter.top_k, "K": adapter.K,
        "routing": {names[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(names)])},
        "routing_entropy": routing_entropy,
        "routing_max_weight": routing_max,
        "routing_per_sample_std": per_sample_std,
        "routing_cross_sample_var": cross_sample_var,
    }

    # Dual-stream: log alpha values
    alphas = adapter.get_alpha_values()
    if alphas:
        out["alpha_values"] = alphas

    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gap-Closing RR-MoA Variants")
    parser.add_argument("--variant", required=True,
                        choices=["dual-stream", "film", "raw-expert", "multi-res"])
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unfreeze", default="frozen",
                        choices=["frozen", "last2", "last4", "all"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Gap-Closing: {args.variant} | {args.dataset} | seed={args.seed}")
    print(f"{'='*60}")

    # Load backbone
    backbone_type = _detect_backbone_type(args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    if args.unfreeze != "frozen":
        n = len(blocks)
        if args.unfreeze == "last2":
            unfreeze_blocks = blocks[-2:]
        elif args.unfreeze == "last4":
            unfreeze_blocks = blocks[-4:]
        elif args.unfreeze == "all":
            unfreeze_blocks = blocks
        for blk in unfreeze_blocks:
            for p in blk.parameters():
                p.requires_grad = True

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon, max_samples=5000)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")

    t0 = time.time()
    result = train_gap_closing(
        model, blocks, X_train, Y_train, X_test, Y_test,
        device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
        batch_size=args.batch_size, backbone_type=backbone_type,
        K=args.K, hidden=args.hidden, top_k=args.top_k,
        variant=args.variant, test_ch=test_ch, scaler=scaler,
    )
    elapsed = time.time() - t0

    result.update({
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "backbone": args.backbone, "unfreeze": args.unfreeze, "elapsed": elapsed,
    })

    print(f"\n  Result: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}, "
          f"entropy={result['routing_entropy']:.3f}, params={result['param_count']:,}")
    if result.get("alpha_values"):
        print(f"  Alpha values: {result['alpha_values']}")

    # Save
    os.makedirs("results/gap_closing", exist_ok=True)
    out_path = f"results/gap_closing/{args.variant}_{args.dataset}_H{args.horizon}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
