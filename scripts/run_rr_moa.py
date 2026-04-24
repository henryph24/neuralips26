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
    compute_denorm_mse,
)
from feasibility.rrmoa_macro_experts import (
    MACRO_EXPERT_CLASSES, MACRO_EXPERT_NAMES,
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

# T3.A: selectable expert pool. ``canonical`` = 5 simple pooling heads
# (current RR-MoA default); ``macro`` = 5 AAS-discovered cross-domain
# motifs from feasibility/rrmoa_macro_experts.py that unify the AAS and
# RR-MoA contributions (W1).
EXPERT_POOLS = {
    "canonical": (HEAD_CLASSES, HEAD_NAMES),
    "macro":     (MACRO_EXPERT_CLASSES, MACRO_EXPERT_NAMES),
    # Diversity ablation: 5 identical experts (different init, same architecture)
    "identical-mean":   ([MeanPoolHead],       ["mean_%d" % i for i in range(5)]),
    "identical-conv1d": ([Conv1dPoolHead],      ["conv1d_%d" % i for i in range(5)]),
    "identical-attn":   ([AttentionPoolHead],   ["attn_%d" % i for i in range(5)]),
}


class RawRoutedMoA(nn.Module):
    """Raw-Routed Mixture of Adapters.

    Routes on RAW input (not hidden states) to avoid normalization cascade
    collapse (RevIN + LayerNorm). The router sees the original time series
    signal; the adapters see the backbone's hidden states.

    Supports Top-K sparse routing: only the top_k experts with highest
    routing probability execute per sample (top_k=K gives dense mode).
    """
    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64, top_k=None,
                 router_input_mode="raw", expert_pool="canonical", entropy_reg_coef=0.0,
                 router_temp=1.0, router_arch="conv", expert_dropout=0.0,
                 rdgf=False, saib_coef=0.0, alpha=0.0):
        super().__init__()
        self.K = K
        self.top_k = top_k if top_k is not None else K  # default: dense
        self.router_temp = router_temp
        self.router_arch = router_arch
        self.expert_dropout = expert_dropout
        self.rdgf = rdgf
        self.output_dim = output_dim
        assert router_input_mode in ("raw", "revin", "uniform", "partial", "shuffled", "hidden_reinjected"), router_input_mode
        self.router_input_mode = router_input_mode
        self.alpha = alpha
        self.entropy_reg_coef = entropy_reg_coef
        assert expert_pool in EXPERT_POOLS, expert_pool
        self.expert_pool = expert_pool

        pool_classes, pool_names = EXPERT_POOLS[expert_pool]
        self._expert_names = pool_names
        # K expert adapter heads (operate on hidden states)
        self.adapters = nn.ModuleList([
            pool_classes[i % len(pool_classes)](d_model, output_dim, hidden)
            for i in range(K)
        ])

        # RAW-INPUT router (operates on unnormalized time series)
        if router_arch == "ssr":
            # Sufficient Statistic Router: route on ONLY [μ, σ] — Prop 2 empirical proof
            self.router_head = nn.Linear(2, K)
        elif router_arch == "stats":
            # Stats router: route on hand-crafted [mean, std, range, slope, autocorr]
            self.router_head = nn.Linear(5, K)
        elif router_arch == "multiscale":
            # Multi-scale router: parallel Conv1d at k=4/16/64
            self.router_ms_4 = nn.Conv1d(1, 8, kernel_size=4, stride=4)
            self.router_ms_16 = nn.Conv1d(1, 8, kernel_size=16, stride=8)
            self.router_ms_64 = nn.Conv1d(1, 8, kernel_size=64, stride=32)
            self.router_head = nn.Linear(24, K)  # 3 scales × 8 channels pooled to 1
        elif router_arch == "fft":
            # Spectral-Temporal Router: parallel time + frequency branches
            self.router_time = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(4),  # (B, 16, 4)
            )
            self.router_fft_proj = nn.Linear(32, 32)
            self.router_head = nn.Linear(96, K)  # 64 time + 32 spectral
        elif router_arch == "linear":
            # Naive linear router: single Linear(input_len, K), no temporal conv
            # V1 ablation: tests whether raw signal alone suffices without
            # temporal feature extraction
            self.router_head = nn.Linear(input_len, K)
        else:
            # Default conv router
            self.router = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(4),  # (B, 16, 4)
            )
            self.router_head = nn.Linear(64, K)  # 16*4 = 64

        # Hidden-state + statistics re-injection router
        if router_input_mode == "hidden_reinjected":
            self.reinjected_head = nn.Linear(d_model + 2, K)

        # Shuffled routing: fixed temporal permutation (seeded at init)
        if router_input_mode == "shuffled":
            self.register_buffer('_shuffle_perm', torch.randperm(input_len))

        self.load_balance_coeff = 0.01

        # SAIB: Statistic-Aware Information Bottleneck auxiliary loss
        self.saib_coef = saib_coef
        if saib_coef > 0 and router_arch in ("conv", "fft"):
            self.saib_head = nn.Linear(64, 2)  # predict [mu, sigma] from router latent

    def _compute_logits(self, raw_input, hidden_states=None):
        if self.router_input_mode == "hidden_reinjected":
            # Re-inject stripped statistics into hidden-state routing input.
            # Tests: "why not just concatenate (μ, σ) to H instead of routing on raw?"
            assert hidden_states is not None, "hidden_reinjected requires hidden_states"
            h_mean = hidden_states.mean(dim=1)  # (B, d_model)
            mu = raw_input.mean(dim=-1, keepdim=True)  # (B, 1)
            sigma = raw_input.std(dim=-1, keepdim=True) + 1e-5  # (B, 1)
            x_cat = torch.cat([h_mean, mu, sigma], dim=-1)  # (B, d_model+2)
            return self.reinjected_head(x_cat)  # (B, K)
        if self.router_input_mode == "uniform":
            # T1.B ensemble-vs-specialization control: force a constant
            # uniform mixture by returning zero logits (softmax -> 1/K).
            # The router's Conv1d/Linear params are still present and get
            # zero gradient; this keeps the rest of the training loop (and
            # the checkpointable param set) identical to the raw / revin
            # variants so any MSE gap is attributable to the routing
            # decision alone, not to capacity differences.
            B = raw_input.shape[0]
            return torch.zeros(B, self.K, device=raw_input.device, dtype=raw_input.dtype)
        if self.router_input_mode == "partial":
            # Dose-response ablation: interpolate between raw and RevIN.
            # alpha=0 → pure raw (RR-MoA default), alpha=1 → pure RevIN.
            # Tests whether MI destruction is continuous and monotonic.
            mu = raw_input.mean(dim=-1, keepdim=True)
            sigma = raw_input.std(dim=-1, keepdim=True) + 1e-5
            revin = (raw_input - mu) / sigma
            x = (1 - self.alpha) * raw_input + self.alpha * revin
        elif self.router_input_mode == "shuffled":
            # Temporal shuffle ablation: destroys temporal ordering but
            # preserves distributional statistics (mean, variance, scale).
            # Tests whether routing depends on temporal patterns vs stats.
            x = raw_input[:, self._shuffle_perm]
        elif self.router_input_mode == "revin":
            # Per-window RevIN-style normalization: zero mean, unit variance
            # along the temporal dimension. This replicates what MOMENT's
            # internal RevIN would do if the router saw post-normalization
            # data, and strips per-window trend/amplitude/volatility. The
            # ablation tests whether routing gains come from rawness of the
            # signal or merely from bypassing the hidden states.
            mu = raw_input.mean(dim=-1, keepdim=True)
            sigma = raw_input.std(dim=-1, keepdim=True) + 1e-5
            x = (raw_input - mu) / sigma
        else:
            x = raw_input

        if self.router_arch == "ssr":
            # Sufficient Statistic Router: just [μ, σ] — 2 scalars, ~10 params
            mu = x.mean(dim=-1)
            sigma = x.std(dim=-1)
            return self.router_head(torch.stack([mu, sigma], dim=-1))

        if self.router_arch == "stats":
            # Hand-crafted statistics: [mean, std, range, slope, autocorr_lag1]
            mu = x.mean(dim=-1)
            sigma = x.std(dim=-1)
            rng = x.max(dim=-1).values - x.min(dim=-1).values
            # Linear regression slope
            t = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)
            t_centered = t - t.mean()
            slope = ((x - mu.unsqueeze(-1)) * t_centered).sum(dim=-1) / (t_centered ** 2).sum()
            # Autocorrelation at lag 1
            x_centered = x - mu.unsqueeze(-1)
            autocorr = (x_centered[:, :-1] * x_centered[:, 1:]).mean(dim=-1) / (sigma ** 2 + 1e-8)
            stats = torch.stack([mu, sigma, rng, slope, autocorr], dim=-1)  # (B, 5)
            return self.router_head(stats)

        if self.router_arch == "multiscale":
            x1 = x.unsqueeze(1)
            f4 = F.adaptive_avg_pool1d(F.gelu(self.router_ms_4(x1)), 1).flatten(1)   # (B, 8)
            f16 = F.adaptive_avg_pool1d(F.gelu(self.router_ms_16(x1)), 1).flatten(1)  # (B, 8)
            f64 = F.adaptive_avg_pool1d(F.gelu(self.router_ms_64(x1)), 1).flatten(1)  # (B, 8)
            return self.router_head(torch.cat([f4, f16, f64], dim=-1))

        if self.router_arch == "fft":
            x1 = x.unsqueeze(1)  # (B, 1, input_len)
            time_feat = self.router_time(x1).flatten(1)  # (B, 64)
            fft_amp = torch.fft.rfft(x, dim=-1).abs()  # (B, input_len//2 + 1)
            fft_top = fft_amp[:, 1:33]  # skip DC component, take bins 1-32
            spec_feat = F.gelu(self.router_fft_proj(fft_top))  # (B, 32)
            return self.router_head(torch.cat([time_feat, spec_feat], dim=-1))

        if self.router_arch == "linear":
            return self.router_head(x)  # (B, input_len) -> (B, K) directly

        # Default conv router
        x = x.unsqueeze(1)  # (B, 1, input_len)
        router_feat = self.router(x).flatten(1)  # (B, 64)
        return self.router_head(router_feat)  # (B, K)

    def forward(self, hidden_states, raw_input):
        """
        hidden_states: (B, T, d_model) from backbone
        raw_input: (B, input_len) raw time series
        """
        logits = self._compute_logits(raw_input, hidden_states=hidden_states)  # (B, K)
        if self.router_temp != 1.0:
            logits = logits / self.router_temp

        # Expert dropout during training: randomly zero out one expert's weight
        if self.training and self.expert_dropout > 0:
            drop_mask = torch.ones(logits.shape, device=logits.device)
            drop_idx = torch.randint(0, self.K, (logits.shape[0],), device=logits.device)
            if torch.rand(1).item() < self.expert_dropout:
                drop_mask[torch.arange(logits.shape[0]), drop_idx] = -1e9
                logits = logits + drop_mask

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

    def get_routing_stats(self, raw_input, hidden_states=None):
        """Full softmax routing weights for analysis (always dense)."""
        with torch.no_grad():
            logits = self._compute_logits(raw_input, hidden_states=hidden_states)
            return F.softmax(logits, dim=-1)

    def load_balance_loss(self, raw_input, hidden_states=None):
        logits = self._compute_logits(raw_input, hidden_states=hidden_states)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        p_i = F.softmax(logits, dim=-1).mean(dim=0)
        return self.K * (f_i * p_i).sum()

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def entropy_regularization(self, raw_input, hidden_states=None):
        """Negative entropy of the router distribution: -H(p).
        Minimizing this maximizes routing entropy (diversity)."""
        logits = self._compute_logits(raw_input, hidden_states=hidden_states)
        probs = F.softmax(logits, dim=-1)
        H = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1).mean()
        return -H

    def saib_loss(self, raw_input):
        """SAIB: force router latent to encode window [mu, sigma].
        Operationalizes Proposition 2 as an explicit training objective."""
        if self.saib_coef <= 0 or self.router_arch not in ("conv", "fft"):
            return torch.tensor(0.0, device=raw_input.device)
        x = raw_input
        if self.router_input_mode == "revin":
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True) + 1e-5
            x = (x - mu) / sigma
        if self.router_arch == "fft":
            router_feat = self.router_time(x.unsqueeze(1)).flatten(1)  # (B, 64)
        else:
            router_feat = self.router(x.unsqueeze(1)).flatten(1)  # (B, 64)
        pred_stats = self.saib_head(router_feat)  # (B, 2)
        true_mu = raw_input.mean(dim=-1)
        true_sigma = raw_input.std(dim=-1)
        true_stats = torch.stack([true_mu, true_sigma], dim=-1)
        return F.mse_loss(pred_stats, true_stats)


def train_rr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", K=5, hidden=64, top_k=None,
                 router_input_mode="raw", test_ch=None, scaler=None,
                 expert_pool="canonical", entropy_reg_coef=0.0, router_temp=1.0,
                 router_arch="conv", expert_dropout=0.0, rdgf=False,
                 saib_coef=0.0, freeze_router=False, alpha=0.0):
    """Train RR-MoA: raw-routed mixture of adapters."""
    hdim = _get_hidden_dim(model)
    adapter = RawRoutedMoA(
        hdim, forecast_horizon, input_len=512, K=K, hidden=hidden, top_k=top_k,
        router_input_mode=router_input_mode, expert_pool=expert_pool,
        entropy_reg_coef=entropy_reg_coef, router_temp=router_temp,
        router_arch=router_arch, expert_dropout=expert_dropout,
        rdgf=rdgf, saib_coef=saib_coef, alpha=alpha,
    ).to(device)

    # V1 ablation: freeze router at random init
    if freeze_router:
        for name, p in adapter.named_parameters():
            if "router" in name:
                p.requires_grad = False

    trainable = list(p for p in adapter.parameters() if p.requires_grad)
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
                if rdgf:
                    # RDGF: main loss uses detached H (no backbone gradient from weighted path)
                    pred = adapter(feat.detach(), bx_raw)
                    loss = mse_fn(pred, by)
                    # Auxiliary: uniform-weighted experts on live H (backbone gets uniform gradient)
                    outputs_live = torch.stack([a(feat) for a in adapter.adapters], dim=1)
                    aux_pred = outputs_live.mean(dim=1)
                    loss = loss + mse_fn(aux_pred, by)
                else:
                    pred = adapter(feat, bx_raw)
                    loss = mse_fn(pred, by)
                loss = loss + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw, hidden_states=feat)
                if adapter.entropy_reg_coef > 0:
                    loss = loss + adapter.entropy_reg_coef * adapter.entropy_regularization(bx_raw, hidden_states=feat)
                if adapter.saib_coef > 0:
                    loss = loss + adapter.saib_coef * adapter.saib_loss(bx_raw)

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
            all_routing.append(adapter.get_routing_stats(bx_raw, hidden_states=feat).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    mean_routing = routing.mean(dim=0).tolist()
    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    routing_max = routing.max(dim=-1).values.mean().item()

    # T1.B analysis support: per-sample routing-weight variance across the
    # test set. A near-zero value indicates the router is effectively
    # constant, in which case RR-MoA is operating as a soft ensemble rather
    # than a per-sample mixture of specialists.
    routing_np = routing.float().numpy()
    per_sample_std = float(np.mean(np.std(routing_np, axis=1)))
    cross_sample_var = float(np.mean(np.var(routing_np, axis=0)))

    names = adapter._expert_names
    out = {
        "mse": mse, "mae": mae, "param_count": adapter.param_count(),
        "top_k": adapter.top_k,
        "router_input_mode": adapter.router_input_mode,
        "expert_pool": adapter.expert_pool,
        "routing": {names[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(names)])},
        "routing_entropy": routing_entropy,
        "routing_max_weight": routing_max,
        "routing_per_sample_std": per_sample_std,
        "routing_cross_sample_var": cross_sample_var,
    }

    # T1.A denormalized MSE in original (un-standardized) units.
    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d

    return out


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
    parser.add_argument("--router-input-mode", default="raw",
                        choices=["raw", "revin", "uniform", "partial", "shuffled", "hidden_reinjected"],
                        help="Signal the gate reads: raw (channel-standardized input, default); "
                             "revin (per-window zero-mean unit-variance); "
                             "uniform (no routing, fixed 1/K weights); "
                             "partial (alpha-interpolated raw/revin, dose-response ablation); "
                             "shuffled (temporally permuted raw, mechanism ablation); "
                             "hidden_reinjected (mean-pooled H + [mu, sigma] from raw).")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Normalization dose for --router-input-mode partial. "
                             "0.0 = pure raw, 1.0 = pure RevIN. Intermediate values "
                             "interpolate linearly.")
    parser.add_argument("--unfreeze", default="last4", choices=["frozen", "last2", "last4", "all"],
                        help="Backbone unfreezing strategy")
    parser.add_argument("--expert-pool", default="canonical",
                        choices=list(EXPERT_POOLS.keys()),
                        help="Which expert pool to populate RR-MoA with. "
                             "'canonical' = 5 simple pooling heads (mean/last/max/attn/conv1d, "
                             "current default). 'macro' = 5 AAS-distilled cross-domain motifs "
                             "from feasibility/rrmoa_macro_experts.py (BN+mean, multi-scale "
                             "conv, Conv1d+BN+residual, depthwise separable, gated conv). "
                             "The 'macro' option is the T3.A integration experiment that "
                             "unifies the AAS and RR-MoA contributions (reviewer W1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training/eval batch size (reduce for large backbones)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline evaluation (faster for ablation sweeps)")
    parser.add_argument("--disable-revin", action="store_true",
                        help="Controlled ablation: disable RevIN inside MOMENT backbone")
    parser.add_argument("--entropy-reg", type=float, default=0.0,
                        help="Entropy reg coefficient for the raw router (B9 symmetry test)")
    parser.add_argument("--router-temp", type=float, default=1.0,
                        help="Router softmax temperature (>1 softer, <1 sharper)")
    parser.add_argument("--router-arch", default="conv",
                        choices=["conv", "stats", "multiscale", "ssr", "fft", "linear"],
                        help="Router architecture: conv (default), stats (hand-crafted), multiscale, fft (spectral-temporal), linear (naive V1 ablation)")
    parser.add_argument("--freeze-router", action="store_true",
                        help="Freeze router weights at random init (V1 ablation: tests whether router must learn)")
    parser.add_argument("--saib-coef", type=float, default=0.0,
                        help="SAIB auxiliary loss coefficient (0 = disabled). Forces router latent to encode window mu/sigma.")
    parser.add_argument("--expert-dropout", type=float, default=0.0,
                        help="Expert dropout probability during training")
    parser.add_argument("--rdgf", action="store_true",
                        help="Router-Detached Gradient Flow: unfreeze backbone with uniform gradient")
    args = parser.parse_args()

    os.makedirs("results/rr_moa", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_backbone(args.backbone, args.device,
                          disable_revin=args.disable_revin)
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
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    # === RR-MoA ===
    top_k_label = "top%d" % args.top_k if args.top_k else "dense"
    print("\nRR-MoA: %s K=%d %s unfreeze=%s seed=%d router_input=%s" % (
        args.dataset, args.K, top_k_label, args.unfreeze, args.seed, args.router_input_mode))
    start = time.time()
    result = train_rr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                          device=args.device, forecast_horizon=args.horizon,
                          backbone_type=bb_type, K=args.K, top_k=args.top_k,
                          n_epochs=args.epochs, router_input_mode=args.router_input_mode,
                          test_ch=test_ch, scaler=scaler,
                          expert_pool=args.expert_pool,
                          batch_size=args.batch_size,
                          entropy_reg_coef=args.entropy_reg,
                          router_temp=args.router_temp,
                          router_arch=args.router_arch,
                          expert_dropout=args.expert_dropout,
                          rdgf=args.rdgf,
                          saib_coef=args.saib_coef,
                          freeze_router=args.freeze_router,
                          alpha=args.alpha)
    elapsed = time.time() - start

    print("RR-MoA: MSE=%.4f  params=%d  time=%.0fs" % (result["mse"], result["param_count"], elapsed))
    if "mse_denorm" in result:
        print("RR-MoA: MSE_denorm=%.4f  MAE_denorm=%.4f (original units)" % (
            result["mse_denorm"], result["mae_denorm"]))
    print("Routing: %s" % result["routing"])
    print("Routing entropy: %.3f / %.3f (max)" % (result["routing_entropy"], np.log(args.K)))
    print("Routing max weight: %.3f (1.0 = collapsed)" % result["routing_max_weight"])
    print("Routing per-sample std: %.4f  cross-sample var: %.6f" % (
        result["routing_per_sample_std"], result["routing_cross_sample_var"]))

    # === Baselines ===
    baseline_results = {}
    if not args.no_baselines:
        print("\nBaselines (%d epochs, unfreeze=%s):" % (args.epochs, args.unfreeze))
        model2 = load_backbone(args.backbone, args.device,
                               disable_revin=args.disable_revin)
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
                                   backbone_type=bb_type, eval_ch=test_ch, scaler=scaler)
                baseline_results[name] = tr
                if "mse_denorm" in tr:
                    print("  %-15s MSE=%.4f  MSE_denorm=%.4f" % (name, tr["mse"], tr["mse_denorm"]))
                else:
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

    # Record the per-channel scale vector so that verify.py / downstream
    # analysis can reconstruct denorm numbers without re-loading the dataset.
    scaler_info = None
    if scaler is not None:
        scaler_info = {
            "scale_": [float(x) for x in scaler.scale_],
            "mean_": [float(x) for x in scaler.mean_],
            "n_features_in_": int(getattr(scaler, "n_features_in_", len(scaler.scale_))),
            "mean_scale_sq": float(np.mean(scaler.scale_ ** 2)),
        }

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "K": args.K, "top_k": args.top_k or args.K, "unfreeze": args.unfreeze,
        "router_input_mode": args.router_input_mode,
        "disable_revin": args.disable_revin,
        "backbone_trainable_params": backbone_trainable,
        "rr_moa": result, "elapsed": elapsed,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner, "delta_pct": delta,
        "alpha": args.alpha,
        "scaler": scaler_info,
    }
    # Append router mode / pool / backbone suffixes only for non-default
    # options so existing raw+canonical JSONs keep their current filenames
    # (and evidence_vm/verify.py's paths keep working).
    suffixes = []
    if args.disable_revin:
        suffixes.append("no-revin")
    if args.router_input_mode != "raw":
        suffixes.append("router-%s" % args.router_input_mode)
    if args.expert_pool != "canonical":
        suffixes.append("pool-%s" % args.expert_pool)
    if args.entropy_reg > 0:
        suffixes.append("entreg-%.3g" % args.entropy_reg)
    if args.router_temp != 1.0:
        suffixes.append("temp-%.3g" % args.router_temp)
    if args.router_arch != "conv":
        suffixes.append("rarch-%s" % args.router_arch)
    if args.expert_dropout > 0:
        suffixes.append("edrop-%.2g" % args.expert_dropout)
    if args.rdgf:
        suffixes.append("rdgf")
    if args.freeze_router:
        suffixes.append("frozenrouter")
    if args.saib_coef > 0:
        suffixes.append("saib-%.3g" % args.saib_coef)
    if args.epochs != 15:
        suffixes.append("ep%d" % args.epochs)
    if args.router_input_mode == "partial":
        suffixes.append("alpha-%.2f" % args.alpha)
    # Backbone suffix for non-default backbones
    bb_lower = args.backbone.lower()
    if "moment" in bb_lower and "large" in bb_lower:
        suffixes.append("bb-moment-large")
    elif "moirai-moe" in bb_lower or "moirai_moe" in bb_lower or "moiraimoe" in bb_lower:
        suffixes.append("bb-moirai-moe")
    elif "moirai" in bb_lower:
        suffixes.append("bb-moirai")
    elif "chronos" in bb_lower:
        suffixes.append("bb-chronos")
    elif args.backbone != "AutonLab/MOMENT-1-small":
        suffixes.append("bb-" + args.backbone.split("/")[-1].lower())
    suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    path = "results/rr_moa/%s_H%d_K%d_%s_%s_%d%s.json" % (
        args.dataset, args.horizon, args.K, top_k_label, args.unfreeze,
        args.seed, suffix)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
