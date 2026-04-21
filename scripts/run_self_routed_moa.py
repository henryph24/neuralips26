"""Self-Routed Mixture of Adapters (SR-MoA).

Eliminates the external router entirely. Each expert has an internal gate
that determines its activation from raw input — experts route themselves.

Inspired by ERMoE (arXiv:2511.10971), AoE (arXiv:2501.13074, ICML 2025),
and Routing-Free MoE (arXiv:2604.00801). Extends the RR-MoA diagnosis:
if normalization kills external routers, the principled fix is to eliminate
the router and let each expert self-determine its activation on raw input.

Three self-routing modes:
  - gated:      per-expert sigmoid gate on raw input (primary)
  - eigenbasis: cosine similarity to learned orthogonal basis vectors
  - hybrid:     sigmoid of cosine similarity (combines both)

Usage:
    python scripts/run_self_routed_moa.py --dataset ETTh1 --routing-mode gated
    python scripts/run_self_routed_moa.py --dataset ETTh1 --routing-mode eigenbasis
"""

import argparse
import json
import math
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


# --- Expert adapter heads (same as RR-MoA for fair comparison) ---

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


class SelfRoutedMoA(nn.Module):
    """Self-Routed Mixture of Adapters.

    No external router. Each expert determines its own activation weight
    from raw input via one of three mechanisms:
      - gated:      sigmoid(MLP_k(x_raw)) per expert
      - eigenbasis: softmax(cos_sim(proj(x_raw), basis_k) / tau)
      - hybrid:     sigmoid(cos_sim(proj(x_raw), basis_k) / tau)
    """

    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64,
                 routing_mode="gated", gate_hidden=16, basis_dim=64,
                 temperature=0.1, gate_init_bias=0.0,
                 routing_input="raw"):
        super().__init__()
        self.K = K
        self.routing_mode = routing_mode
        self.routing_input = routing_input
        self.temperature = temperature
        self.d_model = d_model

        # Expert heads (same canonical pool as RR-MoA)
        self.adapters = nn.ModuleList()
        self._expert_names = []
        for i in range(K):
            cls = HEAD_CLASSES[i % len(HEAD_CLASSES)]
            self.adapters.append(cls(d_model, output_dim, hidden))
            self._expert_names.append(HEAD_NAMES[i % len(HEAD_NAMES)])

        # Self-routing components
        if routing_mode == "gated":
            # Each expert has its own gate: raw_input -> scalar activation
            self.gates = nn.ModuleList()
            for _ in range(K):
                if gate_hidden > 1:
                    gate = nn.Sequential(
                        nn.Linear(input_len, gate_hidden),
                        nn.GELU(),
                        nn.Linear(gate_hidden, 1),
                    )
                else:
                    gate = nn.Linear(input_len, 1)
                self.gates.append(gate)
            # Initialize gate bias
            if gate_init_bias != 0.0:
                for gate in self.gates:
                    last = gate[-1] if isinstance(gate, nn.Sequential) else gate
                    nn.init.constant_(last.bias, gate_init_bias)

        elif routing_mode in ("eigenbasis", "hybrid"):
            # Shared projection + per-expert learned basis vectors
            self.input_proj = nn.Linear(input_len, basis_dim)
            self.basis_vectors = nn.Parameter(torch.empty(K, basis_dim))
            # Initialize orthogonally for diversity
            if K <= basis_dim:
                nn.init.orthogonal_(self.basis_vectors)
            else:
                nn.init.normal_(self.basis_vectors, std=1.0 / math.sqrt(basis_dim))

        else:
            raise ValueError("Unknown routing_mode: %s" % routing_mode)

    def _compute_weights(self, raw_input):
        """Compute per-expert activation weights from raw input."""
        if self.routing_mode == "gated":
            gate_vals = []
            for gate in self.gates:
                g = torch.sigmoid(gate(raw_input))  # (B, 1)
                gate_vals.append(g)
            gates = torch.cat(gate_vals, dim=1)  # (B, K)
            # Renormalize to sum to 1
            weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
            return weights, gates  # weights for mixing, gates for diagnostics

        elif self.routing_mode == "eigenbasis":
            z = self.input_proj(raw_input)  # (B, basis_dim)
            z_norm = F.normalize(z, dim=-1)
            b_norm = F.normalize(self.basis_vectors, dim=-1)  # (K, basis_dim)
            scores = z_norm @ b_norm.T  # (B, K) cosine similarities
            weights = F.softmax(scores / self.temperature, dim=-1)
            return weights, scores

        elif self.routing_mode == "hybrid":
            z = self.input_proj(raw_input)  # (B, basis_dim)
            z_norm = F.normalize(z, dim=-1)
            b_norm = F.normalize(self.basis_vectors, dim=-1)
            scores = z_norm @ b_norm.T  # (B, K) cosine similarities
            gates = torch.sigmoid(scores / self.temperature)  # (B, K)
            weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
            return weights, gates

    def forward(self, hidden_states, raw_input):
        """Forward pass: experts self-route on raw input.

        Args:
            hidden_states: (B, T, d_model) from backbone
            raw_input: (B, input_len) raw time series
        Returns:
            prediction: (B, output_dim)
        """
        # Get routing input (supports hidden-state control experiment)
        if self.routing_input == "hidden":
            routing_in = hidden_states.mean(dim=1)  # (B, d_model) — collapse control
        else:
            routing_in = raw_input  # (B, input_len) — default

        weights, _ = self._compute_weights(routing_in)  # (B, K)
        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)  # (B, K, H)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)  # (B, H)

    def get_routing_stats(self, raw_input):
        """Return normalized routing weights for diagnostics."""
        with torch.no_grad():
            weights, _ = self._compute_weights(raw_input)
        return weights

    def get_gate_diagnostics(self, raw_input):
        """Return raw gate values and detailed diagnostics."""
        with torch.no_grad():
            weights, raw_gates = self._compute_weights(raw_input)
        return {
            "weights": weights,
            "raw_gates": raw_gates,
        }

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def routing_param_count(self):
        """Count parameters used for routing only (not expert heads)."""
        if self.routing_mode == "gated":
            return sum(p.numel() for gate in self.gates for p in gate.parameters())
        else:
            count = sum(p.numel() for p in self.input_proj.parameters())
            count += self.basis_vectors.numel()
            return count


def train_self_routed_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                          device="cuda", n_epochs=15, forecast_horizon=96,
                          batch_size=128, backbone_type="moment", K=5, hidden=64,
                          routing_mode="gated", gate_hidden=16, basis_dim=64,
                          temperature=0.1, gate_init_bias=0.0,
                          routing_input="raw",
                          test_ch=None, scaler=None):
    """Train SR-MoA: self-routed mixture of adapters."""
    hdim = _get_hidden_dim(model)
    input_len = X_train.shape[1]

    adapter = SelfRoutedMoA(
        hdim, forecast_horizon, input_len=input_len, K=K, hidden=hidden,
        routing_mode=routing_mode, gate_hidden=gate_hidden, basis_dim=basis_dim,
        temperature=temperature, gate_init_bias=gate_init_bias,
        routing_input=routing_input,
    ).to(device)

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
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by)
                # No auxiliary losses — self-routing needs no load balancing

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    all_weights = []
    all_raw_gates = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(adapter(feat, bx_raw).float().cpu())
            tgts.append(by.cpu())
            diag = adapter.get_gate_diagnostics(bx_raw)
            all_weights.append(diag["weights"].cpu())
            all_raw_gates.append(diag["raw_gates"].cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    weights = torch.cat(all_weights)  # (N, K) normalized weights
    raw_gates = torch.cat(all_raw_gates)  # (N, K) raw gate values

    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    # Routing diagnostics
    mean_weights = weights.mean(dim=0).tolist()
    # Entropy of normalized weights (analogous to routing entropy in RR-MoA)
    routing_entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
    routing_max = weights.max(dim=-1).values.mean().item()

    weights_np = weights.float().numpy()
    per_sample_std = float(np.mean(np.std(weights_np, axis=1)))
    cross_sample_var = float(np.mean(np.var(weights_np, axis=0)))

    # Gate-specific diagnostics
    raw_gates_np = raw_gates.float().numpy()
    per_expert_gate_mean = [float(raw_gates_np[:, k].mean()) for k in range(K)]
    per_expert_gate_std = [float(raw_gates_np[:, k].std()) for k in range(K)]
    # Activation frequency: fraction of samples where gate > 0.5 (for gated mode)
    if routing_mode == "gated":
        activation_freq = [float((raw_gates_np[:, k] > 0.5).mean()) for k in range(K)]
    else:
        activation_freq = [float((weights_np[:, k] > 1.0 / K).mean()) for k in range(K)]

    names = adapter._expert_names
    out = {
        "mse": mse, "mae": mae,
        "param_count": adapter.param_count(),
        "routing_param_count": adapter.routing_param_count(),
        "routing_mode": routing_mode,
        "routing": {names[i]: round(w, 3) for i, w in enumerate(mean_weights[:len(names)])},
        "routing_entropy": routing_entropy,
        "routing_max_weight": routing_max,
        "routing_per_sample_std": per_sample_std,
        "routing_cross_sample_var": cross_sample_var,
        "gate_diagnostics": {
            "per_expert_gate_mean": {names[i]: round(v, 4) for i, v in enumerate(per_expert_gate_mean)},
            "per_expert_gate_std": {names[i]: round(v, 4) for i, v in enumerate(per_expert_gate_std)},
            "activation_frequency": {names[i]: round(v, 3) for i, v in enumerate(activation_freq)},
        },
    }

    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d

    return out


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
    parser.add_argument("--routing-mode", default="gated",
                        choices=["gated", "eigenbasis", "hybrid"],
                        help="Self-routing mechanism: gated (per-expert sigmoid), "
                             "eigenbasis (cosine to learned basis), hybrid (sigmoid of cosine)")
    parser.add_argument("--gate-hidden", type=int, default=16,
                        help="Hidden dim for gated mode (1 = pure linear gate)")
    parser.add_argument("--basis-dim", type=int, default=64,
                        help="Basis vector dimension for eigenbasis/hybrid modes")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for eigenbasis/hybrid softmax/sigmoid scaling")
    parser.add_argument("--gate-init-bias", type=float, default=0.0,
                        help="Initial bias for gate output (negative = start inactive)")
    parser.add_argument("--routing-input", default="raw",
                        choices=["raw", "hidden"],
                        help="What the self-routing gates read: raw (default) or hidden (collapse control)")
    parser.add_argument("--unfreeze", default="frozen",
                        choices=["frozen", "last2", "last4", "all"],
                        help="Backbone unfreezing strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results/self_routed_moa")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, args.unfreeze)

    backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Unfreeze=%s, backbone trainable params=%d" % (args.unfreeze, backbone_trainable))

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    print("\nSR-MoA: %s K=%d mode=%s unfreeze=%s seed=%d" % (
        args.dataset, args.K, args.routing_mode, args.unfreeze, args.seed))
    if args.routing_mode == "gated":
        print("  gate_hidden=%d, gate_init_bias=%.1f" % (args.gate_hidden, args.gate_init_bias))
    else:
        print("  basis_dim=%d, temperature=%.3f" % (args.basis_dim, args.temperature))

    start = time.time()
    result = train_self_routed_moa(
        model, blocks, X_train, Y_train, X_test, Y_test,
        device=args.device, forecast_horizon=args.horizon,
        backbone_type=bb_type, K=args.K, n_epochs=args.epochs,
        batch_size=args.batch_size,
        routing_mode=args.routing_mode, gate_hidden=args.gate_hidden,
        basis_dim=args.basis_dim, temperature=args.temperature,
        gate_init_bias=args.gate_init_bias,
        routing_input=args.routing_input,
        test_ch=test_ch, scaler=scaler,
    )
    elapsed = time.time() - start

    print("SR-MoA: MSE=%.4f  MAE=%.4f  params=%d (routing=%d)  time=%.0fs" % (
        result["mse"], result["mae"], result["param_count"],
        result["routing_param_count"], elapsed))
    print("Routing: %s" % result["routing"])
    print("Routing entropy: %.3f / %.3f (max)" % (result["routing_entropy"], math.log(args.K)))
    print("Routing max weight: %.3f" % result["routing_max_weight"])
    print("Gate diagnostics: mean=%s" % result["gate_diagnostics"]["per_expert_gate_mean"])
    print("Gate diagnostics: freq=%s" % result["gate_diagnostics"]["activation_frequency"])

    # Build filename
    suffixes = [args.routing_mode]
    if args.routing_mode == "gated":
        suffixes.append("gh%d" % args.gate_hidden)
        if args.gate_init_bias != 0.0:
            suffixes.append("gib%.1f" % args.gate_init_bias)
    else:
        suffixes.append("bd%d" % args.basis_dim)
        suffixes.append("t%.2f" % args.temperature)
    if args.routing_input != "raw":
        suffixes.append("ri-%s" % args.routing_input)

    # Backbone suffix
    bb_lower = args.backbone.lower()
    if "moment" in bb_lower and "large" in bb_lower:
        suffixes.append("bb-moment-large")
    elif "moirai-moe" in bb_lower or "moirai_moe" in bb_lower:
        suffixes.append("bb-moirai-moe")
    elif "moirai" in bb_lower:
        suffixes.append("bb-moirai")
    elif "chronos" in bb_lower:
        suffixes.append("bb-chronos")
    elif args.backbone != "AutonLab/MOMENT-1-small":
        suffixes.append("bb-" + args.backbone.split("/")[-1].lower())

    suffix = "_".join(suffixes)
    path = "%s/%s_H%d_K%d_%s_%d_%s.json" % (
        args.results_dir, args.dataset, args.horizon, args.K,
        args.unfreeze, args.seed, suffix)

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
        "K": args.K, "unfreeze": args.unfreeze,
        "routing_mode": args.routing_mode,
        "routing_input": args.routing_input,
        "gate_hidden": args.gate_hidden if args.routing_mode == "gated" else None,
        "basis_dim": args.basis_dim if args.routing_mode != "gated" else None,
        "temperature": args.temperature if args.routing_mode != "gated" else None,
        "gate_init_bias": args.gate_init_bias,
        "backbone_trainable_params": backbone_trainable,
        "sr_moa": result,
        "elapsed": elapsed,
        "scaler": scaler_info,
    }
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
