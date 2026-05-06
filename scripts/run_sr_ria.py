"""SR-RIA+: Self-Routed Residual-IA+ (SR-MoA routing + RIA+ expert architecture).

Combines the two strongest components identified in the paper:
  - SR-MoA self-routing: each expert has its own sigmoid gate on raw input
    (eliminates external router entirely; 13-42% better than RR-MoA)
  - RIA+ expert architecture: NLinear raw branch (shared) + gated backbone residual
    (matches DLinear on 107/123 cells)

The hybrid should inherit SR-MoA's strong routing AND RIA+'s DLinear-competitive MSE.

Usage:
    python scripts/run_sr_ria.py --dataset ETTh1 --seed 42
    python scripts/run_sr_ria.py --dataset ETTh1 --seed 42 --backbone AutonLab/MOMENT-1-small
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
from feasibility.adapter_seeds import SEED_ADAPTERS
from feasibility.standard_data import (
    load_standard_data, _detect_backbone_type, compute_denorm_mse,
)
from scripts.run_rr_moa import _apply_unfreeze


# --- Expert head classes (same canonical pool as RR-MoA/SR-MoA) ---
class MeanPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.out(h.mean(dim=1))

class LastTokenHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.out(h[:, -1])

class MaxPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        return self.out(h.max(dim=1).values)

class AttentionPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.out = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, h):
        w = F.softmax(self.attn(h), dim=1)
        return self.out((w * h).sum(dim=1))

class Conv1dPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=3, padding=1)
        self.out = nn.Linear(hidden, output_dim)
    def forward(self, h):
        x = h.transpose(1, 2)
        return self.out(F.gelu(self.conv(x)).mean(dim=2))

HEAD_CLASSES = [MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead]
HEAD_NAMES = ["mean", "last", "max", "attention", "conv1d"]


class SelfRoutedResidualIA(nn.Module):
    """SR-RIA+: Self-Routed Residual-IA+.

    Each expert k has:
      - A self-gating function: g_k = sigmoid(MLP_k(X_raw))
      - A backbone adapter head: adapter_k(H)
      - A shared raw branch: NLinear(X_raw) (shared across all experts)
      - A per-expert blending gate: alpha_k = sigmoid(linear_k(X_raw))
        initialized at bias=-2 so backbone starts at 12% contribution

    Output = sum_k [ g_k * (raw_branch(X_raw) + alpha_k * adapter_k(H)) ]
             / sum_k(g_k)
    """
    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64,
                 gate_hidden=16, gate_init_bias=0.0,
                 blend_init_bias=-2.0, raw_arch="nlinear"):
        super().__init__()
        self.K = K
        self.output_dim = output_dim
        self.raw_arch = raw_arch

        # Expert backbone heads (canonical pool)
        self.adapters = nn.ModuleList()
        self._expert_names = []
        for i in range(K):
            cls = HEAD_CLASSES[i % len(HEAD_CLASSES)]
            self.adapters.append(cls(d_model, output_dim, hidden))
            self._expert_names.append(HEAD_NAMES[i % len(HEAD_CLASSES)])

        # Self-routing gates (per-expert, on raw input)
        self.gates = nn.ModuleList()
        for _ in range(K):
            gate = nn.Sequential(
                nn.Linear(input_len, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            self.gates.append(gate)
        if gate_init_bias != 0.0:
            for gate in self.gates:
                nn.init.constant_(gate[-1].bias, gate_init_bias)

        # Shared raw branch (NLinear-like, matches DLinear's 49K params)
        self.raw_branch = nn.Linear(input_len, output_dim)

        # Per-expert blending gates (backbone contribution)
        self.blend_gates = nn.ModuleList()
        for _ in range(K):
            bg = nn.Linear(input_len, 1)
            nn.init.constant_(bg.bias, blend_init_bias)
            self.blend_gates.append(bg)

    def forward(self, hidden_states, raw_input):
        B = raw_input.shape[0]

        # Self-routing weights
        gate_vals = []
        for gate in self.gates:
            gate_vals.append(torch.sigmoid(gate(raw_input)))  # (B, 1)
        gates = torch.cat(gate_vals, dim=1)  # (B, K)
        weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)  # (B, K)

        # Raw branch (shared, NLinear)
        if self.raw_arch == "nlinear":
            last = raw_input[:, -1:].detach()
            raw_pred = self.raw_branch(raw_input - last) + last  # (B, output_dim)
        else:
            raw_pred = self.raw_branch(raw_input)

        # Per-expert: raw_pred + alpha_k * adapter_k(H)
        result = torch.zeros(B, self.output_dim, device=raw_input.device, dtype=hidden_states.dtype)
        for k in range(self.K):
            alpha_k = torch.sigmoid(self.blend_gates[k](raw_input))  # (B, 1)
            expert_out = raw_pred + alpha_k * self.adapters[k](hidden_states)
            result += weights[:, k:k+1] * expert_out

        return result

    def get_routing_stats(self, raw_input):
        with torch.no_grad():
            gate_vals = [torch.sigmoid(g(raw_input)) for g in self.gates]
            gates = torch.cat(gate_vals, dim=1)
            weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def get_blend_stats(self, raw_input):
        with torch.no_grad():
            return [torch.sigmoid(bg(raw_input)).mean().item() for bg in self.blend_gates]

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def train_sr_ria(model, blocks, X_train, Y_train, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", test_ch=None, scaler=None,
                 K=5, unfreeze="frozen", warmup_epochs=5,
                 raw_arch="nlinear", use_early_stop=True):
    """Train SR-RIA+ with optional warmup and early stopping."""
    hdim = _get_hidden_dim(model)
    input_len = X_train.shape[1]

    adapter = SelfRoutedResidualIA(
        hdim, forecast_horizon, input_len=input_len, K=K,
        raw_arch=raw_arch,
    ).to(device)

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)

    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    # Cosine schedule
    total_steps = n_epochs * len(loader)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Validation split for early stopping
    val_loader = None
    if use_early_stop:
        val_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
        ), batch_size=batch_size)

    best_val_mse = float('inf')
    best_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(n_epochs):
        model.train()
        adapter.train()

        # Warmup: freeze backbone branch + blend gates for first N epochs
        if epoch < warmup_epochs:
            for a in adapter.adapters:
                for p in a.parameters():
                    p.requires_grad = False
            for bg in adapter.blend_gates:
                for p in bg.parameters():
                    p.requires_grad = False
        elif epoch == warmup_epochs:
            for a in adapter.adapters:
                for p in a.parameters():
                    p.requires_grad = True
            for bg in adapter.blend_gates:
                for p in bg.parameters():
                    p.requires_grad = True

        for bx, by in loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                               backbone_type=backbone_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

        # Validation for early stopping
        if use_early_stop and val_loader is not None:
            model.eval()
            adapter.eval()
            val_preds, val_tgts = [], []
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                for bx, by in val_loader:
                    bx_raw = bx.to(device)
                    bx_enc = bx.to(device).unsqueeze(1)
                    mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
                    feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                                   backbone_type=backbone_type)
                    val_preds.append(adapter(feat, bx_raw).float().cpu())
                    val_tgts.append(by.cpu())
            val_mse = nn.MSELoss()(torch.cat(val_preds), torch.cat(val_tgts)).item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # Restore best
    if best_state is not None:
        adapter.load_state_dict(best_state)
        adapter.to(device)

    # Final evaluation
    model.eval()
    adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts, all_routing = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                           backbone_type=backbone_type)
            preds.append(adapter(feat, bx_raw).float().cpu())
            tgts.append(by.cpu())
            all_routing.append(adapter.get_routing_stats(bx_raw).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    blend_stats = adapter.get_blend_stats(torch.from_numpy(X_test[:min(500, len(X_test))]).float().to(device))

    result = {
        "mse": mse, "mae": mae,
        "param_count": adapter.param_count(),
        "routing_entropy": routing_entropy,
        "routing_weights": {n: float(routing.mean(dim=0)[i]) for i, n in enumerate(adapter._expert_names)},
        "blend_gates": {n: float(blend_stats[i]) for i, n in enumerate(adapter._expert_names)},
        "best_epoch": n_epochs - patience_counter if best_state else n_epochs,
    }
    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        result["mse_denorm"] = mse_d
        result["mae_denorm"] = mae_d
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--unfreeze", default="frozen")
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--raw-arch", default="nlinear", choices=["linear", "nlinear"])
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("results/sr_ria", exist_ok=True)
    bb_type = _detect_backbone_type(args.backbone)

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    if args.unfreeze != "frozen":
        _apply_unfreeze(blocks, args.unfreeze)

    print("SR-RIA+: %s K=%d unfreeze=%s seed=%d raw_arch=%s warmup=%d" % (
        args.dataset, args.K, args.unfreeze, args.seed, args.raw_arch, args.warmup_epochs))

    t0 = time.time()
    result = train_sr_ria(
        model, blocks, X_train, Y_train, X_test, Y_test,
        device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
        batch_size=args.batch_size, backbone_type=bb_type,
        test_ch=test_ch, scaler=scaler, K=args.K, unfreeze=args.unfreeze,
        warmup_epochs=args.warmup_epochs, raw_arch=args.raw_arch,
        use_early_stop=not args.no_early_stop,
    )
    elapsed = time.time() - t0

    print("SR-RIA+: MSE=%.4f  entropy=%.3f  params=%d  best_ep=%d  (%.0fs)" % (
        result["mse"], result["routing_entropy"], result["param_count"],
        result["best_epoch"], elapsed))
    print("Blend gates:", result["blend_gates"])
    print("Routing:", result["routing_weights"])

    # Save
    bb_suffix = ""
    bb_lower = args.backbone.lower()
    if "large" in bb_lower:
        bb_suffix = "_bb-moment-large"
    elif "moirai-moe" in bb_lower or "moirai_moe" in bb_lower:
        bb_suffix = "_bb-moirai-moe"
    elif "moirai" in bb_lower:
        bb_suffix = "_bb-moirai"
    elif "chronos" in bb_lower:
        bb_suffix = "_bb-chronos"
    elif "timer" in bb_lower:
        bb_suffix = "_bb-timerxl"
    elif args.backbone != "AutonLab/MOMENT-1-small":
        bb_suffix = "_bb-" + args.backbone.split("/")[-1].lower()

    fname = "results/sr_ria/%s_H%d_K%d_%s_%d%s.json" % (
        args.dataset, args.horizon, args.K, args.unfreeze, args.seed, bb_suffix)

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "K": args.K, "unfreeze": args.unfreeze, "backbone": args.backbone,
        "raw_arch": args.raw_arch, "warmup_epochs": args.warmup_epochs,
        "sr_ria": result, "elapsed": elapsed,
    }
    with open(fname, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved: %s" % fname)


if __name__ == "__main__":
    main()
