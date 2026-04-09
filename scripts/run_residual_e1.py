"""
E1: Residual DLinear + Neural FM Correction

Tests whether a NEURAL adapter (trained with backprop on GPU) can extract
complementary signal from backbone features that linear/RF methods cannot.

This is the ONE remaining hypothesis from the CPU diagnostics:
attention-based adapters might exploit inter-patch temporal relationships
(64 patches have ordering) that flat-feature methods like Ridge/RF miss.

Usage:
  python scripts/run_residual_e1.py                    # CPU smoke test
  modal run scripts/run_residual_e1.py                 # GPU full run
  modal run scripts/run_residual_e1.py --dataset weather --seed 43
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from feasibility.data import load_dataset_multihor
from feasibility.model import load_moment, _get_encoder_blocks, _get_hidden_dim
from feasibility.finetune import _extract_features_batch


# ---------------------------------------------------------------------------
# Adapter architectures
# ---------------------------------------------------------------------------

class DLinearBaseline(nn.Module):
    """Vanilla DLinear: raw input -> linear -> forecast."""
    def __init__(self, input_len=512, output_dim=96):
        super().__init__()
        self.linear = nn.Linear(input_len, output_dim)

    def forward(self, raw_input):
        return self.linear(raw_input)


class ResidualFMAdapter(nn.Module):
    """E1: DLinear base + neural FM residual correction.

    pred = DLinear(raw_input) + scale * FM_head(hidden_states)
    """
    def __init__(self, d_model, output_dim, input_len=512):
        super().__init__()
        # DLinear base (jointly trained)
        self.dlinear = nn.Linear(input_len, output_dim)
        # FM residual head (attention over patches + MLP)
        self.patch_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fm_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )
        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states, raw_input):
        # DLinear path
        dl_pred = self.dlinear(raw_input)  # (B, output_dim)
        # FM path: self-attention over patches, then pool + project
        attn_out, _ = self.patch_attn(hidden_states, hidden_states, hidden_states)
        pooled = attn_out.mean(dim=1)  # (B, d_model)
        fm_residual = self.fm_head(pooled)  # (B, output_dim)
        return dl_pred + self.residual_scale * fm_residual


class ResidualFMConv(nn.Module):
    """E1 variant: DLinear + Conv1d over patch sequence."""
    def __init__(self, d_model, output_dim, input_len=512):
        super().__init__()
        self.dlinear = nn.Linear(input_len, output_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fm_head = nn.Linear(64, output_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states, raw_input):
        dl_pred = self.dlinear(raw_input)
        # Conv over patch dimension: (B, n_patches, d_model) -> (B, d_model, n_patches)
        x = hidden_states.transpose(1, 2)
        x = self.conv(x).squeeze(-1)  # (B, 64)
        fm_residual = self.fm_head(x)
        return dl_pred + self.residual_scale * fm_residual


class DualPathGated(nn.Module):
    """A2: Dual-path with learned gate (for comparison)."""
    def __init__(self, d_model, output_dim, input_len=512):
        super().__init__()
        self.raw_path = nn.Linear(input_len, output_dim)
        self.fm_path = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, output_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, hidden_states, raw_input):
        raw_out = self.raw_path(raw_input)
        fm_feat = hidden_states.mean(dim=1)
        fm_out = self.fm_path(fm_feat)
        g = self.gate(fm_feat)
        return g * fm_out + (1 - g) * raw_out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model, encoder_blocks, adapter, samples,
    device="cpu", n_epochs=15, lr=1e-3, batch_size=64, forecast_horizon=96,
):
    """Train a dual-input adapter (hidden_states + raw_input)."""
    # Prepare data
    X = samples[:, :512]
    Y = samples[:, 512:512 + forecast_horizon]
    X_padded = np.zeros((len(X), 512), dtype=samples.dtype)
    X_padded[:, :512] = X

    n = len(X_padded)
    split = int(0.8 * n)
    X_tr, X_te = X_padded[:split], X_padded[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float()),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).float()),
        batch_size=batch_size,
    )

    # Optimizer: adapter params + unfrozen backbone params
    trainable = list(adapter.parameters())
    param_ids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in param_ids:
            trainable.append(p)
            param_ids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    # Training
    # Keep model in eval mode to avoid gradient checkpoint issues,
    # but gradients still flow through unfrozen params (requires_grad=True)
    for epoch in range(n_epochs):
        model.eval()
        adapter.train()
        epoch_loss = 0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x_raw = batch_x.to(device)  # (B, 512) — raw input
            batch_x_enc = batch_x_raw.unsqueeze(1)  # (B, 1, 512) — for MOMENT
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x_enc.shape[0], 512, device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x_enc, input_mask)
            pred = adapter(feat, batch_x_raw)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Check residual scale
            scale = getattr(adapter, 'residual_scale', None)
            scale_str = f"  scale={scale.item():.4f}" if scale is not None else ""
            gate_str = ""
            print(f"  epoch {epoch+1}/{n_epochs}: train_loss={epoch_loss/n_batches:.6f}{scale_str}{gate_str}")

    # Evaluation
    model.eval()
    adapter.eval()
    val_losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x_raw = batch_x.to(device)
            batch_x_enc = batch_x_raw.unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x_enc.shape[0], 512, device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x_enc, input_mask)
            pred = adapter(feat, batch_x_raw)
            val_losses.append(criterion(pred, batch_y).item())
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())

    mse = float(np.mean(val_losses))
    param_count = sum(p.numel() for p in adapter.parameters())
    return {"mse": mse, "param_count": param_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Dataset: {args.dataset}, Seed: {args.seed}, Device: {args.device}")

    # Load data
    data = load_dataset_multihor(args.dataset, input_len=512, forecast_horizon=96, stride=64)
    samples = data["samples"]
    print(f"Samples: {samples.shape}")

    # Load MOMENT
    model = load_moment(device=args.device)
    encoder_blocks = _get_encoder_blocks(model)
    d_model = _get_hidden_dim(model)

    # Unfreeze last 4 encoder layers (same as code_evolution default)
    for param in model.parameters():
        param.requires_grad = False
    for block in encoder_blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True

    results = {}

    # 1. DLinear-only baseline
    print("\n--- DLinear only ---")
    dl = DLinearBaseline(512, 96).to(args.device)
    # Train DLinear without backbone
    X = samples[:, :512]
    Y = samples[:, 512:512+96]
    n = int(0.8 * len(X))
    dl_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[:n]).float(), torch.from_numpy(Y[:n]).float()),
        batch_size=64, shuffle=True)
    dl_opt = torch.optim.Adam(dl.parameters(), lr=1e-3)
    for ep in range(args.n_epochs):
        for bx, by in dl_loader:
            bx, by = bx.to(args.device), by.to(args.device)
            loss = F.mse_loss(dl(bx), by)
            dl_opt.zero_grad(); loss.backward(); dl_opt.step()
    dl.eval()
    with torch.no_grad():
        te_x = torch.from_numpy(X[n:]).float().to(args.device)
        te_y = torch.from_numpy(Y[n:]).float().to(args.device)
        dl_mse = F.mse_loss(dl(te_x), te_y).item()
    print(f"DLinear MSE: {dl_mse:.6f}")
    results["dlinear"] = dl_mse

    # 2. E1: Residual with attention
    print("\n--- E1: Residual FM (attention) ---")
    e1_attn = ResidualFMAdapter(d_model, 96).to(args.device)
    r = train_and_evaluate(model, encoder_blocks, e1_attn, samples,
                           device=args.device, n_epochs=args.n_epochs)
    print(f"E1-Attn MSE: {r['mse']:.6f} (params: {r['param_count']})")
    print(f"  residual_scale: {e1_attn.residual_scale.item():.4f}")
    results["e1_attn"] = r

    # 3. E1: Residual with conv
    print("\n--- E1: Residual FM (conv) ---")
    # Re-freeze/unfreeze backbone for fresh start
    for param in model.parameters():
        param.requires_grad = False
    for block in encoder_blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    e1_conv = ResidualFMConv(d_model, 96).to(args.device)
    r = train_and_evaluate(model, encoder_blocks, e1_conv, samples,
                           device=args.device, n_epochs=args.n_epochs)
    print(f"E1-Conv MSE: {r['mse']:.6f} (params: {r['param_count']})")
    print(f"  residual_scale: {e1_conv.residual_scale.item():.4f}")
    results["e1_conv"] = r

    # 4. A2: Dual-path gated
    print("\n--- A2: Dual-Path Gated ---")
    for param in model.parameters():
        param.requires_grad = False
    for block in encoder_blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    a2 = DualPathGated(d_model, 96).to(args.device)
    r = train_and_evaluate(model, encoder_blocks, a2, samples,
                           device=args.device, n_epochs=args.n_epochs)
    print(f"A2-Gate MSE: {r['mse']:.6f} (params: {r['param_count']})")
    results["a2_gate"] = r

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.dataset} H=96 seed={args.seed}")
    print(f"{'='*60}")
    print(f"DLinear:         {results['dlinear']:.6f}")
    print(f"E1-Attention:    {results['e1_attn']['mse']:.6f}  "
          f"({(results['e1_attn']['mse']/results['dlinear']-1)*100:+.2f}% vs DLinear)  "
          f"scale={e1_attn.residual_scale.item():.4f}")
    print(f"E1-Conv:         {results['e1_conv']['mse']:.6f}  "
          f"({(results['e1_conv']['mse']/results['dlinear']-1)*100:+.2f}% vs DLinear)  "
          f"scale={e1_conv.residual_scale.item():.4f}")
    print(f"A2-Gate:         {results['a2_gate']['mse']:.6f}  "
          f"({(results['a2_gate']['mse']/results['dlinear']-1)*100:+.2f}% vs DLinear)")

    # Save results
    os.makedirs("results/residual_e1", exist_ok=True)
    out_path = f"results/residual_e1/{args.dataset}_H96_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
