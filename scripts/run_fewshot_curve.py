"""Few-shot learning curve: RR-MoA vs DLinear at varying training set sizes.

Shows that frozen RR-MoA dominates at low N (data-scarce deployment)
while DLinear catches up only at high N (full-data regime).

Usage:
    python scripts/run_fewshot_curve.py --dataset ETTh1 --seed 42
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
    RawRoutedMoA, _apply_unfreeze,
    HEAD_CLASSES, HEAD_NAMES,
)


class DLinearModel(nn.Module):
    """Simple DLinear: Y = W @ X."""
    def __init__(self, input_len=512, output_len=96):
        super().__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        return self.linear(x)


def train_dlinear(X_train, Y_train, X_test, Y_test, device="cuda",
                  n_epochs=15, batch_size=128):
    """Train DLinear from scratch on raw input."""
    model = DLinearModel(X_train.shape[1], Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    ), batch_size=min(batch_size, len(X_train)), shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = mse_fn(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in eval_loader:
            preds.append(model(bx.to(device)).cpu())
            tgts.append(by)
    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    params = sum(p.numel() for p in model.parameters())
    return {"mse": mse, "param_count": params}


def train_rrmoa_fewshot(model, blocks, X_train, Y_train, X_test, Y_test,
                        device="cuda", n_epochs=15, batch_size=128,
                        backbone_type="moment", K=5, top_k=2, hidden=64):
    """Train RR-MoA on subsampled data."""
    hdim = _get_hidden_dim(model)
    rrmoa = RawRoutedMoA(hdim, Y_train.shape[1], input_len=X_train.shape[1],
                         K=K, hidden=hidden, top_k=top_k).to(device)

    trainable = list(rrmoa.parameters())
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    ), batch_size=min(batch_size, len(X_train)), shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        rrmoa.train()
        for bx, by in loader:
            bx_raw = bx.to(device)
            by = by.to(device)
            bx_enc = bx_raw.unsqueeze(1)
            mask = torch.ones(bx_raw.shape[0], bx_raw.shape[1], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                               backbone_type=backbone_type)
                pred = rrmoa(feat, bx_raw)
                loss = mse_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    rrmoa.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in eval_loader:
            bx_raw = bx.to(device)
            bx_enc = bx_raw.unsqueeze(1)
            mask = torch.ones(bx_raw.shape[0], bx_raw.shape[1], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                           backbone_type=backbone_type)
            preds.append(rrmoa(feat, bx_raw).float().cpu())
            tgts.append(by)
    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    params = sum(p.numel() for p in rrmoa.parameters())
    return {"mse": mse, "param_count": params}


def train_single_adapter(model, blocks, X_train, Y_train, X_test, Y_test,
                         device="cuda", n_epochs=15, batch_size=128,
                         backbone_type="moment"):
    """Train single conv adapter (frozen backbone, no routing)."""
    from feasibility.code_evolution import SEED_ADAPTERS
    from scripts.run_standard_evolution import train_adapter
    return train_adapter(
        SEED_ADAPTERS[4], model, blocks, X_train, Y_train, X_test, Y_test,
        device=device, n_epochs=n_epochs, forecast_horizon=Y_train.shape[1],
        batch_size=min(batch_size, len(X_train)), backbone_type=backbone_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/fewshot", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bb_type = _detect_backbone_type(args.backbone)

    # Load full data
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train_full, Y_train_full = splits["train"]
    X_test, Y_test = splits["test"]
    N_full = len(X_train_full)
    print("%s H=%d: full_train=%d, test=%d" % (args.dataset, args.horizon, N_full, len(X_test)))

    # Load backbone once (shared across all N)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False

    N_values = [10, 50, 100, 200, 500, 1000, N_full]
    N_values = [n for n in N_values if n <= N_full]

    results = {}

    for N in N_values:
        print("\n=== N=%d ===" % N)

        # Subsample (random, seeded)
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(N_full, size=min(N, N_full), replace=False)
        X_sub = X_train_full[idx]
        Y_sub = Y_train_full[idx]

        n_results = {}

        # 1. DLinear
        torch.manual_seed(args.seed)
        t0 = time.time()
        dl = train_dlinear(X_sub, Y_sub, X_test, Y_test,
                           device=args.device, n_epochs=args.epochs)
        n_results["dlinear"] = {**dl, "time": time.time() - t0}
        print("  DLinear:       MSE=%.4f (params=%d)" % (dl["mse"], dl["param_count"]))

        # 2. Single adapter (conv, frozen backbone)
        torch.manual_seed(args.seed)
        t0 = time.time()
        sa = train_single_adapter(model, blocks, X_sub, Y_sub, X_test, Y_test,
                                  device=args.device, n_epochs=args.epochs,
                                  backbone_type=bb_type)
        n_results["single_adapter"] = {**sa, "time": time.time() - t0}
        print("  Single-adapter: MSE=%.4f (params=%d)" % (sa["mse"], sa["param_count"]))

        # 3. RR-MoA-lite (K=3, hidden=32, ~100K params)
        torch.manual_seed(args.seed)
        t0 = time.time()
        rl = train_rrmoa_fewshot(model, blocks, X_sub, Y_sub, X_test, Y_test,
                                 device=args.device, n_epochs=args.epochs,
                                 backbone_type=bb_type, K=3, top_k=2, hidden=32)
        n_results["rrmoa_lite"] = {**rl, "time": time.time() - t0}
        print("  RR-MoA-lite:   MSE=%.4f (params=%d)" % (rl["mse"], rl["param_count"]))

        # 4. RR-MoA full (K=5, hidden=64, ~426K params)
        torch.manual_seed(args.seed)
        t0 = time.time()
        rf = train_rrmoa_fewshot(model, blocks, X_sub, Y_sub, X_test, Y_test,
                                 device=args.device, n_epochs=args.epochs,
                                 backbone_type=bb_type, K=5, top_k=2, hidden=64)
        n_results["rrmoa_full"] = {**rf, "time": time.time() - t0}
        print("  RR-MoA-full:   MSE=%.4f (params=%d)" % (rf["mse"], rf["param_count"]))

        results["N_%d" % N] = n_results

    # Summary
    print("\n" + "=" * 70)
    print("FEW-SHOT SUMMARY: %s (seed=%d)" % (args.dataset, args.seed))
    print("=" * 70)
    print("%-8s  %-10s %-10s %-10s %-10s" % ("N", "DLinear", "SingleAdp", "RR-lite", "RR-full"))
    for N in N_values:
        r = results["N_%d" % N]
        print("%-8d  %-10.4f %-10.4f %-10.4f %-10.4f" % (
            N, r["dlinear"]["mse"], r["single_adapter"]["mse"],
            r["rrmoa_lite"]["mse"], r["rrmoa_full"]["mse"]))

    # Save
    fname = "results/fewshot/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(fname, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                    "backbone": args.backbone, "results": results}, f, indent=2)
    print("\nSaved: %s" % fname)


if __name__ == "__main__":
    main()
