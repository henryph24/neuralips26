"""DLinear baseline -- no backbone, no adapter, just a linear model trained
from scratch on the same normalized train/val/test protocol as run_rr_moa.py.

This script exists to contextualize our absolute MSE numbers against a
lightweight supervised baseline. A third reviewer asked us to include DLinear
so that the reader can compare the frozen-backbone adapter paradigm against
the simplest possible supervised model. By running DLinear on our exact
normalized evaluation scale (StandardScaler at load time), we get apples-to-
apples numbers rather than having to cite the folkloric "~0.38 on ETTh1"
which is on unnormalized data.

Usage:
    python scripts/run_dlinear_baseline.py --dataset ETTh1 --seed 42 --epochs 15
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

from scripts.run_standard_evolution import load_standard_data


class DLinear(nn.Module):
    """Minimal DLinear: one linear layer from input_len to horizon, no backbone."""
    def __init__(self, input_len, horizon):
        super().__init__()
        self.linear = nn.Linear(input_len, horizon)

    def forward(self, x):
        # x: (B, input_len) -- already StandardScaler-normalized by load_standard_data
        return self.linear(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs("results/dlinear", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Same data pipeline as run_rr_moa.py and run_lora_baseline.py
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    input_len = X_train.shape[1]
    model = DLinear(input_len, args.horizon).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    ), batch_size=128, shuffle=True)

    n_params = sum(p.numel() for p in model.parameters())
    print("DLinear: input_len=%d horizon=%d params=%d" % (input_len, args.horizon, n_params))
    print("%s H=%d: train=%d test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(args.device)
            by = by.to(args.device)
            pred = model(bx)
            loss = mse_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    # Test evaluation
    model.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float(),
    ), batch_size=128)

    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.append(model(bx.to(args.device)).cpu())
            tgts.append(by)
    preds, tgts = torch.cat(preds), torch.cat(tgts)
    test_mse = nn.MSELoss()(preds, tgts).item()
    test_mae = nn.L1Loss()(preds, tgts).item()

    print("DLinear: MSE=%.4f MAE=%.4f  time=%.0fs" % (test_mse, test_mae, elapsed))

    out = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "epochs": args.epochs,
        "input_len": input_len,
        "dlinear_mse": test_mse,
        "dlinear_mae": test_mae,
        "param_count": n_params,
        "elapsed": elapsed,
    }
    path = "results/dlinear/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
