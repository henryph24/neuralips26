"""DLinear vs NLinear anchors alone, many seeds (Pm4m round 2, variance control).

The 3-seed run showed a large NLinear-over-DLinear margin on ETTh2, but our
DLinear there was itself unstable (0.335-0.521) against the paper's n=10 value
of 0.352+/-0.016. Since "NLinear beats a simplified DLinear" is the crux of
Pm4m's objection, that margin has to be estimated with enough seeds to be
trustworthy. No backbone is involved, so this is cheap to run wide.

Both models are Linear(L->H), 49K params, trained under the identical protocol
(Adam lr=1e-3, 15 epochs, batch 128) that scripts/run_dlinear_baseline.py uses
for the published anchor. The only difference is the last-value level anchor.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feasibility.standard_data import load_standard_data


def _train_linear(X_train, Y_train, X_test, Y_test, horizon, device, nlinear,
                  epochs=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_len = X_train.shape[1]
    lin = nn.Linear(input_len, horizon).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    def fwd(x):
        if nlinear:
            last = x[:, -1:].detach()
            return lin(x - last) + last
        return lin(x)

    tl = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=128, shuffle=True)
    for _ in range(epochs):
        lin.train()
        for bx, by in tl:
            loss = mse(fwd(bx.to(device)), by.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    lin.eval()
    el = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=128)
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in el:
            preds.append(fwd(bx.to(device)).cpu()); tgts.append(by)
    return nn.MSELoss()(torch.cat(preds), torch.cat(tgts)).item()


def run_cell(dataset, horizon=96, seeds=(42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
             device="cuda"):
    t0 = time.time()
    splits, _ = load_standard_data(dataset, horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    dl, nl = [], []
    for s in seeds:
        dl.append(_train_linear(X_train, Y_train, X_test, Y_test, horizon,
                                device, nlinear=False, seed=s))
        nl.append(_train_linear(X_train, Y_train, X_test, Y_test, horizon,
                                device, nlinear=True, seed=s))
    dl_a, nl_a = np.array(dl), np.array(nl)
    return {
        "dataset": dataset, "horizon": horizon, "n_seeds": len(seeds),
        "dlinear_mean": float(dl_a.mean()), "dlinear_std": float(dl_a.std()),
        "nlinear_mean": float(nl_a.mean()), "nlinear_std": float(nl_a.std()),
        "dlinear_per_seed": [float(x) for x in dl_a],
        "nlinear_per_seed": [float(x) for x in nl_a],
        # Paired: same seed, same data, only the level anchor differs.
        "nl_vs_dl_pct_mean": float(100.0 * (nl_a - dl_a).mean() / dl_a.mean()),
        "nl_wins": int((nl_a < dl_a).sum()),
        "elapsed": time.time() - t0,
    }
