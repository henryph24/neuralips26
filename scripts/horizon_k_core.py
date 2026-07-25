"""Rebuttal experiment cells (run on Modal A10G):

- run_horizon_cell: longer horizons (jemj Q3 + 8b2Z limitation). RR-MoA (raw)
  vs best fixed adapter vs a from-scratch DLinear at H in {1000, 2000} on a
  strictly frozen MOMENT-small. Tests whether the frozen representation becomes
  limiting and whether routing stays healthy.
- run_k_cell: larger expert pools (8b2Z W3 / scalability). RR-MoA (raw) at
  K in {5, 10, 15, 20}, H=96. Tests that expert-count scaling stays stable
  and routing entropy stays healthy past K=10.

Reuses the proven train_rr_moa_masked / train_fixed_masked loops so the numbers
are comparable to the published harness. input_len=512 (full context, mask is a
no-op). Backbone strictly frozen.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from input_len_ablation import (
    train_rr_moa_masked,
    train_fixed_masked,
    load_backbone,
    _get_encoder_blocks,
    _disable_gradient_checkpointing,
    SEED_ADAPTERS,
    load_standard_data,
    _detect_backbone_type,
)

BACKBONE = "AutonLab/MOMENT-1-small"
# linear / attention / conv fixed adapters (same three as the input-length harness)
_FIXED = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}


def _setup(device):
    model = load_backbone(BACKBONE, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False  # strictly frozen
    return model, blocks, _detect_backbone_type(BACKBONE)


def _best_fixed(model, blocks, Xtr, Ytr, Xte, Yte, input_len, horizon, bb, device, n_epochs):
    bl = {}
    for name, code in _FIXED.items():
        try:
            bl[name] = train_fixed_masked(
                code, model, blocks, Xtr, Ytr, Xte, Yte, input_len,
                device=device, n_epochs=n_epochs, forecast_horizon=horizon, backbone_type=bb,
            )
        except Exception:  # noqa: BLE001
            bl[name] = None
    valid = {k: v for k, v in bl.items() if v is not None}
    if not valid:
        return None, None
    best = min(valid, key=valid.get)
    return best, valid[best]


def _train_dlinear(Xtr, Ytr, Xte, Yte, input_len, horizon, device, n_epochs=15, batch_size=128):
    """From-scratch DLinear: a single Linear(input_len -> horizon) on the raw window."""
    lin = nn.Linear(input_len, horizon).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    tr = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()),
        batch_size=batch_size, shuffle=True,
    )
    for _ in range(n_epochs):
        for bx, by in tr:
            bx = bx.to(device)[:, -input_len:]
            by = by.to(device)
            loss = mse(lin(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    lin.eval()
    te = DataLoader(
        TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(Yte).float()),
        batch_size=batch_size,
    )
    ps, ts = [], []
    with torch.no_grad():
        for bx, by in te:
            ps.append(lin(bx.to(device)[:, -input_len:]).float().cpu())
            ts.append(by)
    return nn.MSELoss()(torch.cat(ps), torch.cat(ts)).item()


def run_horizon_cell(dataset, horizon, seed, n_epochs=15, device="cuda", input_len=512):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, blocks, bb = _setup(device)
    splits, _ = load_standard_data(dataset, horizon)
    Xtr, Ytr = splits["train"]
    Xte, Yte = splits["test"]
    out = {"exp": "horizon", "dataset": dataset, "horizon": int(horizon), "seed": int(seed),
           "input_len": input_len, "n_epochs": n_epochs,
           "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    t0 = time.time()
    raw = train_rr_moa_masked(
        model, blocks, Xtr, Ytr, Xte, Yte, input_len, device=device, n_epochs=n_epochs,
        forecast_horizon=horizon, backbone_type=bb, router_input_mode="raw",
    )
    out.update(rrmoa_raw_mse=raw["mse"], rrmoa_raw_entropy=raw["routing_entropy"],
               rrmoa_raw_max_weight=raw["routing_max_weight"], param_count=raw["param_count"])
    bname, bmse = _best_fixed(model, blocks, Xtr, Ytr, Xte, Yte, input_len, horizon, bb, device, n_epochs)
    if bmse is not None:
        out.update(best_fixed_name=bname, best_fixed_mse=bmse,
                   delta_vs_fixed_pct=100.0 * (raw["mse"] - bmse) / bmse,
                   rrmoa_wins_vs_fixed=bool(raw["mse"] < bmse))
    try:
        dl = _train_dlinear(Xtr, Ytr, Xte, Yte, input_len, horizon, device, n_epochs=n_epochs)
        out.update(dlinear_mse=dl, delta_vs_dlinear_pct=100.0 * (raw["mse"] - dl) / dl)
    except Exception as e:  # noqa: BLE001
        out["dlinear_error"] = str(e)
    out["elapsed"] = time.time() - t0
    return out


def run_k_cell(dataset, K, seed, n_epochs=15, device="cuda", input_len=512, horizon=96):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, blocks, bb = _setup(device)
    splits, _ = load_standard_data(dataset, horizon)
    Xtr, Ytr = splits["train"]
    Xte, Yte = splits["test"]
    out = {"exp": "K", "dataset": dataset, "K": int(K), "seed": int(seed),
           "horizon": horizon, "input_len": input_len, "n_epochs": n_epochs}
    t0 = time.time()
    raw = train_rr_moa_masked(
        model, blocks, Xtr, Ytr, Xte, Yte, input_len, device=device, n_epochs=n_epochs,
        forecast_horizon=horizon, backbone_type=bb, K=K, top_k=2, router_input_mode="raw",
    )
    out.update(rrmoa_raw_mse=raw["mse"], rrmoa_raw_entropy=raw["routing_entropy"],
               rrmoa_raw_max_weight=raw["routing_max_weight"], param_count=raw["param_count"])
    bname, bmse = _best_fixed(model, blocks, Xtr, Ytr, Xte, Yte, input_len, horizon, bb, device, n_epochs)
    if bmse is not None:
        out.update(best_fixed_name=bname, best_fixed_mse=bmse,
                   delta_vs_fixed_pct=100.0 * (raw["mse"] - bmse) / bmse,
                   rrmoa_wins_vs_fixed=bool(raw["mse"] < bmse))
    out["elapsed"] = time.time() - t0
    return out
