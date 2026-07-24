"""Residual-IA+ vs DLinear on off-suite datasets (rebuttal W4).

Directly answers Pm4m W4 "its [Residual-IA+'s] generalization beyond these six
benchmarks is unclear" by running the EXACT Residual-IA+ recipe
(scripts/run_gap_closing.train_gap_closing, variant="residual-ia" + the paper's
RIA+ flags: shared NLinear raw branch, gate init b=-2, 5-epoch warmup, cosine LR,
validation early stopping patience 5, grad clip 1.0) and the EXACT DLinear
baseline (Linear(input_len->horizon)) on datasets *beyond* the primary six —
which Residual-IA+ was never run on. Nothing is re-ported; both training routines
are the canonical scripts.
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

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _disable_gradient_checkpointing,
)
from feasibility.standard_data import load_standard_data, _detect_backbone_type
from run_gap_closing import train_gap_closing


def _train_dlinear(X_train, Y_train, X_test, Y_test, horizon=96, device="cuda", epochs=15):
    """Exact DLinear baseline (scripts/run_dlinear_baseline): one Linear(L->H)."""
    input_len = X_train.shape[1]
    model = nn.Linear(input_len, horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    tl = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for bx, by in tl:
            loss = mse(model(bx.to(device)), by.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    el = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=128)
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in el:
            preds.append(model(bx.to(device)).cpu()); tgts.append(by)
    preds, tgts = torch.cat(preds), torch.cat(tgts)
    return nn.MSELoss()(preds, tgts).item()


def run_cell(dataset, seed, horizon=96, backbone="AutonLab/MOMENT-1-small",
             device="cuda", ria_max_epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_backbone(backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False  # strictly frozen
    bb = _detect_backbone_type(backbone)

    splits, _ = load_standard_data(dataset, horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    X_val, Y_val = splits.get("val", (None, None))
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")

    t0 = time.time()
    # Exact Residual-IA+ recipe.
    ria = train_gap_closing(
        model, blocks, X_train, Y_train, X_test, Y_test,
        device=device, n_epochs=ria_max_epochs, forecast_horizon=horizon,
        batch_size=128, backbone_type=bb, K=5, hidden=64, top_k=2,
        variant="residual-ia", test_ch=test_ch, scaler=scaler,
        raw_hidden=192, lr=1e-3, weight_decay=0.0, cosine_schedule=True,
        raw_depth=2, gate_init=-2.0, warmup_epochs=5, adapter_hidden=None,
        X_val=X_val, Y_val=Y_val, val_early_stop=True, val_patience=5,
        raw_branch_shared=True, raw_arch="nlinear", grad_clip=1.0,
    )
    dlinear_mse = _train_dlinear(X_train, Y_train, X_test, Y_test,
                                 horizon=horizon, device=device)
    ria_mse = ria["mse"]
    gap = 100.0 * (ria_mse - dlinear_mse) / dlinear_mse
    return {
        "dataset": dataset, "seed": int(seed), "horizon": horizon,
        "ria_mse": ria_mse, "ria_entropy": ria.get("routing_entropy"),
        "ria_param_count": ria.get("param_count"),
        "dlinear_mse": dlinear_mse,
        "gap_pct": gap, "match_or_beat": bool(ria_mse <= dlinear_mse),
        "elapsed": time.time() - t0,
    }
