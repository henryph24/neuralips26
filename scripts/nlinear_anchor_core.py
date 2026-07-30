"""Decisive control for Pm4m's NLinear-vs-DLinear confound (rebuttal round 2).

Pm4m's objection: Residual-IA+'s shared raw branch is NLinear
(y = Linear(x - x_T) + x_T, App. P.1 fix (v)), while the from-scratch calibration
anchor is the *simplified* single-Linear DLinear without trend-seasonal
decomposition (App. A). The "65 significant wins vs 11 losses" grid therefore
cannot separate "the frozen backbone adds value" from "NLinear beats a
simplified DLinear", because NLinear's advantage on level-drifting data is a
known from-scratch result (Zeng et al. 2023).

The requested control: a from-scratch NLinear trained under the *same protocol*
as the DLinear anchor. Both anchors here are Linear(L -> H) with an identical
49K parameter budget, identical optimizer, epochs, batch size and loss; the only
difference is the last-value level anchor. RIA+ vs NLinear therefore isolates
the frozen backbone's contribution with the linear branch held fixed:

  DLinear   : y = Linear(x)                       <- current anchor
  NLinear   : y = Linear(x - x_T) + x_T           <- requested anchor (same budget)
  RIA+      : y = NLinear(x) + g(x) * Adapter(H)  <- our method

Every cell trains all three on the same splits in the same process, so the
comparison is internally consistent regardless of environment drift.
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
from ria_offsuite_core import _train_dlinear
from run_gap_closing import train_gap_closing


class _NLinearScratch(nn.Module):
    """NLinear: same Linear(L->H) as the DLinear anchor, on level-anchored input.

    Matches the shared raw branch inside Residual-IA+ exactly
    (run_gap_closing.DualStreamExpert.forward, raw_arch="nlinear").
    """

    def __init__(self, input_len, horizon):
        super().__init__()
        self.linear = nn.Linear(input_len, horizon)

    def forward(self, x):
        last = x[:, -1:].detach()
        return self.linear(x - last) + last


def _train_nlinear(X_train, Y_train, X_test, Y_test, horizon=96, device="cuda",
                   epochs=15):
    """From-scratch NLinear under the DLinear anchor's exact protocol.

    Identical to ria_offsuite_core._train_dlinear (Adam lr=1e-3, 15 epochs,
    batch 128, MSE, same splits) except for the level anchor.
    """
    input_len = X_train.shape[1]
    model = _NLinearScratch(input_len, horizon).to(device)
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
    n_params = sum(p.numel() for p in model.parameters())
    return nn.MSELoss()(preds, tgts).item(), n_params


def run_cell(dataset, seed, horizon=96, backbone="AutonLab/MOMENT-1-small",
             device="cuda", ria_max_epochs=30, weight_decay=0.0,
             raw_hidden=192, raw_depth=2):
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
    # (1) Exact Residual-IA+ recipe (unchanged from ria_offsuite_core).
    ria = train_gap_closing(
        model, blocks, X_train, Y_train, X_test, Y_test,
        device=device, n_epochs=ria_max_epochs, forecast_horizon=horizon,
        batch_size=128, backbone_type=bb, K=5, hidden=64, top_k=2,
        variant="residual-ia", test_ch=test_ch, scaler=scaler,
        raw_hidden=raw_hidden, lr=1e-3, weight_decay=weight_decay,
        cosine_schedule=True,
        raw_depth=raw_depth, gate_init=-2.0, warmup_epochs=5, adapter_hidden=None,
        X_val=X_val, Y_val=Y_val, val_early_stop=True, val_patience=5,
        raw_branch_shared=True, raw_arch="nlinear", grad_clip=1.0,
    )
    # (2) The paper's DLinear anchor.
    dlinear_mse = _train_dlinear(X_train, Y_train, X_test, Y_test,
                                 horizon=horizon, device=device)
    # (3) The control Pm4m asks for: same protocol, level-anchored.
    nlinear_mse, nlinear_params = _train_nlinear(
        X_train, Y_train, X_test, Y_test, horizon=horizon, device=device)

    ria_mse = ria["mse"]
    return {
        "dataset": dataset, "seed": int(seed), "horizon": horizon,
        "backbone": backbone, "epochs": ria_max_epochs,
        "weight_decay": weight_decay,
        "ria_mse": ria_mse, "ria_entropy": ria.get("routing_entropy"),
        "ria_param_count": ria.get("param_count"),
        "dlinear_mse": dlinear_mse,
        "nlinear_mse": nlinear_mse, "nlinear_param_count": nlinear_params,
        # How much of the headline win is level anchoring alone?
        "nlinear_vs_dlinear_pct": 100.0 * (nlinear_mse - dlinear_mse) / dlinear_mse,
        # Does the frozen backbone survive the stronger anchor?
        "ria_vs_nlinear_pct": 100.0 * (ria_mse - nlinear_mse) / nlinear_mse,
        "ria_vs_dlinear_pct": 100.0 * (ria_mse - dlinear_mse) / dlinear_mse,
        "ria_beats_nlinear": bool(ria_mse <= nlinear_mse),
        "ria_beats_dlinear": bool(ria_mse <= dlinear_mse),
        "elapsed": time.time() - t0,
    }
