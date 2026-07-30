"""Single-variable backbone control (Pm4m round 2).

Pm4m's W4 reads App. P.3 as showing the frozen TSFM is dead weight. Our first
response answered with a backbone-free Raw-MLP MoE, but that model also swaps the
raw branch for an MLP, so it varies two things at once and cannot attribute the
difference to the backbone.

This is the clean control. Both arms run the identical Residual-IA+ recipe on the
same splits with the same seed, schedule, warmup, early stopping and shared
NLinear raw branch. The only difference is `backbone_off`, which zeroes the
residual gate so the frozen backbone contributes nothing:

    y = shared_NLinear(x) + g(x) * Adapter(H)      backbone on
    y = shared_NLinear(x) + 0    * Adapter(H)      backbone off

Everything else, including the router and the expert heads, is unchanged. The
paired difference is therefore attributable to the backbone path alone.
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

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _disable_gradient_checkpointing,
)
from feasibility.standard_data import load_standard_data, _detect_backbone_type
from run_gap_closing import train_gap_closing

# The published Residual-IA+ recipe, held fixed across both arms.
_RECIPE = dict(
    batch_size=128, K=5, hidden=64, top_k=2, variant="residual-ia",
    raw_hidden=192, lr=1e-3, weight_decay=0.0, cosine_schedule=True,
    raw_depth=2, gate_init=-2.0, warmup_epochs=5, adapter_hidden=None,
    val_early_stop=True, val_patience=5,
    raw_branch_shared=True, raw_arch="nlinear", grad_clip=1.0,
)


def run_cell(dataset, seed, horizon=96, backbone="AutonLab/MOMENT-1-small",
             device="cuda", ria_max_epochs=30):
    model = load_backbone(backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False  # strictly frozen in both arms
    bb = _detect_backbone_type(backbone)

    splits, _ = load_standard_data(dataset, horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    X_val, Y_val = splits.get("val", (None, None))
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")

    out = {}
    t0 = time.time()
    for arm, off in (("on", False), ("off", True)):
        # Same seed for both arms so initialisation and batch order match.
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = train_gap_closing(
            model, blocks, X_train, Y_train, X_test, Y_test,
            device=device, n_epochs=ria_max_epochs, forecast_horizon=horizon,
            backbone_type=bb, test_ch=test_ch, scaler=scaler,
            X_val=X_val, Y_val=Y_val, backbone_off=off, **_RECIPE,
        )
        out[arm] = res["mse"]

    on, off_ = out["on"], out["off"]
    return {
        "dataset": dataset, "seed": int(seed), "horizon": horizon,
        "backbone": backbone,
        "mse_backbone_on": on, "mse_backbone_off": off_,
        # negative = the backbone helps
        "backbone_gain_pct": 100.0 * (on - off_) / off_,
        "backbone_helps": bool(on <= off_),
        "elapsed": time.time() - t0,
    }
