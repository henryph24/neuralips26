"""TSFM-earns-keep: Residual-IA+ vs DLinear vs backbone-free Raw-MLP MoE (rebuttal Pm4m W1/W4).

Pm4m's sharpest point: even when Residual-IA+ matches DLinear, Appendix P.3 shows a
backbone-free Raw-MLP MoE matches/beats the dual-stream on average, so "the TSFM is
dead weight / a DLinear in a trenchcoat." The decisive rebuttal is a setting where
Residual-IA+ (with the frozen TSFM) beats BOTH DLinear AND the backbone-free Raw-MLP
MoE. The paper reports the TSFM contributes maximally at H=192, so we test there.

Per (dataset, seed) on a strictly frozen MOMENT-small we train, on the SAME split:
  * Residual-IA+  (exact recipe from ria_offsuite_core / run_gap_closing)  -> MSE, entropy
  * DLinear       (exact Linear(L->H) baseline)                            -> MSE
  * Raw-MLP MoE   (no backbone at all; run_raw_mlp_moe)                     -> MSE, entropy
and report whether Residual-IA+ beats both. Reuses the canonical training routines;
nothing is re-ported.
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
from ria_offsuite_core import _train_dlinear
from run_raw_mlp_moe import train_raw_mlp_moe


def run_cell(dataset, seed, horizon=192, backbone="AutonLab/MOMENT-1-small",
             device="cuda", ria_max_epochs=30, rawmlp_epochs=15):
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

    # (1) Exact Residual-IA+ recipe (with frozen TSFM).
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

    # (2) Exact DLinear baseline.
    dlinear_mse = _train_dlinear(X_train, Y_train, X_test, Y_test,
                                 horizon=horizon, device=device)

    # (3) Backbone-free Raw-MLP MoE (NO TSFM forward at all).
    rawmlp = train_raw_mlp_moe(
        X_train, Y_train, X_test, Y_test, device=device, n_epochs=rawmlp_epochs,
        forecast_horizon=horizon, K=5, top_k=2, test_ch=test_ch, scaler=scaler,
    )

    ria_mse = ria["mse"]
    rawmlp_mse = rawmlp["mse"]
    return {
        "dataset": dataset, "seed": int(seed), "horizon": horizon,
        "ria_mse": ria_mse, "ria_entropy": ria.get("routing_entropy"),
        "ria_param_count": ria.get("param_count"),
        "dlinear_mse": dlinear_mse,
        "rawmlp_mse": rawmlp_mse, "rawmlp_entropy": rawmlp.get("routing_entropy"),
        "gap_vs_dlinear_pct": 100.0 * (ria_mse - dlinear_mse) / dlinear_mse,
        "gap_vs_rawmlp_pct": 100.0 * (ria_mse - rawmlp_mse) / rawmlp_mse,
        "beats_dlinear": bool(ria_mse <= dlinear_mse),
        "beats_rawmlp": bool(ria_mse <= rawmlp_mse),
        "beats_both": bool(ria_mse <= dlinear_mse and ria_mse <= rawmlp_mse),
        "elapsed": time.time() - t0,
    }
