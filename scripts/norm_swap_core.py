"""Normalizer-family swap on MOMENT: does a LayerNorm at the input-normalization
position collapse like BatchNorm/GroupNorm? (rebuttal 8b2Z W4 — interventional rigor).

8b2Z notes the BatchNorm swap is a clean intervention on MOMENT, but LayerNorm/RMSNorm
were only OBSERVATIONAL negative controls (different backbones, Chronos/Timer-XL). This
runs a LayerNorm swap AT THE SAME RevIN position on MOMENT to match that rigor.

For each normalizer swapped in at MOMENT's RevIN position, train a hidden-state routed
AdaMix (the collapsing config) and measure final routing entropy:
  * revin      -> paper baseline
  * none       -> disable_revin negative control  (should stay healthy ~0.82)
  * batchnorm  -> App H positive control          (should collapse ~0.004) [self-check]
  * groupnorm  -> App H positive control          (should collapse ~0.000) [self-check]
  * layernorm  -> NEW: LayerNorm at the input-normalization position

If batchnorm/groupnorm reproduce the App H collapse and 'none' stays healthy, the harness
is validated and the layernorm number is trustworthy. Note: this tests LayerNorm at the
*input* position (which strips per-window stats), distinct from LayerNorm used *inside*
the encoder on hidden states (Chronos/Timer-XL), which does not strip input stats. Reuses
the canonical run_adamix training loop; nothing is re-ported.
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
from run_adamix import train_adamix, _apply_unfreeze

NORMS = ["revin", "none", "batchnorm", "groupnorm", "layernorm"]


def _one(dataset, seed, norm, unfreeze, horizon, backbone, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    disable = (norm == "none")
    nt = "revin" if norm in ("revin", "none") else norm
    model = load_backbone(backbone, device, disable_revin=disable, norm_type=nt)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, unfreeze)
    bb = _detect_backbone_type(backbone)

    splits, _ = load_standard_data(dataset, horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]

    res = train_adamix(
        model, blocks, X_train, Y_train, X_val, Y_val, X_test, Y_test,
        device=device, n_epochs=15, forecast_horizon=horizon, backbone_type=bb,
        router_input="hidden",
    )
    return {"routing_entropy": res["routing_entropy"], "mse": res["mse"]}


def run_cell(dataset, seed, unfreeze="last4", horizon=96,
             backbone="AutonLab/MOMENT-1-small", device="cuda"):
    t0 = time.time()
    out = {"dataset": dataset, "seed": int(seed), "unfreeze": unfreeze,
           "horizon": horizon, "norms": {}}
    for norm in NORMS:
        try:
            out["norms"][norm] = _one(dataset, seed, norm, unfreeze, horizon, backbone, device)
        except Exception as exc:  # noqa: BLE001 - record and continue
            out["norms"][norm] = {"error": str(exc)}
    out["elapsed"] = time.time() - t0
    return out
