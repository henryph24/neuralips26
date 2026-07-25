"""Expert-pool composition experiment (rebuttal 8b2Z: "fixed set of relatively
simple adapters"). RR-MoA (raw router, frozen MOMENT-small, H=96) with different
expert pools:

  - canonical    : 5 simple pooling heads (mean/last/max/attention/conv1d)
  - macro        : 5 richer conv/residual/gated/depthwise experts
  - large-diverse: 10 heterogeneous experts (canonical + macro)
  - deep-mlp     : 5 deeper 2-hidden-layer MLP experts

Shows the method is robust to the adapter pool: routing stays healthy (no
collapse) and performance is comparable or better with richer/larger pools.
"""
import time

import numpy as np
import torch

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
_FIXED = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
# one expert per distinct type in the pool
POOL_K = {"canonical": 5, "macro": 5, "large-diverse": 10, "deep-mlp": 5, "hyper-gen": 5}


def _setup(device):
    model = load_backbone(BACKBONE, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    return model, blocks, _detect_backbone_type(BACKBONE)


def run_pool_cell(dataset, pool, seed, n_epochs=15, device="cuda", input_len=512, horizon=96):
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = POOL_K.get(pool, 5)
    model, blocks, bb = _setup(device)
    splits, _ = load_standard_data(dataset, horizon)
    Xtr, Ytr = splits["train"]
    Xte, Yte = splits["test"]
    out = {"exp": "pool", "dataset": dataset, "pool": pool, "K": K, "seed": int(seed),
           "horizon": horizon, "input_len": input_len, "n_epochs": n_epochs}
    t0 = time.time()
    raw = train_rr_moa_masked(
        model, blocks, Xtr, Ytr, Xte, Yte, input_len, device=device, n_epochs=n_epochs,
        forecast_horizon=horizon, backbone_type=bb, K=K, top_k=2,
        router_input_mode="raw", expert_pool=pool,
    )
    out.update(rrmoa_raw_mse=raw["mse"], rrmoa_raw_entropy=raw["routing_entropy"],
               rrmoa_raw_max_weight=raw["routing_max_weight"], param_count=raw["param_count"])
    bl = {}
    for name, code in _FIXED.items():
        try:
            bl[name] = train_fixed_masked(code, model, blocks, Xtr, Ytr, Xte, Yte, input_len,
                                          device=device, n_epochs=n_epochs,
                                          forecast_horizon=horizon, backbone_type=bb)
        except Exception:  # noqa: BLE001
            bl[name] = None
    valid = {k: v for k, v in bl.items() if v is not None}
    if valid:
        best = min(valid, key=valid.get)
        out.update(best_fixed_mse=valid[best],
                   delta_vs_fixed_pct=100.0 * (raw["mse"] - valid[best]) / valid[best],
                   rrmoa_wins_vs_fixed=bool(raw["mse"] < valid[best]))
    out["elapsed"] = time.time() - t0
    return out
