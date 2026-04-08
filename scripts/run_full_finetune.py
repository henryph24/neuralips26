"""Full fine-tuning baseline: unfreeze ALL encoder blocks + single best head.

The most obvious missing baseline for the Frozen Paradox claim.
If frozen RR-MoA beats this, the Frozen Paradox is undeniable.

Usage:
    python scripts/run_full_finetune.py --dataset ETTh1 --seed 42
    python scripts/run_full_finetune.py --dataset ETTh1 --seed 42 --backbone amazon/chronos-t5-small
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

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
    compute_denorm_mse,
)
from scripts.run_rr_moa import _apply_unfreeze

# Conv adapter (SEED_ADAPTERS[4]) — strongest fixed adapter across datasets
CONV_ADAPTER = SEED_ADAPTERS[4]

# All 5 canonical heads for completeness — report the best
HEADS = {
    "mean_linear": SEED_ADAPTERS[0],
    "mean_mlp2": SEED_ADAPTERS[1],
    "last_mlp": SEED_ADAPTERS[2],
    "attention": SEED_ADAPTERS[3],
    "conv": SEED_ADAPTERS[4],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-revin", action="store_true",
                        help="Disable RevIN inside MOMENT backbone (causal ablation)")
    args = parser.parse_args()

    os.makedirs("results/full_finetune", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bb_type = _detect_backbone_type(args.backbone)

    # Load data
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    results = {}

    # Try two learning rates — 1e-3 (standard) and 1e-4 (safer for full-FT)
    for lr in [1e-3, 1e-4]:
        # Reload model fresh for each lr to avoid state leakage
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = load_backbone(args.backbone, args.device,
                              disable_revin=args.disable_revin)
        _disable_gradient_checkpointing(model)
        blocks = _get_encoder_blocks(model)

        # Unfreeze ALL encoder blocks
        for p in model.parameters():
            p.requires_grad = False
        _apply_unfreeze(blocks, "all")

        backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nlr=%.0e, backbone trainable params=%d" % (lr, backbone_trainable))

        # Train each head type with all layers unfrozen
        lr_results = {}
        for head_name, head_code in HEADS.items():
            torch.manual_seed(args.seed)
            # Monkey-patch lr into train_adapter by temporarily modifying the function
            # Actually train_adapter uses lr=1e-3 hardcoded — we need a workaround
            # Use a custom training loop for non-default lr
            if lr == 1e-3:
                out = train_adapter(
                    head_code, model, blocks, X_train, Y_train, X_test, Y_test,
                    device=args.device, n_epochs=args.epochs,
                    forecast_horizon=args.horizon, batch_size=args.batch_size,
                    backbone_type=bb_type, eval_ch=test_ch, scaler=scaler,
                )
            else:
                out = _train_adapter_lr(
                    head_code, model, blocks, X_train, Y_train, X_test, Y_test,
                    device=args.device, n_epochs=args.epochs,
                    forecast_horizon=args.horizon, batch_size=args.batch_size,
                    backbone_type=bb_type, eval_ch=test_ch, scaler=scaler,
                    lr=lr,
                )
            lr_results[head_name] = out
            print("  %s: MSE=%.4f (params=%d)" % (head_name, out["mse"], out["param_count"]))

        results["lr_%.0e" % lr] = lr_results

    # Find the best across all heads and lrs
    best_mse = float("inf")
    best_config = None
    for lr_key, lr_results in results.items():
        for head_name, out in lr_results.items():
            if out["mse"] < best_mse:
                best_mse = out["mse"]
                best_config = {"lr": lr_key, "head": head_name, **out}

    print("\nBest full-FT: %s %s MSE=%.4f" % (best_config["lr"], best_config["head"], best_mse))

    # Save
    bb_suffix = ""
    if "chronos" in args.backbone.lower():
        bb_suffix = "_bb-chronos"
    elif "moirai" in args.backbone.lower():
        bb_suffix = "_bb-moirai"
    elif "large" in args.backbone.lower():
        bb_suffix = "_bb-moment-large"

    revin_suffix = "_no-revin" if args.disable_revin else ""
    fname = "results/full_finetune/%s_H%d_%d%s%s.json" % (
        args.dataset, args.horizon, args.seed, bb_suffix, revin_suffix)
    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "backbone": args.backbone,
        "unfreeze": "all",
        "best": best_config,
        "all_results": {
            lr_key: {h: {"mse": v["mse"], "mae": v["mae"], "param_count": v["param_count"]}
                     for h, v in lr_res.items()}
            for lr_key, lr_res in results.items()
        },
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved: %s" % fname)


def _train_adapter_lr(code, model, blocks, X_train, Y_train, X_eval, Y_eval,
                      device="cuda", n_epochs=3, forecast_horizon=96, batch_size=128,
                      backbone_type="moment", eval_ch=None, scaler=None, lr=1e-4,
                      use_cosine=False, warmup_epochs=0, layerwise_decay=1.0):
    """Train adapter with configurable lr, cosine schedule, warmup, and layer-wise LR decay."""
    from torch.utils.data import DataLoader, TensorDataset
    from feasibility.finetune import _extract_features_batch

    hdim = _get_hidden_dim(model)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)
    adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)
    param_count = sum(p.numel() for p in adapter.parameters())

    # Build parameter groups with optional layer-wise LR decay
    if layerwise_decay < 1.0 and len(blocks) > 0:
        param_groups = [{"params": list(adapter.parameters()), "lr": lr}]
        n_blocks = len(blocks)
        for i, block in enumerate(blocks):
            block_params = [p for p in block.parameters() if p.requires_grad]
            if block_params:
                depth_from_top = n_blocks - 1 - i
                block_lr = lr * (layerwise_decay ** depth_from_top)
                param_groups.append({"params": block_params, "lr": block_lr})
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
    else:
        trainable = list(adapter.parameters())
        pids = {id(p) for p in trainable}
        for p in model.parameters():
            if p.requires_grad and id(p) not in pids:
                trainable.append(p)
                pids.add(id(p))
        optimizer = torch.optim.Adam(trainable, lr=lr)

    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    # Cosine schedule with warmup
    total_steps = n_epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    scheduler = None
    if use_cosine:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        schedulers = []
        if warmup_steps > 0:
            schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps))
        schedulers.append(CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps))
        if len(schedulers) > 1:
            scheduler = SequentialLR(optimizer, schedulers, milestones=[warmup_steps])
        else:
            scheduler = schedulers[0]

    for epoch in range(n_epochs):
        model.train()
        adapter.train()
        for bx, by in loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                loss = mse_fn(adapter(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    # Evaluate
    model.eval()
    adapter.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_eval).float(), torch.from_numpy(Y_eval).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in eval_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            preds.append(adapter(_extract_features_batch(
                model, blocks, bx, mask, backbone_type=backbone_type)).float().cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    out = {"mse": mse, "mae": mae, "param_count": param_count}
    if eval_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, eval_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d
    return out


if __name__ == "__main__":
    main()
