"""Extended full fine-tuning: stronger optimization to address reviewer concern.

Tests whether the Frozen Paradox holds under:
1. More epochs (50 vs 15)
2. Cosine LR schedule with warmup
3. Layer-wise LR decay (deeper layers get smaller lr)
4. AdamW with weight decay

If frozen RR-MoA still wins, the "optimization confound" argument is dead.

Usage:
    python scripts/run_extended_ft.py --dataset ETTh1 --seed 42
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
from feasibility.adapter_seeds import SEED_ADAPTERS
from feasibility.standard_data import (
    load_standard_data, _detect_backbone_type, compute_denorm_mse,
)
from scripts.run_rr_moa import _apply_unfreeze
from scripts.run_full_finetune import _train_adapter_lr

# Best head per dataset from the 15-epoch sweep
BEST_HEADS = {
    "ETTh1": ("last_mlp", SEED_ADAPTERS[2]),
    "ETTm1": ("conv", SEED_ADAPTERS[4]),
    "Weather": ("conv", SEED_ADAPTERS[4]),
    "ETTh2": ("last_mlp", SEED_ADAPTERS[2]),
    "ETTm2": ("attention", SEED_ADAPTERS[3]),
    "Electricity": ("last_mlp", SEED_ADAPTERS[2]),
}
# Fallback
DEFAULT_HEAD = ("conv", SEED_ADAPTERS[4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs("results/extended_ft", exist_ok=True)
    bb_type = _detect_backbone_type("AutonLab/MOMENT-1-small")

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    head_name, head_code = BEST_HEADS.get(args.dataset, DEFAULT_HEAD)
    print("Using head: %s" % head_name)

    configs = [
        # (label, epochs, lr, use_cosine, warmup_epochs, layerwise_decay)
        ("15ep_adam", 15, 1e-4, False, 0, 1.0),
        ("50ep_adam", 50, 1e-4, False, 0, 1.0),
        ("50ep_cosine", 50, 1e-3, True, 3, 1.0),
        ("50ep_cosine_layerwise", 50, 1e-3, True, 3, 0.8),
        ("50ep_1e-5", 50, 1e-5, False, 0, 1.0),
    ]

    results = {}
    for label, epochs, lr, use_cosine, warmup_ep, lw_decay in configs:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = load_backbone("AutonLab/MOMENT-1-small", args.device)
        _disable_gradient_checkpointing(model)
        blocks = _get_encoder_blocks(model)
        for p in model.parameters():
            p.requires_grad = False
        _apply_unfreeze(blocks, "all")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        print("\n[%s] lr=%.0e epochs=%d cosine=%s warmup=%d lw_decay=%.1f trainable=%d" % (
            label, lr, epochs, use_cosine, warmup_ep, lw_decay, trainable_params))

        out = _train_adapter_lr(
            head_code, model, blocks, X_train, Y_train, X_test, Y_test,
            device=args.device, n_epochs=epochs, forecast_horizon=args.horizon,
            batch_size=args.batch_size, backbone_type=bb_type,
            eval_ch=test_ch, scaler=scaler, lr=lr,
            use_cosine=use_cosine, warmup_epochs=warmup_ep,
            layerwise_decay=lw_decay,
        )
        elapsed = time.time() - t0
        results[label] = {**out, "lr": lr, "epochs": epochs,
                          "use_cosine": use_cosine, "warmup": warmup_ep,
                          "layerwise_decay": lw_decay, "elapsed": elapsed}
        print("  MSE=%.4f (%.0fs)" % (out["mse"], elapsed))

    # Summary
    print("\n=== EXTENDED FT SUMMARY: %s ===" % args.dataset)
    for label, r in results.items():
        print("  %-25s MSE=%.4f" % (label, r["mse"]))

    # Save
    fname = "results/extended_ft/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(fname, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                    "head": head_name, "results": results}, f, indent=2)
    print("Saved: %s" % fname)


if __name__ == "__main__":
    main()
