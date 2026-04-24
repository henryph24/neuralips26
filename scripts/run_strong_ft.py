"""Strong full fine-tuning sweep: 10 configurations to make the Frozen Paradox irrefutable.

Tests whether frozen RR-MoA still beats full FT under careful optimization:
  - Very low LR (1e-5, 1e-6) with 100 epochs
  - Long warmup (10 epochs) + cosine decay
  - Layer-wise LR decay (gamma=0.8)
  - Weight decay sweep {0.001, 0.01, 0.1}
  - Gradient accumulation (effective batch 256, 512)
  - Gradient clipping (1.0)

If frozen RR-MoA still wins across all 10 configs, the Frozen Paradox
cannot be dismissed as an optimization artifact.

Usage:
    python scripts/run_strong_ft.py --dataset ETTh1 --seed 42
    python scripts/run_strong_ft.py --dataset ETTh1 --seed 42 --config-idx 0
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
    load_standard_data, _detect_backbone_type, compute_denorm_mse,
)
from scripts.run_rr_moa import _apply_unfreeze
from scripts.run_full_finetune import _train_adapter_lr

# Best head per dataset (same as run_extended_ft.py)
BEST_HEADS = {
    "ETTh1": ("last_mlp", SEED_ADAPTERS[2]),
    "ETTm1": ("conv", SEED_ADAPTERS[4]),
    "Weather": ("conv", SEED_ADAPTERS[4]),
    "ETTh2": ("last_mlp", SEED_ADAPTERS[2]),
    "ETTm2": ("attention", SEED_ADAPTERS[3]),
    "Electricity": ("last_mlp", SEED_ADAPTERS[2]),
}
DEFAULT_HEAD = ("conv", SEED_ADAPTERS[4])

# (label, epochs, lr, use_cosine, warmup_ep, lw_decay, weight_decay, grad_accum, grad_clip)
CONFIGS = [
    # Reference: current best from extended_ft
    ("ref_50ep_cos_lw",    50,  1e-3, True,  3,  0.8, 0.01,  1, 0.0),
    # Very low LR + long training
    ("100ep_1e-5_cos",    100,  1e-5, True, 10,  1.0, 0.01,  1, 0.0),
    ("100ep_1e-6_cos",    100,  1e-6, True, 10,  1.0, 0.01,  1, 0.0),
    ("100ep_1e-5_lw",     100,  1e-5, True, 10,  0.8, 0.01,  1, 0.0),
    # Weight decay sweep at best LR
    ("100ep_1e-5_wd001",  100,  1e-5, True, 10,  0.8, 0.001, 1, 0.0),
    ("100ep_1e-5_wd1",    100,  1e-5, True, 10,  0.8, 0.1,   1, 0.0),
    # Gradient accumulation (effective batch 256, 512)
    ("100ep_1e-5_ga2",    100,  1e-5, True, 10,  0.8, 0.01,  2, 0.0),
    ("100ep_1e-5_ga4",    100,  1e-5, True, 10,  0.8, 0.01,  4, 0.0),
    # Gradient clipping
    ("100ep_1e-5_clip1",  100,  1e-5, True, 10,  0.8, 0.01,  1, 1.0),
    # Best combo: low LR + lw + clip + accum
    ("100ep_1e-5_best",   100,  1e-5, True, 10,  0.8, 0.01,  2, 1.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--config-idx", type=int, default=-1,
                        help="Run only config at this index (-1 = all)")
    args = parser.parse_args()

    os.makedirs("results/strong_ft", exist_ok=True)
    bb_type = _detect_backbone_type("AutonLab/MOMENT-1-small")

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    head_name, head_code = BEST_HEADS.get(args.dataset, DEFAULT_HEAD)
    print("Using head: %s" % head_name)

    configs_to_run = [CONFIGS[args.config_idx]] if args.config_idx >= 0 else CONFIGS

    for label, epochs, lr, use_cosine, warmup_ep, lw_decay, wd, ga, gc in configs_to_run:
        # Check-and-skip
        fname = "results/strong_ft/%s_H%d_%d_%s.json" % (
            args.dataset, args.horizon, args.seed, label)
        if os.path.exists(fname):
            print("[SKIP] %s already exists" % fname)
            continue

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
        print("\n[%s] lr=%.0e ep=%d cos=%s warm=%d lw=%.1f wd=%.3f ga=%d gc=%.1f params=%d" % (
            label, lr, epochs, use_cosine, warmup_ep, lw_decay, wd, ga, gc, trainable_params))

        out = _train_adapter_lr(
            head_code, model, blocks, X_train, Y_train, X_test, Y_test,
            device=args.device, n_epochs=epochs, forecast_horizon=args.horizon,
            batch_size=args.batch_size, backbone_type=bb_type,
            eval_ch=test_ch, scaler=scaler, lr=lr,
            use_cosine=use_cosine, warmup_epochs=warmup_ep,
            layerwise_decay=lw_decay, weight_decay=wd,
            grad_accum_steps=ga, grad_clip=gc,
        )
        elapsed = time.time() - t0
        result = {
            "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
            "config": label, "head": head_name, "lr": lr, "epochs": epochs,
            "use_cosine": use_cosine, "warmup": warmup_ep,
            "layerwise_decay": lw_decay, "weight_decay": wd,
            "grad_accum_steps": ga, "grad_clip": gc,
            "mse": out["mse"], "mae": out["mae"],
            "param_count": out["param_count"], "elapsed": elapsed,
        }
        if "mse_denorm" in out:
            result["mse_denorm"] = out["mse_denorm"]
            result["mae_denorm"] = out["mae_denorm"]

        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print("  MSE=%.4f (%.0fs) -> %s" % (out["mse"], elapsed, fname))

    # Summary
    print("\n=== STRONG FT SUMMARY: %s seed=%d ===" % (args.dataset, args.seed))
    import glob
    pattern = "results/strong_ft/%s_H%d_%d_*.json" % (args.dataset, args.horizon, args.seed)
    best_mse, best_label = float("inf"), ""
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            r = json.load(f)
        tag = " <-- BEST" if r["mse"] < best_mse else ""
        if r["mse"] < best_mse:
            best_mse, best_label = r["mse"], r["config"]
        print("  %-25s MSE=%.4f%s" % (r["config"], r["mse"], tag))
    if best_label:
        print("  Best: %s @ %.4f" % (best_label, best_mse))


if __name__ == "__main__":
    main()
