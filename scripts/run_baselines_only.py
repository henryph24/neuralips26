"""Baseline-only training: linear/attention/conv heads on a frozen MOMENT-small.

Mirrors the baseline branch of `run_rr_moa.py` (lines 625-650) without RR-MoA,
so it can backfill the (cell, seed) baseline gaps the original sweep missed.

Output JSON has the same shape as the `baselines` field of run_rr_moa.py JSONs,
so downstream code (verify.py, table generators) can ingest it directly.

Usage:
    python3 scripts/run_baselines_only.py --dataset ETTh1 --seed 42 --unfreeze frozen
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _apply_unfreeze,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--unfreeze", default="frozen", choices=["frozen", "last2", "last4", "all"])
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/baselines_only")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bb_type = _detect_backbone_type(args.backbone)
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print(f"{args.dataset} H={args.horizon}: train={len(X_train)} test={len(X_test)}")

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p_ in model.parameters():
        p_.requires_grad = False
    _apply_unfreeze(blocks, args.unfreeze)

    baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
    results = {}
    t0 = time.time()
    for name, code in baselines.items():
        tr = train_adapter(code, model, blocks, X_train, Y_train, X_test, Y_test,
                           device=args.device, n_epochs=args.epochs,
                           forecast_horizon=args.horizon, backbone_type=bb_type,
                           eval_ch=test_ch, scaler=scaler)
        results[name] = tr
        msg = f"  {name:10s} MSE={tr['mse']:.4f}"
        if "mse_denorm" in tr:
            msg += f"  MSE_denorm={tr['mse_denorm']:.4f}"
        print(msg)
    elapsed = time.time() - t0

    out = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "unfreeze": args.unfreeze, "K": 5, "top_k": 2,
        "backbone": args.backbone,
        "baselines": results,
        "elapsed": elapsed,
        "source": "run_baselines_only.py",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"{args.dataset}_H{args.horizon}_K5_top2_{args.unfreeze}_{args.seed}_baselines.json"
    path = os.path.join(args.out_dir, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
