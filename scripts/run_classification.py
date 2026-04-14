#!/usr/bin/env python3
"""B7: Cross-task classification on frozen MOMENT-small.

Demonstrates that the same frozen backbone used for forecasting also
supports classification via simple pooling + linear head, validating
the "one backbone, three tasks" deployment claim.

Usage:
    python3 scripts/run_classification.py --dataset EthanolConcentration --seed 42 --device cuda
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feasibility.config import AdapterConfig
from feasibility.finetune import finetune_classification
from feasibility.data import CLASSIFICATION_DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="EthanolConcentration",
                        choices=list(CLASSIFICATION_DATASETS.keys()))
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "max", "last", "cls_mean_max"])
    parser.add_argument("--head", default="linear", choices=["linear", "mlp1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = "results/classification"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.dataset}_{args.pooling}_{args.head}_{args.seed}.json"

    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    # Load dataset
    loader_fn = CLASSIFICATION_DATASETS[args.dataset]
    ds = loader_fn()
    print(f"Dataset: {args.dataset}, samples: {ds['samples'].shape}, "
          f"labels: {len(np.unique(ds['labels']))} classes")

    # Config: frozen backbone, specified pooling + head
    cfg = AdapterConfig(
        adapter_type="linear_probe",
        unfreeze="frozen",
        head_type=args.head,
        pooling=args.pooling,
        config_id=f"frozen_{args.pooling}_{args.head}",
    )

    # Run classification
    result = finetune_classification(
        cfg, ds["samples"], ds["labels"],
        device=args.device, n_epochs=args.epochs, lr=1e-3, batch_size=64,
    )

    elapsed = time.time() - t0
    out = {
        "dataset": args.dataset,
        "pooling": args.pooling,
        "head": args.head,
        "seed": args.seed,
        "epochs": args.epochs,
        "accuracy": result["accuracy"],
        "n_samples": int(ds["samples"].shape[0]),
        "n_classes": int(len(np.unique(ds["labels"]))),
        "elapsed": round(elapsed, 1),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Accuracy: {result['accuracy']:.4f}  ({elapsed:.0f}s)  → {out_path}")


if __name__ == "__main__":
    main()
