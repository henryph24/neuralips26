"""RR-MoA classification: proves routing generalizes beyond forecasting.

Uses the same RawRoutedMoA architecture from run_rr_moa.py but with
CrossEntropyLoss and n_classes output dim instead of forecast horizon.

Usage:
    python scripts/run_rrmoa_classification.py --dataset BasicMotions --seed 42
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.data import CLASSIFICATION_DATASETS
from scripts.run_rr_moa import RawRoutedMoA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BasicMotions",
                        choices=list(CLASSIFICATION_DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = "results/classification"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.dataset}_rrmoa_K{args.K}_top{args.top_k}_{args.seed}.json"

    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    # Load dataset
    loader_fn = CLASSIFICATION_DATASETS[args.dataset]
    ds = loader_fn()
    samples, labels = ds["samples"], ds["labels"]
    n_classes = len(np.unique(labels))
    print(f"Dataset: {args.dataset}, samples: {samples.shape}, classes: {n_classes}")

    # Load backbone
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    for p in model.parameters():
        p.requires_grad = False

    bb_lower = args.backbone.lower()
    bb_type = "moment"
    if "chronos" in bb_lower: bb_type = "chronos"
    elif "timer" in bb_lower: bb_type = "timer"
    elif "moirai" in bb_lower: bb_type = "moirai"

    # Build RR-MoA with classification output
    adapter = RawRoutedMoA(
        d_model=hdim, output_dim=n_classes,
        input_len=samples.shape[-1], K=args.K, hidden=64,
        top_k=args.top_k, router_arch="conv",
    ).to(args.device)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).long()),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).long()),
        batch_size=64,
    )

    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(args.epochs):
        model.eval()
        adapter.train()
        for bx, by in train_loader:
            bx_raw = bx.to(args.device)
            bx_enc = bx.to(args.device).unsqueeze(1)
            by = by.to(args.device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=args.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                               backbone_type=bb_type)
                pred = adapter(feat, bx_raw)
                loss = criterion(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    adapter.eval()
    correct, total = 0, 0
    all_routing = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx_raw = bx.to(args.device)
            bx_enc = bx.to(args.device).unsqueeze(1)
            by = by.to(args.device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=args.device)

            feat = _extract_features_batch(model, blocks, bx_enc, mask,
                                           backbone_type=bb_type)
            pred = adapter(feat, bx_raw)
            correct += (pred.argmax(dim=1) == by).sum().item()
            total += len(by)
            all_routing.append(adapter.get_routing_stats(bx_raw).cpu())

    accuracy = correct / total if total > 0 else 0.0
    routing = torch.cat(all_routing).mean(dim=0)
    routing_entropy = float(-(routing * (routing + 1e-12).log()).sum())
    elapsed = time.time() - t0

    # Also run single-head baselines for comparison
    from feasibility.config import AdapterConfig
    from feasibility.finetune import finetune_classification
    baseline_accs = {}
    for pooling in ["mean", "max", "last"]:
        cfg = AdapterConfig(
            adapter_type="linear_probe", unfreeze="frozen",
            head_type="linear", pooling=pooling,
            config_id=f"frozen_{pooling}_linear",
        )
        r = finetune_classification(cfg, samples, labels,
                                    device=args.device, n_epochs=args.epochs)
        baseline_accs[pooling] = r["accuracy"]

    best_baseline = max(baseline_accs.values())
    best_pooling = max(baseline_accs, key=baseline_accs.get)

    out = {
        "dataset": args.dataset,
        "method": "rr_moa",
        "K": args.K, "top_k": args.top_k,
        "seed": args.seed,
        "accuracy": accuracy,
        "routing_entropy": routing_entropy,
        "routing_weights": {k: float(v) for k, v in
                           zip(["mean", "last", "max", "attention", "conv1d"], routing.tolist())},
        "baselines": baseline_accs,
        "best_baseline_acc": best_baseline,
        "best_baseline_pooling": best_pooling,
        "delta_vs_best": accuracy - best_baseline,
        "n_classes": n_classes,
        "n_samples": int(samples.shape[0]),
        "elapsed": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"RR-MoA: {accuracy:.4f}  Best single: {best_pooling}={best_baseline:.4f}  "
          f"Δ={accuracy-best_baseline:+.4f}  entropy={routing_entropy:.3f}  → {out_path}")


if __name__ == "__main__":
    main()
