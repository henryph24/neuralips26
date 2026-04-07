"""Independent Ensemble Baseline: train 5 canonical experts independently, average predictions.

This is the natural routing-free alternative to RR-MoA: same 5 expert heads,
same frozen backbone, but NO learned router. Each expert is trained independently
and predictions are averaged at test time (uniform 1/5 weights).

Isolates the routing contribution: if RR-MoA > independent ensemble, then
learned per-sample routing adds value beyond simply having diverse experts.

Usage:
    python scripts/run_independent_ensemble.py --dataset ETTh1
    python scripts/run_independent_ensemble.py --dataset ETTh1 --seed 43 --unfreeze frozen
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from scripts.run_standard_evolution import (
    load_standard_data, _detect_backbone_type,
)

# Import the same 5 canonical expert heads used in RR-MoA
from scripts.run_rr_moa import HEAD_CLASSES, HEAD_NAMES, _apply_unfreeze


def train_single_expert(expert_cls, expert_name, model, blocks, X_train, Y_train,
                        X_test, device, n_epochs, forecast_horizon, backbone_type,
                        batch_size=128):
    """Train a single expert head and return test predictions."""
    hdim = _get_hidden_dim(model)
    adapter = expert_cls(hdim, forecast_horizon).to(device)

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in loader:
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat)
                loss = mse_fn(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Get test predictions
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
    ), batch_size=batch_size)

    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for (bx,) in test_loader:
            bx_enc = bx.to(device).unsqueeze(1)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(adapter(feat).float().cpu())

    return torch.cat(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--unfreeze", default="frozen", choices=["frozen", "last2", "last4"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/independent_ensemble", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    Y_test_tensor = torch.from_numpy(Y_test).float()
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    bb_type = _detect_backbone_type(args.backbone)
    all_preds = {}

    start = time.time()
    for cls, name in zip(HEAD_CLASSES, HEAD_NAMES):
        print("\nTraining expert: %s" % name)
        # Reload model fresh for each expert (so backbone state is independent)
        torch.manual_seed(args.seed)
        model = load_backbone(args.backbone, args.device)
        _disable_gradient_checkpointing(model)
        blocks = _get_encoder_blocks(model)
        for p in model.parameters():
            p.requires_grad = False
        _apply_unfreeze(blocks, args.unfreeze)

        preds = train_single_expert(
            cls, name, model, blocks, X_train, Y_train, X_test,
            device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
            backbone_type=bb_type, batch_size=args.batch_size,
        )
        individual_mse = nn.MSELoss()(preds, Y_test_tensor).item()
        all_preds[name] = preds
        print("  %s: MSE=%.4f" % (name, individual_mse))

        # Free GPU memory
        del model, blocks
        torch.cuda.empty_cache()

    elapsed = time.time() - start

    # Ensemble: uniform average of all 5 expert predictions
    ensemble_pred = torch.stack(list(all_preds.values())).mean(dim=0)
    ensemble_mse = nn.MSELoss()(ensemble_pred, Y_test_tensor).item()
    ensemble_mae = nn.L1Loss()(ensemble_pred, Y_test_tensor).item()

    individual_mses = {name: nn.MSELoss()(pred, Y_test_tensor).item()
                       for name, pred in all_preds.items()}
    best_individual = min(individual_mses.values())
    best_individual_name = min(individual_mses, key=individual_mses.get)

    print("\n=== Results ===")
    print("Ensemble (5-expert avg): MSE=%.4f  MAE=%.4f" % (ensemble_mse, ensemble_mae))
    print("Best individual (%s): MSE=%.4f" % (best_individual_name, best_individual))
    for name, mse in individual_mses.items():
        print("  %-12s MSE=%.4f" % (name, mse))

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "unfreeze": args.unfreeze, "backbone": args.backbone,
        "ensemble_mse": ensemble_mse, "ensemble_mae": ensemble_mae,
        "individual_mses": individual_mses,
        "best_individual_name": best_individual_name,
        "best_individual_mse": best_individual,
        "n_experts": len(HEAD_CLASSES),
        "expert_names": HEAD_NAMES,
        "elapsed": elapsed,
    }
    path = "results/independent_ensemble/%s_H%d_%s_%d.json" % (
        args.dataset, args.horizon, args.unfreeze, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
