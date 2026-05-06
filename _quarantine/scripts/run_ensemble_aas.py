"""Ensemble AAS: average predictions from top-K discovered adapters.

Instead of picking the single best adapter, uses all top-K adapters
discovered by AAS and averages their predictions at test time.
Zero additional search cost — the K adapters are already discovered.

Usage:
    python scripts/run_ensemble_aas.py --dataset ETTh1
    python scripts/run_ensemble_aas.py --dataset ETTh1 --top-k 3
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
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


def train_and_predict(code, model, blocks, X_train, Y_train, X_test,
                      device, n_epochs, forecast_horizon, backbone_type, batch_size=128):
    """Train adapter and return test predictions (not just MSE)."""
    hdim = _get_hidden_dim(model)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)
    adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)

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
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                loss = mse_fn(adapter(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Get predictions
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
    ), batch_size=batch_size)
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for (bx,) in test_loader:
            bx = bx.to(device).unsqueeze(1)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
            preds.append(adapter(feat).float().cpu())

    return torch.cat(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/ensemble_aas", exist_ok=True)

    # Load winning adapter codes from evolution results
    evo_path = "results/standard_evolution/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    if not os.path.exists(evo_path):
        print("ERROR: %s not found" % evo_path)
        return

    evo_data = json.load(open(evo_path))
    evolved = evo_data.get("evolved", [])
    if not evolved:
        print("ERROR: no evolved adapters in %s" % evo_path)
        return

    # Sort by test_mse, take top-K
    evolved_sorted = sorted(evolved, key=lambda x: x["test_mse"])
    top_k_codes = [e["code"] for e in evolved_sorted[:args.top_k]]
    print("Loaded top-%d adapters from %s" % (len(top_k_codes), evo_path))
    for i, e in enumerate(evolved_sorted[:args.top_k]):
        print("  #%d: test_mse=%.4f params=%d" % (i+1, e["test_mse"], e["param_count"]))

    # Load model
    print("\nLoading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    Y_test_tensor = torch.from_numpy(Y_test).float()
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    # Train each adapter and collect predictions
    all_preds = []
    individual_mses = []

    for i, code in enumerate(top_k_codes):
        # Reload model fresh for each adapter
        model_i = load_backbone(args.backbone, args.device)
        _disable_gradient_checkpointing(model_i)
        blocks_i = _get_encoder_blocks(model_i)
        for p in model_i.parameters():
            p.requires_grad = False
        for j in range(max(0, len(blocks_i)-4), len(blocks_i)):
            for p in blocks_i[j].parameters():
                p.requires_grad = True

        try:
            preds = train_and_predict(
                code, model_i, blocks_i, X_train, Y_train, X_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            mse_i = nn.MSELoss()(preds, Y_test_tensor).item()
            all_preds.append(preds)
            individual_mses.append(mse_i)
            print("  Adapter %d: MSE=%.4f" % (i+1, mse_i))
        except Exception as e:
            print("  Adapter %d: ERROR %s" % (i+1, e))

    if not all_preds:
        print("ERROR: no valid predictions")
        return

    # Ensemble: average predictions
    ensemble_pred = torch.stack(all_preds).mean(dim=0)
    ensemble_mse = nn.MSELoss()(ensemble_pred, Y_test_tensor).item()
    ensemble_mae = nn.L1Loss()(ensemble_pred, Y_test_tensor).item()

    # Also try top-3 ensemble
    if len(all_preds) >= 3:
        ensemble_top3_pred = torch.stack(all_preds[:3]).mean(dim=0)
        ensemble_top3_mse = nn.MSELoss()(ensemble_top3_pred, Y_test_tensor).item()
    else:
        ensemble_top3_mse = None

    # Get baseline
    baselines = evo_data.get("baselines", {})
    best_baseline_name = min(baselines, key=lambda k: baselines[k]["mse"]) if baselines else "conv"
    best_baseline_mse = baselines.get(best_baseline_name, {}).get("mse", float("inf"))

    best_individual = min(individual_mses)

    print("\n" + "=" * 60)
    print("ENSEMBLE AAS: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    print("Top-1 (best individual):  MSE=%.4f" % best_individual)
    if ensemble_top3_mse:
        print("Top-3 ensemble (avg):     MSE=%.4f  Δ vs top-1: %+.1f%%" % (
            ensemble_top3_mse, (ensemble_top3_mse - best_individual) / best_individual * 100))
    print("Top-%d ensemble (avg):     MSE=%.4f  Δ vs top-1: %+.1f%%" % (
        len(all_preds), ensemble_mse,
        (ensemble_mse - best_individual) / best_individual * 100))
    print("Best baseline (%s):      MSE=%.4f" % (best_baseline_name, best_baseline_mse))

    delta_ens_vs_baseline = (ensemble_mse - best_baseline_mse) / best_baseline_mse * 100
    delta_top1_vs_baseline = (best_individual - best_baseline_mse) / best_baseline_mse * 100
    print("\nEnsemble vs baseline: %+.1f%%" % delta_ens_vs_baseline)
    print("Top-1 vs baseline:    %+.1f%%" % delta_top1_vs_baseline)
    print("Ensemble vs Top-1:    %+.1f%%" % ((ensemble_mse - best_individual) / best_individual * 100))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "top_k": len(all_preds),
        "individual_mses": individual_mses,
        "ensemble_mse": ensemble_mse,
        "ensemble_mae": ensemble_mae,
        "ensemble_top3_mse": ensemble_top3_mse,
        "best_individual_mse": best_individual,
        "best_baseline_mse": best_baseline_mse,
        "best_baseline_name": best_baseline_name,
    }
    path = "results/ensemble_aas/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
