#!/usr/bin/env python3
"""N2a: Per-window regime robustness analysis.

Tests whether RR-MoA's raw-signal router adapts to different temporal
regimes within the test set. Partitions test windows into quartiles by
amplitude and volatility, then compares per-quartile MSE uniformity
(coefficient of variation) for RR-MoA vs best-fixed-adapter.

Usage:
    python3 scripts/run_n2_regime_robustness.py --dataset ETTh1 --seed 42 --device cuda
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feasibility.model import (load_backbone, _get_encoder_blocks, _get_hidden_dim,
                                _disable_gradient_checkpointing)
from feasibility.finetune import _extract_features_batch
from scripts.run_standard_evolution import (load_standard_data, _detect_backbone_type,
                                             SEED_ADAPTERS)
from scripts.run_rr_moa import RawRoutedMoA, HEAD_CLASSES, HEAD_NAMES
from scripts.run_patchwise_analysis import compute_temporal_stats


def train_and_eval_per_sample(adapter, model, blocks, X_train, Y_train, X_test, Y_test,
                               device, backbone_type, n_epochs=15, batch_size=128,
                               raw_input_for_adapter=False, X_train_raw=None, X_test_raw=None):
    """Train adapter and return per-sample MSE on test set.

    Args:
        raw_input_for_adapter: if True, adapter.forward(feat, bx_raw) — for RR-MoA
        X_train_raw, X_test_raw: raw inputs (needed if raw_input_for_adapter)

    Returns:
        per_sample_mse: (N_test,) numpy array
        aggregate_mse: float
    """
    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    # Training
    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                if raw_input_for_adapter:
                    pred = adapter(feat, bx_raw)
                else:
                    pred = adapter(feat)
                loss = mse_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval — collect per-sample MSE
    model.eval(); adapter.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    per_sample_mses = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in eval_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            if raw_input_for_adapter:
                pred = adapter(feat, bx_raw)
            else:
                pred = adapter(feat)
            # Per-sample MSE: mean across forecast horizon
            sample_mse = ((pred.float() - by.float()) ** 2).mean(dim=-1)
            per_sample_mses.append(sample_mse.cpu())

    per_sample_mse = torch.cat(per_sample_mses).numpy()
    aggregate_mse = float(per_sample_mse.mean())
    return per_sample_mse, aggregate_mse


def quartile_analysis(per_sample_mse, stat_values, stat_name):
    """Partition samples by quartiles of stat_values, compute per-quartile MSE."""
    q25, q50, q75 = np.percentile(stat_values, [25, 50, 75])
    boundaries = [-np.inf, q25, q50, q75, np.inf]
    quartile_mses = []
    quartile_counts = []
    for i in range(4):
        mask = (stat_values >= boundaries[i]) & (stat_values < boundaries[i + 1])
        if mask.sum() == 0:
            quartile_mses.append(float('nan'))
            quartile_counts.append(0)
        else:
            quartile_mses.append(float(per_sample_mse[mask].mean()))
            quartile_counts.append(int(mask.sum()))

    quartile_mses = np.array(quartile_mses)
    valid = ~np.isnan(quartile_mses)
    cv = float(np.std(quartile_mses[valid]) / np.mean(quartile_mses[valid])) if valid.sum() > 1 else 0.0

    return {
        f"quartile_mse_by_{stat_name}": [round(x, 6) for x in quartile_mses],
        f"quartile_counts_by_{stat_name}": quartile_counts,
        f"cv_{stat_name}": round(cv, 4),
        f"boundaries_{stat_name}": [round(q25, 4), round(q50, 4), round(q75, 4)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    out_dir = "results/regime_robustness"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.dataset}_H{args.horizon}_{args.seed}.json"

    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    # Load backbone
    bb_type = _detect_backbone_type(args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    hdim = _get_hidden_dim(model)

    # Compute temporal stats for test windows
    print(f"Computing temporal stats for {len(X_test)} test windows...")
    stats = compute_temporal_stats(X_test)

    results = {"dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
               "n_test": len(X_test), "methods": {}}

    # --- Method 1: RR-MoA (raw router) ---
    print("Training RR-MoA (raw router)...")
    torch.manual_seed(args.seed)
    adapter_rrmoa = RawRoutedMoA(hdim, args.horizon, input_len=512, K=5, hidden=64,
                                  top_k=2, router_input_mode="raw").to(args.device)
    ps_mse_rrmoa, agg_mse_rrmoa = train_and_eval_per_sample(
        adapter_rrmoa, model, blocks, X_train, Y_train, X_test, Y_test,
        args.device, bb_type, n_epochs=args.epochs,
        raw_input_for_adapter=True)
    print(f"  RR-MoA MSE={agg_mse_rrmoa:.4f}")

    rrmoa_result = {"mse_overall": round(agg_mse_rrmoa, 6)}
    for stat_name in ["amplitude", "volatility"]:
        rrmoa_result.update(quartile_analysis(ps_mse_rrmoa, stats[stat_name], stat_name))
    results["methods"]["rr_moa_raw"] = rrmoa_result

    # --- Method 2: Best fixed adapter (linear head = SEED_ADAPTERS[0]) ---
    print("Training best-fixed adapter (linear head)...")
    torch.manual_seed(args.seed)
    namespace = {"torch": torch, "nn": nn, "F": F, "math": __import__("math")}
    exec(SEED_ADAPTERS[0], namespace)
    adapter_fixed = namespace["Adapter"](hdim, args.horizon).to(args.device)
    ps_mse_fixed, agg_mse_fixed = train_and_eval_per_sample(
        adapter_fixed, model, blocks, X_train, Y_train, X_test, Y_test,
        args.device, bb_type, n_epochs=args.epochs,
        raw_input_for_adapter=False)
    print(f"  Fixed MSE={agg_mse_fixed:.4f}")

    fixed_result = {"mse_overall": round(agg_mse_fixed, 6)}
    for stat_name in ["amplitude", "volatility"]:
        fixed_result.update(quartile_analysis(ps_mse_fixed, stats[stat_name], stat_name))
    results["methods"]["best_fixed"] = fixed_result

    # --- Save ---
    results["elapsed"] = round(time.time() - t0, 1)
    results["quartile_stats"] = {
        "amplitude_mean": round(float(stats["amplitude"].mean()), 4),
        "amplitude_std": round(float(stats["amplitude"].std()), 4),
        "volatility_mean": round(float(stats["volatility"].mean()), 4),
        "volatility_std": round(float(stats["volatility"].std()), 4),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path} ({results['elapsed']}s)")

    # Quick summary
    for stat_name in ["amplitude", "volatility"]:
        cv_rr = rrmoa_result[f"cv_{stat_name}"]
        cv_fix = fixed_result[f"cv_{stat_name}"]
        print(f"  CV by {stat_name}: RR-MoA={cv_rr:.4f}, Fixed={cv_fix:.4f} "
              f"({'RR-MoA more uniform' if cv_rr < cv_fix else 'Fixed more uniform'})")


if __name__ == "__main__":
    main()
