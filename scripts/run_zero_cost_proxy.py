"""Temporal Zero-Cost Proxies (T-ZCP) for Adapter Architecture Search.

Scores adapter architectures at initialization (Epoch 0) using gradient-flow
metrics, without ANY training. If the proxy correlates with trained MSE,
we can replace 30-minute search with 300ms scoring.

Uses Synflow (Tanaka et al., 2020) adapted for TSFM adapters:
  Score(g) = sum(|theta * dL/dtheta|)

Validates by computing Spearman rank correlation between zero-cost scores
and known 15-epoch test MSEs from our existing experiment logs.

Usage:
    python scripts/run_zero_cost_proxy.py
    python scripts/run_zero_cost_proxy.py --dataset ETTh1
"""

import argparse
import json
import os
import sys
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import validate_adapter_code
from scripts.run_standard_evolution import load_standard_data, _detect_backbone_type


def compute_synflow(adapter, feat, target):
    """Compute Synflow score: sum(|theta * dL/dtheta|) at initialization.

    Higher score = more gradient flow = potentially better architecture.
    """
    adapter.train()

    # Make all params positive (Synflow convention)
    for p in adapter.parameters():
        p.data = p.data.abs()

    pred = adapter(feat)
    # Use sum of outputs as surrogate loss (Synflow uses product of all params)
    loss = pred.sum()

    adapter.zero_grad()
    loss.backward()

    score = 0.0
    for p in adapter.parameters():
        if p.grad is not None:
            score += (p.data * p.grad.data).abs().sum().item()

    return score


def compute_grad_norm(adapter, feat, target):
    """Compute gradient norm at initialization: ||dL/dtheta||.

    Higher norm = steeper landscape = potentially faster learning.
    """
    adapter.train()
    pred = adapter(feat)
    loss = nn.MSELoss()(pred, target)

    adapter.zero_grad()
    loss.backward()

    total_norm = 0.0
    for p in adapter.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_jacobian_score(adapter, feat):
    """Compute Jacobian covariance score (NASWOT-inspired).

    Measures the log-determinant of the correlation matrix of adapter outputs.
    Higher = more diverse feature maps = better architecture.
    """
    adapter.eval()
    with torch.no_grad():
        out = adapter(feat)  # (B, output_dim)

    # Correlation matrix of outputs across batch
    if out.shape[0] < 2:
        return 0.0
    out_centered = out - out.mean(dim=0, keepdim=True)
    cov = (out_centered.T @ out_centered) / (out.shape[0] - 1)

    # Log-determinant (numerically stable)
    try:
        sign, logdet = torch.linalg.slogdet(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
        return logdet.item()
    except Exception:
        return 0.0


def score_adapter(code, model, blocks, X_batch, Y_batch, device, forecast_horizon, backbone_type):
    """Score a single adapter architecture using zero-cost proxies."""
    hdim = _get_hidden_dim(model)

    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)

    scores = {}
    for metric_name, metric_fn in [("synflow", compute_synflow), ("grad_norm", compute_grad_norm)]:
        try:
            # Fresh initialization each time
            torch.manual_seed(42)
            adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)

            # Get features from backbone
            bx = torch.from_numpy(X_batch).float().to(device).unsqueeze(1)
            by = torch.from_numpy(Y_batch).float().to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)

            model.eval()
            with torch.no_grad():
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)

            feat = feat.detach().requires_grad_(False)
            score = metric_fn(adapter, feat.clone().detach().requires_grad_(True), by)
            scores[metric_name] = score
        except Exception as e:
            scores[metric_name] = None

    # Jacobian score (no gradient needed)
    try:
        torch.manual_seed(42)
        adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)
        bx = torch.from_numpy(X_batch).float().to(device).unsqueeze(1)
        mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
        model.eval()
        with torch.no_grad():
            feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
        scores["jacobian"] = compute_jacobian_score(adapter, feat)
    except Exception:
        scores["jacobian"] = None

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/zero_cost_proxy", exist_ok=True)

    # Load model
    print("Loading backbone...")
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False

    # Load data (small batch for scoring)
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    # Use just 32 samples for zero-cost scoring
    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    # Load existing validated adapters with known MSEs
    evo_path = "results/standard_evolution/%s_H%d_42.json" % (args.dataset, args.horizon)
    if not os.path.exists(evo_path):
        print("ERROR: %s not found" % evo_path)
        return

    evo_data = json.load(open(evo_path))
    evolved = evo_data.get("evolved", [])

    # Also include seed adapters as baselines
    from feasibility.code_evolution import SEED_ADAPTERS
    baselines = evo_data.get("baselines", {})

    all_adapters = []
    for e in evolved:
        val = validate_adapter_code(e["code"], d_model=hdim, output_dim=args.horizon)
        if val["valid"]:
            all_adapters.append({"code": e["code"], "mse": e["test_mse"],
                                 "params": e.get("param_count", 0), "source": "evolved"})

    for i, code in enumerate(SEED_ADAPTERS):
        val = validate_adapter_code(code, d_model=hdim, output_dim=args.horizon)
        if val["valid"]:
            bl_name = ["linear", "mlp", "last_token", "attention", "conv"][i]
            bl_mse = baselines.get(bl_name, {}).get("mse", None)
            if bl_mse:
                all_adapters.append({"code": code, "mse": bl_mse,
                                     "params": val["param_count"], "source": "baseline_%s" % bl_name})

    print("Scoring %d adapters on %s (zero-cost, no training)..." % (len(all_adapters), args.dataset))

    # Score each adapter
    results = []
    for i, adapter_info in enumerate(all_adapters):
        scores = score_adapter(
            adapter_info["code"], model, blocks, X_batch, Y_batch,
            args.device, args.horizon, bb_type,
        )
        adapter_info["scores"] = scores
        results.append(adapter_info)

        if (i + 1) % 3 == 0 or i == len(all_adapters) - 1:
            print("  [%d/%d] MSE=%.4f synflow=%.2e grad=%.2e jacobian=%.2f" % (
                i + 1, len(all_adapters), adapter_info["mse"],
                scores.get("synflow", 0) or 0,
                scores.get("grad_norm", 0) or 0,
                scores.get("jacobian", 0) or 0))

    # Compute rank correlations
    print("\n" + "=" * 60)
    print("ZERO-COST PROXY CORRELATIONS: %s" % args.dataset)
    print("=" * 60)

    mses = [r["mse"] for r in results]

    for metric in ["synflow", "grad_norm", "jacobian"]:
        scores = [r["scores"].get(metric) for r in results]
        valid = [(m, s) for m, s in zip(mses, scores) if s is not None and s != 0]
        if len(valid) < 4:
            print("%-12s: insufficient data (%d valid)" % (metric, len(valid)))
            continue

        valid_mses = [v[0] for v in valid]
        valid_scores = [v[1] for v in valid]

        # Spearman correlation (rank-based)
        rho, p_val = scipy_stats.spearmanr(valid_mses, valid_scores)
        # Kendall tau
        tau, tau_p = scipy_stats.kendalltau(valid_mses, valid_scores)

        print("%-12s: Spearman ρ=%+.3f (p=%.4f)  Kendall τ=%+.3f (p=%.4f)  n=%d" % (
            metric, rho, p_val, tau, tau_p, len(valid)))

        # For synflow/grad_norm: NEGATIVE correlation is good (higher score → lower MSE)
        # For jacobian: could be either direction

    # Save
    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "n_adapters": len(results),
        "adapters": [{"mse": r["mse"], "params": r["params"], "source": r["source"],
                       "scores": r["scores"]} for r in results],
    }
    path = "results/zero_cost_proxy/%s_H%d.json" % (args.dataset, args.horizon)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved to %s" % path)


if __name__ == "__main__":
    main()
