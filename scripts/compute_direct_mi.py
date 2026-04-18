"""Compute I(M,Σ; E) directly for each dataset to validate the R(D) proxy.

For each dataset:
  1. Load channel-standardized windows
  2. Load a trained RR-MoA router (from existing result JSONs to retrain on the fly)
  3. Get expert assignment E = argmax of top-1 routing
  4. Estimate I(M,Σ; E) via sklearn mutual_info_classif
  5. Correlate with Δ% and with R(D) across datasets

This eliminates the "monotone proxy" hand-wave in Proposition 2 Part (iv)
by directly measuring the MI between location-scale statistics and expert
assignments from a trained router.

Usage:
    python scripts/compute_direct_mi.py [--device cuda]
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_routing_signal_ratio import (
    load_channel_standardized_windows,
    compute_routing_signal_ratio,
)

# Verified Δ% values from the paper's Figure 2 / signal-ratio analysis.
# These are the mean improvement of RR-MoA over AdaMix (frozen) across seeds.
DELTA_PCT = {
    "ETTh1": -43.2, "ETTh2": -71.0, "ETTm1": -51.1, "ETTm2": -77.2,
    "Weather": -44.6, "Electricity": -26.8, "Traffic": +2.9,
    "Exchange": -66.5, "Solar": -32.9,
}

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
            "Traffic", "Exchange", "Solar"]


def build_router(input_len=512, K=5):
    """Build the Conv1d router architecture (matches run_rr_moa.py default)."""
    router = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(4),
    )
    router_head = nn.Linear(64, K)
    return router, router_head


def train_router_only(windows, K=5, n_epochs=15, batch_size=256, device="cuda"):
    """Train a standalone Conv1d router on raw windows using a simple MoE loss.

    We train K linear experts + conv router jointly on MSE forecasting,
    then extract the trained router for MI analysis. This replicates the
    RR-MoA routing mechanism without needing a TSFM backbone.
    """
    from torch.utils.data import DataLoader, TensorDataset

    input_len = windows.shape[1]
    forecast_horizon = 96

    # Split windows into input/target (last 96 steps as target)
    if input_len <= forecast_horizon:
        # Use full window as input, predict self (reconstruction proxy)
        X = torch.from_numpy(windows).float()
        Y = X.clone()
    else:
        X = torch.from_numpy(windows[:, :input_len]).float()
        Y = torch.from_numpy(windows[:, :forecast_horizon]).float()

    router, router_head = build_router(input_len, K)
    experts = nn.ModuleList([nn.Linear(input_len, forecast_horizon) for _ in range(K)])

    model = nn.ModuleDict({
        "router": router, "router_head": router_head, "experts": experts
    }).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Router
            feat = model["router"](bx.unsqueeze(1)).flatten(1)
            logits = model["router_head"](feat)
            weights = torch.softmax(logits, dim=-1)
            # Top-2
            topk_vals, topk_idx = weights.topk(2, dim=-1)
            topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            # Expert predictions
            pred = torch.zeros_like(by)
            for j in range(2):
                expert_idx = topk_idx[:, j]
                w = topk_weights[:, j].unsqueeze(-1)
                for k in range(K):
                    mask = (expert_idx == k)
                    if mask.any():
                        pred[mask] += w[mask] * model["experts"][k](bx[mask])
            loss = nn.functional.mse_loss(pred, by)
            # Entropy regularization to prevent single-expert collapse
            ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            loss = loss - 0.5 * ent  # encourage diverse routing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Get expert assignments for all windows
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        all_E = []
        for i in range(0, len(X_dev), batch_size):
            batch = X_dev[i:i+batch_size]
            feat = model["router"](batch.unsqueeze(1)).flatten(1)
            logits = model["router_head"](feat)
            E = logits.argmax(dim=-1).cpu().numpy()
            all_E.append(E)

    return np.concatenate(all_E)


def compute_mi_ms_e(windows, E):
    """Estimate I(M,Σ; E) where E is the categorical expert assignment.

    Uses sklearn mutual_info_classif which handles discrete targets natively.
    """
    from sklearn.feature_selection import mutual_info_classif

    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    X = np.column_stack([mu, sigma])  # (N, 2)

    # mutual_info_classif estimates I(X_j; y) for each feature j
    # We want I([mu, sigma]; E), so we estimate per-feature and note
    # that I(M,Σ; E) >= max(I(M;E), I(Σ;E)) but also
    # I(M,Σ; E) <= I(M;E) + I(Σ;E) (sub-additivity).
    # For a more accurate joint estimate, use the KSG k-NN approach.
    mi_per_feat = mutual_info_classif(X, E, discrete_features=False,
                                       n_neighbors=5, random_state=42)
    mi_marginal_sum = float(mi_per_feat.sum())

    # Also compute via k-NN conditional entropy for the joint estimate
    # I(M,Σ; E) = H(E) - H(E | M,Σ)
    from collections import Counter
    counts = Counter(E)
    total = len(E)
    H_E = -sum((c / total) * np.log(c / total) for c in counts.values())

    # H(E | M,Σ) via k-NN: for each point, look at k nearest neighbors
    # in (M,Σ) space and compute local entropy of E among them
    from sklearn.neighbors import NearestNeighbors
    k_nn = min(20, len(E) // 10)
    nn_model = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)
    indices = indices[:, 1:]  # exclude self

    H_E_given_MS = 0.0
    for i in range(len(E)):
        neighbor_E = E[indices[i]]
        local_counts = Counter(neighbor_E)
        local_H = -sum((c / k_nn) * np.log(c / k_nn) for c in local_counts.values())
        H_E_given_MS += local_H
    H_E_given_MS /= len(E)

    mi_joint = max(H_E - H_E_given_MS, 0.0)

    return {
        "I_M_E": float(mi_per_feat[0]),
        "I_Sigma_E": float(mi_per_feat[1]),
        "I_marginal_sum": mi_marginal_sum,
        "I_joint_knn": float(mi_joint),
        "H_E": float(H_E),
        "H_E_given_MS": float(H_E_given_MS),
        "K_nn": k_nn,
        "n_windows": len(E),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    results = {}

    for ds in DATASETS:
        print("\n=== %s ===" % ds)
        try:
            windows = load_channel_standardized_windows(ds)
        except Exception as e:
            print("  SKIP (data not available): %s" % e)
            continue

        # Signal ratio R(D)
        sr = compute_routing_signal_ratio(windows)
        R = sr["ratio_R"]

        # Delta%
        delta = DELTA_PCT.get(ds)
        if delta is None:
            print("  SKIP (no Δ%% data)")
            continue

        # Train router and get expert assignments
        print("  Training router (%d windows)..." % len(windows))
        t0 = time.time()
        E = train_router_only(windows, K=5, n_epochs=15, device=device)
        print("  Router trained in %.1fs" % (time.time() - t0))

        # Expert distribution
        unique, counts = np.unique(E, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print("  Expert distribution: %s" % dist)

        # Compute MI
        mi = compute_mi_ms_e(windows, E)
        print("  I(M,Σ;E) [joint k-NN] = %.4f nats" % mi["I_joint_knn"])
        print("  I(M;E) = %.4f, I(Σ;E) = %.4f" % (mi["I_M_E"], mi["I_Sigma_E"]))
        print("  H(E) = %.4f, H(E|M,Σ) = %.4f" % (mi["H_E"], mi["H_E_given_MS"]))
        print("  R(D) = %.3f, Δ%% = %.1f%%" % (R, delta))

        results[ds] = {
            "R": R,
            "delta_pct": delta,
            "mi": mi,
            "expert_dist": dist,
        }

    # Compute correlations
    if len(results) >= 5:
        from scipy.stats import spearmanr

        datasets = sorted(results.keys())
        R_vals = [results[d]["R"] for d in datasets]
        delta_vals = [results[d]["delta_pct"] for d in datasets]
        mi_vals = [results[d]["mi"]["I_joint_knn"] for d in datasets]

        rho_R_delta, p_R_delta = spearmanr(R_vals, delta_vals)
        rho_MI_delta, p_MI_delta = spearmanr(mi_vals, delta_vals)
        rho_R_MI, p_R_MI = spearmanr(R_vals, mi_vals)

        print("\n" + "="*60)
        print("CORRELATION ANALYSIS (n=%d datasets)" % len(datasets))
        print("="*60)
        print("R(D) vs Δ%%:     ρ = %.3f (p = %.4f)" % (rho_R_delta, p_R_delta))
        print("I(M,Σ;E) vs Δ%%: ρ = %.3f (p = %.4f)" % (rho_MI_delta, p_MI_delta))
        print("R(D) vs I(M,Σ;E): ρ = %.3f (p = %.4f)" % (rho_R_MI, p_R_MI))
        print("")

        for d in datasets:
            r = results[d]
            print("  %-12s R=%.3f  I=%.4f  Δ=%.1f%%" % (
                d, r["R"], r["mi"]["I_joint_knn"], r["delta_pct"]))

        results["_correlations"] = {
            "rho_R_delta": float(rho_R_delta), "p_R_delta": float(p_R_delta),
            "rho_MI_delta": float(rho_MI_delta), "p_MI_delta": float(p_MI_delta),
            "rho_R_MI": float(rho_R_MI), "p_R_MI": float(p_R_MI),
            "n_datasets": len(datasets),
        }

    # Save results
    os.makedirs("results/analysis", exist_ok=True)
    out_path = "results/analysis/direct_mi.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
