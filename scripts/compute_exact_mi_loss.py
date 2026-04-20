"""Exact MI loss measurement for Proposition 2 tightness validation.

For each dataset, measures the EXACT routing information lost to normalization:

    exact_loss = I(X; E) - I(S; E) = H(E|S)

Since E is deterministic given X (argmax router), I(X;E) = H(E), so:

    exact_loss = H(E) - I(S; E) = H(E|S)

H(E|S) is estimated via cross-entropy of an MLP classifier predicting E from S
on a held-out validation set. This bypasses the loose lower bound in Prop 2(ii)
and gives the exact quantity from Prop 2(i).

Additionally computes the bound gap:

    gap = I(M,Sigma; S|E) = sum_e P(E=e) * I(M,Sigma; S | E=e)

via within-expert CCA+KSG decomposition.

Usage:
    python scripts/compute_exact_mi_loss.py --device cuda
    python scripts/compute_exact_mi_loss.py --dataset ETTh1 --seed 42 --device cuda
    python scripts/compute_exact_mi_loss.py --all --device cuda
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_routing_signal_ratio import (
    load_channel_standardized_windows,
    compute_routing_signal_ratio,
)
from scripts.compute_direct_mi import build_router, DELTA_PCT

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
            "Traffic", "Exchange", "Solar"]

LOG_K = math.log(5)  # 1.609 nats


# ---------------------------------------------------------------------------
# Router training (modified from compute_direct_mi.py with stronger entropy reg)
# ---------------------------------------------------------------------------

def train_balanced_router(windows, K=5, n_epochs=20, batch_size=256,
                          entropy_coef=1.0, device="cuda", seed=42):
    """Train Conv1d router with strong entropy regularization for balanced experts.

    Key difference from compute_direct_mi.train_router_only:
    - entropy_coef=1.0 (up from 0.5) to ensure min expert usage >= ~5%
    - 20 epochs (up from 15) for convergence with stronger regularization
    - Seed control for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_len = windows.shape[1]
    forecast_horizon = 96

    X = torch.from_numpy(windows).float()
    Y = X[:, :forecast_horizon].clone()

    router, router_head = build_router(input_len, K)
    experts = nn.ModuleList([nn.Linear(input_len, forecast_horizon) for _ in range(K)])

    model = nn.ModuleDict({
        "router": router, "router_head": router_head, "experts": experts
    }).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model.train()
    for epoch in range(n_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            feat = model["router"](bx.unsqueeze(1)).flatten(1)
            logits = model["router_head"](feat)
            weights = torch.softmax(logits, dim=-1)
            topk_vals, topk_idx = weights.topk(2, dim=-1)
            topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            pred = torch.zeros_like(by)
            for j in range(2):
                expert_idx = topk_idx[:, j]
                w = topk_weights[:, j].unsqueeze(-1)
                for k in range(K):
                    mask = (expert_idx == k)
                    if mask.any():
                        pred[mask] += w[mask] * model["experts"][k](bx[mask])
            loss = nn.functional.mse_loss(pred, by)
            ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            loss = loss - entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract expert assignments
    model.eval()
    all_E = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            feat = model["router"](batch.unsqueeze(1)).flatten(1)
            logits = model["router_head"](feat)
            all_E.append(logits.argmax(dim=-1).cpu().numpy())

    return np.concatenate(all_E)


# ---------------------------------------------------------------------------
# H(E) computation
# ---------------------------------------------------------------------------

def compute_H_E(E, K=5):
    """Shannon entropy of expert assignment distribution in nats."""
    counts = Counter(E)
    total = len(E)
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log(p)
    return H


# ---------------------------------------------------------------------------
# H(E|S) estimation via MLP classifier
# ---------------------------------------------------------------------------

def estimate_H_E_given_S(S, E, K=5, seed=42, device="cuda"):
    """Estimate H(E|S) via held-out cross-entropy of an MLP classifier.

    A well-calibrated classifier's cross-entropy loss on held-out data
    upper-bounds H(E|S), converging to it with sufficient capacity and data.

    Returns dict with H_E_given_S, classifier_accuracy, and diagnostics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, D = S.shape  # D = 512 (shape dimension)

    # Stratified 80/20 split (fall back to non-stratified if a class has <2 members)
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    min_class_count = min(Counter(E).values())
    if min_class_count >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    else:
        sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(S, E))

    S_train, S_val = S[train_idx], S[val_idx]
    E_train, E_val = E[train_idx], E[val_idx]

    # Standardize shape features (important for MLP convergence)
    S_mean = S_train.mean(axis=0)
    S_std = S_train.std(axis=0) + 1e-8
    S_train_norm = (S_train - S_mean) / S_std
    S_val_norm = (S_val - S_mean) / S_std

    X_tr = torch.from_numpy(S_train_norm).float()
    Y_tr = torch.from_numpy(E_train).long()
    X_va = torch.from_numpy(S_val_norm).float()
    Y_va = torch.from_numpy(E_val).long()

    # MLP: 512 -> 256 -> 128 -> K
    classifier = nn.Sequential(
        nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, K),
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=256,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_val_ce = float("inf")
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(50):
        # Train
        classifier.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = classifier(bx)
            loss = nn.functional.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_va.to(device))
            val_ce = nn.functional.cross_entropy(val_logits, Y_va.to(device)).item()
            val_acc = (val_logits.argmax(dim=-1).cpu() == Y_va).float().mean().item()

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Temperature scaling for calibration on best model
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_va.to(device))

    # Learn temperature
    temperature = nn.Parameter(torch.ones(1, device=device))
    temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def temp_closure():
        temp_optimizer.zero_grad()
        scaled = val_logits / temperature
        loss = nn.functional.cross_entropy(scaled, Y_va.to(device))
        loss.backward()
        return loss

    temp_optimizer.step(temp_closure)

    # Calibrated cross-entropy = H(E|S) estimate
    with torch.no_grad():
        calibrated_logits = val_logits / temperature
        calibrated_ce = nn.functional.cross_entropy(
            calibrated_logits, Y_va.to(device)).item()
        val_acc = (calibrated_logits.argmax(dim=-1).cpu() == Y_va).float().mean().item()

    return {
        "H_E_given_S": float(calibrated_ce),
        "H_E_given_S_uncalibrated": float(best_val_ce),
        "temperature": float(temperature.item()),
        "classifier_accuracy": float(val_acc),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }


# ---------------------------------------------------------------------------
# Within-expert I(M,Sigma; S|E) decomposition
# ---------------------------------------------------------------------------

def compute_within_expert_leakage(windows, E, K=5, min_expert_n=50):
    """Compute I(M,Sigma; S|E) = sum_e P(e) * I(M,Sigma; S | E=e).

    Uses PCA dimensionality reduction on S before CCA to avoid spurious
    correlations when n_samples < n_features (512-dim shape).
    Then applies Gaussian reference on the reduced-space canonical correlations.
    Skips experts with fewer than min_expert_n windows.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA

    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    sigma_safe = np.clip(sigma, 1e-8, None)
    S = (windows - mu[:, None]) / sigma_safe[:, None]
    loc_scale = np.column_stack([mu, sigma])

    counts = Counter(E)
    total = len(E)

    weighted_mi = 0.0
    expert_details = {}

    for e in range(K):
        mask = (E == e)
        n_e = mask.sum()
        p_e = n_e / total

        if n_e < min_expert_n:
            expert_details[e] = {
                "n": int(n_e), "p": float(p_e),
                "skipped": True, "reason": "n < %d" % min_expert_n,
            }
            continue

        ls_e = loc_scale[mask]
        S_e = S[mask]

        # Reduce S to at most n_e//3 dimensions (well below n_samples) to
        # prevent CCA from fitting noise when S has 512 dims but expert
        # group has only ~100 samples.
        n_pca = min(10, n_e // 3, S_e.shape[1])
        if n_pca < 2:
            expert_details[e] = {
                "n": int(n_e), "p": float(p_e),
                "skipped": True, "reason": "too few samples for PCA+CCA",
            }
            continue

        try:
            pca = PCA(n_components=n_pca, random_state=42)
            S_e_reduced = pca.fit_transform(S_e)

            n_comp = min(2, n_pca, n_e - 1)
            cca = CCA(n_components=n_comp, max_iter=1000)
            cca.fit(ls_e, S_e_reduced)
            U, V = cca.transform(ls_e, S_e_reduced)
            rho = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1]
                            for i in range(n_comp)])
            # Gaussian reference on reduced space (lower bound on true MI
            # since PCA discards some S variance, but avoids overestimation)
            eps_e = -0.5 * np.sum(np.log(1 - np.clip(rho ** 2, 0, 0.9999)))
        except Exception as ex:
            expert_details[e] = {
                "n": int(n_e), "p": float(p_e),
                "skipped": True, "reason": str(ex),
            }
            continue

        weighted_mi += p_e * eps_e
        expert_details[e] = {
            "n": int(n_e), "p": float(p_e),
            "eps_gauss": float(eps_e),
            "rho": [float(r) for r in rho],
            "n_pca": n_pca,
            "skipped": False,
        }

    return {
        "I_MS_S_given_E": float(weighted_mi),
        "expert_details": expert_details,
        "n_experts_used": sum(1 for d in expert_details.values() if not d.get("skipped", True)),
    }


# ---------------------------------------------------------------------------
# Unconditional I(M,Sigma; E) via CCA+Gaussian (consistent with leakage estimator)
# ---------------------------------------------------------------------------

def compute_I_MS_E_consistent(windows, E, K=5):
    """Estimate I(M,Sigma; E) using a method consistent with the leakage estimator.

    Since E is discrete (K=5), we use:
        I(M,Sigma; E) = H(E) - H(E | M,Sigma)

    H(E|M,Sigma) estimated via k-NN conditional entropy in (mu, sigma) space.
    Uses k=5 to match the KSG leakage estimator's neighbor count.
    """
    from sklearn.neighbors import NearestNeighbors

    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    X = np.column_stack([mu, sigma])

    # Standardize for k-NN
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    H_E = compute_H_E(E, K)

    k_nn = 5  # match KSG k parameter
    nn_model = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
    nn_model.fit(X_norm)
    _, indices = nn_model.kneighbors(X_norm)
    indices = indices[:, 1:]  # exclude self

    H_E_given_MS = 0.0
    for i in range(len(E)):
        neighbor_E = E[indices[i]]
        local_counts = Counter(neighbor_E)
        local_H = -sum((c / k_nn) * math.log(c / k_nn)
                        for c in local_counts.values())
        H_E_given_MS += local_H
    H_E_given_MS /= len(E)

    return {
        "I_MS_E": float(max(H_E - H_E_given_MS, 0.0)),
        "H_E": float(H_E),
        "H_E_given_MS_knn": float(H_E_given_MS),
        "k_nn": k_nn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_dataset(dataset_name, seed=42, device="cuda"):
    """Run full exact MI loss analysis for one dataset and seed."""
    print("  Loading windows...")
    windows = load_channel_standardized_windows(dataset_name)
    N = len(windows)
    print("  %d windows loaded" % N)

    # Signal ratio R(D)
    sr = compute_routing_signal_ratio(windows)
    R = sr["ratio_R"]

    # Train router with balanced entropy
    print("  Training balanced router (seed=%d)..." % seed)
    t0 = time.time()
    E = train_balanced_router(windows, K=5, device=device, seed=seed)
    print("  Router trained in %.1fs" % (time.time() - t0))

    # Expert distribution
    unique, counts = np.unique(E, return_counts=True)
    expert_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    expert_balance = [int(counts[i]) / N if i < len(counts) else 0
                      for i in range(5)]
    min_expert_frac = min(expert_balance) if expert_balance else 0
    print("  Expert dist: %s (min frac=%.3f)" % (expert_dist, min_expert_frac))

    # H(E)
    H_E = compute_H_E(E, K=5)
    print("  H(E) = %.4f nats (max=%.4f)" % (H_E, LOG_K))

    # Compute per-window shape S
    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    sigma_safe = np.clip(sigma, 1e-8, None)
    S = (windows - mu[:, None]) / sigma_safe[:, None]

    # H(E|S) via MLP classifier
    print("  Training shape classifier for H(E|S)...")
    t0 = time.time()
    cls_result = estimate_H_E_given_S(S, E, K=5, seed=seed, device=device)
    H_E_given_S = cls_result["H_E_given_S"]
    print("  H(E|S) = %.4f nats (acc=%.1f%%, T=%.2f) [%.1fs]" % (
        H_E_given_S, cls_result["classifier_accuracy"] * 100,
        cls_result["temperature"], time.time() - t0))

    # Exact MI loss and retention
    exact_mi_loss = H_E_given_S
    I_S_E = H_E - H_E_given_S
    retention = I_S_E / H_E if H_E > 1e-6 else 0.0
    print("  Exact MI loss = %.4f nats (%.1f%% of H(E))" % (
        exact_mi_loss, exact_mi_loss / H_E * 100 if H_E > 1e-6 else 0))
    print("  Retention = %.1f%%" % (retention * 100))

    # Leakage epsilon (unconditional I(M,Sigma; S))
    from scripts.compute_ksg_leakage import compute_leakage
    leakage = compute_leakage(dataset_name)
    epsilon = leakage["eps_gauss"]
    print("  epsilon (Gauss) = %.4f nats" % epsilon)

    # I(M,Sigma; E) with consistent estimator
    mi_result = compute_I_MS_E_consistent(windows, E, K=5)
    I_MS_E = mi_result["I_MS_E"]
    print("  I(M,Sigma;E) [k=5 kNN] = %.4f nats" % I_MS_E)

    # Lower bound
    lower_bound = max(I_MS_E - epsilon, 0.0)
    print("  Lower bound = max(%.4f - %.4f, 0) = %.4f" % (I_MS_E, epsilon, lower_bound))

    # Within-expert gap I(M,Sigma; S|E)
    print("  Computing within-expert I(M,Sigma; S|E)...")
    gap_result = compute_within_expert_leakage(windows, E, K=5)
    I_MS_S_given_E = gap_result["I_MS_S_given_E"]
    print("  I(M,Sigma; S|E) = %.4f nats" % I_MS_S_given_E)

    # Consistency check: H(E|S) should ≈ lower_bound + I(M,Sigma;S|E)
    reconstructed = lower_bound + I_MS_S_given_E
    consistency_gap = abs(exact_mi_loss - reconstructed)
    print("  Consistency: H(E|S)=%.4f vs LB+gap=%.4f (diff=%.4f)" % (
        exact_mi_loss, reconstructed, consistency_gap))

    delta_pct = DELTA_PCT.get(dataset_name)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_windows": N,
        "R": float(R),
        "delta_pct": delta_pct,
        # Core exact quantities
        "H_E": float(H_E),
        "H_E_given_S": float(H_E_given_S),
        "I_S_E": float(I_S_E),
        "exact_mi_loss": float(exact_mi_loss),
        "retention": float(retention),
        # Bound quantities
        "I_MS_E": float(I_MS_E),
        "epsilon": float(epsilon),
        "epsilon_ksg": float(leakage["eps_ksg"]),
        "lower_bound": float(lower_bound),
        # Gap decomposition
        "I_MS_S_given_E": float(I_MS_S_given_E),
        "gap_ratio_logK": float(I_MS_S_given_E / LOG_K),
        # Consistency
        "reconstructed": float(reconstructed),
        "consistency_gap": float(consistency_gap),
        # Diagnostics
        "classifier_accuracy": float(cls_result["classifier_accuracy"]),
        "temperature": float(cls_result["temperature"]),
        "expert_dist": expert_dist,
        "expert_balance": [float(b) for b in expert_balance],
        "min_expert_frac": float(min_expert_frac),
        "n_experts_used_gap": gap_result["n_experts_used"],
    }


def main():
    parser = argparse.ArgumentParser(description="Exact MI loss for Prop 2 tightness")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to run (default: all)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 9 datasets")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: run seeds 42,43,44)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print("Device: %s" % device)

    datasets = [args.dataset] if args.dataset else DATASETS
    seeds = [args.seed] if args.seed is not None else [42, 43, 44]

    all_results = {}

    for ds in datasets:
        print("\n" + "=" * 60)
        print("Dataset: %s" % ds)
        print("=" * 60)

        try:
            seed_results = []
            for s in seeds:
                r = run_one_dataset(ds, seed=s, device=device)
                seed_results.append(r)

            # Aggregate across seeds
            agg = {}
            for key in ["H_E", "H_E_given_S", "I_S_E", "exact_mi_loss", "retention",
                         "I_MS_E", "epsilon", "lower_bound", "I_MS_S_given_E",
                         "classifier_accuracy"]:
                vals = [sr[key] for sr in seed_results]
                agg[key + "_mean"] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))

            agg["R"] = seed_results[0]["R"]  # R is data property, same across seeds
            agg["delta_pct"] = seed_results[0]["delta_pct"]
            agg["n_windows"] = seed_results[0]["n_windows"]
            agg["seeds"] = seeds
            agg["per_seed"] = seed_results

            all_results[ds] = agg

            print("\n  --- Aggregated (n=%d seeds) ---" % len(seeds))
            print("  H(E)     = %.4f +/- %.4f" % (agg["H_E_mean"], agg["H_E_std"]))
            print("  H(E|S)   = %.4f +/- %.4f" % (agg["H_E_given_S_mean"], agg["H_E_given_S_std"]))
            print("  Exact MI = %.4f +/- %.4f" % (agg["exact_mi_loss_mean"], agg["exact_mi_loss_std"]))
            print("  Retention = %.1f%% +/- %.1f%%" % (
                agg["retention_mean"] * 100, agg["retention_std"] * 100))
            print("  LB       = %.4f +/- %.4f" % (agg["lower_bound_mean"], agg["lower_bound_std"]))

        except Exception as e:
            print("  FAILED: %s" % e)
            import traceback
            traceback.print_exc()

    # Correlation analysis across datasets
    if len(all_results) >= 5:
        from scipy.stats import spearmanr

        ds_list = sorted(all_results.keys())
        R_vals = [all_results[d]["R"] for d in ds_list]
        delta_vals = [all_results[d]["delta_pct"] for d in ds_list]
        exact_vals = [all_results[d]["exact_mi_loss_mean"] for d in ds_list]
        retention_vals = [all_results[d]["retention_mean"] for d in ds_list]

        rho_exact_delta, p_exact_delta = spearmanr(exact_vals, delta_vals)
        rho_ret_delta, p_ret_delta = spearmanr(retention_vals, delta_vals)
        rho_R_delta, p_R_delta = spearmanr(R_vals, delta_vals)

        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS (n=%d datasets)" % len(ds_list))
        print("=" * 60)
        print("H(E|S) vs Δ%%:      ρ = %.3f (p = %.4f)" % (rho_exact_delta, p_exact_delta))
        print("Retention vs Δ%%:   ρ = %.3f (p = %.4f)" % (rho_ret_delta, p_ret_delta))
        print("R(D) vs Δ%%:        ρ = %.3f (p = %.4f)" % (rho_R_delta, p_R_delta))

        all_results["_correlations"] = {
            "rho_exact_delta": float(rho_exact_delta),
            "p_exact_delta": float(p_exact_delta),
            "rho_retention_delta": float(rho_ret_delta),
            "p_retention_delta": float(p_ret_delta),
            "rho_R_delta": float(rho_R_delta),
            "p_R_delta": float(p_R_delta),
            "n_datasets": len(ds_list),
        }

        # Summary table
        print("\n%-12s  %5s  %6s  %6s  %6s  %6s  %5s  %6s" % (
            "Dataset", "R", "H(E)", "H(E|S)", "Ret%", "LB", "Gap", "Δ%"))
        print("-" * 72)
        for d in ds_list:
            r = all_results[d]
            print("%-12s  %5.2f  %6.3f  %6.3f  %5.1f%%  %6.3f  %5.3f  %+5.1f%%" % (
                d, r["R"], r["H_E_mean"], r["H_E_given_S_mean"],
                r["retention_mean"] * 100, r["lower_bound_mean"],
                r["I_MS_S_given_E_mean"], r["delta_pct"]))

    # Save
    os.makedirs("results/analysis", exist_ok=True)
    out_path = "results/analysis/exact_mi_loss.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
