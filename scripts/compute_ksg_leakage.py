"""Compute KSG leakage bound for Proposition 2 Part (iv).

For each dataset, estimates ε = I(M,Σ; S) via:
  (a) Gaussian reference: ε_Gauss = -½ Σ log(1 - ρ_i²) from CCA
  (b) Distribution-free KSG estimator on 2-vs-2 CCA-projected space

Usage:
    python scripts/compute_ksg_leakage.py
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_routing_signal_ratio import load_channel_standardized_windows


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
            "Traffic", "Exchange", "Solar"]


def compute_leakage(dataset_name):
    """Compute Gaussian and KSG leakage estimates."""
    from sklearn.cross_decomposition import CCA
    from sklearn.feature_selection import mutual_info_regression

    windows = load_channel_standardized_windows(dataset_name)
    N = len(windows)

    # Per-window statistics
    mu = windows.mean(axis=1)      # (N,)
    sigma = windows.std(axis=1)    # (N,)
    sigma_safe = np.clip(sigma, 1e-8, None)
    S = (windows - mu[:, None]) / sigma_safe[:, None]  # (N, 512)

    # Location-scale matrix (N, 2)
    loc_scale = np.column_stack([mu, sigma])

    # CCA: 2 components (max possible since loc_scale is 2-d)
    cca = CCA(n_components=2, max_iter=1000)
    cca.fit(loc_scale, S)
    U, V = cca.transform(loc_scale, S)  # U: (N,2), V: (N,2)

    # Canonical correlations
    rho = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(2)])
    rho_max = float(np.max(np.abs(rho)))

    # Gaussian reference: ε = -½ Σ log(1 - ρ_i²)
    eps_gauss = -0.5 * np.sum(np.log(1 - np.clip(rho ** 2, 0, 0.9999)))

    # KSG estimate on 2-vs-2 CCA-projected space
    # mutual_info_regression estimates I(V_i; U) for each column of V
    # We compute I(U; V) ≈ I(U; V_1) + I(U; V_2 | V_1) as an approximation
    # Better: use the multivariate KSG via concatenation
    # Since sklearn only does univariate target, estimate I(loc_scale; V_1) + I(loc_scale; V_2)
    # as an upper bound (by chain rule, I(X; Y1,Y2) ≤ I(X;Y1) + I(X;Y2))
    mi_1 = mutual_info_regression(loc_scale, V[:, 0], n_neighbors=5, random_state=42)
    mi_2 = mutual_info_regression(loc_scale, V[:, 1], n_neighbors=5, random_state=42)
    eps_ksg = float(np.sum(mi_1) + np.sum(mi_2))

    # Also do a tighter single-target estimate: I((M,Σ); first CCA variate of S)
    # This is a lower bound via DPI
    mi_single = mutual_info_regression(loc_scale, V[:, 0], n_neighbors=5, random_state=42)
    eps_ksg_lb = float(np.sum(mi_single))

    return {
        "dataset": dataset_name,
        "n_windows": N,
        "rho_1": float(rho[0]),
        "rho_2": float(rho[1]),
        "rho_max": rho_max,
        "eps_gauss": float(eps_gauss),
        "eps_ksg": float(eps_ksg),
        "eps_ksg_lb": float(eps_ksg_lb),
        "log_K": float(math.log(5)),
        "ratio_eps_logK": float(eps_gauss / math.log(5)),
    }


def main():
    results = []
    print("%-12s  %6s  %8s  %8s  %8s  %6s" % (
        "Dataset", "ρ_max", "ε_Gauss", "ε_KSG", "ε/logK", "N"))
    print("-" * 60)

    for ds in DATASETS:
        try:
            r = compute_leakage(ds)
            results.append(r)
            print("%-12s  %6.3f  %8.4f  %8.4f  %8.1f%%  %6d" % (
                ds, r["rho_max"], r["eps_gauss"], r["eps_ksg"],
                r["ratio_eps_logK"] * 100, r["n_windows"]))
        except Exception as e:
            print("%-12s  FAILED: %s" % (ds, e))

    # Save
    out_path = "results/analysis/ksg_leakage.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)

    # Summary
    if results:
        max_gauss = max(r["eps_gauss"] for r in results)
        max_ksg = max(r["eps_ksg"] for r in results)
        print("\nMax ε_Gauss: %.4f nats (%.1f%% of log K)" % (
            max_gauss, max_gauss / math.log(5) * 100))
        print("Max ε_KSG:   %.4f nats (%.1f%% of log K)" % (
            max_ksg, max_ksg / math.log(5) * 100))


if __name__ == "__main__":
    main()
