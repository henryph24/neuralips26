"""Compute per-dataset routing signal ratio R and correlate with RR-MoA improvement.

Validates Theorem 1's prediction: datasets where RevIN strips more information
(higher R = Var(mu,sigma) / Var(shape)) should show larger RR-MoA improvements.

Usage:
    python scripts/analyze_routing_signal_ratio.py
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd
from urllib.request import urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
INPUT_LEN = 512
STRIDE = 64


def load_channel_standardized_windows(dataset_name):
    """Load channel-standardized windows (what the RR-MoA router actually sees).

    The pipeline applies StandardScaler per-channel on the training split, then
    creates sliding windows. RevIN additionally normalizes per-window on top.
    We compute the routing signal ratio on these channel-standardized windows
    to measure how much per-window variation RevIN strips.

    Returns array of shape (n_windows, seq_len).
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if dataset_name.startswith("ETT"):
        url = "%s/%s.csv" % (ETT_BASE, dataset_name)
        df = pd.read_csv(io.BytesIO(urlopen(url).read()))
    elif dataset_name == "Weather":
        df = pd.read_csv(os.path.join(data_dir, "weather.csv"))
    elif dataset_name == "Electricity":
        df = pd.read_csv(os.path.join(data_dir, "electricity.csv"))
    elif dataset_name == "Traffic":
        df = pd.read_csv(os.path.join(data_dir, "traffic.csv"))
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    from sklearn.preprocessing import StandardScaler

    values = df.iloc[:, 1:].values.astype(np.float32)
    n_ch = values.shape[1]

    # Fit scaler on first 60% (training split, matching pipeline)
    n_train = int(0.6 * len(values))
    scaler = StandardScaler()
    scaler.fit(values[:n_train])
    values_scaled = scaler.transform(values).astype(np.float32)

    # Create sliding windows per channel on scaled data
    windows = []
    for ch in range(n_ch):
        s = values_scaled[:, ch]
        for i in range(0, len(s) - INPUT_LEN + 1, STRIDE):
            windows.append(s[i:i + INPUT_LEN])

    windows = np.stack(windows)  # (n_windows, INPUT_LEN)

    # Subsample if too large
    if len(windows) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(windows), 10000, replace=False)
        windows = windows[idx]

    return windows


def compute_routing_signal_ratio(windows):
    """Compute R = Var(mu, sigma) / Var(shape) for a set of raw windows.

    mu_w = per-window mean (what RevIN strips)
    sigma_w = per-window std (what RevIN strips)
    s_w = (x_w - mu_w) / sigma_w = shape (what RevIN preserves)
    """
    # Per-window location-scale statistics
    mu_w = windows.mean(axis=1)       # (n_windows,)
    sigma_w = windows.std(axis=1)     # (n_windows,)

    # Shape after RevIN
    sigma_safe = np.clip(sigma_w, 1e-8, None)
    shape_w = (windows - mu_w[:, None]) / sigma_safe[:, None]

    # Variance of location-scale statistics
    var_mu = np.var(mu_w)
    var_sigma = np.var(sigma_w)
    location_scale_var = var_mu + var_sigma

    # Variance of shape (average variance across temporal positions)
    shape_var = np.mean(np.var(shape_w, axis=0))

    ratio = location_scale_var / max(shape_var, 1e-8)

    return {
        "var_mu": float(var_mu),
        "var_sigma": float(var_sigma),
        "location_scale_var": float(location_scale_var),
        "shape_var": float(shape_var),
        "ratio_R": float(ratio),
    }


def load_rrmoa_improvement(dataset_name):
    """Load RR-MoA improvement (Δ%) from result JSONs, averaged over 3 seeds."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "rr_moa")
    deltas = []
    for seed in [42, 43, 44]:
        path = os.path.join(results_dir, "%s_H96_K5_top2_frozen_%d.json" % (dataset_name, seed))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        rrmoa_mse = d["rr_moa"]["mse"]
        # Get best baseline MSE
        if d.get("baselines"):
            best_bl = min(d["baselines"].values(), key=lambda x: x["mse"])["mse"]
            delta = (rrmoa_mse - best_bl) / best_bl * 100
            deltas.append(delta)

    if not deltas:
        return None
    return float(np.mean(deltas))


def main():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]

    print("=" * 80)
    print("ROUTING SIGNAL RATIO ANALYSIS (Theorem 1 Prediction)")
    print("=" * 80)
    print()

    results = []
    for ds in datasets:
        print("Processing %s..." % ds)
        try:
            windows = load_channel_standardized_windows(ds)
            stats = compute_routing_signal_ratio(windows)
            delta = load_rrmoa_improvement(ds)
            stats["dataset"] = ds
            stats["delta_pct"] = delta
            stats["n_windows"] = len(windows)
            results.append(stats)
            print("  R = %.4f  (Var_mu=%.4f, Var_sigma=%.4f, Var_shape=%.4f)  Δ%% = %s" % (
                stats["ratio_R"], stats["var_mu"], stats["var_sigma"],
                stats["shape_var"],
                "%.1f%%" % delta if delta is not None else "N/A"))
        except Exception as e:
            print("  ERROR: %s" % e)

    print()

    # Correlation analysis
    valid = [(r["ratio_R"], r["delta_pct"]) for r in results if r["delta_pct"] is not None]
    if len(valid) >= 4:
        from scipy.stats import spearmanr, pearsonr
        Rs = [v[0] for v in valid]
        Ds = [v[1] for v in valid]  # Note: Δ is negative (improvement), more negative = better
        # We expect: higher R → more negative Δ (larger improvement)
        rho_s, p_s = spearmanr(Rs, Ds)
        rho_p, p_p = pearsonr(Rs, Ds)
        print("CORRELATION (R vs Δ%%, n=%d):" % len(valid))
        print("  Spearman ρ = %.3f  (p = %.4f)" % (rho_s, p_s))
        print("  Pearson  r = %.3f  (p = %.4f)" % (rho_p, p_p))
        print()
        print("  Interpretation: negative correlation means higher R → larger improvement")
        print("  (more negative Δ%%), validating Theorem 1.")
    else:
        print("Not enough data points for correlation (need ≥ 4, have %d)" % len(valid))

    # Save results
    out_path = "results/analysis/routing_signal_ratio.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
