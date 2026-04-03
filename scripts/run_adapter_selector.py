"""Zero-Shot Adapter Selector: predict optimal adapter mixture from dataset features.

Phase 1 (this script): Feasibility validation
- Extract time series features (NOT hidden states) for each dataset
- Check if features discriminate datasets
- Build adapter portfolio from best per-dataset adapters
- Cross-evaluate portfolio on all datasets (6×5 matrix)
- Test: does a simple feature-weighted mixture beat uniform mixture?

Usage:
    python scripts/run_adapter_selector.py
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
from scipy import stats as scipy_stats

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import validate_adapter_code
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]


def extract_ts_features(X, Y):
    """Extract time series features from raw data (NOT hidden states).

    These features characterize the dataset's temporal structure.
    """
    # X: (N, 512) input windows, Y: (N, H) targets
    # Use a sample of series for efficiency
    n_sample = min(200, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    features = {}

    # 1. Autocorrelation at multiple lags
    acfs = []
    for lag in [1, 5, 10, 24, 48, 96]:
        acf_vals = []
        for series in X_sample[:50]:
            if lag < len(series):
                c = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(c):
                    acf_vals.append(c)
        acfs.append(np.mean(acf_vals) if acf_vals else 0.0)
    features["acf_1"] = acfs[0]
    features["acf_5"] = acfs[1]
    features["acf_10"] = acfs[2]
    features["acf_24"] = acfs[3]
    features["acf_48"] = acfs[4]
    features["acf_96"] = acfs[5]

    # 2. Trend strength (variance of differenced vs original)
    diffs = np.diff(X_sample, axis=1)
    var_orig = np.var(X_sample, axis=1).mean()
    var_diff = np.var(diffs, axis=1).mean()
    features["trend_strength"] = max(0, 1 - var_diff / (var_orig + 1e-10))

    # 3. Seasonality via FFT
    fft_vals = np.abs(np.fft.rfft(X_sample, axis=1))
    mean_fft = fft_vals.mean(axis=0)
    # Spectral entropy (of raw data, not hidden states)
    psd = mean_fft ** 2
    psd_norm = psd / (psd.sum() + 1e-10)
    features["spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))

    # Dominant period
    dominant_freq_idx = np.argmax(mean_fft[1:]) + 1  # skip DC
    features["dominant_period"] = float(len(X_sample[0]) / dominant_freq_idx) if dominant_freq_idx > 0 else 0

    # 4. Stationarity (ADF-like: ratio of mean to std across windows)
    n_windows = 4
    window_size = X_sample.shape[1] // n_windows
    window_means = []
    for w in range(n_windows):
        window_means.append(X_sample[:, w*window_size:(w+1)*window_size].mean(axis=1).mean())
    features["stationarity"] = float(np.std(window_means) / (np.mean(np.abs(window_means)) + 1e-10))

    # 5. Kurtosis and skewness
    features["kurtosis"] = float(scipy_stats.kurtosis(X_sample.flatten()))
    features["skewness"] = float(scipy_stats.skew(X_sample.flatten()))

    # 6. Value range and scale
    features["value_range"] = float(X_sample.max() - X_sample.min())
    features["value_std"] = float(X_sample.std())

    # 7. Forecast difficulty proxy (naive MSE: predict last value)
    naive_pred = np.repeat(X_sample[:, -1:], Y.shape[1], axis=1)
    features["naive_mse"] = float(np.mean((naive_pred - Y[idx]) ** 2))

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/adapter_selector", exist_ok=True)

    # === Step 1: Extract time series features ===
    print("=" * 60)
    print("STEP 1: Time Series Features (raw data, NOT hidden states)")
    print("=" * 60)

    all_features = {}
    all_data = {}
    for dataset in DATASETS:
        try:
            splits, _ = load_standard_data(dataset, args.horizon)
            X_train, Y_train = splits["train"]
            all_data[dataset] = splits
            fp = extract_ts_features(X_train, Y_train)
            all_features[dataset] = fp
            print("%-12s acf1=%.3f acf24=%.3f trend=%.3f entropy=%.2f period=%.0f kurt=%.2f naive=%.3f" % (
                dataset, fp["acf_1"], fp["acf_24"], fp["trend_strength"],
                fp["spectral_entropy"], fp["dominant_period"], fp["kurtosis"], fp["naive_mse"]))
        except Exception as e:
            print("%-12s ERROR: %s" % (dataset, e))

    # Check discrimination
    print("\n--- Feature discrimination check ---")
    for feat_name in ["acf_1", "acf_24", "trend_strength", "spectral_entropy", "kurtosis", "naive_mse"]:
        vals = [all_features[d][feat_name] for d in DATASETS if d in all_features]
        cv = np.std(vals) / (np.abs(np.mean(vals)) + 1e-10)
        print("  %-20s range=[%.3f, %.3f]  CV=%.2f  %s" % (
            feat_name, min(vals), max(vals), cv,
            "DISCRIMINATES" if cv > 0.3 else "weak"))

    # === Step 2: Load adapter portfolio ===
    print("\n" + "=" * 60)
    print("STEP 2: Build Adapter Portfolio")
    print("=" * 60)

    portfolio = {}
    for dataset in DATASETS:
        evo_path = "results/standard_evolution/%s_H%d_%d.json" % (dataset, args.horizon, args.seed)
        if not os.path.exists(evo_path):
            print("%-12s: no evolution results" % dataset)
            continue
        d = json.load(open(evo_path))
        evolved = d.get("evolved", [])
        if evolved:
            best = min(evolved, key=lambda x: x["test_mse"])
            portfolio[dataset] = {
                "code": best["code"],
                "source_mse": best["test_mse"],
                "param_count": best.get("param_count", 0),
            }
            print("%-12s MSE=%.4f params=%d" % (dataset, best["test_mse"], best.get("param_count", 0)))

    if len(portfolio) < 3:
        print("ERROR: need at least 3 portfolio adapters")
        return

    # === Step 3: Cross-evaluate portfolio ===
    print("\n" + "=" * 60)
    print("STEP 3: Cross-Evaluate Portfolio (P×D matrix)")
    print("=" * 60)

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True

    # Cross-evaluation matrix: mse_matrix[adapter_source][target_dataset]
    mse_matrix = {}
    portfolio_names = list(portfolio.keys())

    for src in portfolio_names:
        mse_matrix[src] = {}
        code = portfolio[src]["code"]

        for tgt in DATASETS:
            if tgt not in all_data:
                continue
            splits = all_data[tgt]
            X_train, Y_train = splits["train"]
            X_test, Y_test = splits["test"]

            # Reload model fresh
            model_fresh = load_backbone(args.backbone, args.device)
            _disable_gradient_checkpointing(model_fresh)
            blocks_fresh = _get_encoder_blocks(model_fresh)
            for p in model_fresh.parameters():
                p.requires_grad = False
            for i in range(max(0, len(blocks_fresh)-4), len(blocks_fresh)):
                for p in blocks_fresh[i].parameters():
                    p.requires_grad = True

            val = validate_adapter_code(code, d_model=hdim, output_dim=args.horizon)
            if not val["valid"]:
                mse_matrix[src][tgt] = float("inf")
                continue

            try:
                tr = train_adapter(code, model_fresh, blocks_fresh,
                                   X_train, Y_train, X_test, Y_test,
                                   device=args.device, n_epochs=15,
                                   forecast_horizon=args.horizon,
                                   backbone_type=bb_type)
                mse_matrix[src][tgt] = tr["mse"]
                diag = " (DIAG)" if src == tgt else ""
                print("  %s → %-12s MSE=%.4f%s" % (src, tgt, tr["mse"], diag))
            except Exception as e:
                mse_matrix[src][tgt] = float("inf")
                print("  %s → %-12s ERROR: %s" % (src, tgt, e))

    # === Step 4: Compute optimal mixture weights per dataset ===
    print("\n" + "=" * 60)
    print("STEP 4: Optimal & Uniform Mixture Analysis")
    print("=" * 60)

    for tgt in DATASETS:
        if tgt not in all_data:
            continue

        mses = []
        for src in portfolio_names:
            mse = mse_matrix.get(src, {}).get(tgt, float("inf"))
            mses.append(mse)

        mses = np.array(mses)
        valid = mses < float("inf")

        if not valid.any():
            continue

        best_single_idx = np.argmin(mses)
        best_single_mse = mses[best_single_idx]
        best_single_src = portfolio_names[best_single_idx]

        # Softmin weights (optimal mixture)
        temp = 1.0
        weights_softmin = np.exp(-mses[valid] / temp)
        weights_softmin = weights_softmin / weights_softmin.sum()

        # Uniform weights
        n_valid = valid.sum()
        weights_uniform = np.ones(n_valid) / n_valid

        print("%-12s best_single=%s (%.4f)  weights=[%s]" % (
            tgt, best_single_src, best_single_mse,
            ", ".join("%.2f" % w for w in weights_softmin)))

    # === Step 5: Feature-based selector feasibility ===
    print("\n" + "=" * 60)
    print("STEP 5: Can Features Predict Best Adapter?")
    print("=" * 60)

    # For each dataset, find which portfolio adapter is best
    best_adapter_per_dataset = {}
    for tgt in DATASETS:
        mses = {src: mse_matrix.get(src, {}).get(tgt, float("inf")) for src in portfolio_names}
        best_src = min(mses, key=mses.get)
        best_adapter_per_dataset[tgt] = best_src

    print("\nDataset → Best portfolio adapter:")
    for tgt in DATASETS:
        if tgt in best_adapter_per_dataset and tgt in all_features:
            fp = all_features[tgt]
            print("  %-12s → %-12s  (acf1=%.3f, trend=%.3f, entropy=%.2f)" % (
                tgt, best_adapter_per_dataset[tgt],
                fp["acf_1"], fp["trend_strength"], fp["spectral_entropy"]))

    # Check: do datasets that prefer the same adapter have similar features?
    print("\nFeature clustering by preferred adapter:")
    from collections import defaultdict
    adapter_groups = defaultdict(list)
    for tgt, src in best_adapter_per_dataset.items():
        if tgt in all_features:
            adapter_groups[src].append(tgt)

    for src, tgts in adapter_groups.items():
        if len(tgts) > 1:
            features_in_group = [all_features[t] for t in tgts]
            for feat in ["acf_1", "acf_24", "trend_strength", "spectral_entropy"]:
                vals = [f[feat] for f in features_in_group]
                print("  Adapter %-12s → datasets %s: %s=[%s]" % (
                    src, tgts, feat, ", ".join("%.3f" % v for v in vals)))

    # Save
    save_data = {
        "features": all_features,
        "portfolio": {k: {"source_mse": v["source_mse"], "param_count": v["param_count"]}
                      for k, v in portfolio.items()},
        "mse_matrix": mse_matrix,
        "best_adapter_per_dataset": best_adapter_per_dataset,
    }
    path = "results/adapter_selector/feasibility_H%d.json" % args.horizon
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved to %s" % path)


if __name__ == "__main__":
    main()
