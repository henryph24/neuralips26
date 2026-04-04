"""Analyze calibration data: per-statistic correlations, scatter plots, heatmaps.

Usage:
    python scripts/analyze_calibration.py
    python scripts/analyze_calibration.py --calibration-dir results/calibration/ --seed 42
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from feasibility.proxy_search import load_calibration_data
from feasibility.proxy_gp import STAT_NAMES, deserialize_tree, evaluate_proxy


def load_all_calibrations(cal_dir, seed, datasets):
    """Load calibration data for all datasets."""
    cals = {}
    for ds in datasets:
        path = os.path.join(cal_dir, f"{ds}_{seed}.json")
        if os.path.exists(path):
            cals[ds] = load_calibration_data(path)
            print(f"Loaded {ds}: {len(cals[ds].configs)} configs, metric={cals[ds].metric_key}")
        else:
            print(f"Skipping {ds}: {path} not found")
    return cals


def compute_per_stat_correlations(cals):
    """Compute Kendall τ for each statistic vs performance on each dataset."""
    all_stats = set()
    for cal in cals.values():
        for s in cal.statistics:
            all_stats.update(s.keys())
    stat_names = sorted(all_stats)

    results = {}
    for ds_name, cal in cals.items():
        perfs = cal.performances
        sign = 1.0 if cal.higher_is_better else -1.0  # negate for MSE so positive τ = good

        ds_results = {}
        for stat in stat_names:
            vals = [s.get(stat, 0.0) for s in cal.statistics]
            if len(set(vals)) <= 1:
                ds_results[stat] = {"tau": 0.0, "p": 1.0, "spearman": 0.0}
                continue
            # For MSE (lower=better), negate vals so positive τ means higher stat → lower MSE
            tau, p = kendalltau([sign * v for v in vals], perfs)
            rho, _ = spearmanr([sign * v for v in vals], perfs)
            ds_results[stat] = {"tau": float(tau), "p": float(p), "spearman": float(rho)}
        results[ds_name] = ds_results

    return results, stat_names


def print_correlation_table(corr_results, stat_names, cals):
    """Print formatted correlation table."""
    ds_names = list(corr_results.keys())

    print(f"\n{'='*80}")
    print(f"PER-STATISTIC KENDALL TAU (positive τ = higher stat predicts better performance)")
    print(f"{'='*80}")

    header = f"{'Statistic':<25}"
    for ds in ds_names:
        header += f" {ds:>12}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for stat in stat_names:
        taus = [corr_results[ds].get(stat, {}).get("tau", 0.0) for ds in ds_names]
        mean_tau = np.mean(taus)
        rows.append((stat, taus, mean_tau))

    # Sort by absolute mean tau
    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    for stat, taus, mean_tau in rows:
        line = f"{stat:<25}"
        for tau in taus:
            marker = "*" if abs(tau) > 0.3 else " "
            line += f" {tau:>11.3f}{marker}"
        line += f" {mean_tau:>10.3f}"
        print(line)

    print(f"\n* = |τ| > 0.3")

    # Summary
    print(f"\n{'='*80}")
    print(f"PERFORMANCE DISTRIBUTIONS")
    print(f"{'='*80}")
    for ds_name, cal in cals.items():
        perfs = cal.performances
        print(f"\n{ds_name} ({cal.metric_key}, {'higher' if cal.higher_is_better else 'lower'} is better):")
        print(f"  min={min(perfs):.4f}  max={max(perfs):.4f}  mean={np.mean(perfs):.4f}  std={np.std(perfs):.4f}")
        print(f"  CV = {np.std(perfs)/abs(np.mean(perfs)):.3f}")


def plot_correlation_heatmap(corr_results, stat_names, out_path):
    """Plot heatmap of τ values: statistics × datasets."""
    ds_names = list(corr_results.keys())

    # Sort stats by absolute mean tau
    mean_taus = {}
    for stat in stat_names:
        taus = [corr_results[ds].get(stat, {}).get("tau", 0.0) for ds in ds_names]
        mean_taus[stat] = np.mean(np.abs(taus))
    sorted_stats = sorted(stat_names, key=lambda s: mean_taus[s], reverse=True)

    data = np.zeros((len(sorted_stats), len(ds_names)))
    for i, stat in enumerate(sorted_stats):
        for j, ds in enumerate(ds_names):
            data[i, j] = corr_results[ds].get(stat, {}).get("tau", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(ds_names) * 2.5), max(6, len(sorted_stats) * 0.4)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_yticks(range(len(sorted_stats)))
    ax.set_yticklabels(sorted_stats)

    # Annotate cells
    for i in range(len(sorted_stats)):
        for j in range(len(ds_names)):
            val = data[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, label="Kendall τ")
    ax.set_title("Per-Statistic Correlation with Downstream Performance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved heatmap to {out_path}")


def plot_scatter_panels(cals, corr_results, out_path):
    """Scatter plots: top-4 statistics vs performance for each dataset."""
    ds_names = list(cals.keys())
    n_ds = len(ds_names)

    # Find top 4 stats by mean |τ| across datasets
    all_stats = set()
    for ds_corr in corr_results.values():
        all_stats.update(ds_corr.keys())

    mean_abs_tau = {}
    for stat in all_stats:
        taus = [abs(corr_results[ds].get(stat, {}).get("tau", 0.0)) for ds in ds_names]
        mean_abs_tau[stat] = np.mean(taus)
    top_stats = sorted(mean_abs_tau, key=mean_abs_tau.get, reverse=True)[:4]

    fig, axes = plt.subplots(4, n_ds, figsize=(5 * n_ds, 16))
    if n_ds == 1:
        axes = axes.reshape(-1, 1)

    for col, ds_name in enumerate(ds_names):
        cal = cals[ds_name]
        perfs = cal.performances

        for row, stat in enumerate(top_stats):
            ax = axes[row, col]
            vals = [s.get(stat, 0.0) for s in cal.statistics]
            tau = corr_results[ds_name].get(stat, {}).get("tau", 0.0)

            ax.scatter(vals, perfs, alpha=0.7, s=40, edgecolors="k", linewidth=0.5)
            ax.set_xlabel(stat)
            if col == 0:
                ax.set_ylabel(cal.metric_key)
            ax.set_title(f"{ds_name}: {stat}\nτ = {tau:.3f}", fontsize=10)

            # Add trend line
            if len(set(vals)) > 1:
                z = np.polyfit(vals, perfs, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(vals), max(vals), 50)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5)

    plt.suptitle("Top Statistics vs Downstream Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plots to {out_path}")


def plot_proxy_vs_performance(cals, gp_result_path, out_path):
    """Scatter: GP proxy score vs actual performance for each dataset."""
    if not os.path.exists(gp_result_path):
        print(f"GP result not found at {gp_result_path}, skipping proxy scatter")
        return

    with open(gp_result_path) as f:
        gp_data = json.load(f)
    tree = deserialize_tree(gp_data["best_tree"])
    formula = gp_data["best_formula"]
    best_tau = gp_data["best_fitness"]

    ds_names = list(cals.keys())
    n_ds = len(ds_names)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        cal = cals[ds_name]

        proxy_scores = [evaluate_proxy(tree, s) for s in cal.statistics]
        perfs = cal.performances

        # Sign-correct for MSE
        sign = 1.0 if cal.higher_is_better else -1.0
        corrected_scores = [sign * s for s in proxy_scores]

        tau, p = kendalltau(corrected_scores, perfs)

        ax.scatter(proxy_scores, perfs, alpha=0.7, s=40, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("GP Proxy Score")
        ax.set_ylabel(cal.metric_key)
        ax.set_title(f"{ds_name}\nτ = {tau:.3f} (p = {p:.3f})")

        # Trend line
        if len(set(proxy_scores)) > 1:
            z = np.polyfit(proxy_scores, perfs, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(min(proxy_scores), max(proxy_scores), 50)
            ax.plot(x_line, p_line(x_line), "r--", alpha=0.5)

    plt.suptitle(f"GP Proxy vs Downstream Performance\n{formula[:80]}...\nCross-dataset τ = {best_tau:.3f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved proxy scatter to {out_path}")


def analyze_performance_variance(cals):
    """Check if the search space has enough performance variance to be worth optimizing."""
    print(f"\n{'='*80}")
    print(f"SEARCH SPACE ANALYSIS: Is adapter selection even useful?")
    print(f"{'='*80}")

    for ds_name, cal in cals.items():
        perfs = np.array(cal.performances)
        best = max(perfs) if cal.higher_is_better else min(perfs)
        worst = min(perfs) if cal.higher_is_better else max(perfs)
        median = np.median(perfs)
        iqr = np.percentile(perfs, 75) - np.percentile(perfs, 25)

        print(f"\n{ds_name} ({cal.metric_key}):")
        print(f"  Best:   {best:.4f}")
        print(f"  Median: {median:.4f}")
        print(f"  Worst:  {worst:.4f}")
        print(f"  IQR:    {iqr:.4f}")
        improvement = abs(best - median) / abs(median) * 100
        print(f"  Best vs median improvement: {improvement:.1f}%")

        if improvement < 10:
            print(f"  → LOW VARIANCE: Best config only {improvement:.1f}% better than median.")
            print(f"    Random selection likely sufficient — proxy search has limited upside.")
        else:
            print(f"  → MEANINGFUL VARIANCE: {improvement:.1f}% gap suggests proxy search could help.")


def main():
    parser = argparse.ArgumentParser(description="Analyze calibration data")
    parser.add_argument("--calibration-dir", type=str, default="results/calibration/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gp-result", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/analysis/")
    args = parser.parse_args()

    datasets = ["ETTh1", "ETTm1", "EthanolConcentration"]
    cals = load_all_calibrations(args.calibration_dir, args.seed, datasets)

    if not cals:
        print("No calibration data found. Run proxy-search first.")
        return

    # Per-statistic correlations
    corr_results, stat_names = compute_per_stat_correlations(cals)
    print_correlation_table(corr_results, stat_names, cals)

    # Performance variance analysis
    analyze_performance_variance(cals)

    # Save correlation results
    os.makedirs(args.out_dir, exist_ok=True)
    corr_path = os.path.join(args.out_dir, f"stat_correlations_{args.seed}.json")
    with open(corr_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print(f"\nSaved correlations to {corr_path}")

    # Plots
    plot_correlation_heatmap(corr_results, stat_names, os.path.join(args.out_dir, "heatmap_tau.png"))
    plot_scatter_panels(cals, corr_results, os.path.join(args.out_dir, "scatter_top_stats.png"))

    gp_path = args.gp_result or f"results/proxy_search/gp_result_{args.seed}.json"
    plot_proxy_vs_performance(cals, gp_path, os.path.join(args.out_dir, "scatter_proxy.png"))

    # Early-stopping proxy analysis hint
    print(f"\n{'='*80}")
    print(f"NEXT: Consider 3-epoch fine-tuning as proxy (early-stopping approach)")
    print(f"  The calibration data already has 15-epoch results.")
    print(f"  Re-run finetune with n_epochs=3 for same configs to test if")
    print(f"  3-epoch loss predicts 15-epoch performance (likely τ > 0.8).")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
