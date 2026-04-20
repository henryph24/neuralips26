"""Analyze normalization dose-response and shuffle ablation results.

Reads JSON results from the dose-response sweep and produces:
  1. Per-dataset dose-response table (alpha vs MSE, entropy)
  2. Cross-dataset summary with monotonicity test
  3. Shuffle ablation comparison

Usage:
    python scripts/analyze_dose_response.py [--latex]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]


def load_results():
    """Load dose-response and shuffle results from results/rr_moa/."""
    results_dir = "results/rr_moa"

    # Dose-response: alpha=0.0 is the standard raw result (no suffix)
    # alpha>0 uses router-partial_alpha-{alpha} suffix
    dose = defaultdict(lambda: defaultdict(list))  # dose[dataset][alpha] = [mse, ...]
    entropy = defaultdict(lambda: defaultdict(list))  # entropy[dataset][alpha] = [ent, ...]

    for ds in DATASETS:
        for seed in SEEDS:
            # alpha=0.0: from the dose-response sweep (router-partial_alpha-0.00)
            raw_path = os.path.join(results_dir,
                "%s_H96_K5_dense_frozen_%d_router-partial_alpha-0.00.json" % (ds, seed))
            if os.path.exists(raw_path):
                with open(raw_path) as f:
                    data = json.load(f)
                dose[ds][0.0].append(data["rr_moa"]["mse"])
                entropy[ds][0.0].append(data["rr_moa"]["routing_entropy"])

            # alpha > 0: partial suffix
            for alpha in ALPHAS[1:]:
                path = os.path.join(results_dir,
                    "%s_H96_K5_dense_frozen_%d_router-partial_alpha-%.2f.json" % (ds, seed, alpha))
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    dose[ds][alpha].append(data["rr_moa"]["mse"])
                    entropy[ds][alpha].append(data["rr_moa"]["routing_entropy"])

    # Shuffle results
    shuffle = defaultdict(list)  # shuffle[dataset] = [(mse, entropy), ...]
    for ds in DATASETS:
        for seed in SEEDS:
            path = os.path.join(results_dir,
                "%s_H96_K5_dense_frozen_%d_router-shuffled.json" % (ds, seed))
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                shuffle[ds].append({
                    "mse": data["rr_moa"]["mse"],
                    "entropy": data["rr_moa"]["routing_entropy"],
                })

    return dose, entropy, shuffle


def check_monotonicity(values):
    """Check if a sequence is monotonically non-decreasing."""
    for i in range(1, len(values)):
        if values[i] < values[i-1] - 1e-6:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Emit LaTeX table")
    args = parser.parse_args()

    dose, entropy, shuffle = load_results()

    print("=" * 80)
    print("NORMALIZATION DOSE-RESPONSE ANALYSIS")
    print("=" * 80)

    # Per-dataset table
    all_monotone_mse = 0
    all_monotone_ent = 0
    total_ds = 0

    for ds in DATASETS:
        if not dose[ds]:
            print("\n%s: NO DATA" % ds)
            continue

        total_ds += 1
        print("\n%s:" % ds)
        print("  %-8s  %8s  %8s  %8s  %8s  %3s  %3s" % (
            "alpha", "MSE_mean", "MSE_std", "Ent_mean", "Ent_std", "n_m", "n_e"))

        mse_means = []
        ent_means = []
        for alpha in ALPHAS:
            mse_vals = dose[ds].get(alpha, [])
            ent_vals = entropy[ds].get(alpha, [])
            if mse_vals:
                m_mean, m_std = np.mean(mse_vals), np.std(mse_vals)
                e_mean, e_std = np.mean(ent_vals), np.std(ent_vals)
                mse_means.append(m_mean)
                ent_means.append(e_mean)
                print("  %-8.2f  %8.4f  %8.4f  %8.3f  %8.3f  %3d  %3d" % (
                    alpha, m_mean, m_std, e_mean, e_std, len(mse_vals), len(ent_vals)))
            else:
                print("  %-8.2f  %8s  %8s  %8s  %8s  %3d  %3d" % (
                    alpha, "---", "---", "---", "---", 0, 0))

        if len(mse_means) >= 3:
            mono_mse = check_monotonicity(mse_means)
            mono_ent = check_monotonicity([-e for e in ent_means])  # entropy should decrease
            all_monotone_mse += int(mono_mse)
            all_monotone_ent += int(mono_ent)
            print("  Monotone MSE (increasing): %s" % ("YES" if mono_mse else "NO"))
            print("  Monotone entropy (decreasing): %s" % ("YES" if mono_ent else "NO"))

            if mse_means[0] > 0 and mse_means[-1] > 0:
                degradation = (mse_means[-1] - mse_means[0]) / mse_means[0] * 100
                print("  MSE degradation (alpha=0 -> 1): +%.1f%%" % degradation)

    print("\n" + "=" * 80)
    print("MONOTONICITY SUMMARY: MSE %d/%d, Entropy %d/%d" % (
        all_monotone_mse, total_ds, all_monotone_ent, total_ds))

    # Shuffle analysis
    print("\n" + "=" * 80)
    print("SHUFFLE ABLATION")
    print("=" * 80)
    print("  %-12s  %8s  %8s  %8s  %8s  %8s  %3s" % (
        "Dataset", "Raw_MSE", "Shuf_MSE", "Shuf_Ent", "Rev_MSE", "Delta%", "n"))

    for ds in DATASETS:
        raw_mse = dose[ds].get(0.0, [])
        rev_mse = dose[ds].get(1.0, [])
        shuf_data = shuffle.get(ds, [])

        if raw_mse and shuf_data:
            raw_m = np.mean(raw_mse)
            shuf_m = np.mean([s["mse"] for s in shuf_data])
            shuf_e = np.mean([s["entropy"] for s in shuf_data])
            rev_m = np.mean(rev_mse) if rev_mse else float('nan')
            delta = (shuf_m - raw_m) / raw_m * 100
            print("  %-12s  %8.4f  %8.4f  %8.3f  %8.4f  %+7.1f%%  %3d" % (
                ds, raw_m, shuf_m, shuf_e, rev_m, delta, len(shuf_data)))
        else:
            print("  %-12s  %8s  %8s  %8s  %8s  %8s  %3d" % (
                ds, "---", "---", "---", "---", "---", len(shuf_data)))

    # Cross-dataset summary for paper
    print("\n" + "=" * 80)
    print("PAPER-READY SUMMARY")
    print("=" * 80)

    # Compute mean degradation per alpha step
    all_degradations = []
    for ds in DATASETS:
        mse_means = []
        for alpha in ALPHAS:
            vals = dose[ds].get(alpha, [])
            if vals:
                mse_means.append(np.mean(vals))
        if len(mse_means) == len(ALPHAS):
            for i in range(1, len(ALPHAS)):
                step_deg = (mse_means[i] - mse_means[0]) / mse_means[0] * 100
                all_degradations.append((ALPHAS[i], ds, step_deg))

    if all_degradations:
        print("\nMean MSE degradation vs alpha=0 (across datasets):")
        for alpha in ALPHAS[1:]:
            degs = [d for a, _, d in all_degradations if a == alpha]
            if degs:
                print("  alpha=%.2f: +%.1f%% mean (+%.1f%% to +%.1f%%)" % (
                    alpha, np.mean(degs), np.min(degs), np.max(degs)))

    if args.latex:
        print("\n% LaTeX table (dose-response)")
        print("\\begin{tabular}{l" + "cc" * len(ALPHAS) + "}")
        print("\\toprule")
        header = "Dataset"
        for a in ALPHAS:
            header += " & MSE & Ent"
        print(header + " \\\\")
        alpha_header = ""
        for a in ALPHAS:
            alpha_header += " & $\\alpha=%.2f$" % a + " &"
        print("\\midrule")

        for ds in DATASETS:
            row = ds
            for alpha in ALPHAS:
                mse_vals = dose[ds].get(alpha, [])
                ent_vals = entropy[ds].get(alpha, [])
                if mse_vals:
                    row += " & %.3f & %.2f" % (np.mean(mse_vals), np.mean(ent_vals))
                else:
                    row += " & --- & ---"
            print(row + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
