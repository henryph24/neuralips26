#!/usr/bin/env python3
"""Analyze N2a regime robustness results.

Compares per-quartile MSE coefficient of variation for RR-MoA vs best-fixed.

Usage:
    python3 scripts/analyze_n2_regime.py
"""
import json, glob, os
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [42, 43, 44]


def main():
    results = []
    for ds in DATASETS:
        for seed in SEEDS:
            path = f"results/regime_robustness/{ds}_H96_{seed}.json"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                d = json.load(f)
            results.append(d)

    if not results:
        print("No results found in results/regime_robustness/")
        return

    print(f"Loaded {len(results)} result files.\n")

    # Per-dataset summary
    for stat_name in ["amplitude", "volatility"]:
        print(f"=== Quartile analysis by {stat_name} ===\n")
        print(f"{'Dataset':10s} {'Seed':>5s} {'RR-MoA CV':>10s} {'Fixed CV':>10s} {'Winner':>12s} "
              f"{'RR-MoA MSE':>11s} {'Fixed MSE':>11s}")
        print("-" * 72)

        cv_rr_all, cv_fix_all = [], []

        for ds in DATASETS:
            ds_results = [r for r in results if r["dataset"] == ds]
            for r in ds_results:
                rr = r["methods"]["rr_moa_raw"]
                fix = r["methods"]["best_fixed"]
                cv_rr = rr[f"cv_{stat_name}"]
                cv_fix = fix[f"cv_{stat_name}"]
                winner = "RR-MoA" if cv_rr < cv_fix else "Fixed"
                print(f"{ds:10s} {r['seed']:5d} {cv_rr:10.4f} {cv_fix:10.4f} {winner:>12s} "
                      f"{rr['mse_overall']:11.4f} {fix['mse_overall']:11.4f}")
                cv_rr_all.append(cv_rr)
                cv_fix_all.append(cv_fix)

        if cv_rr_all:
            print("-" * 72)
            rr_wins = sum(1 for r, f in zip(cv_rr_all, cv_fix_all) if r < f)
            print(f"{'Average':10s} {'':5s} {np.mean(cv_rr_all):10.4f} {np.mean(cv_fix_all):10.4f} "
                  f"{'RR-MoA' if np.mean(cv_rr_all) < np.mean(cv_fix_all) else 'Fixed':>12s} "
                  f"(RR-MoA wins {rr_wins}/{len(cv_rr_all)})")
        print()

    # Per-quartile MSE breakdown for the first dataset/seed (illustrative)
    if results:
        r = results[0]
        print(f"=== Illustrative quartile breakdown: {r['dataset']} seed={r['seed']} ===\n")
        for stat_name in ["amplitude", "volatility"]:
            print(f"  {stat_name} quartiles (Q1=lowest, Q4=highest):")
            rr_q = r["methods"]["rr_moa_raw"][f"quartile_mse_by_{stat_name}"]
            fix_q = r["methods"]["best_fixed"][f"quartile_mse_by_{stat_name}"]
            counts = r["methods"]["rr_moa_raw"][f"quartile_counts_by_{stat_name}"]
            for i in range(4):
                print(f"    Q{i+1} (n={counts[i]:4d}): RR-MoA={rr_q[i]:.4f}  Fixed={fix_q[i]:.4f}")
            print()


if __name__ == "__main__":
    main()
