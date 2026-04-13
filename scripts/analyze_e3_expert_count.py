#!/usr/bin/env python3
"""Analyze E3 expert count scaling results.

Compares RR-MoA MSE across K={3,5,7,10} experts.

Usage:
    python3 scripts/analyze_e3_expert_count.py
"""
import json, glob, os
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [42, 43, 44]
K_VALUES = [3, 5, 7, 10]


def main():
    data = defaultdict(lambda: defaultdict(list))

    for K in K_VALUES:
        for ds in DATASETS:
            for seed in SEEDS:
                path = f"results/rr_moa/{ds}_H96_K{K}_top2_frozen_{seed}.json"
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    d = json.load(f)
                data[K][ds].append({
                    "mse": d["rr_moa"]["mse"],
                    "entropy": d["rr_moa"]["routing_entropy"],
                    "params": d["rr_moa"]["param_count"],
                })

    print(f"{'K':>3s} {'Dataset':>8s} {'MSE (mean±std)':>16s} {'Entropy':>8s} {'Params':>8s} {'Δ vs K=5':>10s} {'n':>3s}")
    print("-" * 62)

    k5_means = {}
    for ds in DATASETS:
        if data[5][ds]:
            k5_means[ds] = np.mean([x["mse"] for x in data[5][ds]])

    for K in K_VALUES:
        for ds in DATASETS:
            entries = data[K][ds]
            if not entries:
                print(f"{K:3d} {ds:>8s} {'---':>16s}")
                continue
            mse_m = np.mean([x["mse"] for x in entries])
            mse_s = np.std([x["mse"] for x in entries])
            ent_m = np.mean([x["entropy"] for x in entries])
            params = entries[0]["params"]
            n = len(entries)

            delta = (mse_m - k5_means.get(ds, mse_m)) / k5_means.get(ds, mse_m) * 100 if ds in k5_means else 0
            delta_str = f"{delta:+.1f}%" if K != 5 else "baseline"

            print(f"{K:3d} {ds:>8s} {mse_m:.4f}±{mse_s:.4f} {ent_m:8.3f} {params:8d} {delta_str:>10s} {n:3d}")

    # Summary row
    print("-" * 62)
    print("\nSummary (averaged across datasets):")
    print(f"{'K':>3s} {'Avg MSE':>10s} {'Avg Δ%':>10s} {'Avg Entropy':>12s}")
    print("-" * 40)
    for K in K_VALUES:
        all_mse, all_delta, all_ent = [], [], []
        for ds in DATASETS:
            if data[K][ds] and ds in k5_means:
                m = np.mean([x["mse"] for x in data[K][ds]])
                all_mse.append(m)
                all_delta.append((m - k5_means[ds]) / k5_means[ds] * 100)
                all_ent.append(np.mean([x["entropy"] for x in data[K][ds]]))
        if all_mse:
            print(f"{K:3d} {np.mean(all_mse):10.4f} {np.mean(all_delta):+9.1f}% {np.mean(all_ent):12.3f}")


if __name__ == "__main__":
    main()
