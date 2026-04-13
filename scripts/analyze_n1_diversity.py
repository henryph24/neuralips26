#!/usr/bin/env python3
"""Analyze N1 expert diversity ablation results.

Compares identical-expert pools against the canonical diverse pool.
Prints a markdown table and optionally a LaTeX row.

Usage:
    python3 scripts/analyze_n1_diversity.py [--latex]
"""
import json, glob, sys, os
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [42, 43, 44]
POOLS = ["canonical", "identical-mean", "identical-conv1d", "identical-attn"]


def load_result(dataset, seed, pool):
    """Load a single RR-MoA result JSON."""
    if pool == "canonical":
        path = f"results/rr_moa/{dataset}_H96_K5_top2_frozen_{seed}.json"
    else:
        path = f"results/rr_moa/{dataset}_H96_K5_top2_frozen_{seed}_pool-{pool}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {
        "mse": d["rr_moa"]["mse"],
        "entropy": d["rr_moa"]["routing_entropy"],
        "per_sample_std": d["rr_moa"].get("routing_per_sample_std", None),
    }


def main():
    latex_mode = "--latex" in sys.argv

    # Collect per-pool, per-dataset stats
    pool_stats = defaultdict(lambda: defaultdict(lambda: {"mse": [], "entropy": []}))

    for pool in POOLS:
        for ds in DATASETS:
            for seed in SEEDS:
                r = load_result(ds, seed, pool)
                if r:
                    pool_stats[pool][ds]["mse"].append(r["mse"])
                    pool_stats[pool][ds]["entropy"].append(r["entropy"])

    # Print table
    print("| Pool | Dataset | MSE (mean±std) | Entropy | n | Δ vs canonical |")
    print("|------|---------|----------------|---------|---|----------------|")

    for pool in POOLS:
        for ds in DATASETS:
            info = pool_stats[pool][ds]
            if not info["mse"]:
                print(f"| {pool:18s} | {ds:7s} | --- | --- | 0 | --- |")
                continue
            mse_m = np.mean(info["mse"])
            mse_s = np.std(info["mse"])
            ent_m = np.mean(info["entropy"])
            n = len(info["mse"])

            # Compute delta vs canonical
            can = pool_stats["canonical"][ds]
            if can["mse"]:
                can_m = np.mean(can["mse"])
                delta = (mse_m - can_m) / can_m * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "---"

            print(f"| {pool:18s} | {ds:7s} | {mse_m:.4f}±{mse_s:.4f} | {ent_m:.3f} | {n} | {delta_str:>14s} |")

    # Summary: per-pool average across datasets
    print("\n=== Summary (averaged across datasets) ===\n")
    print(f"{'Pool':20s} {'Avg MSE':>10s} {'Avg Δ%':>10s} {'Avg Entropy':>12s} {'n':>5s}")
    print("-" * 60)

    for pool in POOLS:
        all_mse, all_delta, all_ent = [], [], []
        for ds in DATASETS:
            info = pool_stats[pool][ds]
            can = pool_stats["canonical"][ds]
            if info["mse"] and can["mse"]:
                mse_m = np.mean(info["mse"])
                can_m = np.mean(can["mse"])
                all_mse.append(mse_m)
                all_delta.append((mse_m - can_m) / can_m * 100)
                all_ent.append(np.mean(info["entropy"]))

        if all_mse:
            n_total = sum(len(pool_stats[pool][ds]["mse"]) for ds in DATASETS)
            print(f"{pool:20s} {np.mean(all_mse):10.4f} {np.mean(all_delta):+9.1f}% {np.mean(all_ent):12.3f} {n_total:5d}")

    if latex_mode:
        print("\n=== LaTeX rows ===\n")
        for pool in POOLS:
            if pool == "canonical":
                continue
            all_delta = []
            for ds in DATASETS:
                info = pool_stats[pool][ds]
                can = pool_stats["canonical"][ds]
                if info["mse"] and can["mse"]:
                    all_delta.append((np.mean(info["mse"]) - np.mean(can["mse"])) / np.mean(can["mse"]) * 100)
            if all_delta:
                avg_d = np.mean(all_delta)
                print(f"5$\\times${pool.replace('identical-', '')} & "
                      f"${avg_d:+.1f}\\%$ & {len(all_delta)*3}/9 \\\\")


if __name__ == "__main__":
    main()
