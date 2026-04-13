#!/usr/bin/env python3
"""Analyze B9 entropy regularization on RR-MoA results.

Compares entropy-reg RR-MoA vs vanilla RR-MoA.
Expected: no significant difference (the raw router doesn't collapse).

Usage:
    python3 scripts/analyze_b9_entropy_reg.py
"""
import json, glob, os
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["ETTh1", "ETTm1", "Weather"]
SEEDS = [42, 43, 44]
LAMBDAS = [0.0, 0.01, 0.1, 1.0]


def main():
    data = defaultdict(lambda: defaultdict(list))

    # Vanilla (λ=0)
    for ds in DATASETS:
        for seed in SEEDS:
            path = f"results/rr_moa/{ds}_H96_K5_top2_frozen_{seed}.json"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                d = json.load(f)
            data[0.0][ds].append({
                "mse": d["rr_moa"]["mse"],
                "entropy": d["rr_moa"]["routing_entropy"],
            })

    # Entropy-reg variants
    for lam in [0.01, 0.1, 1.0]:
        for ds in DATASETS:
            for seed in SEEDS:
                path = f"results/rr_moa/{ds}_H96_K5_top2_frozen_{seed}_entreg-{lam}.json"
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    d = json.load(f)
                data[lam][ds].append({
                    "mse": d["rr_moa"]["mse"],
                    "entropy": d["rr_moa"]["routing_entropy"],
                })

    print(f"{'λ':>6s} {'Dataset':>8s} {'MSE (mean±std)':>16s} {'Entropy':>8s} {'Δ vs λ=0':>10s} {'n':>3s}")
    print("-" * 56)

    baseline_means = {}
    for ds in DATASETS:
        if data[0.0][ds]:
            baseline_means[ds] = np.mean([x["mse"] for x in data[0.0][ds]])

    for lam in LAMBDAS:
        for ds in DATASETS:
            entries = data[lam][ds]
            if not entries:
                continue
            mse_m = np.mean([x["mse"] for x in entries])
            mse_s = np.std([x["mse"] for x in entries])
            ent_m = np.mean([x["entropy"] for x in entries])
            n = len(entries)
            delta = (mse_m - baseline_means.get(ds, mse_m)) / baseline_means.get(ds, mse_m) * 100
            delta_str = f"{delta:+.1f}%" if lam > 0 else "baseline"
            lam_str = f"{lam:.2f}" if lam > 0 else "0 (vanilla)"
            print(f"{lam_str:>10s} {ds:>8s} {mse_m:.4f}±{mse_s:.4f} {ent_m:8.3f} {delta_str:>10s} {n:3d}")

    # Summary
    print("\nSummary (averaged across datasets):")
    for lam in LAMBDAS:
        all_delta = []
        for ds in DATASETS:
            if data[lam][ds] and ds in baseline_means:
                m = np.mean([x["mse"] for x in data[lam][ds]])
                all_delta.append((m - baseline_means[ds]) / baseline_means[ds] * 100)
        if all_delta:
            lam_str = f"{lam:.2f}" if lam > 0 else "0 (vanilla)"
            print(f"  λ={lam_str}: avg Δ = {np.mean(all_delta):+.1f}%")


if __name__ == "__main__":
    main()
