"""Statistical significance tests for NeurIPS paper claims.

Computes Wilcoxon signed-rank tests, Cohen's d effect sizes, and
Bonferroni-corrected p-values across all method comparisons.

Usage:
    python scripts/compute_significance.py
"""

import json
import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rrmoa_mses(datasets, seeds, unfreeze="frozen", suffix=""):
    """Load RR-MoA MSE values from result JSONs."""
    mses = {}
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            path = "results/rr_moa/%s_H96_K5_top2_%s_%d%s.json" % (ds, unfreeze, seed, suffix)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                mses[ds].append(d["rr_moa"]["mse"])
            else:
                print("WARNING: missing %s" % path)
    return mses


def load_baseline_mses(datasets, seeds, unfreeze="frozen"):
    """Load best-fixed baseline MSEs from RR-MoA results (baselines field)."""
    mses = {}
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            path = "results/rr_moa/%s_H96_K5_top2_%s_%d.json" % (ds, unfreeze, seed)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                bl = d.get("baselines", {})
                if bl:
                    best = min(bl.values(), key=lambda x: x["mse"])["mse"]
                    mses[ds].append(best)
    return mses


def load_trace_mses(datasets, seeds, unfreeze="frozen"):
    """Load TRACE baseline MSEs."""
    mses = {}
    uf_suffix = "" if unfreeze == "last4" else "_%s" % unfreeze
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            path = "results/trace_baseline/%s_H96_%d%s.json" % (ds, seed, uf_suffix)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                # Use trace_full (strongest variant)
                tf = d.get("results", {}).get("trace_full", {}).get("test_mse")
                if tf is not None:
                    mses[ds].append(tf)
    return mses


def load_ensemble_mses(datasets, seeds, unfreeze="frozen"):
    """Load independent ensemble MSEs."""
    mses = {}
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            path = "results/independent_ensemble/%s_H96_%s_%d.json" % (ds, unfreeze, seed)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                mses[ds].append(d["ensemble_mse"])
    return mses


def load_adamix_mses(datasets, seeds, unfreeze="frozen"):
    """Load AdaMix MSEs."""
    mses = {}
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            path = "results/adamix/%s_H96_K5_%s_%d.json" % (ds, unfreeze, seed)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                mses[ds].append(d.get("adamix", {}).get("mse", d.get("mse")))
    return mses


def load_lora_best_mses(datasets, seeds):
    """Load best LoRA config MSEs from sweep results."""
    mses = {}
    for ds in datasets:
        mses[ds] = []
        for seed in seeds:
            best_mse = float("inf")
            for f in glob.glob("results/lora_baseline/%s_H96_*_frozen_%d.json" % (ds, seed)):
                with open(f) as fh:
                    d = json.load(fh)
                m = d.get("lora_mse", d.get("test_mse", d.get("mse")))
                if m is not None and m < best_mse:
                    best_mse = m
            if best_mse < float("inf"):
                mses[ds].append(best_mse)
    return mses


def cohens_d(a, b):
    """Cohen's d effect size (paired)."""
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def wilcoxon_test(rrmoa_vals, baseline_vals):
    """Wilcoxon signed-rank test (one-sided: RR-MoA < baseline)."""
    try:
        from scipy.stats import wilcoxon
        if len(rrmoa_vals) < 3:
            return float("nan"), float("nan")
        stat, p = wilcoxon(rrmoa_vals, baseline_vals, alternative="less")
        return stat, p
    except ImportError:
        print("WARNING: scipy not available, using sign test fallback")
        wins = sum(1 for a, b in zip(rrmoa_vals, baseline_vals) if a < b)
        # Simple sign test p-value
        n = len(rrmoa_vals)
        from math import comb
        p = sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
        return wins, p


def main():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
    seeds = [42, 43, 44]

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    # Load all methods
    rrmoa = load_rrmoa_mses(datasets, seeds)
    baseline = load_baseline_mses(datasets, seeds)
    trace = load_trace_mses(datasets, seeds, unfreeze="frozen")
    ensemble = load_ensemble_mses(datasets, seeds)
    adamix = load_adamix_mses(datasets, seeds)
    lora = load_lora_best_mses(datasets, seeds)

    methods = {
        "Best Fixed": baseline,
        "LoRA (best)": lora,
        "TRACE (frozen)": trace,
        "Ind. Ensemble": ensemble,
        "AdaMix (frozen)": adamix,
    }

    n_comparisons = len([m for m in methods.values() if any(len(v) > 0 for v in m.values())])
    bonferroni = n_comparisons if n_comparisons > 0 else 1

    print("\nRR-MoA vs each baseline (Wilcoxon signed-rank, Bonferroni-corrected for %d comparisons):" % bonferroni)
    print("-" * 80)

    results_summary = {}

    for method_name, method_mses in methods.items():
        # Pool across datasets for omnibus test
        rr_pooled = []
        bl_pooled = []
        for ds in datasets:
            rr_vals = rrmoa.get(ds, [])
            bl_vals = method_mses.get(ds, [])
            n = min(len(rr_vals), len(bl_vals))
            rr_pooled.extend(rr_vals[:n])
            bl_pooled.extend(bl_vals[:n])

        if len(rr_pooled) < 3:
            print("  %-20s: insufficient data (%d paired observations)" % (method_name, len(rr_pooled)))
            continue

        stat, p_raw = wilcoxon_test(rr_pooled, bl_pooled)
        p_corrected = min(p_raw * bonferroni, 1.0)
        d = cohens_d(bl_pooled, rr_pooled)  # positive d = RR-MoA better
        wins = sum(1 for a, b in zip(rr_pooled, bl_pooled) if a < b)

        sig = ""
        if p_corrected < 0.001: sig = "***"
        elif p_corrected < 0.01: sig = "**"
        elif p_corrected < 0.05: sig = "*"

        rr_mean = np.mean(rr_pooled)
        bl_mean = np.mean(bl_pooled)
        delta = (rr_mean - bl_mean) / bl_mean * 100

        print("  %-20s: RR-MoA=%.3f vs %.3f  delta=%+.1f%%  wins=%d/%d  "
              "p=%.4f (raw=%.4f)  d=%.2f  %s" % (
                  method_name, rr_mean, bl_mean, delta, wins, len(rr_pooled),
                  p_corrected, p_raw, d, sig))

        results_summary[method_name] = {
            "rrmoa_mean": rr_mean, "baseline_mean": bl_mean,
            "delta_pct": delta, "wins": wins, "n": len(rr_pooled),
            "p_raw": p_raw, "p_corrected": p_corrected,
            "cohens_d": d, "significance": sig,
        }

    # Per-dataset breakdown
    print("\n\nPer-dataset breakdown:")
    print("-" * 80)
    for ds in datasets:
        print("\n%s:" % ds)
        rr = rrmoa.get(ds, [])
        if rr:
            print("  RR-MoA: %.4f ± %.4f" % (np.mean(rr), np.std(rr)))
        for method_name, method_mses in methods.items():
            bl = method_mses.get(ds, [])
            if bl:
                print("  %-20s: %.4f ± %.4f" % (method_name, np.mean(bl), np.std(bl)))

    # Save
    save_path = "results/significance_tests.json"
    with open(save_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved to %s" % save_path)


if __name__ == "__main__":
    main()
