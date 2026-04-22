"""Analyze SR-MoA strengthening sweep results.

Produces:
  1. Raw-vs-hidden controlled comparison (6 datasets, MOMENT-small)
  2. Timer-XL negative control (raw ≈ hidden, no RevIN)
  3. Frozen Paradox replication (freeze ablation, 6 datasets)
  4. Cross-backbone SR-MoA grid (6 backbones × 3 datasets)

Usage:
    python3 scripts/analyze_srmoa_strengthen.py [--results-dir results/self_routed_moa]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats


def load_results(results_dir):
    """Load all SR-MoA JSON results."""
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        d["_path"] = os.path.basename(path)
        results.append(d)
    return results


def analyze_raw_vs_hidden(results, backbone_filter="MOMENT-1-small"):
    """E1: Raw-vs-hidden controlled comparison."""
    print("=" * 60)
    print("E1: Raw-vs-Hidden Control (MOMENT-small, frozen, H=96)")
    print("=" * 60)

    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]

    for ds in datasets:
        raw_mses = []
        hid_mses = []
        raw_ents = []
        hid_ents = []

        for r in results:
            if r["dataset"] != ds or r["horizon"] != 96 or r["unfreeze"] != "frozen":
                continue
            bb = r.get("_path", "")
            # Filter to MOMENT-small only (no bb- suffix means MOMENT-small)
            if "bb-" in bb:
                continue
            if r.get("routing_mode") != "gated" or r.get("gate_hidden") != 16:
                continue
            # Skip non-default gate_init_bias
            if r.get("gate_init_bias", 0.0) != 0.0:
                continue

            mse = r["sr_moa"]["mse"]
            ent = r["sr_moa"]["routing_entropy"]
            if r.get("routing_input", "raw") == "hidden":
                hid_mses.append(mse)
                hid_ents.append(ent)
            else:
                raw_mses.append(mse)
                raw_ents.append(ent)

        if not raw_mses or not hid_mses:
            print(f"  {ds:12s}  INCOMPLETE (raw={len(raw_mses)}, hidden={len(hid_mses)})")
            continue

        raw_mean = np.mean(raw_mses)
        raw_std = np.std(raw_mses, ddof=1) if len(raw_mses) > 1 else 0
        hid_mean = np.mean(hid_mses)
        hid_std = np.std(hid_mses, ddof=1) if len(hid_mses) > 1 else 0
        delta_pct = (hid_mean - raw_mean) / raw_mean * 100

        # Welch's t-test
        if len(raw_mses) >= 2 and len(hid_mses) >= 2:
            t_stat, p_val = stats.ttest_ind(raw_mses, hid_mses, equal_var=False)
        else:
            p_val = float("nan")

        print(f"  {ds:12s}  Raw={raw_mean:.3f}±{raw_std:.3f} (n={len(raw_mses)})  "
              f"Hidden={hid_mean:.3f}±{hid_std:.3f} (n={len(hid_mses)})  "
              f"Δ={delta_pct:+.1f}%  p={p_val:.4f}")
        print(f"  {'':12s}  Raw ent={np.mean(raw_ents):.3f}  Hidden ent={np.mean(hid_ents):.3f}")

    # Paired Wilcoxon across datasets (using means)
    raw_means = []
    hid_means = []
    for ds in datasets:
        raw_vals = [r["sr_moa"]["mse"] for r in results
                    if r["dataset"] == ds and r["horizon"] == 96
                    and r["unfreeze"] == "frozen" and "bb-" not in r.get("_path", "")
                    and r.get("routing_mode") == "gated" and r.get("gate_hidden") == 16
                    and r.get("gate_init_bias", 0.0) == 0.0
                    and r.get("routing_input", "raw") == "raw"]
        hid_vals = [r["sr_moa"]["mse"] for r in results
                    if r["dataset"] == ds and r["horizon"] == 96
                    and r["unfreeze"] == "frozen" and "bb-" not in r.get("_path", "")
                    and r.get("routing_mode") == "gated" and r.get("gate_hidden") == 16
                    and r.get("gate_init_bias", 0.0) == 0.0
                    and r.get("routing_input", "raw") == "hidden"]
        if raw_vals and hid_vals:
            raw_means.append(np.mean(raw_vals))
            hid_means.append(np.mean(hid_vals))

    if len(raw_means) >= 3:
        w_stat, w_p = stats.wilcoxon(raw_means, hid_means, alternative="less")
        print(f"\n  Wilcoxon (raw < hidden, n={len(raw_means)} datasets): W={w_stat:.1f}, p={w_p:.6f}")
        print(f"  Raw wins: {sum(r < h for r, h in zip(raw_means, hid_means))}/{len(raw_means)}")


def analyze_timer_negative_control(results):
    """E2: Timer-XL negative control."""
    print("\n" + "=" * 60)
    print("E2: Timer-XL Negative Control (no RevIN → raw ≈ hidden)")
    print("=" * 60)

    datasets = ["ETTh1", "ETTm1", "Weather"]

    for ds in datasets:
        raw_mses = []
        hid_mses = []

        for r in results:
            if r["dataset"] != ds or r["horizon"] != 96 or r["unfreeze"] != "frozen":
                continue
            if "bb-timer" not in r.get("_path", ""):
                continue
            if r.get("routing_mode") != "gated" or r.get("gate_hidden") != 16:
                continue

            mse = r["sr_moa"]["mse"]
            if r.get("routing_input", "raw") == "hidden":
                hid_mses.append(mse)
            else:
                raw_mses.append(mse)

        if not raw_mses or not hid_mses:
            print(f"  {ds:12s}  INCOMPLETE (raw={len(raw_mses)}, hidden={len(hid_mses)})")
            continue

        raw_mean = np.mean(raw_mses)
        hid_mean = np.mean(hid_mses)
        delta_pct = (hid_mean - raw_mean) / raw_mean * 100

        if len(raw_mses) >= 2 and len(hid_mses) >= 2:
            t_stat, p_val = stats.ttest_ind(raw_mses, hid_mses, equal_var=False)
        else:
            p_val = float("nan")

        print(f"  {ds:12s}  Raw={raw_mean:.3f}(n={len(raw_mses)})  "
              f"Hidden={hid_mean:.3f}(n={len(hid_mses)})  "
              f"Δ={delta_pct:+.1f}%  p={p_val:.4f}")

    # Summary
    print(f"\n  Expected: p>0.05 (no significant difference) → confirms Prop 2 prediction")


def analyze_freeze_ablation(results):
    """E3: Frozen Paradox replication."""
    print("\n" + "=" * 60)
    print("E3: Frozen Paradox Replication (SR-MoA, MOMENT-small, H=96)")
    print("=" * 60)

    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
    freeze_levels = ["frozen", "last2", "last4"]

    frozen_wins = 0
    total = 0

    for ds in datasets:
        row = {}
        for fl in freeze_levels:
            mses = [r["sr_moa"]["mse"] for r in results
                    if r["dataset"] == ds and r["horizon"] == 96
                    and r["unfreeze"] == fl and "bb-" not in r.get("_path", "")
                    and r.get("routing_mode") == "gated" and r.get("gate_hidden") == 16
                    and r.get("gate_init_bias", 0.0) == 0.0
                    and r.get("routing_input", "raw") in ("raw", None)]
            if mses:
                row[fl] = (np.mean(mses), np.std(mses, ddof=1) if len(mses) > 1 else 0, len(mses))

        if "frozen" not in row:
            print(f"  {ds:12s}  INCOMPLETE")
            continue

        parts = []
        best_fl = min(row.keys(), key=lambda k: row[k][0])
        for fl in freeze_levels:
            if fl in row:
                m, s, n = row[fl]
                marker = " ←BEST" if fl == best_fl else ""
                parts.append(f"{fl}={m:.3f}±{s:.3f}(n={n}){marker}")
        print(f"  {ds:12s}  " + "  ".join(parts))

        if best_fl == "frozen":
            frozen_wins += 1
        total += 1

    print(f"\n  Frozen wins: {frozen_wins}/{total}")


def analyze_cross_backbone(results):
    """E4: Cross-backbone SR-MoA grid."""
    print("\n" + "=" * 60)
    print("E4: Cross-Backbone SR-MoA Grid (raw, frozen, H=96)")
    print("=" * 60)

    datasets = ["ETTh1", "ETTm1", "Weather"]
    backbone_suffixes = {
        "MOMENT-sm": lambda p: "bb-" not in p,
        "MOMENT-lg": lambda p: "bb-moment-large" in p,
        "Moirai": lambda p: "bb-moirai" in p and "moe" not in p,
        "Moirai-MoE": lambda p: "bb-moirai-moe" in p,
        "Chronos": lambda p: "bb-chronos" in p,
        "Timer-XL": lambda p: "bb-timer" in p,
    }

    header = f"  {'Dataset':12s}" + "".join(f"  {bb:>12s}" for bb in backbone_suffixes.keys())
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds in datasets:
        parts = [f"  {ds:12s}"]
        for bb_name, bb_filter in backbone_suffixes.items():
            mses = [r["sr_moa"]["mse"] for r in results
                    if r["dataset"] == ds and r["horizon"] == 96
                    and r["unfreeze"] == "frozen" and bb_filter(r.get("_path", ""))
                    and r.get("routing_mode") == "gated" and r.get("gate_hidden") == 16
                    and r.get("gate_init_bias", 0.0) == 0.0
                    and r.get("routing_input", "raw") in ("raw", None)]
            if mses:
                parts.append(f"  {np.mean(mses):>9.3f}({len(mses)})")
            else:
                parts.append(f"  {'—':>12s}")
        print("".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/self_routed_moa")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} SR-MoA results from {args.results_dir}\n")

    analyze_raw_vs_hidden(results)
    analyze_timer_negative_control(results)
    analyze_freeze_ablation(results)
    analyze_cross_backbone(results)


if __name__ == "__main__":
    main()
