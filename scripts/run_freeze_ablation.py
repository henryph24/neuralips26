"""Freeze-level and Top-K sparsity ablation runner.

Orchestrates the critical experiment matrix:
  - 3 datasets x 3 freeze levels x {RR-MoA, baselines} = 18 runs
  - Top-K ablation: 4 sparsity levels x 1 dataset = 4 runs
  - AdaMix (routing collapse control): 3 datasets x 3 freeze levels = 9 runs

Usage (on RACE VM with GPU):
    python scripts/run_freeze_ablation.py                          # full matrix
    python scripts/run_freeze_ablation.py --experiment topk        # Top-K only
    python scripts/run_freeze_ablation.py --experiment freeze      # freeze-level only
    python scripts/run_freeze_ablation.py --experiment adamix      # AdaMix control
    python scripts/run_freeze_ablation.py --experiment all         # everything
"""

import argparse
import json
import os
import subprocess
import sys
import time


DATASETS = ["ETTh1", "ETTm1", "Weather"]
FREEZE_LEVELS = ["frozen", "last2", "last4"]
TOPK_VALUES = [1, 2, 3, None]  # None = dense (K=5)
DEFAULT_SEED = 42
DEFAULT_K = 5


def run_cmd(cmd, label):
    """Run a command and return success/failure."""
    print("\n" + "=" * 70)
    print("RUNNING: %s" % label)
    print("CMD: %s" % " ".join(cmd))
    print("=" * 70)
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED (rc=%d)" % result.returncode
    print("[%s] %s  (%.0fs)" % (status, label, elapsed))
    return result.returncode == 0


def run_freeze_ablation(device="cuda", seed=DEFAULT_SEED, epochs=15):
    """Run freeze-level ablation: 3 datasets x 3 freeze levels x {RR-MoA + baselines}."""
    results = []
    for dataset in DATASETS:
        for unfreeze in FREEZE_LEVELS:
            label = "RR-MoA %s unfreeze=%s seed=%d" % (dataset, unfreeze, seed)
            cmd = [
                sys.executable, "scripts/run_rr_moa.py",
                "--dataset", dataset,
                "--unfreeze", unfreeze,
                "--top-k", "2",
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--device", device,
            ]
            ok = run_cmd(cmd, label)
            results.append({"label": label, "ok": ok})
    return results


def run_topk_ablation(device="cuda", seed=DEFAULT_SEED, epochs=15, dataset="ETTh1", unfreeze="last2"):
    """Run Top-K sparsity ablation on one dataset."""
    results = []
    for top_k in TOPK_VALUES:
        tk_label = "top%d" % top_k if top_k else "dense"
        label = "RR-MoA %s %s unfreeze=%s" % (dataset, tk_label, unfreeze)
        cmd = [
            sys.executable, "scripts/run_rr_moa.py",
            "--dataset", dataset,
            "--unfreeze", unfreeze,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--device", device,
            "--no-baselines",
        ]
        if top_k is not None:
            cmd += ["--top-k", str(top_k)]
        ok = run_cmd(cmd, label)
        results.append({"label": label, "ok": ok})
    return results


def run_adamix_control(device="cuda", seed=DEFAULT_SEED, epochs=15):
    """Run AdaMix (hidden-state routing) as collapse control across freeze levels."""
    results = []
    for dataset in DATASETS:
        for unfreeze in FREEZE_LEVELS:
            label = "AdaMix %s unfreeze=%s seed=%d" % (dataset, unfreeze, seed)
            cmd = [
                sys.executable, "scripts/run_adamix.py",
                "--dataset", dataset,
                "--unfreeze", unfreeze,
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--device", device,
            ]
            ok = run_cmd(cmd, label)
            results.append({"label": label, "ok": ok})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["freeze", "topk", "adamix", "all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    all_results = []
    t0 = time.time()

    if args.experiment in ("freeze", "all"):
        print("\n### FREEZE-LEVEL ABLATION ###")
        all_results += run_freeze_ablation(args.device, args.seed, args.epochs)

    if args.experiment in ("topk", "all"):
        print("\n### TOP-K SPARSITY ABLATION ###")
        all_results += run_topk_ablation(args.device, args.seed, args.epochs)

    if args.experiment in ("adamix", "all"):
        print("\n### ADAMIX ROUTING COLLAPSE CONTROL ###")
        all_results += run_adamix_control(args.device, args.seed, args.epochs)

    total = time.time() - t0
    n_ok = sum(1 for r in all_results if r["ok"])
    n_total = len(all_results)

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE: %d/%d succeeded in %.0fs (%.1f min)" % (
        n_ok, n_total, total, total / 60))
    print("=" * 70)

    for r in all_results:
        status = "OK" if r["ok"] else "FAIL"
        print("  [%s] %s" % (status, r["label"]))

    summary_path = "results/freeze_ablation_summary_%d.json" % args.seed
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "total_seconds": total}, f, indent=2)
    print("\nSummary: %s" % summary_path)


if __name__ == "__main__":
    main()
