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


DEFAULT_DATASETS = ["ETTh1", "ETTm1", "Weather"]
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


def run_freeze_ablation(device="cuda", seed=DEFAULT_SEED, epochs=15,
                        datasets=None, freeze_levels=None):
    """Run freeze-level ablation: N datasets x M freeze levels x {RR-MoA + baselines}."""
    datasets = datasets or DEFAULT_DATASETS
    freeze_levels = freeze_levels or FREEZE_LEVELS
    results = []
    for dataset in datasets:
        for unfreeze in freeze_levels:
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


def run_adamix_control(device="cuda", seed=DEFAULT_SEED, epochs=15,
                       datasets=None, freeze_levels=None):
    """Run AdaMix (hidden-state routing) as collapse control across freeze levels."""
    datasets = datasets or DEFAULT_DATASETS
    freeze_levels = freeze_levels or FREEZE_LEVELS
    results = []
    for dataset in datasets:
        for unfreeze in freeze_levels:
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


def run_uniform_control(device="cuda", seed=DEFAULT_SEED, epochs=15,
                        datasets=None):
    """T1.B ensemble-vs-specialization control: RR-MoA with uniform router
    (fixed 1/K weights) under the strictly frozen backbone on the datasets
    where raw-routed RR-MoA is strongest. If uniform matches raw, the paper
    must reframe from ``per-sample specialization'' to ``soft ensemble.''
    """
    datasets = datasets or DEFAULT_DATASETS
    results = []
    for dataset in datasets:
        label = "RR-MoA-uniform %s frozen seed=%d" % (dataset, seed)
        cmd = [
            sys.executable, "scripts/run_rr_moa.py",
            "--dataset", dataset,
            "--unfreeze", "frozen",
            "--top-k", "2",
            "--router-input-mode", "uniform",
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--device", device,
            "--no-baselines",
        ]
        ok = run_cmd(cmd, label)
        results.append({"label": label, "ok": ok})
    return results


def run_macro_pool(device="cuda", seed=DEFAULT_SEED, epochs=15, datasets=None):
    """T3.A: RR-MoA with the AAS-distilled macro expert pool, strictly
    frozen backbone, Top-2 sparse. Deliverable is a head-to-head row against
    the canonical-pool RR-MoA on the same 3 main datasets + extended set,
    demonstrating that the AAS discoveries actually feed into the RR-MoA
    experiments (closing reviewer W1).
    """
    datasets = datasets or DEFAULT_DATASETS
    results = []
    for dataset in datasets:
        label = "RR-MoA-macro %s frozen seed=%d" % (dataset, seed)
        cmd = [
            sys.executable, "scripts/run_rr_moa.py",
            "--dataset", dataset,
            "--unfreeze", "frozen",
            "--top-k", "2",
            "--expert-pool", "macro",
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--device", device,
            "--no-baselines",
        ]
        ok = run_cmd(cmd, label)
        results.append({"label": label, "ok": ok})
    return results


def run_dlinear_baseline(device="cuda", seed=DEFAULT_SEED, epochs=15,
                         datasets=None):
    """T1.A DLinear reference: compute DLinear MSE on the same normalized
    scale and also on the original (denormalized) scale for each dataset
    in this seed so that the paper can show a direct side-by-side row."""
    datasets = datasets or DEFAULT_DATASETS
    results = []
    for dataset in datasets:
        label = "DLinear %s seed=%d" % (dataset, seed)
        cmd = [
            sys.executable, "scripts/run_dlinear_baseline.py",
            "--dataset", dataset,
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
                        choices=["freeze", "topk", "adamix", "uniform",
                                 "dlinear", "macro", "all"])
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset list (default: ETTh1,ETTm1,Weather). "
                             "Use e.g. 'ETTh2,ETTm2,Electricity' to extend the grid.")
    parser.add_argument("--freeze-levels", default=None,
                        help="Comma-separated freeze levels (default: frozen,last2,last4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    datasets = args.datasets.split(",") if args.datasets else None
    freeze_levels = args.freeze_levels.split(",") if args.freeze_levels else None

    all_results = []
    t0 = time.time()

    if args.experiment in ("freeze", "all"):
        print("\n### FREEZE-LEVEL ABLATION ###")
        all_results += run_freeze_ablation(args.device, args.seed, args.epochs,
                                           datasets=datasets, freeze_levels=freeze_levels)

    if args.experiment in ("topk", "all"):
        print("\n### TOP-K SPARSITY ABLATION ###")
        all_results += run_topk_ablation(args.device, args.seed, args.epochs)

    if args.experiment in ("adamix", "all"):
        print("\n### ADAMIX ROUTING COLLAPSE CONTROL ###")
        all_results += run_adamix_control(args.device, args.seed, args.epochs,
                                          datasets=datasets, freeze_levels=freeze_levels)

    if args.experiment in ("uniform", "all"):
        print("\n### T1.B UNIFORM ROUTER CONTROL ###")
        all_results += run_uniform_control(args.device, args.seed, args.epochs,
                                           datasets=datasets)

    if args.experiment in ("dlinear", "all"):
        print("\n### T1.A DLINEAR REFERENCE ###")
        all_results += run_dlinear_baseline(args.device, args.seed, args.epochs,
                                            datasets=datasets)

    if args.experiment in ("macro", "all"):
        print("\n### T3.A AAS MACRO-EXPERT POOL ###")
        all_results += run_macro_pool(args.device, args.seed, args.epochs,
                                      datasets=datasets)

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
