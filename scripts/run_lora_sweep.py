"""T3.B -- LoRA sweep orchestrator.

Addresses reviewer W6b: the current paper's LoRA comparison uses a single
config (rank=8, q+v projections, linear head). A NeurIPS reviewer will
reasonably ask whether a stronger LoRA configuration closes the gap. This
script sweeps LoRA across the cells that matter:

    rank          in {8, 16, 32}           -- low-rank expressivity
    target_modules in {qv, qkvo}            -- coverage of attention projections
    head          in {linear, mlp2}         -- trainable capacity on top
    seed          in {42, 43, 44}           -- multi-seed stability

Under the ``strictly frozen`` backbone (matching the paper's main
comparison) this gives 3*2*2*3 = 36 runs per dataset and 108 runs total
across {ETTh1, ETTm1, Weather}. Runtime: ~2 min per run on A10G = roughly
3.6 GPU-hours. Results land in ``results/lora_baseline/`` under descriptive
filenames; the default (rank=8, qv, linear) filename format is preserved
so ``evidence_vm/verify.py`` continues to pass.

After the sweep completes, run the summary helper to pick the strongest
LoRA per dataset for the paper's Table 6::

    python scripts/run_lora_sweep.py --summarize

Usage::

    python scripts/run_lora_sweep.py                               # full sweep (3 datasets)
    python scripts/run_lora_sweep.py --datasets ETTh1              # single dataset
    python scripts/run_lora_sweep.py --dry-run                      # print commands only
    python scripts/run_lora_sweep.py --summarize                    # best-per-dataset table
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

RANKS = [8, 16, 32]
TARGETS = ["qv", "qkvo"]
HEADS = ["linear", "mlp2"]
SEEDS = [42, 43, 44]
DEFAULT_DATASETS = ["ETTh1", "ETTm1", "Weather"]
DEFAULT_EPOCHS = 15


def sweep_commands(datasets, epochs, device):
    """Enumerate every (dataset, rank, targets, head, seed) config."""
    for dataset in datasets:
        for rank in RANKS:
            for targets in TARGETS:
                for head in HEADS:
                    for seed in SEEDS:
                        yield {
                            "dataset": dataset, "rank": rank,
                            "target_modules": targets, "head": head,
                            "seed": seed,
                            "cmd": [
                                sys.executable, "scripts/run_lora_baseline.py",
                                "--dataset", dataset,
                                "--rank", str(rank),
                                "--target-modules", targets,
                                "--head", head,
                                "--unfreeze", "frozen",
                                "--seed", str(seed),
                                "--epochs", str(epochs),
                                "--device", device,
                            ],
                        }


def run_sweep(datasets, epochs, device, dry_run=False, skip_existing=True):
    configs = list(sweep_commands(datasets, epochs, device))
    print("[LoRA sweep] %d runs queued (datasets=%s)" % (len(configs), datasets))
    t0 = time.time()
    results = []
    for i, c in enumerate(configs):
        label = "[%3d/%d] %s r=%d %s head=%s seed=%d" % (
            i + 1, len(configs), c["dataset"], c["rank"], c["target_modules"], c["head"], c["seed"])
        # Skip if a JSON already exists (idempotent sweep restart).
        if skip_existing:
            expected = _expected_path(c)
            if expected and os.path.exists(expected):
                print("SKIP %s (exists: %s)" % (label, expected))
                results.append({"label": label, "ok": True, "skipped": True})
                continue
        print("\n" + "=" * 70)
        print("RUN %s" % label)
        print("=" * 70)
        if dry_run:
            print("DRY: " + " ".join(c["cmd"]))
            results.append({"label": label, "ok": True, "dry": True})
            continue
        rc = subprocess.run(c["cmd"], capture_output=False).returncode
        ok = rc == 0
        results.append({"label": label, "ok": ok})
        if not ok:
            print("FAILED rc=%d" % rc)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print("\n[LoRA sweep] %d/%d ok in %.1f min" % (n_ok, len(results), elapsed / 60))
    return results


def _expected_path(c):
    dataset, rank, tm, head, seed = c["dataset"], c["rank"], c["target_modules"], c["head"], c["seed"]
    if tm == "qv" and head == "linear":
        return "results/lora_baseline/%s_H96_r%d_frozen_%d.json" % (dataset, rank, seed)
    return "results/lora_baseline/%s_H96_r%d_%s_%s_frozen_%d.json" % (
        dataset, rank, tm, head, seed)


def summarize():
    """Pick the strongest LoRA config per dataset and print a camera-ready row."""
    paths = sorted(glob.glob("results/lora_baseline/*.json"))
    by_dataset = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ds = d.get("dataset")
        if ds is None or "lora_mse" not in d:
            continue
        if d.get("unfreeze") != "frozen":
            continue
        key = (ds, d.get("rank"), d.get("target_modules", "qv"), d.get("head", "linear"))
        by_dataset.setdefault(key, []).append((p, d["lora_mse"]))

    # Aggregate across seeds for each (dataset, rank, targets, head).
    aggregates = {}
    for key, entries in by_dataset.items():
        if len(entries) < 2:
            continue
        mses = [e[1] for e in entries]
        mean = sum(mses) / len(mses)
        std = (sum((m - mean) ** 2 for m in mses) / len(mses)) ** 0.5
        aggregates[key] = (mean, std, len(mses))

    # Per-dataset best (by mean MSE).
    best_by_ds = {}
    for key, (mean, std, n) in aggregates.items():
        ds = key[0]
        if ds not in best_by_ds or mean < best_by_ds[ds][1][0]:
            best_by_ds[ds] = (key, (mean, std, n))

    print("\n# LoRA sweep summary (strictly frozen backbone, seed-averaged)")
    print("%-10s  %-6s  %-6s  %-8s  %-8s  %-8s  %s" % (
        "Dataset", "rank", "target", "head", "mean MSE", "std", "n seeds"))
    print("-" * 72)
    for ds, (key, (mean, std, n)) in sorted(best_by_ds.items()):
        print("%-10s  r=%-4d  %-6s  %-8s  %-8.4f  %-8.4f  %d" % (
            ds, key[1], key[2], key[3], mean, std, n))

    print("\n# Full sweep grid (for appendix):")
    for key, (mean, std, n) in sorted(aggregates.items()):
        print("  %-10s r=%-4d %-6s %-8s  %.4f +- %.4f  (n=%d)" % (
            key[0], key[1], key[2], key[3], mean, std, n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip running and print the best-per-dataset table from existing JSONs.")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        return

    datasets = args.datasets.split(",")
    run_sweep(datasets, args.epochs, args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
