"""Orchestrator: run all configs through TEMPLATE scoring.

Usage:
    Local (CPU, smoke test):  python scripts/run_sweep.py --local --max-configs 3
    Modal (GPU, full sweep):  modal run feasibility/modal_app.py
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import generate_configs
from feasibility.data import load_etth1, load_ethanol_concentration
from feasibility.features import extract_features
from feasibility.scores import compute_template_scores


def run_local(max_configs: int = 3, datasets: list = None):
    """Run a local smoke test on CPU with a few configs."""
    configs = generate_configs()[:max_configs]
    print(f"Running {len(configs)} configs locally (CPU)")

    if datasets is None:
        print("Loading ETTh1...")
        datasets = [load_etth1()]

    results = []
    for ds in datasets:
        # Use a small subset for speed
        samples = ds["samples"][:100]
        ds_name = ds["name"]
        print(f"\nDataset: {ds_name} ({samples.shape[0]} samples)")

        for i, cfg in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {cfg.config_id}...", end=" ", flush=True)
            try:
                features = extract_features(cfg, samples, device="cpu", batch_size=32)
                scores = compute_template_scores(
                    features["feature"],
                    features["first_feature"],
                    features["trend_feature"],
                    device="cpu",
                )
                result = {
                    "config": cfg.to_dict(),
                    "scores": scores,
                    "dataset": ds_name,
                }
                results.append(result)
                print(f"DL={scores['dl']:.4f} PL={scores['pl']:.4f} TA={scores['ta']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/sweep_results_local.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Quick variance check
    if results:
        composites = [r["scores"]["composite"] for r in results]
        print(f"\nComposite score stats: mean={sum(composites)/len(composites):.4f}, "
              f"std={__import__('numpy').std(composites):.4f}, "
              f"range=[{min(composites):.4f}, {max(composites):.4f}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally on CPU")
    parser.add_argument("--max-configs", type=int, default=3, help="Max configs for local run")
    parser.add_argument("--dataset", choices=["etth1", "ethanol", "both"], default="etth1")
    args = parser.parse_args()

    if args.local:
        datasets = []
        if args.dataset in ("etth1", "both"):
            datasets.append(load_etth1())
        if args.dataset in ("ethanol", "both"):
            datasets.append(load_ethanol_concentration())
        run_local(max_configs=args.max_configs, datasets=datasets)
    else:
        print("For GPU sweep, use: modal run feasibility/modal_app.py")
