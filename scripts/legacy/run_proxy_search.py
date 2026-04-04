"""End-to-end pipeline: calibration data collection + GP proxy search + validation.

Usage:
    # Full pipeline on Modal
    modal run feasibility/modal_app.py --mode proxy-search

    # Local smoke test (CPU, small scale)
    python scripts/run_proxy_search.py --local --n-configs 5 --n-generations 5

    # GP search only (from cached calibration data)
    python scripts/run_proxy_search.py --gp-only --calibration-dir results/calibration/
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from feasibility.config import AdapterConfig
from feasibility.proxy_gp import serialize_tree, deserialize_tree
from feasibility.proxy_search import (
    CalibrationData,
    ProxySearchResult,
    select_calibration_configs,
    collect_calibration_data,
    save_calibration_data,
    load_calibration_data,
    run_proxy_search,
    make_proxy_evaluate_fn,
)


def run_local_calibration(
    n_configs: int = 5,
    n_epochs: int = 5,
    datasets: list = None,
    seed: int = 42,
):
    """Phase 1: Collect calibration data locally (CPU, small scale)."""
    from feasibility.statistics import compute_all_statistics
    from feasibility.finetune import finetune_forecasting, finetune_classification
    from feasibility.data import load_etth1, load_ettm1, load_ethanol_concentration

    if datasets is None:
        datasets = ["ETTh1"]

    loaders = {
        "ETTh1": load_etth1,
        "ETTm1": load_ettm1,
        "EthanolConcentration": load_ethanol_concentration,
    }

    configs = select_calibration_configs(n=n_configs, seed=seed)
    print(f"Selected {len(configs)} calibration configs")

    calibrations = []
    for ds_name in datasets:
        print(f"\n--- Calibrating on {ds_name} ---")
        ds = loaders[ds_name]()
        samples = ds["samples"]
        labels = ds.get("labels")
        is_clf = ds.get("task") == "classification"

        def compute_stats_fn(cfg):
            return compute_all_statistics(cfg, samples, labels=labels, device="cpu", batch_size=32)

        def finetune_fn(cfg):
            if is_clf:
                return finetune_classification(cfg, samples, labels, device="cpu", n_epochs=n_epochs)
            return finetune_forecasting(cfg, samples, device="cpu", n_epochs=n_epochs)

        cal = collect_calibration_data(configs, ds, compute_stats_fn, finetune_fn, n_epochs)

        path = f"results/calibration/{ds_name}_{seed}.json"
        save_calibration_data(cal, path)
        print(f"Saved calibration to {path}")
        calibrations.append(cal)

    return calibrations


def run_gp_search(
    calibrations: list,
    n_generations: int = 50,
    pop_size: int = 100,
    seed: int = 42,
):
    """Phase 2: GP proxy search on cached calibration data (CPU)."""
    print(f"\n{'='*60}")
    print(f"GP Proxy Search: {n_generations} generations, pop={pop_size}")
    print(f"Calibration datasets: {[c.dataset_name for c in calibrations]}")
    print(f"{'='*60}")

    result = run_proxy_search(
        calibration=calibrations[0],
        multi_calibrations=calibrations if len(calibrations) > 1 else None,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
    )

    # Save results
    os.makedirs("results/proxy_search", exist_ok=True)
    out = result.to_dict()
    out["calibration_datasets"] = [c.dataset_name for c in calibrations]
    with open(f"results/proxy_search/gp_result_{seed}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nBest proxy: {result.best_formula}")
    print(f"Best fitness (Kendall tau): {result.best_fitness:.4f}")
    print(f"Saved to results/proxy_search/gp_result_{seed}.json")

    return result


def run_validation(
    gp_result: ProxySearchResult,
    datasets: list = None,
    n_evo_generations: int = 12,
    validate_top: int = 5,
    seed: int = 42,
):
    """Phase 3: Compare GP proxy vs TEMPLATE vs random on validation datasets."""
    from feasibility.evolution import run_evolution, random_config
    from feasibility.finetune import finetune_forecasting, finetune_classification
    from feasibility.data import load_etth1, load_ettm1, load_ethanol_concentration

    if datasets is None:
        datasets = ["ETTh1"]

    loaders = {
        "ETTh1": load_etth1,
        "ETTm1": load_ettm1,
        "EthanolConcentration": load_ethanol_concentration,
    }

    best_tree = gp_result.best_tree

    for ds_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# Validation — {ds_name}")
        print(f"{'#'*60}")

        ds = loaders[ds_name]()
        is_clf = ds.get("task") == "classification"
        metric_key = "accuracy" if is_clf else "mse"

        # --- Method 1: Evo-GP-Proxy ---
        print("\n[1/3] Evo-GP-Proxy...")
        proxy_eval_fn = make_proxy_evaluate_fn(best_tree, ds, device="cpu", batch_size=32)
        evo_proxy_result = run_evolution(
            evaluate_fn=proxy_eval_fn,
            n_generations=n_evo_generations,
            pop_size=20,
            seed=seed,
        )

        # --- Method 2: Random search ---
        print("\n[2/3] Random search...")
        rng = np.random.default_rng(seed)
        random_configs = [random_config(rng) for _ in range(20)]
        random_results = proxy_eval_fn(random_configs)
        random_results.sort(key=lambda r: r["scores"]["composite"], reverse=True)

        # --- Collect top configs from each method ---
        proxy_top = [
            AdapterConfig.from_dict(r["config"])
            for r in evo_proxy_result["final_population"][:validate_top]
        ]
        random_top = [
            AdapterConfig.from_dict(r["config"])
            for r in random_results[:validate_top]
        ]

        # --- Fine-tune all ---
        print(f"\nFine-tuning top-{validate_top} from each method...")
        all_methods = {
            "Evo-GP-Proxy": proxy_top,
            "Random": random_top,
        }

        validation_results = {}
        for method_name, configs in all_methods.items():
            results = []
            for cfg in configs:
                if is_clf:
                    r = finetune_classification(cfg, ds["samples"], ds["labels"], device="cpu", n_epochs=10)
                else:
                    r = finetune_forecasting(cfg, ds["samples"], device="cpu", n_epochs=10)
                results.append({
                    "config_id": cfg.config_id,
                    metric_key: r[metric_key],
                })
            validation_results[method_name] = results

        # --- Print comparison ---
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS — {ds_name} ({metric_key})")
        print(f"{'='*60}")

        for method_name, results in validation_results.items():
            vals = [r[metric_key] for r in results]
            best_val = max(vals) if is_clf else min(vals)
            mean_val = np.mean(vals)
            print(f"\n{method_name}:")
            print(f"  Best {metric_key}: {best_val:.4f}")
            print(f"  Mean {metric_key}: {mean_val:.4f}")
            for r in results:
                print(f"    {r['config_id']}: {r[metric_key]:.4f}")

        # Save
        os.makedirs("results/proxy_search", exist_ok=True)
        out = {
            "dataset": ds_name,
            "metric_key": metric_key,
            "proxy_formula": gp_result.best_formula,
            "proxy_fitness": gp_result.best_fitness,
            "validation": validation_results,
        }
        with open(f"results/proxy_search/validation_{ds_name}_{seed}.json", "w") as f:
            json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GP proxy search pipeline")
    parser.add_argument("--local", action="store_true", help="Run locally on CPU")
    parser.add_argument("--gp-only", action="store_true", help="GP search only (from cached calibration)")
    parser.add_argument("--validate-only", action="store_true", help="Validation only (from cached GP result)")
    parser.add_argument("--n-configs", type=int, default=30, help="Number of calibration configs")
    parser.add_argument("--n-generations", type=int, default=50, help="GP generations")
    parser.add_argument("--pop-size", type=int, default=100, help="GP population size")
    parser.add_argument("--n-epochs", type=int, default=15, help="Fine-tuning epochs for calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"], help="Datasets to use")
    parser.add_argument("--calibration-dir", type=str, default="results/calibration/", help="Calibration data dir")
    parser.add_argument("--gp-result", type=str, default=None, help="Path to GP result JSON")
    parser.add_argument("--validate-top", type=int, default=5, help="Top-k configs to validate")
    args = parser.parse_args()

    if args.validate_only:
        # Load GP result and run validation
        gp_path = args.gp_result or f"results/proxy_search/gp_result_{args.seed}.json"
        with open(gp_path) as f:
            gp_data = json.load(f)
        best_tree = deserialize_tree(gp_data["best_tree"])
        gp_result = ProxySearchResult(
            best_tree=best_tree,
            best_fitness=gp_data["best_fitness"],
            best_formula=gp_data["best_formula"],
            generation_log=gp_data["generation_log"],
            final_population=gp_data["final_population"],
        )
        run_validation(gp_result, datasets=args.datasets, validate_top=args.validate_top, seed=args.seed)
        return

    if args.gp_only:
        # Load calibration data
        calibrations = []
        for ds_name in args.datasets:
            path = os.path.join(args.calibration_dir, f"{ds_name}_{args.seed}.json")
            cal = load_calibration_data(path)
            calibrations.append(cal)
            print(f"Loaded calibration: {ds_name} ({len(cal.configs)} configs)")

        gp_result = run_gp_search(calibrations, args.n_generations, args.pop_size, args.seed)
        run_validation(gp_result, datasets=args.datasets, validate_top=args.validate_top, seed=args.seed)
        return

    if args.local:
        # Full local pipeline
        calibrations = run_local_calibration(
            n_configs=args.n_configs,
            n_epochs=args.n_epochs,
            datasets=args.datasets,
            seed=args.seed,
        )
        gp_result = run_gp_search(calibrations, args.n_generations, args.pop_size, args.seed)
        run_validation(gp_result, datasets=args.datasets, validate_top=args.validate_top, seed=args.seed)
        return

    print("Specify --local, --gp-only, or --validate-only. For Modal, use modal_app.py --mode proxy-search")


if __name__ == "__main__":
    main()
