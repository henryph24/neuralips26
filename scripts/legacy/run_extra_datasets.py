"""Extend early-stopping proxy experiment to ETTh2 (and Weather if available).

Phase 1: Calibration (30 configs) + early-stopping correlation (1/3/5 vs 15 epoch)
Phase 2: Evo-3epoch vs Evo-GP-Proxy vs Random, fine-tune top 5 at 15 epochs

Usage:
    modal run scripts/run_extra_datasets.py::app.extra_datasets --seed 42
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
import numpy as np
from scipy.stats import kendalltau, spearmanr

from feasibility.modal_app import app, finetune_config, compute_statistics_remote
from feasibility.proxy_search import (
    CalibrationData,
    select_calibration_configs,
    save_calibration_data,
    load_calibration_data,
)
from feasibility.proxy_gp import deserialize_tree, evaluate_proxy
from feasibility.data import load_etth2, serialize_dataset
from feasibility.evolution import run_evolution, random_config

# Check if weather data exists
_weather_available = os.path.exists(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "weather.csv")
)
if _weather_available:
    from feasibility.data import load_weather

EARLY_EPOCHS = [1, 3, 5]


def _build_datasets():
    """Build dataset registry for available extra datasets."""
    datasets = {"ETTh2": load_etth2}
    if _weather_available:
        datasets["Weather"] = load_weather
    return datasets


@app.local_entrypoint()
def extra_datasets(seed: int = 42, validate_top: int = 5,
                   evo_generations: int = 12, evo_pop: int = 20):
    os.makedirs("results/early_stopping", exist_ok=True)
    os.makedirs("results/calibration", exist_ok=True)

    datasets = _build_datasets()
    print(f"Datasets to process: {list(datasets.keys())}")

    # Load GP proxy tree
    gp_path = f"results/proxy_search/gp_result_{seed}.json"
    gp_tree = None
    if os.path.exists(gp_path):
        with open(gp_path) as f:
            gp_tree = deserialize_tree(json.load(f)["best_tree"])
        print(f"Loaded GP proxy tree from {gp_path}")
    else:
        print(f"WARNING: No GP tree at {gp_path}, Evo-GP-Proxy will be skipped")

    # =========================================================================
    # Phase 1: Calibration + Early-Stopping Correlation
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# Phase 1: Calibration + Early-Stopping ({EARLY_EPOCHS} epochs)")
    print(f"{'#'*60}")

    all_correlation_results = {}

    for ds_name, loader in datasets.items():
        cal_path = f"results/calibration/{ds_name}_{seed}.json"

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)
        is_clf = meta.get("task") == "classification"
        mk = "accuracy" if is_clf else "mse"

        # --- Calibration: collect 30-config data if not cached ---
        if os.path.exists(cal_path):
            print(f"\n--- {ds_name}: Loading cached calibration from {cal_path} ---")
            cal = load_calibration_data(cal_path)
        else:
            print(f"\n--- {ds_name}: Running calibration (30 configs, 15 epochs) ---")
            configs = select_calibration_configs(n=30, seed=seed)
            print(f"  Computing statistics for {len(configs)} configs...")
            stat_results = list(compute_statistics_remote.starmap(
                [(c.to_dict(), data_bytes, meta) for c in configs]
            ))
            statistics = [sr["statistics"] for sr in stat_results]

            print(f"  Fine-tuning {len(configs)} configs for 15 epochs...")
            ft_results_15 = list(finetune_config.starmap(
                [(c.to_dict(), data_bytes, meta, 15) for c in configs]
            ))
            performances = [float(r[mk]) for r in ft_results_15]

            cal = CalibrationData(
                configs=configs,
                statistics=statistics,
                performances=performances,
                dataset_name=ds_name,
                metric_key=mk,
                higher_is_better=is_clf,
            )
            save_calibration_data(cal, cal_path)
            print(f"  Saved calibration to {cal_path}")

        configs = cal.configs
        perfs_15 = cal.performances
        print(f"  {len(configs)} configs, 15-epoch {mk}: "
              f"min={min(perfs_15):.4f} max={max(perfs_15):.4f}")

        # --- Early-stopping correlation: 1, 3, 5 epochs ---
        ds_results = {"15_epoch": perfs_15}

        for n_ep in EARLY_EPOCHS:
            print(f"  Fine-tuning {len(configs)} configs for {n_ep} epochs...")
            ft_results = list(finetune_config.starmap(
                [(cfg.to_dict(), data_bytes, meta, n_ep) for cfg in configs]
            ))
            perfs_early = [float(r[mk]) for r in ft_results]
            ds_results[f"{n_ep}_epoch"] = perfs_early

            tau, p = kendalltau(perfs_early, perfs_15)
            rho, _ = spearmanr(perfs_early, perfs_15)
            print(f"    {n_ep}-epoch vs 15-epoch: τ={tau:.3f} (p={p:.4f}), ρ={rho:.3f}")

        all_correlation_results[ds_name] = {
            "metric_key": mk,
            "higher_is_better": is_clf,
            "epochs_data": {k: [float(v) for v in vs] for k, vs in ds_results.items()},
            "configs": [c.to_dict() for c in configs],
        }

    # Save correlation data
    corr_path = f"results/early_stopping/epoch_correlations_extra_{seed}.json"
    with open(corr_path, "w") as f:
        json.dump(all_correlation_results, f, indent=2, default=str)
    print(f"\nSaved correlation data to {corr_path}")

    # --- Correlation summary ---
    print(f"\n{'='*70}")
    print(f"EARLY-STOPPING CORRELATION SUMMARY (Extra Datasets)")
    print(f"{'='*70}")

    header = f"{'Dataset':<25} {'Epochs':>8} {'Kendall τ':>12} {'Spearman ρ':>12}"
    print(header)
    print("-" * len(header))

    for ds_name, ds_data in all_correlation_results.items():
        perfs_15 = ds_data["epochs_data"]["15_epoch"]
        is_clf = ds_data["higher_is_better"]

        for n_ep in EARLY_EPOCHS:
            perfs_early = ds_data["epochs_data"][f"{n_ep}_epoch"]
            tau, _ = kendalltau(perfs_early, perfs_15)
            rho, _ = spearmanr(perfs_early, perfs_15)
            print(f"{ds_name:<25} {n_ep:>8} {tau:>12.3f} {rho:>12.3f}")

        # GP proxy correlation
        if gp_tree:
            cal = load_calibration_data(f"results/calibration/{ds_name}_{seed}.json")
            proxy_scores = [evaluate_proxy(gp_tree, s) for s in cal.statistics]
            sign = 1.0 if is_clf else -1.0
            gp_tau, _ = kendalltau([sign * s for s in proxy_scores], perfs_15)
            print(f"{ds_name:<25} {'GP-proxy':>8} {gp_tau:>12.3f} {'':>12}")

    # =========================================================================
    # Phase 2: Evo-3epoch vs Evo-GP-Proxy vs Random
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Evo-3epoch vs Evo-GP-Proxy vs Random")
    print(f"{'#'*60}")

    for ds_name, loader in datasets.items():
        if ds_name not in all_correlation_results:
            continue

        comp_path = f"results/early_stopping/comparison_{ds_name}_{seed}.json"
        if os.path.exists(comp_path):
            print(f"\nSkipping {ds_name}: already completed at {comp_path}")
            continue

        ds_data = all_correlation_results[ds_name]
        mk = ds_data["metric_key"]
        is_clf = ds_data["higher_is_better"]

        print(f"\n--- {ds_name} ---")
        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        # Method 1: Evo-3epoch
        print("  [1/3] Evo-3epoch...")
        def make_3ep_eval(d_bytes, d_meta, metric_key, higher):
            def eval_fn(cfgs):
                fts = list(finetune_config.starmap(
                    [(c.to_dict(), d_bytes, d_meta, 3) for c in cfgs]
                ))
                results = []
                for c, ft in zip(cfgs, fts):
                    val = float(ft[metric_key])
                    score = val if higher else -val
                    results.append({"config": c.to_dict(), "scores": {"composite": score}})
                return results
            return eval_fn

        ep3_eval = make_3ep_eval(data_bytes, meta, mk, is_clf)
        evo_3ep = run_evolution(ep3_eval, n_generations=evo_generations, pop_size=evo_pop, seed=seed)

        # Method 2: Evo-GP-Proxy
        evo_gp = None
        if gp_tree:
            print("  [2/3] Evo-GP-Proxy...")
            def make_gp_eval(d_bytes, d_meta, tree):
                def eval_fn(cfgs):
                    srs = list(compute_statistics_remote.starmap(
                        [(c.to_dict(), d_bytes, d_meta) for c in cfgs]
                    ))
                    return [
                        {"config": c.to_dict(), "scores": {"composite": evaluate_proxy(tree, sr["statistics"])}}
                        for c, sr in zip(cfgs, srs)
                    ]
                return eval_fn
            gp_eval = make_gp_eval(data_bytes, meta, gp_tree)
            evo_gp = run_evolution(gp_eval, n_generations=evo_generations, pop_size=evo_pop, seed=seed)
        else:
            print("  [2/3] Evo-GP-Proxy... SKIPPED (no GP tree)")

        # Method 3: Random search
        print("  [3/3] Random search...")
        rng = np.random.default_rng(seed)
        rand_cfgs = [random_config(rng) for _ in range(evo_pop)]

        # Fine-tune top-k from each method (15 epochs)
        top_n = min(validate_top, 5)
        methods = {
            "Evo-3epoch": [r["config"] for r in evo_3ep["final_population"][:top_n]],
        }
        if evo_gp:
            methods["Evo-GP-Proxy"] = [r["config"] for r in evo_gp["final_population"][:top_n]]
        methods["Random"] = [c.to_dict() for c in rand_cfgs[:top_n]]

        print(f"\n  Fine-tuning top-{top_n} from each method (15 epochs)...")
        comparison = {}
        for method_name, top_cfgs in methods.items():
            ft = list(finetune_config.starmap(
                [(c, data_bytes, meta, 15) for c in top_cfgs]
            ))
            comparison[method_name] = ft

        # Print comparison
        print(f"\n  {'='*60}")
        print(f"  COMPARISON - {ds_name} ({mk})")
        print(f"  {'='*60}")
        for method_name, results in comparison.items():
            vals = [float(r.get(mk, float("nan"))) for r in results]
            best_v = max(vals) if is_clf else min(vals)
            mean_v = float(np.mean(vals))
            print(f"    {method_name:<20} best={best_v:.4f}  mean={mean_v:.4f}")

        # Save
        with open(comp_path, "w") as f:
            json.dump({
                "dataset": ds_name,
                "metric_key": mk,
                "comparison": {
                    m: [{"config": r["config"], mk: float(r.get(mk, 0))} for r in rs]
                    for m, rs in comparison.items()
                },
            }, f, indent=2, default=str)
        print(f"  Saved to {comp_path}")

    print(f"\n{'#'*60}")
    print(f"# All done!")
    print(f"{'#'*60}")
