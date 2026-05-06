"""Test 3-epoch fine-tuning as a proxy for 15-epoch performance.

Hypothesis: 3-epoch validation loss/accuracy predicts 15-epoch performance
with τ > 0.8, making it a cheap but effective proxy for adapter selection.

Usage:
    modal run scripts/test_early_stopping_proxy.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
import numpy as np
from scipy.stats import kendalltau, spearmanr

from feasibility.modal_app import app, finetune_config, compute_statistics_remote
from feasibility.proxy_search import load_calibration_data, select_calibration_configs
from feasibility.proxy_gp import deserialize_tree, evaluate_proxy
from feasibility.data import (
    load_etth1, load_ettm1, load_ethanol_concentration, serialize_dataset,
)
from feasibility.evolution import run_evolution, random_config


DATASETS = {
    "ETTh1": load_etth1,
    "ETTm1": load_ettm1,
    "EthanolConcentration": load_ethanol_concentration,
}

EARLY_EPOCHS = [1, 3, 5]


@app.local_entrypoint()
def early_stop(seed: int = 42, validate_top: int = 5, evo_generations: int = 12, evo_pop: int = 20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results/early_stopping", exist_ok=True)

    # =========================================================================
    # Phase 1: Collect early-stopping data for calibration configs
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# Phase 1: Early-Stopping Proxy — {EARLY_EPOCHS} epochs")
    print(f"{'#'*60}")

    all_results = {}

    for ds_name, loader in DATASETS.items():
        cal_path = f"results/calibration/{ds_name}_{seed}.json"
        if not os.path.exists(cal_path):
            print(f"Skipping {ds_name}: no calibration data")
            continue

        cal = load_calibration_data(cal_path)
        configs = cal.configs
        perfs_15 = cal.performances
        is_clf = cal.higher_is_better
        mk = cal.metric_key

        print(f"\n--- {ds_name} ({mk}) ---")
        print(f"  {len(configs)} configs, 15-epoch {mk}: "
              f"min={min(perfs_15):.4f} max={max(perfs_15):.4f}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        ds_results = {"15_epoch": perfs_15}

        for n_ep in EARLY_EPOCHS:
            print(f"  Fine-tuning {len(configs)} configs for {n_ep} epochs...")
            ft_results = list(finetune_config.starmap(
                [(cfg.to_dict(), data_bytes, meta, n_ep) for cfg in configs]
            ))
            perfs_early = [r[mk] for r in ft_results]
            ds_results[f"{n_ep}_epoch"] = perfs_early

            # Compute correlation with 15-epoch
            if is_clf:
                tau, p = kendalltau(perfs_early, perfs_15)
            else:
                # For MSE, both are "lower is better" — direct correlation
                tau, p = kendalltau(perfs_early, perfs_15)
            rho, _ = spearmanr(perfs_early, perfs_15)

            print(f"    {n_ep}-epoch vs 15-epoch: τ={tau:.3f} (p={p:.4f}), ρ={rho:.3f}")

        all_results[ds_name] = {
            "metric_key": mk,
            "higher_is_better": is_clf,
            "epochs_data": {k: [float(v) for v in vs] for k, vs in ds_results.items()},
            "configs": [c.to_dict() for c in configs],
        }

    # Save raw data
    with open(f"results/early_stopping/epoch_correlations_{seed}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # =========================================================================
    # Phase 1b: Print summary + plot
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"EARLY-STOPPING CORRELATION SUMMARY")
    print(f"{'='*70}")

    header = f"{'Dataset':<25} {'Epochs':>8} {'Kendall τ':>12} {'Spearman ρ':>12}"
    print(header)
    print("-" * len(header))

    # Also load GP proxy for comparison
    gp_path = f"results/proxy_search/gp_result_{seed}.json"
    gp_tree = None
    if os.path.exists(gp_path):
        with open(gp_path) as f:
            gp_data = json.load(f)
        gp_tree = deserialize_tree(gp_data["best_tree"])

    fig, axes = plt.subplots(len(all_results), len(EARLY_EPOCHS),
                              figsize=(5 * len(EARLY_EPOCHS), 5 * len(all_results)))
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)

    best_proxy_epoch = {}

    for row, (ds_name, ds_data) in enumerate(all_results.items()):
        mk = ds_data["metric_key"]
        is_clf = ds_data["higher_is_better"]
        perfs_15 = ds_data["epochs_data"]["15_epoch"]
        best_tau = -1
        best_ep = None

        for col, n_ep in enumerate(EARLY_EPOCHS):
            perfs_early = ds_data["epochs_data"][f"{n_ep}_epoch"]
            tau, p = kendalltau(perfs_early, perfs_15)
            rho, _ = spearmanr(perfs_early, perfs_15)
            print(f"{ds_name:<25} {n_ep:>8} {tau:>12.3f} {rho:>12.3f}")

            if tau > best_tau:
                best_tau = tau
                best_ep = n_ep

            ax = axes[row, col]
            ax.scatter(perfs_early, perfs_15, alpha=0.7, s=40, edgecolors="k", linewidth=0.5)
            ax.set_xlabel(f"{n_ep}-epoch {mk}")
            ax.set_ylabel(f"15-epoch {mk}")
            ax.set_title(f"{ds_name}: {n_ep}ep vs 15ep\nτ={tau:.3f}, ρ={rho:.3f}")

            # 45-degree reference line
            lims = [min(min(perfs_early), min(perfs_15)),
                    max(max(perfs_early), max(perfs_15))]
            ax.plot(lims, lims, "k--", alpha=0.3)

        # Compare with GP proxy
        if gp_tree and ds_name in all_results:
            cal = load_calibration_data(f"results/calibration/{ds_name}_{seed}.json")
            proxy_scores = [evaluate_proxy(gp_tree, s) for s in cal.statistics]
            sign = 1.0 if is_clf else -1.0
            gp_tau, _ = kendalltau([sign * s for s in proxy_scores], perfs_15)
            print(f"{ds_name:<25} {'GP-proxy':>8} {gp_tau:>12.3f} {'':>12}")

        best_proxy_epoch[ds_name] = best_ep

    plt.tight_layout()
    plt.savefig("results/early_stopping/epoch_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved scatter plots to results/early_stopping/epoch_scatter.png")

    # =========================================================================
    # Phase 2: Use best early-stopping proxy in evolution (3-epoch)
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Evo-3epoch vs Evo-TEMPLATE vs Random")
    print(f"{'#'*60}")

    for ds_name, loader in DATASETS.items():
        if ds_name not in all_results:
            continue

        ds_data = all_results[ds_name]
        mk = ds_data["metric_key"]
        is_clf = ds_data["higher_is_better"]

        print(f"\n--- {ds_name} ---")
        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        # Method 1: Evo-3epoch (3-epoch finetune as fitness)
        print("  [1/3] Evo-3epoch...")
        def make_3ep_eval(d_bytes, d_meta, metric_key, higher):
            def eval_fn(cfgs):
                fts = list(finetune_config.starmap(
                    [(c.to_dict(), d_bytes, d_meta, 3) for c in cfgs]
                ))
                results = []
                for c, ft in zip(cfgs, fts):
                    val = float(ft[metric_key])
                    # Evolution maximizes composite — for MSE, negate
                    score = val if higher else -val
                    results.append({"config": c.to_dict(), "scores": {"composite": score}})
                return results
            return eval_fn

        ep3_eval = make_3ep_eval(data_bytes, meta, mk, is_clf)
        evo_3ep = run_evolution(ep3_eval, n_generations=evo_generations, pop_size=evo_pop, seed=seed)

        # Method 2: Evo-GP-Proxy
        print("  [2/3] Evo-GP-Proxy...")
        if gp_tree:
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
            evo_gp = None

        # Method 3: Random search
        print("  [3/3] Random search...")
        rng = np.random.default_rng(seed)
        rand_cfgs = [random_config(rng) for _ in range(evo_pop)]
        # No proxy needed — just fine-tune directly

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
        comp_path = f"results/early_stopping/comparison_{ds_name}_{seed}.json"
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
