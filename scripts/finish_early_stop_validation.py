"""Complete Phase 2 validation for ETTm1 and EthanolConcentration."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
import numpy as np

from feasibility.modal_app import app, finetune_config, compute_statistics_remote
from feasibility.proxy_search import load_calibration_data
from feasibility.proxy_gp import deserialize_tree, evaluate_proxy
from feasibility.data import load_ettm1, load_ethanol_concentration, serialize_dataset
from feasibility.evolution import run_evolution, random_config


@app.local_entrypoint()
def finish_validation(seed: int = 42, validate_top: int = 5,
                      evo_generations: int = 12, evo_pop: int = 20):
    os.makedirs("results/early_stopping", exist_ok=True)

    gp_path = f"results/proxy_search/gp_result_{seed}.json"
    gp_tree = None
    if os.path.exists(gp_path):
        with open(gp_path) as f:
            gp_tree = deserialize_tree(json.load(f)["best_tree"])

    datasets = {
        "ETTm1": (load_ettm1, "mse", False),
        "EthanolConcentration": (load_ethanol_concentration, "accuracy", True),
    }

    for ds_name, (loader, mk, is_clf) in datasets.items():
        comp_path = f"results/early_stopping/comparison_{ds_name}_{seed}.json"
        if os.path.exists(comp_path):
            print(f"Skipping {ds_name}: already completed")
            continue

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
                return [
                    {"config": c.to_dict(),
                     "scores": {"composite": float(ft[metric_key]) if higher else -float(ft[metric_key])}}
                    for c, ft in zip(cfgs, fts)
                ]
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

        # Method 3: Random
        print("  [3/3] Random search...")
        rng = np.random.default_rng(seed)
        rand_cfgs = [random_config(rng) for _ in range(evo_pop)]

        top_n = min(validate_top, 5)
        methods = {"Evo-3epoch": [r["config"] for r in evo_3ep["final_population"][:top_n]]}
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

        print(f"\n  {'='*60}")
        print(f"  COMPARISON - {ds_name} ({mk})")
        print(f"  {'='*60}")
        for method_name, results in comparison.items():
            vals = [float(r.get(mk, float("nan"))) for r in results]
            best_v = max(vals) if is_clf else min(vals)
            mean_v = float(np.mean(vals))
            print(f"    {method_name:<20} best={best_v:.4f}  mean={mean_v:.4f}")

        with open(comp_path, "w") as f:
            json.dump({
                "dataset": ds_name, "metric_key": mk,
                "comparison": {
                    m: [{"config": r["config"], mk: float(r.get(mk, 0))} for r in rs]
                    for m, rs in comparison.items()
                },
            }, f, indent=2, default=str)
        print(f"  Saved to {comp_path}")
