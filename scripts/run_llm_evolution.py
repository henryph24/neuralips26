"""Head-to-head comparison: LLM-guided Evo vs Traditional Evo vs Random.

Reuses cached baselines when available, otherwise runs them fresh.

Usage:
    modal run scripts/run_llm_evolution.py
    modal run scripts/run_llm_evolution.py --n-generations 3  # smoke test
    modal run scripts/run_llm_evolution.py --seed 43          # different seed
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
import numpy as np

from feasibility.modal_app import app, finetune_config
from feasibility.data import load_etth1, load_ettm1, serialize_dataset
from feasibility.llm_operators import run_llm_evolution
from feasibility.evolution import run_evolution, random_config

DATASETS = {
    "ETTh1": load_etth1,
    "ETTm1": load_ettm1,
}


def load_cached_results(ds_name: str, seed: int) -> dict | None:
    """Load cached comparison results from early_stopping experiment."""
    path = f"results/early_stopping/comparison_{ds_name}_{seed}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_3ep_eval(data_bytes, data_meta, metric_key, higher_is_better):
    """Build a 3-epoch evaluate_fn for evolution."""
    def eval_fn(cfgs):
        fts = list(finetune_config.starmap(
            [(c.to_dict(), data_bytes, data_meta, 3) for c in cfgs]
        ))
        results = []
        for c, ft in zip(cfgs, fts):
            val = float(ft[metric_key])
            score = val if higher_is_better else -val
            results.append({"config": c.to_dict(), "scores": {"composite": score}})
        return results
    return eval_fn


@app.local_entrypoint()
def llm_evo(
    seed: int = 42,
    n_generations: int = 12,
    pop_size: int = 20,
    validate_top: int = 5,
    model: str = "gpt-4o-mini",
):
    os.makedirs("results/llm_evolution", exist_ok=True)

    for ds_name, loader in DATASETS.items():
        print(f"\n{'#'*60}")
        print(f"# {ds_name} — LLM-Guided Evo Comparison")
        print(f"# Generations: {n_generations}, Pop: {pop_size}, Seed: {seed}")
        print(f"{'#'*60}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)
        is_clf = ds.get("task") == "classification"
        mk = "accuracy" if is_clf else "mse"
        ep3_eval = make_3ep_eval(data_bytes, meta, mk, is_clf)

        # =====================================================================
        # Method 1: LLM-Guided Evolution (always fresh)
        # =====================================================================
        print(f"\n  [1/3] LLM-Evo...")
        llm_result = run_llm_evolution(
            ep3_eval,
            n_generations=n_generations,
            pop_size=pop_size,
            seed=seed,
            model=model,
        )

        # Save evolution log
        llm_logger = llm_result["logger"]
        log_data = llm_logger.to_dict()
        log_data["params"] = {
            "n_generations": n_generations,
            "pop_size": pop_size,
            "seed": seed,
            "dataset": ds_name,
            "model": model,
        }
        log_data["llm_reasonings"] = llm_result["llm_reasonings"]
        log_path = f"results/llm_evolution/evo_log_{ds_name}_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"  Saved evolution log to {log_path}")

        # =====================================================================
        # Method 2 & 3: Traditional Evo + Random (cached or fresh)
        # =====================================================================
        cached = load_cached_results(ds_name, seed)

        if cached and "Evo-3epoch" in cached.get("comparison", {}):
            print(f"  [2/3] Evo-3epoch: using cached results (seed={seed})")
            trad_ft = cached["comparison"]["Evo-3epoch"]
        else:
            print(f"  [2/3] Evo-3epoch: running fresh...")
            trad_result = run_evolution(
                ep3_eval,
                n_generations=n_generations,
                pop_size=pop_size,
                seed=seed,
            )
            top_n = min(validate_top, 5)
            trad_top_cfgs = [r["config"] for r in trad_result["final_population"][:top_n]]
            trad_ft = list(finetune_config.starmap(
                [(c, data_bytes, meta, 15) for c in trad_top_cfgs]
            ))

        if cached and "Random" in cached.get("comparison", {}):
            print(f"  [3/3] Random: using cached results (seed={seed})")
            rand_ft = cached["comparison"]["Random"]
        else:
            print(f"  [3/3] Random: running fresh...")
            rng = np.random.default_rng(seed)
            top_n = min(validate_top, 5)
            rand_cfgs = [random_config(rng) for _ in range(pop_size)]
            rand_ft = list(finetune_config.starmap(
                [(c.to_dict(), data_bytes, meta, 15) for c in rand_cfgs[:top_n]]
            ))

        # =====================================================================
        # Validate top LLM-Evo configs at 15 epochs
        # =====================================================================
        top_n = min(validate_top, 5)
        llm_top_cfgs = [r["config"] for r in llm_result["final_population"][:top_n]]

        print(f"\n  Fine-tuning top-{top_n} LLM-Evo configs (15 epochs)...")
        llm_ft = list(finetune_config.starmap(
            [(c, data_bytes, meta, 15) for c in llm_top_cfgs]
        ))

        # =====================================================================
        # Print comparison
        # =====================================================================
        comparison = {
            "LLM-Evo": llm_ft,
            "Evo-3epoch": trad_ft,
            "Random": rand_ft,
        }

        print(f"\n  {'='*60}")
        print(f"  COMPARISON — {ds_name} ({mk}, seed={seed})")
        print(f"  {'='*60}")

        for method_name, results in comparison.items():
            vals = [float(r.get(mk, float("nan"))) for r in results]
            valid_vals = [v for v in vals if not np.isnan(v)]
            if not valid_vals:
                print(f"    {method_name:<20} no valid results")
                continue
            best_v = max(valid_vals) if is_clf else min(valid_vals)
            mean_v = float(np.mean(valid_vals))
            print(f"    {method_name:<20} best={best_v:.4f}  mean={mean_v:.4f}")

        # =====================================================================
        # Save full comparison
        # =====================================================================
        save_data = {
            "dataset": ds_name,
            "metric_key": mk,
            "seed": seed,
            "llm_model": model,
            "n_generations": n_generations,
            "pop_size": pop_size,
            "comparison": {
                "LLM-Evo": [
                    {"config": r.get("config", llm_top_cfgs[i]), mk: float(r.get(mk, 0))}
                    for i, r in enumerate(llm_ft)
                ],
                "Evo-3epoch": [
                    {"config": r.get("config", {}), mk: float(r.get(mk, 0))}
                    for r in trad_ft
                ],
                "Random": [
                    {"config": r.get("config", {}), mk: float(r.get(mk, 0))}
                    for r in rand_ft
                ],
            },
        }

        comp_path = f"results/llm_evolution/comparison_{ds_name}_{seed}.json"
        with open(comp_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"  Saved to {comp_path}")

    # =========================================================================
    # Cross-seed summary (if multiple seeds exist)
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# CROSS-SEED SUMMARY")
    print(f"{'#'*60}")

    for ds_name in DATASETS:
        print(f"\n  {ds_name}:")
        seed_results = {}
        for s in range(42, 50):
            path = f"results/llm_evolution/comparison_{ds_name}_{s}.json"
            if os.path.exists(path):
                with open(path) as f:
                    seed_results[s] = json.load(f)

        if len(seed_results) < 2:
            print(f"    Only {len(seed_results)} seed(s) available, need 2+ for comparison")
            continue

        mk = list(seed_results.values())[0]["metric_key"]
        is_clf = mk == "accuracy"

        for method in ["LLM-Evo", "Evo-3epoch", "Random"]:
            bests = []
            for s, data in seed_results.items():
                vals = [float(r.get(mk, float("nan"))) for r in data["comparison"].get(method, [])]
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    bests.append(max(valid) if is_clf else min(valid))
            if bests:
                print(f"    {method:<20} seeds={list(seed_results.keys())}  "
                      f"best_per_seed={[round(b, 4) for b in bests]}  "
                      f"mean={np.mean(bests):.4f}  std={np.std(bests):.4f}")

    print(f"\n{'='*60}")
    print("Done. Results in results/llm_evolution/")
