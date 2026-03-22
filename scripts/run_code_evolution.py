"""Code-level adapter architecture evolution via LLM.

The LLM generates actual PyTorch nn.Module code instead of picking from a
fixed menu of hyperparameters. Evaluated on Modal GPUs.

Usage:
    modal run scripts/run_code_evolution.py
    modal run scripts/run_code_evolution.py --n-generations 2  # smoke test
    modal run scripts/run_code_evolution.py --seed 43
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import modal
import numpy as np

from feasibility.modal_app import app, image, finetune_config
from feasibility.data import load_etth1, load_ettm1, serialize_dataset
from feasibility.code_evolution import (
    run_code_evolution,
    validate_adapter_code,
    SEED_ADAPTERS,
)

DATASETS = {
    "ETTh1": load_etth1,
    "ETTm1": load_ettm1,
}


@app.function(gpu="A10G", timeout=600)
def evaluate_adapter_code_remote(
    code_str: str,
    data_bytes: bytes,
    data_meta: dict,
    n_epochs: int = 3,
) -> dict:
    """Evaluate a single adapter code string on GPU.

    Loads MOMENT, unfreezes last-4 layers, trains adapter from code,
    returns MSE.
    """
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn as nn

    from feasibility.model import (
        load_moment,
        _get_encoder_blocks,
        _disable_gradient_checkpointing,
    )
    from feasibility.data import deserialize_dataset
    from feasibility.code_evolution import train_adapter_from_code

    try:
        dataset = deserialize_dataset(data_bytes, data_meta)
        model = load_moment("cuda")
        _disable_gradient_checkpointing(model)

        # Freeze all, then unfreeze last-4
        for param in model.parameters():
            param.requires_grad = False

        encoder_blocks = _get_encoder_blocks(model)
        n_blocks = len(encoder_blocks)
        unfreeze_from = max(0, n_blocks - 4)
        for i in range(unfreeze_from, n_blocks):
            for param in encoder_blocks[i].parameters():
                param.requires_grad = True

        result = train_adapter_from_code(
            code=code_str,
            model=model,
            encoder_blocks=encoder_blocks,
            samples=dataset["samples"],
            device="cuda",
            n_epochs=n_epochs,
            lr=1e-3,
            forecast_horizon=96,
            batch_size=64,
        )
        return result

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def load_cached_baselines(ds_name: str, seed: int) -> dict | None:
    """Load cached baselines from early_stopping or llm_evolution experiments."""
    for prefix in ["results/early_stopping/comparison", "results/llm_evolution/comparison"]:
        path = f"{prefix}_{ds_name}_{seed}.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


@app.local_entrypoint()
def code_evo(
    seed: int = 42,
    n_generations: int = 12,
    pop_size: int = 20,
    validate_top: int = 5,
    model: str = "gpt-4o-mini",
    dataset: str = "both",
):
    os.makedirs("results/code_evolution", exist_ok=True)

    # Quick local validation of seed adapters
    print("Validating seed adapters locally...")
    for i, code in enumerate(SEED_ADAPTERS):
        val = validate_adapter_code(code)
        status = "OK" if val["valid"] else f"FAIL: {val['error']}"
        print(f"  Seed {i+1}: {status} (params={val['param_count']})")
        if not val["valid"]:
            print(f"    WARNING: Seed adapter {i+1} is invalid!")

    datasets_to_run = {}
    if dataset in ("etth1", "both"):
        datasets_to_run["ETTh1"] = DATASETS["ETTh1"]
    if dataset in ("ettm1", "both"):
        datasets_to_run["ETTm1"] = DATASETS["ETTm1"]

    for ds_name, loader in datasets_to_run.items():
        print(f"\n{'#'*60}")
        print(f"# {ds_name} — Code-Level Architecture Evolution")
        print(f"# Generations: {n_generations}, Pop: {pop_size}, Seed: {seed}")
        print(f"{'#'*60}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        # Build batch evaluate_fn that dispatches to Modal in parallel
        def make_evaluate_fn(d_bytes, d_meta):
            def eval_fn(code_strs):
                results = list(evaluate_adapter_code_remote.starmap(
                    [(code, d_bytes, d_meta, 3) for code in code_strs]
                ))
                return results
            return eval_fn

        evaluate_fn = make_evaluate_fn(data_bytes, meta)

        # Run code evolution
        result = run_code_evolution(
            evaluate_fn=evaluate_fn,
            n_generations=n_generations,
            pop_size=pop_size,
            seed=seed,
            model=model,
        )

        # Save evolution log
        log_data = result["logger"].to_dict()
        log_data["params"] = {
            "n_generations": n_generations,
            "pop_size": pop_size,
            "seed": seed,
            "dataset": ds_name,
            "model": model,
        }
        log_data["all_reasonings"] = result["all_reasonings"]
        log_data["best_code"] = result["best_code"]
        log_data["best_mse"] = result["best_mse"]
        log_data["best_param_count"] = result["best_param_count"]

        log_path = f"results/code_evolution/evo_log_{ds_name}_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"\nSaved evolution log to {log_path}")

        # Validate top-N at 15 epochs
        top_n = min(validate_top, 5)
        valid_pop = [
            ind for ind in result["final_population"]
            if ind["error"] is None and ind["mse"] < float('inf')
        ][:top_n]

        if valid_pop:
            print(f"\nValidating top-{len(valid_pop)} adapters at 15 epochs...")
            validated = []
            for ind in valid_pop:
                val_result = evaluate_adapter_code_remote.remote(
                    ind["code"], data_bytes, meta, n_epochs=15,
                )
                validated.append({
                    "code": ind["code"],
                    "mse_3ep": ind["mse"],
                    "mse_15ep": val_result.get("mse", float('inf')),
                    "param_count": val_result.get("param_count", ind["param_count"]),
                    "error": val_result.get("error"),
                    "reasoning": ind["reasoning"],
                })
                mse_str = f"{val_result.get('mse', 'ERR'):.4f}" if "mse" in val_result else val_result.get("error", "ERR")
                print(f"  mse_3ep={ind['mse']:.4f} -> mse_15ep={mse_str}")

            val_path = f"results/code_evolution/validated_{ds_name}_{seed}.json"
            with open(val_path, "w") as f:
                json.dump(validated, f, indent=2, default=str)

        # Load cached baselines for comparison
        cached = load_cached_baselines(ds_name, seed)

        print(f"\n{'='*60}")
        print(f"COMPARISON — {ds_name} (MSE, seed={seed})")
        print(f"{'='*60}")

        # Code-Evo results
        if valid_pop:
            code_evo_mses = [v["mse_15ep"] for v in validated if v.get("error") is None]
            if code_evo_mses:
                print(f"  {'Code-Evo':<20} best={min(code_evo_mses):.4f}  mean={np.mean(code_evo_mses):.4f}")
            else:
                print(f"  {'Code-Evo':<20} all failed at 15ep validation")
        else:
            print(f"  {'Code-Evo':<20} no valid adapters found")

        # Cached baselines
        if cached:
            mk = cached.get("metric_key", "mse")
            for method_name in ["Evo-3epoch", "Random", "LLM-Evo"]:
                results = cached.get("comparison", {}).get(method_name, [])
                if results:
                    vals = [float(r.get(mk, float("nan"))) for r in results]
                    valid_vals = [v for v in vals if not np.isnan(v)]
                    if valid_vals:
                        best_v = min(valid_vals)
                        mean_v = float(np.mean(valid_vals))
                        print(f"  {method_name:<20} best={best_v:.4f}  mean={mean_v:.4f}")
        else:
            print(f"  (No cached baselines found for comparison)")

        # Save full comparison
        save_data = {
            "dataset": ds_name,
            "metric_key": "mse",
            "seed": seed,
            "model": model,
            "n_generations": n_generations,
            "pop_size": pop_size,
            "code_evolution": {
                "best_code": result["best_code"],
                "best_mse_3ep": result["best_mse"],
                "best_param_count": result["best_param_count"],
                "validated": validated if valid_pop else [],
            },
            "cached_baselines": cached.get("comparison", {}) if cached else {},
        }

        comp_path = f"results/code_evolution/comparison_{ds_name}_{seed}.json"
        with open(comp_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {comp_path}")

    print(f"\n{'='*60}")
    print("Done. Results in results/code_evolution/")
