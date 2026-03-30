"""Multi-horizon evaluation of Code Evolution adapters.

Takes winning adapters discovered at H=96 and evaluates them at
H=192, 336, 720. Also evaluates baselines at each horizon.

This avoids re-evolving per horizon (~4x cheaper) while providing
the standard NeurIPS multi-horizon comparison table.

Usage:
    modal run scripts/run_multi_horizon.py --dataset etth1 --seed 42
    modal run scripts/run_multi_horizon.py --dataset etth1 --seed 42 --horizons 96,192  # subset
    modal run scripts/run_multi_horizon.py --dataset all --seed 42
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
from feasibility.data import (
    load_dataset_multihor,
    serialize_dataset,
    FORECAST_HORIZONS,
)
from feasibility.code_evolution import validate_adapter_code
from feasibility.evolution import random_config


DATASETS = ["ETTh1", "ETTm1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic"]


@app.function(
    gpu="A10G",
    timeout=900,
    scaledown_window=2,
    max_containers=20,
)
def evaluate_adapter_code_remote(
    code_str: str,
    data_bytes: bytes,
    data_meta: dict,
    n_epochs: int = 15,
    forecast_horizon: int = 96,
) -> dict:
    """Evaluate adapter code at a specific forecast horizon."""
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
            forecast_horizon=forecast_horizon,
            batch_size=64,
        )
        return result

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def load_code_evo_adapters(ds_name, seed, top_n=5):
    """Load top validated adapters from Code Evolution results."""
    val_path = f"results/code_evolution/validated_{ds_name}_{seed}.json"
    if not os.path.exists(val_path):
        print(f"  WARNING: No validated results at {val_path}")
        return []
    with open(val_path) as f:
        validated = json.load(f)
    # Take top_n by mse_15ep
    valid = [v for v in validated if v.get("error") is None]
    valid.sort(key=lambda v: v.get("mse_15ep", float('inf')))
    return valid[:top_n]


def parse_datasets(dataset_arg):
    if dataset_arg == "all":
        return DATASETS
    return [d for d in DATASETS if d.lower() == dataset_arg.lower()]


@app.local_entrypoint()
def multi_horizon(
    seed: int = 42,
    dataset: str = "both",
    horizons: str = "96,192,336,720",
    n_epochs: int = 15,
    top_n: int = 5,
):
    os.makedirs("results/code_evolution", exist_ok=True)

    horizon_list = [int(h.strip()) for h in horizons.split(",")]

    if dataset == "both":
        ds_list = ["ETTh1", "ETTm1"]
    else:
        ds_list = parse_datasets(dataset)

    if not ds_list:
        print(f"ERROR: Unknown dataset '{dataset}'")
        return

    for ds_name in ds_list:
        print(f"\n{'#'*60}")
        print(f"# {ds_name} — Multi-Horizon Evaluation")
        print(f"# Horizons: {horizon_list}, Seed: {seed}, Epochs: {n_epochs}")
        print(f"{'#'*60}")

        # Load winning adapters from H=96 Code Evolution
        adapters = load_code_evo_adapters(ds_name, seed, top_n)
        if not adapters:
            print(f"  Skipping {ds_name} — no Code Evolution results found")
            continue
        print(f"  Loaded {len(adapters)} Code Evolution adapters")

        results_by_horizon = {}

        for H in horizon_list:
            print(f"\n  --- Horizon H={H} ---")

            # Load dataset with decoupled windows for this horizon
            try:
                ds = load_dataset_multihor(ds_name, input_len=512, forecast_horizon=H)
            except (FileNotFoundError, ValueError) as e:
                print(f"  ERROR loading {ds_name} for H={H}: {e}")
                continue
            data_bytes, meta = serialize_dataset(ds)
            print(f"  Data: {ds['samples'].shape[0]} samples, shape={ds['samples'].shape}")

            # Validate adapters for this horizon (output_dim changes with H)
            valid_codes = []
            for adapter in adapters:
                code = adapter["code"]
                val = validate_adapter_code(code, output_dim=H)
                if val["valid"]:
                    valid_codes.append(code)
                else:
                    print(f"    Adapter invalid for H={H}: {val['error']}")

            if not valid_codes:
                print(f"  No valid adapters for H={H}")
                results_by_horizon[str(H)] = {"code_evo": []}
                continue

            print(f"  Evaluating {len(valid_codes)} adapters at H={H}, {n_epochs} epochs (parallel)...")
            eval_results = list(evaluate_adapter_code_remote.starmap(
                [(code, data_bytes, meta, n_epochs, H) for code in valid_codes]
            ))

            code_evo_results = []
            for code, result in zip(valid_codes, eval_results):
                mse = result.get("mse", float('inf'))
                error = result.get("error")
                code_evo_results.append({
                    "code": code,
                    "mse": mse,
                    "param_count": result.get("param_count", 0),
                    "error": error,
                })
                mse_str = f"{mse:.4f}" if error is None else f"ERR: {error}"
                print(f"    MSE={mse_str}")

            valid_mses = [r["mse"] for r in code_evo_results if r.get("error") is None]
            if valid_mses:
                print(f"  Code-Evo H={H}: best={min(valid_mses):.4f} mean={np.mean(valid_mses):.4f}")

            results_by_horizon[str(H)] = {"code_evo": code_evo_results}

        # Print summary table
        print(f"\n{'='*60}")
        print(f"MULTI-HORIZON SUMMARY — {ds_name} (seed={seed})")
        print(f"{'='*60}")
        print(f"  {'Horizon':<10} {'Best MSE':>10} {'Mean MSE':>10} {'Valid':>8}")
        for H in horizon_list:
            h_key = str(H)
            if h_key not in results_by_horizon:
                print(f"  H={H:<7} {'—':>10} {'—':>10} {'—':>8}")
                continue
            evo = results_by_horizon[h_key].get("code_evo", [])
            valid_mses = [r["mse"] for r in evo if r.get("error") is None]
            if valid_mses:
                print(f"  H={H:<7} {min(valid_mses):>10.4f} {np.mean(valid_mses):>10.4f} {len(valid_mses):>5}/{len(evo)}")
            else:
                print(f"  H={H:<7} {'—':>10} {'—':>10} {'0':>5}/{len(evo)}")

        # Save results
        save_data = {
            "dataset": ds_name,
            "seed": seed,
            "n_epochs": n_epochs,
            "horizons": results_by_horizon,
        }
        save_path = f"results/code_evolution/multi_horizon_{ds_name}_{seed}.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {save_path}")

    print(f"\n{'='*60}")
    print("Done. Multi-horizon results in results/code_evolution/")
