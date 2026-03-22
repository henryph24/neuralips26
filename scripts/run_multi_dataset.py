"""Run evolutionary search + baselines across multiple forecasting datasets.

Produces the comparison table:
  | Method        | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
  |---------------|-------|-------|-------|-------|
  | Hand-designed | ...   | ...   | ...   | ...   |
  | Random search | ...   | ...   | ...   | ...   |
  | Evo-TEMPLATE  | ...   | ...   | ...   | ...   |

Usage:
    modal run scripts/run_multi_dataset.py
"""

import json
import os
import sys
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import AdapterConfig, generate_configs
from feasibility.data import FORECASTING_DATASETS, serialize_dataset
from feasibility.evolution import run_evolution, random_config

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "momentfm",
        "peft>=0.7.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
    )
    .add_local_dir("template_code", "/root/template_code")
    .add_local_dir("feasibility", "/root/feasibility")
)

app = modal.App("template-multi-dataset", image=image)


@app.function(gpu="A10G", timeout=600)
def evaluate_config(config_dict: dict, data_bytes: bytes, data_meta: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.features import extract_features
    from feasibility.scores import compute_template_scores

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    features = extract_features(adapter_cfg, dataset["samples"], device="cuda", batch_size=64)
    scores = compute_template_scores(
        feature=features["feature"],
        first_feature=features["first_feature"],
        trend_feature=features["trend_feature"],
        device="cuda",
    )
    return {"config": config_dict, "scores": scores, "dataset": data_meta.get("name", "unknown")}


@app.function(gpu="A10G", timeout=900)
def finetune_config(config_dict: dict, data_bytes: bytes, data_meta: dict,
                    n_epochs: int = 10) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.finetune import finetune_forecasting

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    result = finetune_forecasting(adapter_cfg, dataset["samples"], device="cuda", n_epochs=n_epochs)
    return {"config": config_dict, **result, "dataset": data_meta.get("name", "unknown")}


# --- Baselines ---

HAND_DESIGNED_CONFIGS = [
    # Standard LoRA config commonly used in literature
    AdapterConfig(
        adapter_type="lora", lora_rank=8, lora_alpha=16,
        target_modules_key="qv", layer_placement="all",
        config_id="hand_lora_r8_qv_all",
    ),
    # Larger LoRA
    AdapterConfig(
        adapter_type="lora", lora_rank=16, lora_alpha=32,
        target_modules_key="qkvo", layer_placement="all",
        config_id="hand_lora_r16_qkvo_all",
    ),
    # Linear probe (no adapter)
    AdapterConfig(
        adapter_type="linear_probe",
        config_id="linear_probe_baseline",
    ),
]


def run_random_search(data_bytes, data_meta, n_configs=20, seed=42):
    """Random search baseline: sample N configs, pick best by TEMPLATE score."""
    rng = np.random.default_rng(seed)
    seen = set()
    configs = []
    while len(configs) < n_configs:
        cfg = random_config(rng)
        if cfg.config_id not in seen:
            seen.add(cfg.config_id)
            configs.append(cfg)

    results = list(evaluate_config.starmap(
        [(cfg.to_dict(), data_bytes, data_meta) for cfg in configs]
    ))

    # Return best by composite score
    best = max(results, key=lambda r: r["scores"]["composite"])
    return best, results


@app.local_entrypoint()
def main(generations: int = 12, pop_size: int = 20, seed: int = 42,
         n_epochs: int = 10, random_budget: int = 20):

    os.makedirs("results", exist_ok=True)

    all_results = {}

    for ds_name, loader in FORECASTING_DATASETS.items():
        print(f"\n{'#'*60}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*60}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)
        print(f"  Loaded: {ds['samples'].shape}")

        results_for_ds = {}

        # --- 1. Hand-designed baselines ---
        print(f"\n  [1/3] Hand-designed baselines ({len(HAND_DESIGNED_CONFIGS)} configs)...")
        hand_ft = list(finetune_config.starmap(
            [(cfg.to_dict(), data_bytes, meta, n_epochs) for cfg in HAND_DESIGNED_CONFIGS]
        ))
        best_hand = min(hand_ft, key=lambda r: r["mse"])
        results_for_ds["hand_designed"] = {
            "best_mse": best_hand["mse"],
            "best_config": best_hand["config"]["config_id"],
            "all_results": hand_ft,
        }
        print(f"    Best: {best_hand['config']['config_id']} MSE={best_hand['mse']:.4f}")

        # --- 2. Random search ---
        print(f"\n  [2/3] Random search ({random_budget} configs)...")
        best_random_score, random_results = run_random_search(
            data_bytes, meta, n_configs=random_budget, seed=seed
        )
        # Fine-tune top-3 random configs
        random_sorted = sorted(random_results, key=lambda r: r["scores"]["composite"], reverse=True)
        top3_random = [r["config"] for r in random_sorted[:3]]
        random_ft = list(finetune_config.starmap(
            [(cfg, data_bytes, meta, n_epochs) for cfg in top3_random]
        ))
        best_random_ft = min(random_ft, key=lambda r: r["mse"])
        results_for_ds["random_search"] = {
            "best_mse": best_random_ft["mse"],
            "best_config": best_random_ft["config"]["config_id"],
            "n_evaluated": random_budget,
            "all_results": random_ft,
        }
        print(f"    Best: {best_random_ft['config']['config_id']} MSE={best_random_ft['mse']:.4f}")

        # --- 3. Evo-TEMPLATE ---
        print(f"\n  [3/3] Evo-TEMPLATE ({generations} gens × {pop_size} pop)...")

        def modal_evaluate(configs):
            return list(evaluate_config.starmap(
                [(cfg.to_dict(), data_bytes, meta) for cfg in configs]
            ))

        evo_result = run_evolution(
            evaluate_fn=modal_evaluate,
            n_generations=generations,
            pop_size=pop_size,
            seed=seed,
        )

        # Fine-tune top-5 evolved configs
        top5_evo = [r["config"] for r in evo_result["final_population"][:5]]
        evo_ft = list(finetune_config.starmap(
            [(cfg, data_bytes, meta, n_epochs) for cfg in top5_evo]
        ))
        best_evo_ft = min(evo_ft, key=lambda r: r["mse"])
        results_for_ds["evo_template"] = {
            "best_mse": best_evo_ft["mse"],
            "best_config": best_evo_ft["config"]["config_id"],
            "best_fitness": evo_result["best_fitness"],
            "all_results": evo_ft,
        }
        print(f"    Best: {best_evo_ft['config']['config_id']} MSE={best_evo_ft['mse']:.4f}")

        # Save evolution log
        logger = evo_result["logger"]
        log_data = logger.to_dict()
        log_data["params"] = {
            "n_generations": generations, "pop_size": pop_size, "seed": seed,
            "dataset": ds_name,
        }
        with open(f"results/evolution_log_{ds_name}_{seed}.json", "w") as f:
            json.dump(log_data, f, indent=2)

        all_results[ds_name] = results_for_ds

    # --- Print comparison table ---
    print(f"\n\n{'='*70}")
    print(f"COMPARISON TABLE — Forecasting MSE (lower is better)")
    print(f"{'='*70}")

    datasets = list(all_results.keys())
    header = f"{'Method':<20}" + "".join(f"{ds:>10}" for ds in datasets)
    print(header)
    print("-" * len(header))

    for method, label in [("hand_designed", "Hand-designed"),
                          ("random_search", "Random search"),
                          ("evo_template", "Evo-TEMPLATE")]:
        row = f"{label:<20}"
        for ds in datasets:
            mse = all_results[ds][method]["best_mse"]
            row += f"{mse:>10.4f}"
        print(row)

    print(f"{'='*70}")

    # Save full results
    # Strip non-serializable data for JSON
    save_results = {}
    for ds_name, ds_results in all_results.items():
        save_results[ds_name] = {}
        for method, data in ds_results.items():
            save_results[ds_name][method] = {
                "best_mse": data["best_mse"],
                "best_config": data["best_config"],
                "all_results": data["all_results"],
            }
            if "best_fitness" in data:
                save_results[ds_name][method]["best_fitness"] = data["best_fitness"]

    with open("results/multi_dataset_comparison.json", "w") as f:
        json.dump(save_results, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\nFull results saved to results/multi_dataset_comparison.json")
