"""Run evolutionary search + baselines for classification across datasets.

Uses expanded search space: unfreeze strategy, head architecture, pooling.

Usage:
    modal run scripts/run_classification_sweep.py
"""

import json
import os
import sys
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import AdapterConfig, _make_id
from feasibility.data import CLASSIFICATION_DATASETS, serialize_dataset
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
        "aeon",
    )
    .add_local_dir("template_code", "/root/template_code")
    .add_local_dir("feasibility", "/root/feasibility")
)

app = modal.App("template-classification", image=image)


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


@app.function(gpu="A10G", timeout=1200)
def finetune_config(config_dict: dict, data_bytes: bytes, data_meta: dict,
                    n_epochs: int = 15) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.finetune import finetune_classification

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    result = finetune_classification(
        adapter_cfg, dataset["samples"], dataset["labels"],
        device="cuda", n_epochs=n_epochs,
    )
    return {"config": config_dict, **result, "dataset": data_meta.get("name", "unknown")}


# --- Baselines ---

HAND_DESIGNED_CONFIGS = [
    # Standard LoRA, frozen, linear head
    AdapterConfig(
        adapter_type="lora", lora_rank=8, lora_alpha=16,
        target_modules_key="qv", layer_placement="all",
        unfreeze="frozen", head_type="linear", pooling="mean",
    ),
    # LoRA with unfreezing + MLP head
    AdapterConfig(
        adapter_type="lora", lora_rank=16, lora_alpha=32,
        target_modules_key="qkvo", layer_placement="all",
        unfreeze="last2", head_type="mlp1", pooling="mean",
    ),
    # Linear probe, frozen
    AdapterConfig(
        adapter_type="linear_probe",
        unfreeze="frozen", head_type="linear", pooling="mean",
    ),
    # Linear probe with partial unfreezing
    AdapterConfig(
        adapter_type="linear_probe",
        unfreeze="last2", head_type="mlp1", pooling="mean",
    ),
]

# Set config IDs
for cfg in HAND_DESIGNED_CONFIGS:
    cfg.config_id = _make_id(cfg)


@app.local_entrypoint()
def main(generations: int = 15, pop_size: int = 20, seed: int = 42,
         n_epochs: int = 15, random_budget: int = 20):

    os.makedirs("results", exist_ok=True)

    all_results = {}

    for ds_name, loader in CLASSIFICATION_DATASETS.items():
        print(f"\n{'#'*60}")
        print(f"# Classification: {ds_name}")
        print(f"{'#'*60}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)
        n_classes = len(np.unique(ds["labels"]))
        print(f"  Loaded: {ds['samples'].shape}, {n_classes} classes")

        results_for_ds = {}

        # --- 1. Hand-designed baselines ---
        print(f"\n  [1/3] Hand-designed baselines ({len(HAND_DESIGNED_CONFIGS)} configs)...")
        hand_ft = list(finetune_config.starmap(
            [(cfg.to_dict(), data_bytes, meta, n_epochs) for cfg in HAND_DESIGNED_CONFIGS]
        ))
        best_hand = max(hand_ft, key=lambda r: r["accuracy"])
        results_for_ds["hand_designed"] = {
            "best_accuracy": best_hand["accuracy"],
            "best_config": best_hand["config"]["config_id"],
            "all_results": [{
                "config_id": r["config"]["config_id"],
                "accuracy": r["accuracy"],
            } for r in hand_ft],
        }
        for r in hand_ft:
            print(f"    {r['config']['config_id']}: acc={r['accuracy']:.4f}")
        print(f"    ** Best: {best_hand['config']['config_id']} acc={best_hand['accuracy']:.4f}")

        # --- 2. Random search ---
        print(f"\n  [2/3] Random search ({random_budget} configs)...")
        rng = np.random.default_rng(seed)
        seen = set()
        rand_configs = []
        while len(rand_configs) < random_budget:
            cfg = random_config(rng)
            if cfg.config_id not in seen:
                seen.add(cfg.config_id)
                rand_configs.append(cfg)

        # Evaluate TEMPLATE scores
        rand_scores = list(evaluate_config.starmap(
            [(cfg.to_dict(), data_bytes, meta) for cfg in rand_configs]
        ))
        # Fine-tune top-5 by TEMPLATE score
        rand_sorted = sorted(rand_scores, key=lambda r: r["scores"]["composite"], reverse=True)
        top5_rand = [r["config"] for r in rand_sorted[:5]]
        rand_ft = list(finetune_config.starmap(
            [(cfg, data_bytes, meta, n_epochs) for cfg in top5_rand]
        ))
        best_rand = max(rand_ft, key=lambda r: r["accuracy"])
        results_for_ds["random_search"] = {
            "best_accuracy": best_rand["accuracy"],
            "best_config": best_rand["config"]["config_id"],
            "n_evaluated": random_budget,
            "all_results": [{
                "config_id": r["config"]["config_id"],
                "accuracy": r["accuracy"],
            } for r in rand_ft],
        }
        print(f"    Best: {best_rand['config']['config_id']} acc={best_rand['accuracy']:.4f}")

        # --- 3. Evo-TEMPLATE ---
        print(f"\n  [3/3] Evo-TEMPLATE ({generations} gens x {pop_size} pop)...")

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
        best_evo = max(evo_ft, key=lambda r: r["accuracy"])
        results_for_ds["evo_template"] = {
            "best_accuracy": best_evo["accuracy"],
            "best_config": best_evo["config"]["config_id"],
            "best_fitness": float(evo_result["best_fitness"]),
            "all_results": [{
                "config_id": r["config"]["config_id"],
                "accuracy": r["accuracy"],
            } for r in evo_ft],
        }
        print(f"    Best: {best_evo['config']['config_id']} acc={best_evo['accuracy']:.4f}")

        # Save evolution log
        logger = evo_result["logger"]
        log_data = logger.to_dict()
        log_data["params"] = {
            "n_generations": generations, "pop_size": pop_size, "seed": seed,
            "dataset": ds_name, "task": "classification",
        }
        with open(f"results/cls_evolution_log_{ds_name}_{seed}.json", "w") as f:
            json.dump(log_data, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)

        all_results[ds_name] = results_for_ds

    # --- Print comparison table ---
    print(f"\n\n{'='*70}")
    print(f"COMPARISON TABLE — Classification Accuracy (higher is better)")
    print(f"{'='*70}")

    datasets = list(all_results.keys())
    header = f"{'Method':<20}" + "".join(f"{ds[:12]:>14}" for ds in datasets) + f"{'Average':>10}"
    print(header)
    print("-" * len(header))

    for method, label in [("hand_designed", "Hand-designed"),
                          ("random_search", "Random search"),
                          ("evo_template", "Evo-TEMPLATE")]:
        row = f"{label:<20}"
        accs = []
        for ds in datasets:
            acc = all_results[ds][method]["best_accuracy"]
            accs.append(acc)
            row += f"{acc:>14.4f}"
        row += f"{np.mean(accs):>10.4f}"
        print(row)

    print(f"{'='*70}")

    # Save
    with open("results/classification_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nSaved to results/classification_comparison.json")
