"""Run evolutionary search for adapter configurations.

Usage:
    # Full run on Modal GPUs
    python scripts/run_evolution.py --dataset etth1 --generations 12 --pop-size 20

    # Smoke test locally
    python scripts/run_evolution.py --dataset etth1 --local --generations 3 --pop-size 5

    # Both datasets
    python scripts/run_evolution.py --dataset both --generations 12 --pop-size 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import AdapterConfig
from feasibility.data import (
    load_etth1,
    load_ethanol_concentration,
    serialize_dataset,
)
from feasibility.evolution import run_evolution


def make_local_evaluator(dataset: dict, max_samples: int = 200):
    """Create a local CPU evaluator for smoke testing."""
    from feasibility.features import extract_features
    from feasibility.scores import compute_template_scores

    samples = dataset["samples"][:max_samples]

    def evaluate(configs: List[AdapterConfig]) -> List[dict]:
        results = []
        for cfg in configs:
            features = extract_features(cfg, samples, device="cpu", batch_size=32)
            scores = compute_template_scores(
                feature=features["feature"],
                first_feature=features["first_feature"],
                trend_feature=features["trend_feature"],
                device="cpu",
            )
            results.append({
                "config": cfg.to_dict(),
                "scores": scores,
                "dataset": dataset.get("name", "unknown"),
            })
        return results

    return evaluate


def make_modal_evaluator(data_bytes: bytes, data_meta: dict, app_context):
    """Create a Modal GPU evaluator that runs configs in parallel."""
    from feasibility.modal_app import evaluate_config

    def evaluate(configs: List[AdapterConfig]) -> List[dict]:
        results = list(evaluate_config.starmap(
            [(cfg.to_dict(), data_bytes, data_meta) for cfg in configs]
        ))
        return results

    return evaluate


def validate_top_configs(
    configs: List[dict],
    data_bytes: bytes,
    data_meta: dict,
    n_epochs: int = 10,
    local: bool = False,
    dataset: dict = None,
) -> List[dict]:
    """Fine-tune top configs to validate evolutionary search results."""
    if local:
        from feasibility.finetune import finetune_forecasting, finetune_classification
        results = []
        for cfg_dict in configs:
            adapter_cfg = AdapterConfig.from_dict(cfg_dict)
            if data_meta.get("task") == "forecasting":
                samples = dataset["samples"][:200]
                metrics = finetune_forecasting(
                    adapter_cfg, samples, device="cpu", n_epochs=3
                )
            else:
                samples = dataset["samples"][:200]
                labels = dataset["labels"][:200]
                metrics = finetune_classification(
                    adapter_cfg, samples, labels, device="cpu", n_epochs=3
                )
            results.append({
                "config": cfg_dict,
                **metrics,
                "dataset": data_meta.get("name", "unknown"),
            })
        return results
    else:
        from feasibility.modal_app import finetune_config
        results = list(finetune_config.starmap(
            [(cfg_dict, data_bytes, data_meta, n_epochs) for cfg_dict in configs]
        ))
        return results


def run_for_dataset(dataset_name: str, args):
    """Run evolution for a single dataset."""
    print(f"\n{'#'*60}")
    print(f"# Evolutionary Search — {dataset_name}")
    print(f"# Generations: {args.generations}, Pop: {args.pop_size}, Seed: {args.seed}")
    print(f"{'#'*60}")

    if dataset_name == "etth1":
        ds = load_etth1()
    else:
        ds = load_ethanol_concentration()

    if args.local:
        evaluate_fn = make_local_evaluator(ds)
    else:
        data_bytes, data_meta = serialize_dataset(ds)
        evaluate_fn = make_modal_evaluator(data_bytes, data_meta)

    result = run_evolution(
        evaluate_fn=evaluate_fn,
        n_generations=args.generations,
        pop_size=args.pop_size,
        elite_count=args.elite_count,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
    )

    logger = result["logger"]

    # Save evolution log
    os.makedirs("results", exist_ok=True)
    log_path = f"results/evolution_log_{dataset_name}_{args.seed}.json"
    log_data = logger.to_dict()
    log_data["params"] = {
        "n_generations": args.generations,
        "pop_size": args.pop_size,
        "elite_count": args.elite_count,
        "mutation_rate": args.mutation_rate,
        "crossover_rate": args.crossover_rate,
        "seed": args.seed,
        "dataset": dataset_name,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nEvolution log saved to {log_path}")

    # Validate top configs
    top_n = min(args.validate_top, len(result["final_population"]))
    top_configs = [r["config"] for r in result["final_population"][:top_n]]

    print(f"\nValidating top-{top_n} configs via fine-tuning...")
    if args.local:
        ft_results = validate_top_configs(
            top_configs, None, {"name": ds["name"], "task": ds.get("task", "forecasting")},
            local=True, dataset=ds,
        )
    else:
        ft_results = validate_top_configs(
            top_configs, data_bytes, data_meta,
            n_epochs=10,
        )

    ft_path = f"results/evolution_finetune_{dataset_name}_{args.seed}.json"
    with open(ft_path, "w") as f:
        json.dump(ft_results, f, indent=2)
    print(f"Validation results saved to {ft_path}")

    # Summary
    metric_key = "mse" if ds.get("task") == "forecasting" else "accuracy"
    print(f"\n{'='*60}")
    print(f"EVOLUTION SUMMARY — {dataset_name}")
    print(f"{'='*60}")
    print(f"Best TEMPLATE fitness: {result['best_fitness']:.4f}")
    print(f"Best config: {result['best_config']['config_id']}")
    print(f"\nTop-{top_n} fine-tuning validation:")
    for r in ft_results:
        cfg_id = r["config"]["config_id"]
        metric = r.get(metric_key, "N/A")
        print(f"  {cfg_id}: {metric_key}={metric}")
    print(f"{'='*60}")

    return result, ft_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etth1", choices=["etth1", "ethanol", "both"])
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--crossover-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-top", type=int, default=5)
    parser.add_argument("--local", action="store_true", help="Run on local CPU (smoke test)")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("etth1", "both"):
        datasets.append("etth1")
    if args.dataset in ("ethanol", "both"):
        datasets.append("ethanol")

    for ds_name in datasets:
        run_for_dataset(ds_name, args)


if __name__ == "__main__":
    main()
