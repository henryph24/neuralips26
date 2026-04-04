"""Fine-tune validation subset: select ~10 configs spanning the score range.

Usage:
    Local:  python scripts/run_finetune.py --local --results results/sweep_results.json
    Modal:  python scripts/run_finetune.py --modal --results results/sweep_results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import AdapterConfig
from feasibility.data import load_etth1, load_ethanol_concentration, serialize_dataset


def select_validation_configs(results: list, n_total: int = 10) -> list:
    """Select configs spanning the score range for fine-tuning validation.

    Picks: 2 lowest, 3 mid, 2 highest, 1 bottleneck, 1 baseline, 1 random.
    """
    # Separate by type
    lora = [r for r in results if r["config"]["adapter_type"] == "lora"]
    bottleneck = [r for r in results if r["config"]["adapter_type"] == "bottleneck"]
    baseline = [r for r in results if r["config"]["adapter_type"] == "linear_probe"]

    # Sort LoRA by composite score
    lora_sorted = sorted(lora, key=lambda r: r["scores"]["composite"])

    selected = []

    # 2 lowest
    selected.extend(lora_sorted[:2])

    # 2 highest
    selected.extend(lora_sorted[-2:])

    # 3 mid (evenly spaced from middle)
    mid_start = len(lora_sorted) // 4
    mid_end = 3 * len(lora_sorted) // 4
    mid_indices = np.linspace(mid_start, mid_end, 3, dtype=int)
    for idx in mid_indices:
        if idx < len(lora_sorted) and lora_sorted[idx] not in selected:
            selected.append(lora_sorted[idx])

    # 1 bottleneck (middle score)
    if bottleneck:
        bn_sorted = sorted(bottleneck, key=lambda r: r["scores"]["composite"])
        selected.append(bn_sorted[len(bn_sorted) // 2])

    # 1 baseline
    if baseline:
        selected.append(baseline[0])

    # 1 random LoRA (if not already selected)
    remaining = [r for r in lora if r not in selected]
    if remaining:
        rng = np.random.default_rng(42)
        selected.append(remaining[rng.integers(len(remaining))])

    return selected[:n_total]


def run_local_finetune(results_path: str, dataset_name: str = "etth1"):
    """Run fine-tuning locally on CPU for validation configs."""
    from feasibility.finetune import finetune_forecasting, finetune_classification

    with open(results_path) as f:
        all_results = json.load(f)

    # Filter to requested dataset
    ds_results = [r for r in all_results if r.get("dataset", "").lower().startswith(dataset_name[:4])]
    if not ds_results:
        ds_results = all_results  # fallback

    selected = select_validation_configs(ds_results)
    print(f"Selected {len(selected)} configs for fine-tuning validation")

    # Load dataset
    if dataset_name == "etth1":
        ds = load_etth1()
        samples = ds["samples"][:200]  # subset for local
    else:
        ds = load_ethanol_concentration()
        samples = ds["samples"][:200]

    ft_results = []
    for i, r in enumerate(selected):
        cfg = AdapterConfig.from_dict(r["config"])
        print(f"  [{i+1}/{len(selected)}] {cfg.config_id}...", end=" ", flush=True)

        try:
            if ds.get("task") == "forecasting":
                metrics = finetune_forecasting(cfg, samples, device="cpu", n_epochs=3)
            else:
                labels = ds.get("labels", np.zeros(len(samples), dtype=int))[:200]
                metrics = finetune_classification(cfg, samples, labels, device="cpu", n_epochs=3)

            ft_results.append({
                "config": r["config"],
                **metrics,
                "dataset": ds["name"],
            })
            print(metrics)
        except Exception as e:
            print(f"FAILED: {e}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/finetune_results_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(ft_results, f, indent=2)
    print(f"\nSaved {len(ft_results)} fine-tune results to {out_path}")
    return ft_results


def run_modal_finetune(results_path: str):
    """Run fine-tuning on Modal GPUs."""
    import modal

    with open(results_path) as f:
        all_results = json.load(f)

    # Group by dataset
    by_dataset = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    datasets_loaders = {
        "ETTh1": load_etth1,
        "EthanolConcentration": load_ethanol_concentration,
    }

    from feasibility.modal_app import finetune_config

    all_ft_results = []
    for ds_name, ds_results in by_dataset.items():
        selected = select_validation_configs(ds_results)
        print(f"\nFine-tuning {len(selected)} configs on {ds_name}")

        loader = datasets_loaders.get(ds_name)
        if loader is None:
            print(f"  Skipping unknown dataset: {ds_name}")
            continue

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        with modal.enable_output():
            results = list(finetune_config.starmap(
                [(r["config"], data_bytes, meta) for r in selected]
            ))
        all_ft_results.extend(results)

    os.makedirs("results", exist_ok=True)
    out_path = "results/finetune_results.json"
    with open(out_path, "w") as f:
        json.dump(all_ft_results, f, indent=2)
    print(f"\nSaved {len(all_ft_results)} fine-tune results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to sweep results JSON")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--dataset", default="etth1")
    args = parser.parse_args()

    if args.local:
        run_local_finetune(args.results, args.dataset)
    elif args.modal:
        run_modal_finetune(args.results)
    else:
        print("Specify --local or --modal")
