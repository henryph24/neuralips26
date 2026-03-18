"""Modal.com GPU entry point for parallel config evaluation."""

import json
import modal

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

app = modal.App("template-feasibility", image=image)


@app.function(gpu="A10G", timeout=600)
def evaluate_config(config_dict: dict, data_bytes: bytes, data_meta: dict) -> dict:
    """Evaluate a single adapter config on GPU.

    Args:
        config_dict: serialized AdapterConfig
        data_bytes: serialized numpy dataset
        data_meta: dataset metadata (name, task)

    Returns:
        dict with config, scores, and metadata
    """
    import sys
    sys.path.insert(0, "/root")

    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.features import extract_features
    from feasibility.scores import compute_template_scores

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    features = extract_features(
        adapter_cfg,
        dataset["samples"],
        device="cuda",
        batch_size=64,
    )

    scores = compute_template_scores(
        feature=features["feature"],
        first_feature=features["first_feature"],
        trend_feature=features["trend_feature"],
        device="cuda",
    )

    return {
        "config": config_dict,
        "scores": scores,
        "dataset": data_meta.get("name", "unknown"),
    }


@app.function(gpu="A10G", timeout=900)
def finetune_config(config_dict: dict, data_bytes: bytes, data_meta: dict) -> dict:
    """Fine-tune a single adapter config on GPU.

    Returns dict with config and downstream metric.
    """
    import sys
    sys.path.insert(0, "/root")

    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.finetune import finetune_forecasting, finetune_classification

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    if data_meta.get("task") == "forecasting":
        result = finetune_forecasting(
            adapter_cfg, dataset["samples"], device="cuda"
        )
    else:
        result = finetune_classification(
            adapter_cfg, dataset["samples"], dataset["labels"], device="cuda"
        )

    return {
        "config": config_dict,
        **result,
        "dataset": data_meta.get("name", "unknown"),
    }


@app.local_entrypoint()
def main(mode: str = "sweep", sweep_results: str = "results/sweep_results.json"):
    """Run sweep or fine-tuning.

    Usage:
        modal run feasibility/modal_app.py                         # sweep
        modal run feasibility/modal_app.py --mode finetune         # finetune
    """
    import os
    import sys
    sys.path.insert(0, ".")

    from feasibility.data import load_etth1, load_ethanol_concentration, serialize_dataset

    if mode == "sweep":
        from feasibility.config import generate_configs
        configs = generate_configs()
        print(f"Generated {len(configs)} adapter configs")

        datasets = []
        print("Loading ETTh1...")
        datasets.append(load_etth1())
        print("Loading EthanolConcentration...")
        datasets.append(load_ethanol_concentration())

        all_results = []
        for ds in datasets:
            data_bytes, meta = serialize_dataset(ds)
            print(f"\nEvaluating {len(configs)} configs on {meta.get('name', ds.get('name'))}...")
            results = list(evaluate_config.starmap(
                [(cfg.to_dict(), data_bytes, meta) for cfg in configs]
            ))
            all_results.extend(results)
            print(f"  Completed {len(results)} evaluations")

        os.makedirs("results", exist_ok=True)
        with open("results/sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} results to results/sweep_results.json")

    elif mode == "finetune":
        from scripts.run_finetune import select_validation_configs

        with open(sweep_results) as f:
            all_sweep = json.load(f)

        datasets_loaders = {
            "ETTh1": load_etth1,
            "EthanolConcentration": load_ethanol_concentration,
        }

        # Group sweep results by dataset
        by_dataset = {}
        for r in all_sweep:
            by_dataset.setdefault(r["dataset"], []).append(r)

        all_ft_results = []
        for ds_name, ds_results in by_dataset.items():
            selected = select_validation_configs(ds_results)
            print(f"\nFine-tuning {len(selected)} configs on {ds_name}")
            for s in selected:
                print(f"  {s['config']['config_id']}: composite={s['scores']['composite']:.4f}")

            loader = datasets_loaders.get(ds_name)
            if loader is None:
                print(f"  Skipping unknown dataset: {ds_name}")
                continue

            ds = loader()
            data_bytes, meta = serialize_dataset(ds)

            results = list(finetune_config.starmap(
                [(s["config"], data_bytes, meta) for s in selected]
            ))
            all_ft_results.extend(results)
            print(f"  Completed {len(results)} fine-tuning runs")

        os.makedirs("results", exist_ok=True)
        with open("results/finetune_results.json", "w") as f:
            json.dump(all_ft_results, f, indent=2)
        print(f"\nSaved {len(all_ft_results)} finetune results to results/finetune_results.json")
