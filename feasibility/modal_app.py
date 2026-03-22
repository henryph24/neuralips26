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
def finetune_config(config_dict: dict, data_bytes: bytes, data_meta: dict,
                    n_epochs: int = 10) -> dict:
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
            adapter_cfg, dataset["samples"], device="cuda", n_epochs=n_epochs
        )
    else:
        result = finetune_classification(
            adapter_cfg, dataset["samples"], dataset["labels"],
            device="cuda", n_epochs=n_epochs
        )

    return {
        "config": config_dict,
        **result,
        "dataset": data_meta.get("name", "unknown"),
    }


@app.function(gpu="A10G", timeout=600)
def compute_statistics_remote(config_dict: dict, data_bytes: bytes, data_meta: dict) -> dict:
    """Compute all statistics for a single adapter config on GPU.

    Returns dict with config and statistics.
    """
    import sys
    sys.path.insert(0, "/root")

    from feasibility.config import AdapterConfig
    from feasibility.data import deserialize_dataset
    from feasibility.statistics import compute_all_statistics

    adapter_cfg = AdapterConfig.from_dict(config_dict)
    dataset = deserialize_dataset(data_bytes, data_meta)

    labels = dataset.get("labels")
    stats = compute_all_statistics(
        adapter_cfg,
        dataset["samples"],
        labels=labels,
        device="cuda",
        batch_size=64,
    )

    return {
        "config": config_dict,
        "statistics": stats,
        "dataset": data_meta.get("name", "unknown"),
    }


@app.local_entrypoint()
def main(mode: str = "sweep", sweep_results: str = "results/sweep_results.json",
         n_epochs: int = 10, n_select: int = 3,
         generations: int = 12, pop_size: int = 20, seed: int = 42,
         dataset: str = "etth1", validate_top: int = 5,
         n_configs: int = 30, gp_generations: int = 50, gp_pop_size: int = 100):
    """Run sweep, fine-tuning, classification diagnosis, evolution, or proxy search.

    Usage:
        modal run feasibility/modal_app.py                                    # sweep
        modal run feasibility/modal_app.py --mode finetune                    # finetune
        modal run feasibility/modal_app.py --mode diagnose --n-epochs 10      # classification diagnosis
        modal run feasibility/modal_app.py --mode evolve --dataset etth1      # evolutionary search
        modal run feasibility/modal_app.py --mode proxy-search --n-configs 30 # GP proxy search
    """
    import os
    import sys
    import numpy as np
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

    elif mode == "diagnose":
        import numpy as np
        from scipy.stats import kendalltau, spearmanr

        with open(sweep_results) as f:
            all_sweep = json.load(f)

        ec_results = [r for r in all_sweep if r.get("dataset") == "EthanolConcentration"]
        if not ec_results:
            print("No EthanolConcentration results found")
            return

        # Select top-n and bottom-n by composite score
        sorted_ec = sorted(ec_results, key=lambda r: r["scores"]["composite"])
        selected = sorted_ec[:n_select] + sorted_ec[-n_select:]

        print(f"Classification diagnosis: top-{n_select} + bottom-{n_select} configs, {n_epochs} epochs")
        for r in selected:
            print(f"  {r['config']['config_id']}: composite={r['scores']['composite']:.4f}")

        print("\nLoading EthanolConcentration...")
        ds = load_ethanol_concentration()
        data_bytes, meta = serialize_dataset(ds)
        print(f"  Samples: {ds['samples'].shape}, Classes: {np.bincount(ds['labels'])}")

        print(f"\nFine-tuning {len(selected)} configs on GPU...")
        results = list(finetune_config.starmap(
            [(r["config"], data_bytes, meta, n_epochs) for r in selected]
        ))

        # Pair with TEMPLATE scores
        ft_results = []
        for r, sel in zip(results, selected):
            accuracy = r.get("accuracy", 0.0)
            composite = sel["scores"]["composite"]
            cfg_id = sel["config"]["config_id"]
            print(f"  {cfg_id}: TEMPLATE={composite:.4f}, accuracy={accuracy:.4f}")
            ft_results.append({
                "config_id": cfg_id,
                "composite": composite,
                "accuracy": accuracy,
                "config": sel["config"],
                "scores": sel["scores"],
            })

        composites = [r["composite"] for r in ft_results]
        accuracies = [r["accuracy"] for r in ft_results]

        tau, tau_p = kendalltau(composites, accuracies)
        rho, rho_p = spearmanr(composites, accuracies)

        print(f"\n{'='*70}")
        print(f"DIAGNOSIS RESULTS")
        print(f"{'='*70}")
        print(f"{'Config ID':<35} {'TEMPLATE':>10} {'Accuracy':>10}")
        print(f"{'-'*55}")
        for r in sorted(ft_results, key=lambda x: x["composite"]):
            print(f"{r['config_id']:<35} {r['composite']:>10.4f} {r['accuracy']:>10.4f}")

        print(f"\nKendall tau  = {tau:.3f} (p={tau_p:.4f})")
        print(f"Spearman rho = {rho:.3f} (p={rho_p:.4f})")

        if tau > 0.3 and tau_p < 0.1:
            print("\nVERDICT: Correlation exists. Classification viable for the paper.")
        elif all(a < 0.1 for a in accuracies):
            print("\nVERDICT: All accuracies near zero. Problem deeper than init.")
        else:
            print("\nVERDICT: Weak/no correlation. Consider scoping out classification.")

        os.makedirs("results", exist_ok=True)
        out = {
            "configs": ft_results,
            "kendall_tau": tau, "kendall_p": tau_p,
            "spearman_rho": rho, "spearman_p": rho_p,
            "n_epochs": n_epochs,
        }
        with open("results/classification_diagnosis.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to results/classification_diagnosis.json")

    elif mode == "evolve":
        from feasibility.evolution import run_evolution

        datasets_to_run = []
        if dataset in ("etth1", "both"):
            datasets_to_run.append(("etth1", load_etth1))
        if dataset in ("ethanol", "both"):
            datasets_to_run.append(("ethanol", load_ethanol_concentration))

        for ds_name, loader in datasets_to_run:
            print(f"\n{'#'*60}")
            print(f"# Evolutionary Search — {ds_name}")
            print(f"# Generations: {generations}, Pop: {pop_size}, Seed: {seed}")
            print(f"{'#'*60}")

            ds = loader()
            data_bytes, meta = serialize_dataset(ds)

            def modal_evaluate(configs):
                return list(evaluate_config.starmap(
                    [(cfg.to_dict(), data_bytes, meta) for cfg in configs]
                ))

            result = run_evolution(
                evaluate_fn=modal_evaluate,
                n_generations=generations,
                pop_size=pop_size,
                seed=seed,
            )

            logger = result["logger"]
            os.makedirs("results", exist_ok=True)

            log_data = logger.to_dict()
            log_data["params"] = {
                "n_generations": generations,
                "pop_size": pop_size,
                "seed": seed,
                "dataset": ds_name,
            }
            log_path = f"results/evolution_log_{ds_name}_{seed}.json"
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)
            print(f"\nEvolution log saved to {log_path}")

            # Validate top configs
            top_n = min(validate_top, len(result["final_population"]))
            top_configs = [r["config"] for r in result["final_population"][:top_n]]

            print(f"\nValidating top-{top_n} configs via fine-tuning...")
            ft_results = list(finetune_config.starmap(
                [(cfg, data_bytes, meta, n_epochs) for cfg in top_configs]
            ))

            ft_path = f"results/evolution_finetune_{ds_name}_{seed}.json"
            with open(ft_path, "w") as f:
                json.dump(ft_results, f, indent=2)

            metric_key = "mse" if ds.get("task") == "forecasting" else "accuracy"
            print(f"\n{'='*60}")
            print(f"EVOLUTION SUMMARY — {ds_name}")
            print(f"{'='*60}")
            print(f"Best TEMPLATE fitness: {result['best_fitness']:.4f}")
            print(f"Best config: {result['best_config']['config_id']}")
            print(f"\nTop-{top_n} fine-tuning validation:")
            for r in ft_results:
                cfg_id = r["config"]["config_id"]
                metric = r.get(metric_key, "N/A")
                print(f"  {cfg_id}: {metric_key}={metric}")
            print(f"{'='*60}")

    elif mode == "proxy-search":
        from feasibility.config import AdapterConfig
        from feasibility.data import (
            load_etth1, load_ettm1, load_ethanol_concentration, serialize_dataset
        )
        from feasibility.proxy_search import (
            select_calibration_configs, CalibrationData,
            save_calibration_data, run_proxy_search, make_proxy_evaluate_fn,
        )
        from feasibility.proxy_gp import serialize_tree
        from feasibility.evolution import run_evolution
        from scipy.stats import kendalltau

        calibration_datasets = {
            "ETTh1": load_etth1,
            "ETTm1": load_ettm1,
            "EthanolConcentration": load_ethanol_concentration,
        }

        ds_names = ["ETTh1", "ETTm1", "EthanolConcentration"]

        # Phase 1: Calibration
        print(f"\n{'#'*60}")
        print(f"# Phase 1: Calibration ({n_configs} configs x {len(ds_names)} datasets)")
        print(f"{'#'*60}")

        configs = select_calibration_configs(n=n_configs, seed=seed)
        print(f"Selected {len(configs)} calibration configs")

        calibrations = []
        for ds_name in ds_names:
            print(f"\n--- {ds_name} ---")
            ds = calibration_datasets[ds_name]()
            data_bytes, meta = serialize_dataset(ds)

            print(f"  Computing statistics for {len(configs)} configs...")
            stat_results = list(compute_statistics_remote.starmap(
                [(cfg.to_dict(), data_bytes, meta) for cfg in configs]
            ))

            print(f"  Fine-tuning {len(configs)} configs for {n_epochs} epochs...")
            ft_results = list(finetune_config.starmap(
                [(cfg.to_dict(), data_bytes, meta, n_epochs) for cfg in configs]
            ))

            is_clf = ds.get("task") == "classification"
            mk = "accuracy" if is_clf else "mse"

            cal = CalibrationData(
                configs=configs,
                statistics=[r["statistics"] for r in stat_results],
                performances=[r[mk] for r in ft_results],
                dataset_name=ds_name,
                metric_key=mk,
                higher_is_better=is_clf,
            )

            cal_path = f"results/calibration/{ds_name}_{seed}.json"
            save_calibration_data(cal, cal_path)
            print(f"  Saved calibration to {cal_path}")
            calibrations.append(cal)

        # Phase 2: GP search (local CPU)
        print(f"\n{'#'*60}")
        print(f"# Phase 2: GP Proxy Search ({gp_generations} gens, pop={gp_pop_size})")
        print(f"{'#'*60}")

        gp_result = run_proxy_search(
            calibration=calibrations[0],
            multi_calibrations=calibrations,
            pop_size=gp_pop_size,
            n_generations=gp_generations,
            seed=seed,
        )

        os.makedirs("results/proxy_search", exist_ok=True)
        gp_out = gp_result.to_dict()
        gp_out["calibration_datasets"] = ds_names
        with open(f"results/proxy_search/gp_result_{seed}.json", "w") as f:
            json.dump(gp_out, f, indent=2)

        print(f"\nBest proxy: {gp_result.best_formula}")
        print(f"Best fitness (Kendall tau): {gp_result.best_fitness:.4f}")

        # Phase 3: Validation (GPU)
        print(f"\n{'#'*60}")
        print(f"# Phase 3: Validation")
        print(f"{'#'*60}")

        best_tree = gp_result.best_tree

        for ds_name in ds_names:
            print(f"\n--- Validating on {ds_name} ---")
            ds = calibration_datasets[ds_name]()
            data_bytes, meta = serialize_dataset(ds)
            is_clf = ds.get("task") == "classification"
            mk = "accuracy" if is_clf else "mse"

            def make_modal_proxy_eval(d_bytes, d_meta, tree):
                def eval_fn(cfgs):
                    srs = list(compute_statistics_remote.starmap(
                        [(c.to_dict(), d_bytes, d_meta) for c in cfgs]
                    ))
                    from feasibility.proxy_gp import evaluate_proxy as ep
                    return [
                        {"config": c.to_dict(), "scores": {"composite": ep(tree, sr["statistics"])}}
                        for c, sr in zip(cfgs, srs)
                    ]
                return eval_fn

            print("  [1/3] Evo-GP-Proxy...")
            proxy_eval = make_modal_proxy_eval(data_bytes, meta, best_tree)
            evo_proxy = run_evolution(proxy_eval, n_generations=generations, pop_size=pop_size, seed=seed)

            print("  [2/3] Evo-TEMPLATE...")
            def modal_template_eval(cfgs):
                return list(evaluate_config.starmap(
                    [(c.to_dict(), data_bytes, meta) for c in cfgs]
                ))
            evo_template = run_evolution(modal_template_eval, n_generations=generations, pop_size=pop_size, seed=seed)

            print("  [3/3] Random search...")
            from feasibility.evolution import random_config as gen_random_config
            rng = np.random.default_rng(seed)
            rand_cfgs = [gen_random_config(rng) for _ in range(pop_size)]
            rand_results = proxy_eval(rand_cfgs)
            rand_results.sort(key=lambda r: r["scores"]["composite"], reverse=True)

            top_n = min(validate_top, 5)
            methods = {
                "Evo-GP-Proxy": [r["config"] for r in evo_proxy["final_population"][:top_n]],
                "Evo-TEMPLATE": [r["config"] for r in evo_template["final_population"][:top_n]],
                "Random": [r["config"] for r in rand_results[:top_n]],
            }

            print(f"\n  Fine-tuning top-{top_n} from each method...")
            comparison = {}
            for method_name, top_cfgs in methods.items():
                ft = list(finetune_config.starmap(
                    [(c, data_bytes, meta, n_epochs) for c in top_cfgs]
                ))
                comparison[method_name] = ft

            print(f"\n  {'='*70}")
            print(f"  COMPARISON - {ds_name} ({mk})")
            print(f"  {'='*70}")
            for method_name, results in comparison.items():
                vals = [r.get(mk, float("nan")) for r in results]
                best_v = max(vals) if is_clf else min(vals)
                mean_v = float(np.mean(vals))
                print(f"    {method_name:<20} best={best_v:.4f}  mean={mean_v:.4f}")

            comp_path = f"results/proxy_search/comparison_{ds_name}_{seed}.json"
            with open(comp_path, "w") as f:
                json.dump({
                    "dataset": ds_name,
                    "metric_key": mk,
                    "proxy_formula": gp_result.best_formula,
                    "proxy_tau": gp_result.best_fitness,
                    "comparison": {
                        m: [{"config": r["config"], mk: float(r.get(mk, 0))} for r in rs]
                        for m, rs in comparison.items()
                    },
                }, f, indent=2, default=str)
            print(f"  Saved to {comp_path}")
