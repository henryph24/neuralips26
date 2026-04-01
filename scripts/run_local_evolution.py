"""Run code evolution locally on GPU (no Modal required).

For RACE VM or any machine with a GPU. Runs sequentially (not parallel)
but avoids Modal overhead and costs.

Usage:
    python scripts/run_local_evolution.py --dataset etth1 --n-generations 3 --pop-size 10
    python scripts/run_local_evolution.py --dataset etth1 --mode random  # random code + evo
    python scripts/run_local_evolution.py --dataset etth1 --mode llm     # LLM code + evo
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except Exception:
    pass

import numpy as np
import torch

from feasibility.model import (
    load_moment, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import (
    validate_adapter_code, SEED_ADAPTERS, train_adapter_from_code,
    run_code_evolution, CodeIndividual, CodeEvolutionLogger,
)
from feasibility.data import (
    load_etth1, load_ettm1, load_etth2, load_ettm2,
    load_weather, load_electricity,
)

# Import random code generator (no Modal dependency)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_ablations import (
    run_random_code_ablation,
    generate_valid_random_adapter,
    generate_random_adapter_code,
)

DATASETS = {
    "etth1": load_etth1,
    "ettm1": load_ettm1,
    "etth2": load_etth2,
    "ettm2": load_ettm2,
    "weather": load_weather,
    "electricity": load_electricity,
}


def setup_model(device="cuda"):
    """Load MOMENT, freeze backbone, unfreeze last 4 blocks."""
    print("Loading MOMENT on %s..." % device)
    model = load_moment(device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    n_blocks = len(blocks)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, n_blocks - 4), n_blocks):
        for p in blocks[i].parameters():
            p.requires_grad = True

    hdim = _get_hidden_dim(model)
    print("MOMENT: %d blocks, d_model=%d, last %d unfrozen" % (
        n_blocks, hdim, min(4, n_blocks)))
    return model, blocks


def make_local_evaluate_fn(model, blocks, samples, n_epochs=3, device="cuda"):
    """Build evaluate_fn that runs locally on GPU (sequential)."""
    def evaluate_fn(code_strs):
        results = []
        for code in code_strs:
            try:
                result = train_adapter_from_code(
                    code=code, model=model, encoder_blocks=blocks,
                    samples=samples, device=device,
                    n_epochs=n_epochs, batch_size=64,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": "%s: %s" % (type(e).__name__, e)})
        return results
    return evaluate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etth1", choices=list(DATASETS.keys()))
    parser.add_argument("--mode", default="random", choices=["random", "llm", "local_llm", "random_no_evo"])
    parser.add_argument("--n-generations", type=int, default=12)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model (--mode llm) or local path (--mode local_llm)")
    parser.add_argument("--adapter-path", default="models/qwen-adapter-coder", help="Path to fine-tuned LoRA adapter (--mode local_llm)")
    parser.add_argument("--validate-top", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/local_evolution", exist_ok=True)

    # Setup
    model, blocks = setup_model(args.device)
    ds = DATASETS[args.dataset]()
    ds_name = ds["name"]
    print("%s: %s samples" % (ds_name, ds["samples"].shape))

    evaluate_fn = make_local_evaluate_fn(
        model, blocks, ds["samples"], n_epochs=3, device=args.device
    )

    # Run evolution
    print("\n" + "=" * 60)
    print("Mode: %s | Dataset: %s | Gens: %d | Pop: %d | Seed: %d" % (
        args.mode, ds_name, args.n_generations, args.pop_size, args.seed))
    print("=" * 60)

    start = time.time()
    if args.mode == "random_no_evo":
        # Generate N random adapters, evaluate all, pick best (no evolution)
        rng = np.random.default_rng(args.seed)
        total = args.n_generations * args.pop_size
        print("Generating %d random adapters (no evolution)..." % total)
        codes = [generate_valid_random_adapter(rng) for _ in range(total)]
        all_results = evaluate_fn(codes)
        logger = CodeEvolutionLogger()
        all_individuals = []
        for code, res in zip(codes, all_results):
            ind = CodeIndividual(code=code, generation=0)
            if "error" in res:
                ind.error = res["error"]
                ind.fitness = float('-inf')
            else:
                ind.mse = res["mse"]
                ind.fitness = -res["mse"]
                ind.param_count = res.get("param_count", 0)
            all_individuals.append(ind)
        logger.log_generation(0, all_individuals)
        valid = [i for i in all_individuals if i.error is None and i.mse < float('inf')]
        valid.sort(key=lambda i: i.mse)
        best = valid[0] if valid else all_individuals[0]
        result = {
            "best_code": best.code, "best_mse": best.mse, "best_fitness": best.fitness,
            "best_param_count": best.param_count, "best_reasoning": "",
            "logger": logger, "all_reasonings": [],
            "final_population": [
                {"code": i.code, "mse": i.mse, "fitness": i.fitness,
                 "param_count": i.param_count, "error": i.error, "reasoning": ""}
                for i in sorted(all_individuals, key=lambda i: i.fitness, reverse=True)
            ],
        }
    elif args.mode == "random":
        result = run_random_code_ablation(
            evaluate_fn=evaluate_fn,
            n_generations=args.n_generations,
            pop_size=args.pop_size,
            seed=args.seed,
        )
    elif args.mode == "local_llm":
        # Load fine-tuned local model before evolution starts
        from feasibility.code_evolution import load_local_llm
        print("Pre-loading fine-tuned LLM from %s..." % args.adapter_path)
        load_local_llm(args.adapter_path)
        result = run_code_evolution(
            evaluate_fn=evaluate_fn,
            n_generations=args.n_generations,
            pop_size=args.pop_size,
            seed=args.seed,
            model="local:%s" % args.adapter_path,
        )
    else:
        result = run_code_evolution(
            evaluate_fn=evaluate_fn,
            n_generations=args.n_generations,
            pop_size=args.pop_size,
            seed=args.seed,
            model=args.model,
        )
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("RESULT: best_mse=%.4f, params=%d, time=%.0fs" % (
        result["best_mse"], result["best_param_count"], elapsed))
    print("=" * 60)

    # Validate top-N at 15 epochs
    top_n = min(args.validate_top, 5)
    valid_pop = [
        ind for ind in result["final_population"]
        if ind["error"] is None and ind["mse"] < float('inf')
    ][:top_n]

    validated = []
    if valid_pop:
        print("\nValidating top-%d at 15 epochs..." % len(valid_pop))
        val_fn = make_local_evaluate_fn(
            model, blocks, ds["samples"], n_epochs=15, device=args.device
        )
        for ind in valid_pop:
            val_results = val_fn([ind["code"]])
            vr = val_results[0]
            validated.append({
                "code": ind["code"],
                "mse_3ep": ind["mse"],
                "mse_15ep": vr.get("mse", float('inf')),
                "param_count": vr.get("param_count", ind["param_count"]),
                "error": vr.get("error"),
            })
            mse_str = "%.4f" % vr["mse"] if "mse" in vr else vr.get("error", "ERR")
            print("  mse_3ep=%.4f -> mse_15ep=%s" % (ind["mse"], mse_str))

    # Save
    save_data = {
        "mode": args.mode,
        "dataset": ds_name,
        "seed": args.seed,
        "n_generations": args.n_generations,
        "pop_size": args.pop_size,
        "elapsed_seconds": elapsed,
        "best_mse_3ep": result["best_mse"],
        "best_param_count": result["best_param_count"],
        "best_code": result["best_code"],
        "generations": result["logger"].to_dict()["generations"],
        "validated": validated,
        "final_population": result["final_population"][:10],
    }

    tag_map = {"local_llm": "self_improving", "random_no_evo": "random_no_evo"}
    tag = tag_map.get(args.mode, args.mode)
    path = "results/local_evolution/%s_%s_%d.json" % (tag, ds_name, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
