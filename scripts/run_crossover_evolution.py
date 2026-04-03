"""Crossover Evolution: block-level recombination of adapter architectures.

Novel search operator that recombines architectural blocks from high-fitness
parents, unlike random template generation (no parent information) or LLM
generation (generates from scratch).

The key insight: adapter architectures decompose into 3 functional blocks:
  1. Pre-processing (normalization, projection)
  2. Pooling (temporal aggregation strategy)
  3. Head (post-pooling MLP/projection)

Crossover takes blocks from different parents. Mutation perturbs a single
block (swap activation, change hidden dim, add/remove norm).

Usage:
    python scripts/run_crossover_evolution.py --dataset ETTh1
    python scripts/run_crossover_evolution.py --dataset ETTh1 --mode crossover  # crossover + mutation
    python scripts/run_crossover_evolution.py --dataset ETTh1 --mode random     # baseline (current method)
"""

import argparse
import json
import os
import sys
import time
import re
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from run_ablations import (
    generate_random_adapter_code,
    generate_valid_random_adapter,
    run_random_code_ablation,
    POOLING_TYPES, ACTIVATIONS, HIDDEN_DIMS, DROPOUT_RATES,
    _build_pooling_init,
)
from run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)
from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import (
    validate_adapter_code, SEED_ADAPTERS, CodeIndividual, CodeEvolutionLogger,
)


# --- Block-level code decomposition ---

def decompose_adapter(code):
    """Decompose adapter code into functional blocks.

    Returns dict with:
      - pre_norm: bool (has LayerNorm before pooling)
      - pooling_type: str (mean/max/last/attention/conv1d)
      - pooling_init: list of init lines
      - pooling_fwd: list of forward lines
      - mlp_init: list of init lines
      - mlp_fwd: list of forward lines
      - activation: str (nn.GELU/nn.ReLU/nn.SiLU)
      - dropout: float
      - hidden_dims: list of int
    """
    info = {
        "pre_norm": "LayerNorm" in code,
        "pooling_type": "mean",
        "activation": "nn.GELU()",
        "dropout": 0.1,
        "hidden_dims": [128],
    }

    code_lower = code.lower()
    if "conv1d" in code_lower:
        info["pooling_type"] = "conv_pool"
    elif "softmax" in code_lower or "attn" in code_lower:
        info["pooling_type"] = "attention_pool"
    elif "[:, -1" in code or "last" in code_lower:
        info["pooling_type"] = "last_token"
    elif ".max(" in code:
        info["pooling_type"] = "max_pool"
    else:
        info["pooling_type"] = "mean"

    for act in ["nn.GELU()", "nn.ReLU()", "nn.SiLU()", "nn.Tanh()"]:
        if act in code:
            info["activation"] = act
            break

    # Extract hidden dims from nn.Linear calls
    linear_pattern = r"nn\.Linear\(\s*(\w+)\s*,\s*(\w+)\s*\)"
    matches = re.findall(linear_pattern, code)
    dims = []
    for in_d, out_d in matches:
        try:
            dims.append(int(out_d))
        except ValueError:
            pass
    if dims:
        info["hidden_dims"] = dims[:-1] if len(dims) > 1 else dims

    # Extract dropout
    drop_match = re.search(r"Dropout\(([0-9.]+)\)", code)
    if drop_match:
        info["dropout"] = float(drop_match.group(1))

    return info


def crossover(parent_a_info, parent_b_info, rng):
    """Block-level crossover: take blocks from different parents.

    Randomly assigns each block (pre_norm, pooling, activation, dims, dropout)
    from either parent A or parent B.
    """
    child = {}
    for key in ["pre_norm", "pooling_type", "activation", "hidden_dims", "dropout"]:
        if rng.random() < 0.5:
            child[key] = copy.deepcopy(parent_a_info[key])
        else:
            child[key] = copy.deepcopy(parent_b_info[key])
    return child


def mutate(info, rng, mutation_rate=0.3):
    """Mutate one or more blocks with probability mutation_rate."""
    info = copy.deepcopy(info)

    # Mutate pooling
    if rng.random() < mutation_rate:
        info["pooling_type"] = str(rng.choice(POOLING_TYPES))

    # Mutate activation
    if rng.random() < mutation_rate:
        info["activation"] = str(rng.choice(ACTIVATIONS))

    # Mutate hidden dims
    if rng.random() < mutation_rate:
        n_layers = int(rng.choice([1, 1, 2, 2, 3]))
        info["hidden_dims"] = [int(rng.choice(HIDDEN_DIMS)) for _ in range(max(1, n_layers - 1))]

    # Mutate dropout
    if rng.random() < mutation_rate:
        info["dropout"] = float(rng.choice(DROPOUT_RATES))

    # Mutate pre_norm
    if rng.random() < mutation_rate:
        info["pre_norm"] = not info["pre_norm"]

    return info


def assemble_adapter(info, rng):
    """Reassemble a complete adapter code string from block info."""
    pooling_type = info["pooling_type"]
    pool_init, pool_fwd, out_dim_str = _build_pooling_init(pooling_type, rng)

    pre_init = []
    pre_fwd = []
    if info["pre_norm"]:
        pre_init.append("self.pre_norm = nn.LayerNorm(d_model)")
        pre_fwd.append("hidden_states = self.pre_norm(hidden_states)")

    activation = info["activation"]
    dropout = info["dropout"]
    hidden_dims = info["hidden_dims"]

    if out_dim_str == "d_model":
        first_in = "d_model"
    else:
        first_in = out_dim_str

    mlp_init = []
    mlp_fwd = []

    dims = [first_in] + [str(d) for d in hidden_dims] + ["output_dim"]
    for i in range(len(dims) - 1):
        mlp_init.append("self.fc%d = nn.Linear(%s, %s)" % (i, dims[i], dims[i+1]))
        mlp_fwd.append("x = self.fc%d(x)" % i)
        if i < len(dims) - 2:
            mlp_init.append("self.act%d = %s" % (i, activation))
            mlp_fwd.append("x = self.act%d(x)" % i)
            if dropout > 0:
                mlp_init.append("self.drop%d = nn.Dropout(%s)" % (i, dropout))
                mlp_fwd.append("x = self.drop%d(x)" % i)

    mlp_fwd.append("return x")

    all_init = pre_init + pool_init + mlp_init
    all_fwd = pre_fwd + pool_fwd + mlp_fwd

    init_body = "\n        ".join(all_init) if all_init else "pass"
    fwd_body = "\n        ".join(all_fwd)

    code = """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        %s

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        %s""" % (init_body, fwd_body)

    return code


def run_crossover_evolution(
    evaluate_fn,
    n_generations=12,
    pop_size=20,
    elite_count=2,
    seed=42,
    crossover_rate=0.7,
    mutation_rate=0.3,
):
    """Evolutionary search with block-level crossover and mutation.

    Unlike random template generation (no parent info) or LLM generation
    (full regeneration), this operator recombines functional blocks from
    high-fitness parents — a novel code-level recombination operator.
    """
    rng = np.random.default_rng(seed)
    logger = CodeEvolutionLogger()

    # Initialize population
    population = [CodeIndividual(code=c, generation=0) for c in SEED_ADAPTERS]
    while len(population) < pop_size:
        code = generate_valid_random_adapter(rng)
        population.append(CodeIndividual(code=code, generation=0))

    for gen in range(n_generations):
        # Validate
        for ind in population:
            if ind.fitness == 0.0 and ind.mse == float('inf') and ind.error is None:
                val = validate_adapter_code(ind.code)
                if not val["valid"]:
                    ind.error = val["error"]
                    ind.fitness = float('-inf')
                    ind.param_count = val.get("param_count", 0)
                    continue
                ind.param_count = val["param_count"]

        # GPU evaluation
        to_eval = [ind for ind in population if ind.error is None and ind.mse == float('inf')]
        if to_eval:
            codes = [ind.code for ind in to_eval]
            try:
                results = evaluate_fn(codes)
            except Exception as e:
                results = [{"error": "batch error: %s" % e}] * len(codes)

            for ind, result in zip(to_eval, results):
                if "error" in result:
                    ind.error = result["error"]
                    ind.fitness = float('-inf')
                else:
                    ind.mse = result["mse"]
                    ind.fitness = -result["mse"]
                    ind.param_count = result.get("param_count", ind.param_count)

        # Sort and log
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        logger.log_generation(gen, population)

        valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
        if gen == n_generations - 1:
            break

        # Elitism
        next_gen = []
        seen_codes = set()
        for ind in population[:elite_count]:
            if ind.error is None and ind.mse < float('inf'):
                elite = CodeIndividual(
                    code=ind.code,
                    fitness=ind.fitness,
                    mse=ind.mse,
                    param_count=ind.param_count,
                    generation=gen + 1,
                )
                next_gen.append(elite)
                seen_codes.add(ind.code.strip())

        # Decompose all valid parents for crossover
        parent_pool = [ind for ind in valid_pop[:10]]  # top-10 parents
        parent_infos = [(ind, decompose_adapter(ind.code)) for ind in parent_pool]

        # Generate offspring via crossover + mutation
        attempts = 0
        max_attempts = pop_size * 5
        while len(next_gen) < pop_size and attempts < max_attempts:
            attempts += 1

            if len(parent_infos) >= 2 and rng.random() < crossover_rate:
                # Tournament selection: pick 2 parents from top-10
                idx_a, idx_b = rng.choice(len(parent_infos), size=2, replace=False)
                _, info_a = parent_infos[idx_a]
                _, info_b = parent_infos[idx_b]
                child_info = crossover(info_a, info_b, rng)
                child_info = mutate(child_info, rng, mutation_rate)
            else:
                # Mutation only: pick one parent, mutate heavily
                if parent_infos:
                    idx = int(rng.integers(len(parent_infos)))
                    _, info = parent_infos[idx]
                    child_info = mutate(info, rng, mutation_rate=0.6)
                else:
                    # Fallback to random
                    code = generate_valid_random_adapter(rng)
                    if code.strip() not in seen_codes:
                        seen_codes.add(code.strip())
                        next_gen.append(CodeIndividual(code=code, generation=gen + 1))
                    continue

            code = assemble_adapter(child_info, rng)
            val = validate_adapter_code(code)
            if val["valid"] and code.strip() not in seen_codes:
                seen_codes.add(code.strip())
                next_gen.append(CodeIndividual(code=code, generation=gen + 1))

        # Fill remaining with random if needed
        while len(next_gen) < pop_size:
            code = generate_valid_random_adapter(rng)
            if code.strip() not in seen_codes:
                seen_codes.add(code.strip())
                next_gen.append(CodeIndividual(code=code, generation=gen + 1))

        population = next_gen

    valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
    best = min(valid_pop, key=lambda ind: ind.mse) if valid_pop else population[0]

    return {
        "best_code": best.code,
        "best_mse": best.mse,
        "best_fitness": best.fitness,
        "best_param_count": best.param_count,
        "logger": logger,
        "final_population": [
            {"code": ind.code, "mse": ind.mse, "error": ind.error, "param_count": ind.param_count}
            for ind in population
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--n-generations", type=int, default=12)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="both", choices=["crossover", "random", "both"])
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/crossover_ablation", exist_ok=True)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True

    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, val=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_val), len(X_test)))

    def evaluate_fn(code_strs):
        results = []
        for code in code_strs:
            try:
                result = train_adapter(
                    code, model, blocks, X_train, Y_train, X_val, Y_val,
                    device=args.device, n_epochs=3, forecast_horizon=args.horizon,
                    backbone_type=bb_type,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": "%s: %s" % (type(e).__name__, e)})
        return results

    all_results = {}

    modes = [args.mode] if args.mode != "both" else ["crossover", "random"]

    for mode in modes:
        print("\n" + "=" * 60)
        print("%s EVOLUTION: %s H=%d seed=%d" % (mode.upper(), args.dataset, args.horizon, args.seed))
        print("=" * 60)

        start = time.time()
        if mode == "crossover":
            result = run_crossover_evolution(
                evaluate_fn=evaluate_fn,
                n_generations=args.n_generations,
                pop_size=args.pop_size,
                seed=args.seed,
            )
        else:
            result = run_random_code_ablation(
                evaluate_fn=evaluate_fn,
                n_generations=args.n_generations,
                pop_size=args.pop_size,
                seed=args.seed,
            )
        elapsed = time.time() - start

        # Evaluate top-5 on TEST
        valid_pop = [
            ind for ind in result["final_population"]
            if ind["error"] is None and ind["mse"] < float('inf')
        ][:5]

        test_results = []
        for ind in valid_pop:
            try:
                tr = train_adapter(
                    ind["code"], model, blocks, X_train, Y_train, X_test, Y_test,
                    device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                    backbone_type=bb_type,
                )
                test_results.append(tr)
            except Exception:
                pass

        best_test = min((r["mse"] for r in test_results), default=float("inf"))
        all_results[mode] = {
            "best_val_mse": result["best_mse"],
            "best_test_mse": best_test,
            "elapsed": elapsed,
        }
        print("  best_val=%.4f  best_test=%.4f  time=%.0fs" % (result["best_mse"], best_test, elapsed))

    # Summary
    if len(all_results) == 2:
        c = all_results["crossover"]["best_test_mse"]
        r = all_results["random"]["best_test_mse"]
        delta = (c - r) / r * 100
        winner = "CROSSOVER" if c < r else "RANDOM"
        print("\n%s wins: crossover=%.4f random=%.4f delta=%+.1f%%" % (winner, c, r, delta))

    # Save
    path = "results/crossover_ablation/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
                    "results": all_results}, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
