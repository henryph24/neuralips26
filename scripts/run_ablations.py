"""Ablation experiments for Code Evolution.

Two ablations to isolate what drives Code Evolution's improvement:

1. Random Code (--ablation random_code):
   Keep evolutionary loop (elitism, selection pressure), replace LLM with
   random template-based code generator. Tests whether the unbounded code
   space alone is sufficient, or LLM guidance is essential.

2. LLM Without Evolution (--ablation llm_no_evo):
   Keep LLM generation, remove evolutionary loop (no fitness feedback,
   no population context, no elitism). Tests whether the evolutionary
   loop adds value beyond one-shot LLM generation.

Usage:
    modal run scripts/run_ablations.py --ablation random_code --dataset etth1
    modal run scripts/run_ablations.py --ablation llm_no_evo --dataset etth1
    modal run scripts/run_ablations.py --ablation random_code --n-generations 2 --pop-size 5  # smoke test
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

try:
    import modal
    from feasibility.modal_app import app, image
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

import numpy as np
from feasibility.data import (
    load_etth1, load_ettm1, load_etth2, load_ettm2,
    load_weather, load_electricity, load_traffic,
    serialize_dataset,
)
from feasibility.code_evolution import (
    validate_adapter_code,
    SEED_ADAPTERS,
    CodeIndividual,
    CodeEvolutionLogger,
    CODE_SYSTEM_PROMPT,
    _call_llm,
)

DATASETS = {
    "ETTh1": load_etth1,
    "ETTm1": load_ettm1,
    "ETTh2": load_etth2,
    "ETTm2": load_ettm2,
    "Weather": load_weather,
    "Electricity": load_electricity,
    "Traffic": load_traffic,
}


# --- Modal GPU evaluation (identical to run_code_evolution.py) ---
# Guarded: only define when Modal is available (not needed for local runs)

if not MODAL_AVAILABLE:
    def _noop_decorator(*args, **kw):
        def wrapper(f):
            return f
        return wrapper
    app = type('FakeApp', (), {'function': staticmethod(_noop_decorator), 'local_entrypoint': staticmethod(_noop_decorator)})()

@app.function(
    gpu="A10G",
    timeout=600,
    scaledown_window=2,
    max_containers=20,
)
def evaluate_adapter_code_remote(
    code_str: str,
    data_bytes: bytes,
    data_meta: dict,
    n_epochs: int = 3,
    forecast_horizon: int = 96,
) -> dict:
    """Evaluate a single adapter code string on GPU."""
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


# --- Random code generation ---

# Building blocks for template-based random adapter code generation.
# Each adapter is composed of: optional pre-pooling → pooling → MLP head.
# Shape flow is guaranteed correct by construction.

POOLING_TYPES = ["mean", "last_token", "max_pool", "attention_pool", "conv_pool"]
ACTIVATIONS = ["nn.GELU()", "nn.ReLU()", "nn.SiLU()", "nn.Tanh()"]
HIDDEN_DIMS = [64, 128, 192, 256, 384]
DROPOUT_RATES = [0.0, 0.0, 0.1, 0.1, 0.2]
CONV_CHANNELS = [64, 128, 256]
CONV_KERNELS = [3, 5, 7, 9]
CONV_STRIDES = [2, 4]


def _build_pooling_init(pooling_type, rng):
    """Return (init_lines, forward_lines, out_dim_expr) for a pooling strategy."""
    if pooling_type == "mean":
        return [], ["x = hidden_states.mean(dim=1)"], "d_model"

    if pooling_type == "last_token":
        return [], ["x = hidden_states[:, -1, :]"], "d_model"

    if pooling_type == "max_pool":
        return [], ["x = hidden_states.max(dim=1)[0]"], "d_model"

    if pooling_type == "attention_pool":
        init = ["self.attn_w = nn.Linear(d_model, 1)"]
        fwd = [
            "scores = self.attn_w(hidden_states).squeeze(-1)",
            "weights = torch.softmax(scores, dim=1).unsqueeze(-1)",
            "x = (hidden_states * weights).sum(dim=1)",
        ]
        return init, fwd, "d_model"

    if pooling_type == "conv_pool":
        ch = int(rng.choice(CONV_CHANNELS))
        k = int(rng.choice(CONV_KERNELS))
        stride = int(rng.choice(CONV_STRIDES))
        act = str(rng.choice(ACTIVATIONS))
        init = [
            f"self.proj_conv = nn.Linear(d_model, {ch})",
            f"self.conv = nn.Conv1d({ch}, {ch}, kernel_size={k}, stride={stride})",
            f"self.act_conv = {act}",
        ]
        fwd = [
            "x = self.proj_conv(hidden_states)",
            "x = x.transpose(1, 2)",
            "x = self.act_conv(self.conv(x))",
            "x = x.mean(dim=2)",
        ]
        return init, fwd, str(ch)

    raise ValueError(f"Unknown pooling type: {pooling_type}")


def generate_random_adapter_code(rng):
    """Generate a random adapter code string using template composition.

    Composes: optional pre-pooling layers → pooling → MLP head.
    Shape correctness is guaranteed by construction.
    """
    pooling_type = str(rng.choice(POOLING_TYPES))
    pool_init, pool_fwd, out_dim_str = _build_pooling_init(pooling_type, rng)

    # Optional pre-pooling: LayerNorm applied per-timestep before pooling
    use_pre_norm = bool(rng.choice([True, False]))
    pre_init = []
    pre_fwd = []
    if use_pre_norm:
        pre_init.append("self.pre_norm = nn.LayerNorm(d_model)")
        pre_fwd.append("hidden_states = self.pre_norm(hidden_states)")

    # Post-pooling MLP (1-3 layers)
    n_mlp_layers = int(rng.choice([1, 1, 2, 2, 3]))
    activation = str(rng.choice(ACTIVATIONS))
    dropout = float(rng.choice(DROPOUT_RATES))

    # Build dimension chain: post_pool_dim → hidden → ... → output_dim
    # out_dim_str is either "d_model" (768) or a concrete int string like "128"
    if out_dim_str == "d_model":
        first_in = "d_model"
    else:
        first_in = out_dim_str

    mlp_init = []
    mlp_fwd = []

    dims = [first_in]
    for i in range(n_mlp_layers - 1):
        dim = int(rng.choice(HIDDEN_DIMS))
        dims.append(str(dim))
    dims.append("output_dim")

    for i in range(len(dims) - 1):
        mlp_init.append(f"self.fc{i} = nn.Linear({dims[i]}, {dims[i+1]})")
        mlp_fwd.append(f"x = self.fc{i}(x)")
        if i < len(dims) - 2:  # no activation/dropout on final layer
            mlp_init.append(f"self.act{i} = {activation}")
            mlp_fwd.append(f"x = self.act{i}(x)")
            if dropout > 0:
                mlp_init.append(f"self.drop{i} = nn.Dropout({dropout})")
                mlp_fwd.append(f"x = self.drop{i}(x)")

    mlp_fwd.append("return x")

    # Assemble
    all_init = pre_init + pool_init + mlp_init
    all_fwd = pre_fwd + pool_fwd + mlp_fwd

    init_body = "\n        ".join(all_init) if all_init else "pass"
    fwd_body = "\n        ".join(all_fwd)

    code = f"""class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        {init_body}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        {fwd_body}"""

    return code


def generate_valid_random_adapter(rng, max_retries=10):
    """Generate random adapter code, retrying until valid."""
    for _ in range(max_retries):
        code = generate_random_adapter_code(rng)
        val = validate_adapter_code(code)
        if val["valid"]:
            return code
    # Fallback to a random seed adapter
    return SEED_ADAPTERS[int(rng.integers(len(SEED_ADAPTERS)))]


# --- Ablation 1: Random Code + Evolution ---

def run_random_code_ablation(
    evaluate_fn,
    n_generations=12,
    pop_size=20,
    elite_count=2,
    seed=42,
):
    """Evolutionary search with random code generation instead of LLM.

    Same evolutionary loop as Code Evolution (elitism, selection pressure),
    but offspring are generated randomly instead of by LLM.
    """
    rng = np.random.default_rng(seed)
    logger = CodeEvolutionLogger()
    best_fitness_history = []

    # Initialize: 5 seed adapters + random adapters
    population = [CodeIndividual(code=c, generation=0) for c in SEED_ADAPTERS]
    while len(population) < pop_size:
        code = generate_valid_random_adapter(rng)
        population.append(CodeIndividual(code=code, generation=0))

    for gen in range(n_generations):
        # Step 1: Validate locally
        for ind in population:
            if ind.fitness == 0.0 and ind.mse == float('inf') and ind.error is None:
                val = validate_adapter_code(ind.code)
                if not val["valid"]:
                    ind.error = val["error"]
                    ind.fitness = float('-inf')
                    ind.param_count = val["param_count"]
                    continue
                ind.param_count = val["param_count"]

        # Step 2: Batch GPU evaluation
        to_eval = [ind for ind in population if ind.error is None and ind.mse == float('inf')]
        if to_eval:
            codes = [ind.code for ind in to_eval]
            try:
                results = evaluate_fn(codes)
            except Exception as e:
                results = [{"error": f"batch evaluate error: {e}"}] * len(codes)

            for ind, result in zip(to_eval, results):
                if "error" in result:
                    ind.error = result["error"]
                    ind.fitness = float('-inf')
                else:
                    ind.mse = result["mse"]
                    ind.fitness = -result["mse"]
                    ind.param_count = result.get("param_count", ind.param_count)

        # Step 3: Sort and log
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        logger.log_generation(gen, population)

        valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
        best_fitness_history.append(valid_pop[0].fitness if valid_pop else float('-inf'))

        if gen == n_generations - 1:
            break

        # Step 4: Elitism — top elites pass unchanged
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

        # Step 5: Random offspring (replaces LLM generation)
        while len(next_gen) < pop_size:
            code = generate_valid_random_adapter(rng)
            if code.strip() not in seen_codes:
                seen_codes.add(code.strip())
                next_gen.append(CodeIndividual(code=code, generation=gen + 1))

        population = next_gen

    # Final result
    valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
    if valid_pop:
        best = min(valid_pop, key=lambda ind: ind.mse)
    else:
        best = population[0] if population else CodeIndividual(code="", mse=float('inf'))

    return {
        "best_code": best.code,
        "best_mse": best.mse,
        "best_fitness": best.fitness,
        "best_param_count": best.param_count,
        "best_reasoning": "",
        "logger": logger,
        "all_reasonings": [],
        "final_population": [
            {
                "code": ind.code,
                "mse": ind.mse,
                "fitness": ind.fitness,
                "param_count": ind.param_count,
                "error": ind.error,
                "reasoning": "",
            }
            for ind in sorted(population, key=lambda ind: ind.fitness, reverse=True)
        ],
    }


# --- Ablation 2: LLM Without Evolution ---

LLM_NO_EVO_USER_PROMPT = """Generate {n_adapters} diverse adapter architectures for a time series forecasting model.

You are designing from scratch — there is no existing population or fitness feedback.

Generate architectures that are structurally diverse:
- Try different pooling strategies (mean, attention, conv-based, learned weights)
- Try different layer types (Linear, Conv1d, multi-head attention)
- Try different depths and widths
- Try different activation functions and normalization approaches

Return a JSON object with key "adapters", containing a list of objects each with "reasoning" (string) and "code" (string containing the full Adapter class definition)."""


def llm_generate_code_no_evo(n_adapters, model="gpt-4o-mini"):
    """Generate adapters with NO evolutionary feedback.

    Uses the same system prompt (adapter contract + design principles)
    but gives no population info, no fitness history, no error feedback.
    """
    try:
        user_prompt = LLM_NO_EVO_USER_PROMPT.format(n_adapters=n_adapters)
        content = _call_llm(CODE_SYSTEM_PROMPT, user_prompt, model=model)
        parsed = json.loads(content)
        adapters = parsed.get("adapters", [])

        print(f"    LLM proposed {len(adapters)} adapters (requested {n_adapters})")
        return [{"reasoning": a.get("reasoning", ""), "code": a.get("code", "")} for a in adapters]

    except Exception as e:
        print(f"    LLM call failed ({type(e).__name__}: {e})")
        return []


def run_llm_no_evo_ablation(
    evaluate_fn,
    n_generations=12,
    pop_size=20,
    seed=42,
    model="gpt-4o-mini",
):
    """LLM generates adapters in independent batches with no evolutionary feedback.

    Same total eval budget as Code Evolution (~n_generations × pop_size),
    but each batch is generated independently — no elitism, no fitness
    feedback, no population context between batches.
    """
    logger = CodeEvolutionLogger()
    all_individuals = []

    for batch_idx in range(n_generations):
        print(f"\n  Batch {batch_idx+1}/{n_generations}: requesting {pop_size} adapters...")

        proposals = llm_generate_code_no_evo(pop_size, model=model)

        batch_pop = []
        for item in proposals:
            ind = CodeIndividual(
                code=item["code"],
                reasoning=item.get("reasoning", ""),
                generation=batch_idx,
            )
            val = validate_adapter_code(ind.code)
            if not val["valid"]:
                ind.error = val["error"]
                ind.fitness = float('-inf')
            ind.param_count = val["param_count"]
            batch_pop.append(ind)

        # Evaluate valid ones on GPU
        to_eval = [ind for ind in batch_pop if ind.error is None]
        if to_eval:
            codes = [ind.code for ind in to_eval]
            try:
                results = evaluate_fn(codes)
            except Exception as e:
                results = [{"error": f"batch evaluate error: {e}"}] * len(codes)

            for ind, result in zip(to_eval, results):
                if "error" in result:
                    ind.error = result["error"]
                    ind.fitness = float('-inf')
                else:
                    ind.mse = result["mse"]
                    ind.fitness = -result["mse"]
                    ind.param_count = result.get("param_count", ind.param_count)

        all_individuals.extend(batch_pop)
        logger.log_generation(batch_idx, batch_pop)

    # Pick best overall — no selection pressure was applied during generation
    valid = [ind for ind in all_individuals if ind.error is None and ind.mse < float('inf')]
    valid.sort(key=lambda ind: ind.mse)
    best = valid[0] if valid else (all_individuals[0] if all_individuals else CodeIndividual(code="", mse=float('inf')))

    return {
        "best_code": best.code,
        "best_mse": best.mse,
        "best_fitness": best.fitness,
        "best_param_count": best.param_count,
        "best_reasoning": best.reasoning,
        "logger": logger,
        "all_reasonings": [],
        "final_population": [
            {
                "code": ind.code,
                "mse": ind.mse,
                "fitness": ind.fitness,
                "param_count": ind.param_count,
                "error": ind.error,
                "reasoning": ind.reasoning,
            }
            for ind in sorted(all_individuals, key=lambda ind: ind.fitness, reverse=True)
        ],
    }


# --- Main entrypoint ---

def parse_datasets(dataset_arg):
    """Parse --dataset arg into dict of {name: loader}."""
    datasets_to_run = {}
    if dataset_arg in ("etth1", "both", "all"):
        datasets_to_run["ETTh1"] = DATASETS["ETTh1"]
    if dataset_arg in ("ettm1", "both", "all"):
        datasets_to_run["ETTm1"] = DATASETS["ETTm1"]
    if dataset_arg in ("etth2", "all"):
        datasets_to_run["ETTh2"] = DATASETS["ETTh2"]
    if dataset_arg in ("ettm2", "all"):
        datasets_to_run["ETTm2"] = DATASETS["ETTm2"]
    if dataset_arg in ("weather", "all"):
        datasets_to_run["Weather"] = DATASETS["Weather"]
    if dataset_arg in ("electricity", "all"):
        datasets_to_run["Electricity"] = DATASETS["Electricity"]
    if dataset_arg in ("traffic", "all"):
        datasets_to_run["Traffic"] = DATASETS["Traffic"]
    return datasets_to_run


@app.local_entrypoint()
def ablation(
    ablation: str = "random_code",
    seed: int = 42,
    n_generations: int = 12,
    pop_size: int = 20,
    validate_top: int = 5,
    model: str = "gpt-4o-mini",
    dataset: str = "both",
):
    if ablation not in ("random_code", "llm_no_evo"):
        print(f"ERROR: --ablation must be 'random_code' or 'llm_no_evo', got '{ablation}'")
        return

    os.makedirs("results/code_evolution", exist_ok=True)
    datasets_to_run = parse_datasets(dataset)

    ablation_label = {
        "random_code": "Random Code + Evolution",
        "llm_no_evo": "LLM Without Evolution",
    }[ablation]

    for ds_name, loader in datasets_to_run.items():
        print(f"\n{'#'*60}")
        print(f"# {ds_name} — Ablation: {ablation_label}")
        print(f"# Generations: {n_generations}, Pop: {pop_size}, Seed: {seed}")
        print(f"{'#'*60}")

        ds = loader()
        data_bytes, meta = serialize_dataset(ds)

        # Build batch evaluate_fn
        def make_evaluate_fn(d_bytes, d_meta):
            def eval_fn(code_strs):
                results = list(evaluate_adapter_code_remote.starmap(
                    [(code, d_bytes, d_meta, 3) for code in code_strs]
                ))
                return results
            return eval_fn

        evaluate_fn = make_evaluate_fn(data_bytes, meta)

        # Run ablation
        if ablation == "random_code":
            result = run_random_code_ablation(
                evaluate_fn=evaluate_fn,
                n_generations=n_generations,
                pop_size=pop_size,
                seed=seed,
            )
        else:
            result = run_llm_no_evo_ablation(
                evaluate_fn=evaluate_fn,
                n_generations=n_generations,
                pop_size=pop_size,
                seed=seed,
                model=model,
            )

        # Save ablation log
        log_data = result["logger"].to_dict()
        log_data["params"] = {
            "ablation": ablation,
            "n_generations": n_generations,
            "pop_size": pop_size,
            "seed": seed,
            "dataset": ds_name,
            "model": model if ablation == "llm_no_evo" else "N/A",
        }
        log_data["best_code"] = result["best_code"]
        log_data["best_mse"] = result["best_mse"]
        log_data["best_param_count"] = result["best_param_count"]

        log_path = f"results/code_evolution/ablation_{ablation}_{ds_name}_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"\nSaved ablation log to {log_path}")

        # Validate top-N at 15 epochs
        top_n = min(validate_top, 5)
        valid_pop = [
            ind for ind in result["final_population"]
            if ind["error"] is None and ind["mse"] < float('inf')
        ][:top_n]

        validated = []
        if valid_pop:
            print(f"\nValidating top-{len(valid_pop)} adapters at 15 epochs (parallel)...")
            val_results = list(evaluate_adapter_code_remote.starmap(
                [(ind["code"], data_bytes, meta, 15) for ind in valid_pop]
            ))
            for ind, val_result in zip(valid_pop, val_results):
                validated.append({
                    "code": ind["code"],
                    "mse_3ep": ind["mse"],
                    "mse_15ep": val_result.get("mse", float('inf')),
                    "param_count": val_result.get("param_count", ind["param_count"]),
                    "error": val_result.get("error"),
                    "reasoning": ind.get("reasoning", ""),
                })
                mse_str = f"{val_result.get('mse', 'ERR'):.4f}" if "mse" in val_result else val_result.get("error", "ERR")
                print(f"  mse_3ep={ind['mse']:.4f} -> mse_15ep={mse_str}")

            val_path = f"results/code_evolution/ablation_{ablation}_validated_{ds_name}_{seed}.json"
            with open(val_path, "w") as f:
                json.dump(validated, f, indent=2, default=str)

        # Print comparison with cached Code Evolution results
        print(f"\n{'='*60}")
        print(f"COMPARISON — {ds_name} (MSE, seed={seed})")
        print(f"{'='*60}")

        # Ablation results
        if validated:
            abl_mses = [v["mse_15ep"] for v in validated if v.get("error") is None]
            if abl_mses:
                print(f"  {ablation_label:<25} best={min(abl_mses):.4f}  mean={np.mean(abl_mses):.4f}")
            else:
                print(f"  {ablation_label:<25} all failed at 15ep validation")
        else:
            print(f"  {ablation_label:<25} no valid adapters found")

        # Code Evolution results (if cached)
        evo_val_path = f"results/code_evolution/validated_{ds_name}_{seed}.json"
        if os.path.exists(evo_val_path):
            with open(evo_val_path) as f:
                evo_validated = json.load(f)
            evo_mses = [v["mse_15ep"] for v in evo_validated if v.get("error") is None]
            if evo_mses:
                print(f"  {'Code Evolution':<25} best={min(evo_mses):.4f}  mean={np.mean(evo_mses):.4f}")

        # Baselines (if cached)
        for baseline_prefix in [
            f"results/code_evolution/baselines_{ds_name}_{seed}.json",
            f"results/early_stopping/comparison_{ds_name}_{seed}.json",
        ]:
            if os.path.exists(baseline_prefix):
                with open(baseline_prefix) as f:
                    cached = json.load(f)
                mk = cached.get("metric_key", "mse")
                for method_name in ["Evo-3epoch", "Random"]:
                    method_results = cached.get("comparison", {}).get(method_name, [])
                    if method_results:
                        vals = [float(r.get(mk, float("nan"))) for r in method_results]
                        valid_vals = [v for v in vals if not np.isnan(v)]
                        if valid_vals:
                            print(f"  {method_name:<25} best={min(valid_vals):.4f}  mean={np.mean(valid_vals):.4f}")
                break

        # Save full comparison
        save_data = {
            "ablation": ablation,
            "dataset": ds_name,
            "metric_key": "mse",
            "seed": seed,
            "model": model if ablation == "llm_no_evo" else "N/A",
            "n_generations": n_generations,
            "pop_size": pop_size,
            "total_evaluated": sum(
                g.get("n_valid", 0) + g.get("n_invalid", 0)
                for g in log_data["generations"]
            ),
            "n_valid": sum(g.get("n_valid", 0) for g in log_data["generations"]),
            "best_code": result["best_code"],
            "best_mse_3ep": result["best_mse"],
            "best_param_count": result["best_param_count"],
            "validated": validated,
        }

        comp_path = f"results/code_evolution/ablation_{ablation}_comparison_{ds_name}_{seed}.json"
        with open(comp_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {comp_path}")

    print(f"\n{'='*60}")
    print(f"Done. Ablation '{ablation}' results in results/code_evolution/")
