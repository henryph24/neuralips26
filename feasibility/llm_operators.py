"""LLM-guided evolutionary operators for adapter configuration search."""

import copy
import json
import traceback
from typing import Callable, List

import numpy as np

from feasibility.config import (
    AdapterConfig,
    LORA_RANKS,
    BOTTLENECK_DIMS,
    LAYER_PLACEMENTS,
    TARGET_MODULE_SETS,
    UNFREEZE_STRATEGIES,
    HEAD_TYPES,
    POOLING_STRATEGIES,
    _make_id,
)
from feasibility.evolution import (
    Individual,
    EvolutionLogger,
    random_config,
    mutate,
    tournament_select,
)

ADAPTER_TYPES = ["lora", "bottleneck"]
PLACEMENT_KEYS = list(LAYER_PLACEMENTS.keys())
TARGET_KEYS = list(TARGET_MODULE_SETS.keys())

# OpenAI function calling schema for structured output
PROPOSE_CONFIGS_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_configs",
        "description": "Propose new adapter configurations for the next generation of evolutionary search.",
        "parameters": {
            "type": "object",
            "properties": {
                "configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "One sentence explaining why this config is promising.",
                            },
                            "adapter_type": {
                                "type": "string",
                                "enum": ["lora", "bottleneck"],
                            },
                            "lora_rank": {
                                "type": "integer",
                                "enum": [2, 4, 8, 16, 32, 64],
                                "description": "Only for lora adapter_type.",
                            },
                            "target_modules_key": {
                                "type": "string",
                                "enum": ["qv", "qkvo"],
                                "description": "Only for lora adapter_type.",
                            },
                            "bottleneck_dim": {
                                "type": "integer",
                                "enum": [32, 64, 128],
                                "description": "Only for bottleneck adapter_type.",
                            },
                            "layer_placement": {
                                "type": "string",
                                "enum": ["all", "first_half", "last_half"],
                            },
                            "unfreeze": {
                                "type": "string",
                                "enum": ["frozen", "last2", "last4", "all"],
                            },
                            "head_type": {
                                "type": "string",
                                "enum": ["linear", "mlp1", "mlp2"],
                            },
                            "pooling": {
                                "type": "string",
                                "enum": ["mean", "max", "last", "cls_mean_max"],
                            },
                        },
                        "required": [
                            "reasoning",
                            "adapter_type",
                            "layer_placement",
                            "unfreeze",
                            "head_type",
                            "pooling",
                        ],
                    },
                },
            },
            "required": ["configs"],
        },
    },
}

SYSTEM_PROMPT = """You are an expert in neural architecture search (NAS) for time series foundation models. You are guiding an evolutionary search over adapter configurations for a pre-trained time series model (MOMENT).

Your goal: propose diverse, promising adapter configurations that improve upon the current population. Learn from both high-performing and low-performing configs to identify which design choices matter most."""

USER_PROMPT_TEMPLATE = """# Adapter Search Space

8 dimensions with valid values:
1. adapter_type: "lora" or "bottleneck"
2. lora_rank (lora only): {lora_ranks}
3. target_modules_key (lora only): "qv" (query+value) or "qkvo" (query+key+value+output)
4. bottleneck_dim (bottleneck only): {bottleneck_dims}
5. layer_placement: "all" (layers 0-7), "first_half" (0-3), "last_half" (4-7)
6. unfreeze: "frozen" (adapter only), "last2", "last4", "all" (backbone layers)
7. head_type: "linear", "mlp1" (1 hidden layer), "mlp2" (2 hidden layers)
8. pooling: "mean", "max", "last", "cls_mean_max"

Constraints: lora_alpha is always 2 × lora_rank (set automatically).

# Current Population (Generation {generation})

## Top 5 (best fitness):
{top_configs}

## Bottom 3 (worst fitness):
{bottom_configs}

## Best fitness trajectory: {fitness_history}

# Task

Propose {n_offspring} new adapter configurations. For each:
- Provide 1-sentence reasoning explaining your design choice
- Specify all required fields

Balance exploitation (variations of top configs) with exploration (novel combinations). Avoid duplicating existing configs."""


def _format_individual(ind: Individual) -> dict:
    """Format an individual for the LLM prompt."""
    cfg = ind.config
    d = {
        "fitness": round(ind.fitness, 4),
        "adapter_type": cfg.adapter_type,
        "layer_placement": cfg.layer_placement,
        "unfreeze": cfg.unfreeze,
        "head_type": cfg.head_type,
        "pooling": cfg.pooling,
    }
    if cfg.adapter_type == "lora":
        d["lora_rank"] = cfg.lora_rank
        d["target_modules_key"] = cfg.target_modules_key
    elif cfg.adapter_type == "bottleneck":
        d["bottleneck_dim"] = cfg.bottleneck_dim
    return d


def _validate_and_fix_config(raw: dict, rng: np.random.Generator) -> AdapterConfig:
    """Validate LLM-proposed config and fix invalid fields."""
    adapter_type = raw.get("adapter_type", "lora")
    if adapter_type not in ADAPTER_TYPES:
        adapter_type = rng.choice(ADAPTER_TYPES)

    placement = raw.get("layer_placement", "all")
    if placement not in PLACEMENT_KEYS:
        placement = rng.choice(PLACEMENT_KEYS)

    unfreeze = raw.get("unfreeze", "frozen")
    if unfreeze not in UNFREEZE_STRATEGIES:
        unfreeze = rng.choice(UNFREEZE_STRATEGIES)

    head_type = raw.get("head_type", "linear")
    if head_type not in HEAD_TYPES:
        head_type = rng.choice(HEAD_TYPES)

    pooling = raw.get("pooling", "mean")
    if pooling not in POOLING_STRATEGIES:
        pooling = rng.choice(POOLING_STRATEGIES)

    if adapter_type == "lora":
        rank = raw.get("lora_rank", 8)
        if rank not in LORA_RANKS:
            rank = int(rng.choice(LORA_RANKS))
        tm_key = raw.get("target_modules_key", "qv")
        if tm_key not in TARGET_KEYS:
            tm_key = rng.choice(TARGET_KEYS)
        cfg = AdapterConfig(
            adapter_type="lora",
            lora_rank=rank,
            lora_alpha=rank * 2,
            target_modules_key=tm_key,
            layer_placement=placement,
            unfreeze=unfreeze,
            head_type=head_type,
            pooling=pooling,
        )
    else:
        dim = raw.get("bottleneck_dim", 64)
        if dim not in BOTTLENECK_DIMS:
            dim = int(rng.choice(BOTTLENECK_DIMS))
        cfg = AdapterConfig(
            adapter_type="bottleneck",
            bottleneck_dim=dim,
            layer_placement=placement,
            lora_rank=None,
            lora_alpha=None,
            target_modules_key=None,
            unfreeze=unfreeze,
            head_type=head_type,
            pooling=pooling,
        )

    cfg.config_id = _make_id(cfg)
    return cfg


def llm_generate_offspring(
    population: List[Individual],
    n_offspring: int,
    generation: int,
    best_fitness_history: List[float],
    model: str = "gpt-4o-mini",
    rng: np.random.Generator = None,
) -> tuple[List[AdapterConfig], List[str]]:
    """Use LLM to propose offspring configs.

    Returns:
        Tuple of (configs, reasonings). On LLM failure, falls back to
        random mutations and returns empty reasonings.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build prompt
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    top_5 = [_format_individual(ind) for ind in sorted_pop[:5]]
    bottom_3 = [_format_individual(ind) for ind in sorted_pop[-3:]]

    history_str = [round(f, 4) for f in best_fitness_history]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        lora_ranks=LORA_RANKS,
        bottleneck_dims=BOTTLENECK_DIMS,
        generation=generation,
        top_configs=json.dumps(top_5, indent=2),
        bottom_configs=json.dumps(bottom_3, indent=2),
        fitness_history=history_str,
        n_offspring=n_offspring,
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[PROPOSE_CONFIGS_TOOL],
            tool_choice={"type": "function", "function": {"name": "propose_configs"}},
        )

        # Extract function call result
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("No tool_calls in LLM response")

        tool_call = message.tool_calls[0]
        if tool_call.function.name != "propose_configs":
            raise ValueError(f"Unexpected function: {tool_call.function.name}")

        parsed = json.loads(tool_call.function.arguments)
        raw_configs = parsed.get("configs", [])
        if not raw_configs:
            raise ValueError("Empty configs in tool response")

        configs = []
        reasonings = []
        for raw in raw_configs:
            cfg = _validate_and_fix_config(raw, rng)
            configs.append(cfg)
            reasonings.append(raw.get("reasoning", ""))

        print(f"    LLM proposed {len(configs)} configs (requested {n_offspring})")
        return configs, reasonings

    except Exception as e:
        print(f"    LLM call failed ({type(e).__name__}: {e}), falling back to mutations")
        traceback.print_exc()
        # Fallback: mutate random parents
        configs = []
        for _ in range(n_offspring):
            parent = population[rng.integers(len(population))]
            configs.append(mutate(parent.config, rng))
        return configs, []


def run_llm_evolution(
    evaluate_fn: Callable[[List[AdapterConfig]], List[dict]],
    n_generations: int = 12,
    pop_size: int = 20,
    elite_count: int = 2,
    seed: int = 42,
    model: str = "gpt-4o-mini",
) -> dict:
    """Run evolutionary search with LLM-guided mutation/crossover.

    Same interface as run_evolution but uses LLM to propose offspring
    instead of random crossover/mutation.
    """
    rng = np.random.default_rng(seed)
    logger = EvolutionLogger()
    best_fitness_history = []
    all_reasonings = []

    # Initialize population
    seen_ids = set()
    population = []
    while len(population) < pop_size:
        cfg = random_config(rng)
        if cfg.config_id not in seen_ids:
            seen_ids.add(cfg.config_id)
            population.append(Individual(config=cfg, generation=0))

    # Main loop
    for gen in range(n_generations):
        # Evaluate population
        configs = [ind.config for ind in population]
        results = evaluate_fn(configs)

        for ind, result in zip(population, results):
            ind.fitness = result["scores"]["composite"]
            ind.scores = result["scores"]
            ind.generation = gen

        # Sort by fitness descending
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        logger.log_generation(gen, population)
        best_fitness_history.append(population[0].fitness)

        if gen == n_generations - 1:
            break

        # Build next generation
        next_gen = []

        # Elitism
        for i in range(elite_count):
            elite = copy.deepcopy(population[i])
            next_gen.append(elite)

        gen_seen = set(ind.config.config_id for ind in next_gen)

        # LLM-guided offspring
        n_needed = pop_size - elite_count
        llm_configs, reasonings = llm_generate_offspring(
            population, n_needed, gen, best_fitness_history, model=model, rng=rng,
        )
        all_reasonings.append({
            "generation": gen,
            "reasonings": reasonings,
        })

        # Dedup and add LLM configs
        for cfg in llm_configs:
            if len(next_gen) >= pop_size:
                break
            if cfg.config_id not in gen_seen:
                gen_seen.add(cfg.config_id)
                next_gen.append(Individual(config=cfg, generation=gen + 1))

        # Fill remaining slots with traditional mutations (fallback)
        attempts = 0
        while len(next_gen) < pop_size and attempts < pop_size * 5:
            attempts += 1
            parent = tournament_select(population, k=3, rng=rng)
            child_cfg = mutate(parent.config, rng)
            if child_cfg.config_id not in gen_seen:
                gen_seen.add(child_cfg.config_id)
                next_gen.append(Individual(config=child_cfg, generation=gen + 1))

        # Last resort: random configs
        while len(next_gen) < pop_size:
            cfg = random_config(rng)
            if cfg.config_id not in gen_seen:
                gen_seen.add(cfg.config_id)
                next_gen.append(Individual(config=cfg, generation=gen + 1))

        population = next_gen

    best = max(population, key=lambda ind: ind.fitness)
    return {
        "best_config": best.config.to_dict(),
        "best_fitness": best.fitness,
        "best_scores": best.scores,
        "final_population": [
            {"config": ind.config.to_dict(), "fitness": ind.fitness, "scores": ind.scores}
            for ind in sorted(population, key=lambda ind: ind.fitness, reverse=True)
        ],
        "logger": logger,
        "llm_reasonings": all_reasonings,
    }
