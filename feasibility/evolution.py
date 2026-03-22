"""Evolutionary search for adapter configurations using TEMPLATE scores as fitness."""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

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

ADAPTER_TYPES = ["lora", "bottleneck"]
PLACEMENT_KEYS = list(LAYER_PLACEMENTS.keys())
TARGET_KEYS = list(TARGET_MODULE_SETS.keys())


@dataclass
class Individual:
    config: AdapterConfig
    fitness: float = 0.0
    scores: dict = field(default_factory=dict)
    generation: int = 0


def random_config(rng: np.random.Generator) -> AdapterConfig:
    """Generate a random valid adapter config."""
    adapter_type = rng.choice(ADAPTER_TYPES, p=[0.7, 0.3])
    placement = rng.choice(PLACEMENT_KEYS)
    unfreeze = rng.choice(UNFREEZE_STRATEGIES)
    head_type = rng.choice(HEAD_TYPES)
    pooling = rng.choice(POOLING_STRATEGIES)

    if adapter_type == "lora":
        rank = rng.choice(LORA_RANKS)
        tm_key = rng.choice(TARGET_KEYS)
        cfg = AdapterConfig(
            adapter_type="lora",
            lora_rank=int(rank),
            lora_alpha=int(rank) * 2,
            target_modules_key=tm_key,
            layer_placement=placement,
            unfreeze=unfreeze,
            head_type=head_type,
            pooling=pooling,
        )
    else:
        dim = rng.choice(BOTTLENECK_DIMS)
        cfg = AdapterConfig(
            adapter_type="bottleneck",
            bottleneck_dim=int(dim),
            layer_placement=placement,
            unfreeze=unfreeze,
            head_type=head_type,
            pooling=pooling,
        )
    cfg.config_id = _make_id(cfg)
    return cfg


def _regenerate_id(cfg: AdapterConfig) -> AdapterConfig:
    """Regenerate config_id from parameters."""
    cfg.config_id = _make_id(cfg)
    return cfg


def mutate(config: AdapterConfig, rng: np.random.Generator) -> AdapterConfig:
    """Apply one random mutation to a config."""
    cfg = copy.deepcopy(config)

    if cfg.adapter_type == "linear_probe":
        return random_config(rng)

    # Shared mutations that apply to any adapter type
    shared_ops = ["unfreeze", "head_type", "pooling"]

    if cfg.adapter_type == "lora":
        type_ops = ["rank", "target_modules", "placement", "type"]
    else:
        type_ops = ["dim", "placement", "type"]

    all_ops = type_ops + shared_ops
    op = rng.choice(all_ops)

    # Shared mutations
    if op == "unfreeze":
        others = [u for u in UNFREEZE_STRATEGIES if u != cfg.unfreeze]
        cfg.unfreeze = rng.choice(others)
    elif op == "head_type":
        others = [h for h in HEAD_TYPES if h != cfg.head_type]
        cfg.head_type = rng.choice(others)
    elif op == "pooling":
        others = [p for p in POOLING_STRATEGIES if p != cfg.pooling]
        cfg.pooling = rng.choice(others)
    # Type-specific mutations
    elif op == "rank":
        idx = LORA_RANKS.index(cfg.lora_rank)
        direction = rng.choice([-1, 1])
        new_idx = max(0, min(len(LORA_RANKS) - 1, idx + direction))
        cfg.lora_rank = LORA_RANKS[new_idx]
        cfg.lora_alpha = cfg.lora_rank * 2
    elif op == "target_modules":
        cfg.target_modules_key = "qkvo" if cfg.target_modules_key == "qv" else "qv"
    elif op == "placement":
        others = [p for p in PLACEMENT_KEYS if p != cfg.layer_placement]
        cfg.layer_placement = rng.choice(others)
    elif op == "dim":
        idx = BOTTLENECK_DIMS.index(cfg.bottleneck_dim)
        direction = rng.choice([-1, 1])
        new_idx = max(0, min(len(BOTTLENECK_DIMS) - 1, idx + direction))
        cfg.bottleneck_dim = BOTTLENECK_DIMS[new_idx]
    elif op == "type":
        if cfg.adapter_type == "lora":
            cfg.adapter_type = "bottleneck"
            cfg.bottleneck_dim = int(rng.choice(BOTTLENECK_DIMS))
            cfg.lora_rank = None
            cfg.lora_alpha = None
            cfg.target_modules_key = None
        else:
            cfg.adapter_type = "lora"
            cfg.lora_rank = int(rng.choice(LORA_RANKS))
            cfg.lora_alpha = cfg.lora_rank * 2
            cfg.target_modules_key = rng.choice(TARGET_KEYS)
            cfg.bottleneck_dim = None

    return _regenerate_id(cfg)


def crossover(
    parent1: AdapterConfig, parent2: AdapterConfig, rng: np.random.Generator
) -> AdapterConfig:
    """Uniform crossover between two parents."""
    cfg = copy.deepcopy(parent1)

    # Shared genes: crossover independently
    cfg.layer_placement = rng.choice([parent1.layer_placement, parent2.layer_placement])
    cfg.unfreeze = rng.choice([parent1.unfreeze, parent2.unfreeze])
    cfg.head_type = rng.choice([parent1.head_type, parent2.head_type])
    cfg.pooling = rng.choice([parent1.pooling, parent2.pooling])

    if parent1.adapter_type == parent2.adapter_type:
        if cfg.adapter_type == "lora":
            cfg.lora_rank = rng.choice([parent1.lora_rank, parent2.lora_rank])
            cfg.lora_alpha = cfg.lora_rank * 2
            cfg.target_modules_key = rng.choice(
                [parent1.target_modules_key, parent2.target_modules_key]
            )
        elif cfg.adapter_type == "bottleneck":
            cfg.bottleneck_dim = rng.choice(
                [parent1.bottleneck_dim, parent2.bottleneck_dim]
            )
    else:
        chosen = rng.choice([parent1, parent2])
        cfg.adapter_type = chosen.adapter_type
        cfg.lora_rank = chosen.lora_rank
        cfg.lora_alpha = chosen.lora_alpha
        cfg.target_modules_key = chosen.target_modules_key
        cfg.bottleneck_dim = chosen.bottleneck_dim

    return _regenerate_id(cfg)


def tournament_select(
    population: List[Individual], k: int, rng: np.random.Generator
) -> Individual:
    """Tournament selection: pick k random individuals, return the fittest."""
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    candidates = [population[i] for i in indices]
    return max(candidates, key=lambda ind: ind.fitness)


class EvolutionLogger:
    """Track evolution progress across generations."""

    def __init__(self):
        self.generations = []
        self.all_evaluated = []

    def log_generation(self, gen: int, population: List[Individual]):
        fitnesses = [ind.fitness for ind in population]
        best = max(population, key=lambda ind: ind.fitness)

        stats = {
            "gen": gen,
            "best_fitness": best.fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_config_id": best.config.config_id,
            "population": [
                {
                    "config_id": ind.config.config_id,
                    "fitness": ind.fitness,
                    "scores": ind.scores,
                }
                for ind in population
            ],
        }
        self.generations.append(stats)

        for ind in population:
            self.all_evaluated.append({
                "config_id": ind.config.config_id,
                "fitness": ind.fitness,
                "generation": gen,
            })

        print(
            f"Gen {gen:2d}: best={best.fitness:.4f} "
            f"mean={stats['mean_fitness']:.4f} "
            f"std={stats['std_fitness']:.4f} "
            f"[{best.config.config_id}]"
        )

    def to_dict(self) -> dict:
        unique_configs = len(set(e["config_id"] for e in self.all_evaluated))
        best = max(self.all_evaluated, key=lambda e: e["fitness"])
        return {
            "generations": self.generations,
            "all_evaluated_unique": unique_configs,
            "best_overall": best,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_evolution(
    evaluate_fn: Callable[[List[AdapterConfig]], List[dict]],
    n_generations: int = 12,
    pop_size: int = 20,
    elite_count: int = 2,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.5,
    seed: int = 42,
) -> dict:
    """Run evolutionary search for adapter configurations.

    Args:
        evaluate_fn: Takes list of AdapterConfigs, returns list of result dicts
                     with 'scores' containing 'composite'.
        n_generations: Number of generations to run.
        pop_size: Population size.
        elite_count: Number of top individuals to pass unchanged.
        mutation_rate: Probability of mutating an offspring.
        crossover_rate: Probability of crossover vs mutation-only.
        seed: Random seed.

    Returns:
        Dict with best_config, best_fitness, logger data, final_population.
    """
    rng = np.random.default_rng(seed)
    logger = EvolutionLogger()

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

        if gen == n_generations - 1:
            break

        # Build next generation
        next_gen = []

        # Elitism
        for i in range(elite_count):
            elite = copy.deepcopy(population[i])
            next_gen.append(elite)

        # Fill remaining slots
        gen_seen = set(ind.config.config_id for ind in next_gen)
        attempts = 0
        while len(next_gen) < pop_size and attempts < pop_size * 5:
            attempts += 1

            if rng.random() < crossover_rate:
                p1 = tournament_select(population, k=3, rng=rng)
                p2 = tournament_select(population, k=3, rng=rng)
                child_cfg = crossover(p1.config, p2.config, rng)
                if rng.random() < mutation_rate:
                    child_cfg = mutate(child_cfg, rng)
            else:
                parent = tournament_select(population, k=3, rng=rng)
                child_cfg = mutate(parent.config, rng)

            if child_cfg.config_id not in gen_seen:
                gen_seen.add(child_cfg.config_id)
                next_gen.append(Individual(config=child_cfg, generation=gen + 1))

        # If still under pop_size, fill with random
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
    }
