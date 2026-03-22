"""Adapter search space definitions for TEMPLATE feasibility experiment."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from itertools import product


LAYER_PLACEMENTS = {
    "all": [0, 1, 2, 3, 4, 5, 6, 7],
    "first_half": [0, 1, 2, 3],
    "last_half": [4, 5, 6, 7],
}

LORA_RANKS = [2, 4, 8, 16, 32, 64]
TARGET_MODULE_SETS = {
    "qv": ["q", "v"],
    "qkvo": ["q", "k", "v", "o"],
}
BOTTLENECK_DIMS = [32, 64, 128]

# --- New search dimensions ---
UNFREEZE_STRATEGIES = ["frozen", "last2", "last4", "all"]
HEAD_TYPES = ["linear", "mlp1", "mlp2"]
POOLING_STRATEGIES = ["mean", "max", "last", "cls_mean_max"]


@dataclass
class AdapterConfig:
    adapter_type: str  # "lora", "bottleneck", "linear_probe"
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    target_modules_key: Optional[str] = None  # "qv" or "qkvo"
    layer_placement: str = "all"
    bottleneck_dim: Optional[int] = None
    unfreeze: str = "frozen"  # "frozen", "last2", "last4", "all"
    head_type: str = "linear"  # "linear", "mlp1", "mlp2"
    pooling: str = "mean"  # "mean", "max", "last", "cls_mean_max"
    config_id: str = ""

    @property
    def layer_indices(self) -> List[int]:
        return LAYER_PLACEMENTS[self.layer_placement]

    @property
    def target_modules(self) -> Optional[List[str]]:
        if self.target_modules_key is None:
            return None
        return TARGET_MODULE_SETS[self.target_modules_key]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_indices"] = self.layer_indices
        d["target_modules"] = self.target_modules
        return d

    @staticmethod
    def from_dict(d: dict) -> "AdapterConfig":
        filtered = {
            k: v for k, v in d.items()
            if k in AdapterConfig.__dataclass_fields__
        }
        return AdapterConfig(**filtered)


def _make_id(cfg: "AdapterConfig") -> str:
    """Generate a deterministic config_id from parameters."""
    parts = [cfg.adapter_type]
    if cfg.adapter_type == "lora":
        parts.append(f"r{cfg.lora_rank}")
        parts.append(cfg.target_modules_key or "qv")
        parts.append(cfg.layer_placement)
    elif cfg.adapter_type == "bottleneck":
        parts.append(f"d{cfg.bottleneck_dim}")
        parts.append(cfg.layer_placement)
    # Add new dimensions only if non-default
    if cfg.unfreeze != "frozen":
        parts.append(f"uf{cfg.unfreeze}")
    if cfg.head_type != "linear":
        parts.append(cfg.head_type)
    if cfg.pooling != "mean":
        parts.append(f"pool_{cfg.pooling}")
    return "_".join(parts)


def generate_configs() -> List[AdapterConfig]:
    configs = []

    # LoRA configs: 6 ranks × 2 target sets × 3 placements = 36
    for rank, tm_key, placement in product(LORA_RANKS, TARGET_MODULE_SETS, LAYER_PLACEMENTS):
        cfg = AdapterConfig(
            adapter_type="lora",
            lora_rank=rank,
            lora_alpha=rank * 2,
            target_modules_key=tm_key,
            layer_placement=placement,
            config_id=f"lora_r{rank}_{tm_key}_{placement}",
        )
        configs.append(cfg)

    # Bottleneck configs: 3 dims × 3 placements = 9
    for dim, placement in product(BOTTLENECK_DIMS, LAYER_PLACEMENTS):
        cfg = AdapterConfig(
            adapter_type="bottleneck",
            bottleneck_dim=dim,
            layer_placement=placement,
            config_id=f"bottleneck_d{dim}_{placement}",
        )
        configs.append(cfg)

    # Baseline: linear probe (no adapter)
    configs.append(AdapterConfig(
        adapter_type="linear_probe",
        config_id="linear_probe_baseline",
    ))

    return configs
