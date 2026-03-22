"""MOMENT loading, adapter attachment, and hook registration."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from functools import partial

from feasibility.config import AdapterConfig


# Will be populated after first model load via discover_module_names()
_MOMENT_MODULE_MAP = None


def load_moment(device: str = "cpu") -> nn.Module:
    """Load MOMENT-small pretrained model."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={"task_name": "reconstruction"},
    )
    model.init()
    model = model.to(device)
    model.eval()
    return model


def discover_module_names(model: nn.Module) -> Dict[str, List[str]]:
    """Discover T5 encoder module paths for LoRA targeting.

    Returns a dict mapping short names (q, k, v, o) to full module paths.
    """
    global _MOMENT_MODULE_MAP
    module_map = {"q": [], "k": [], "v": [], "o": []}

    for name, _ in model.named_modules():
        for key in module_map:
            # T5 attention modules typically end with .q, .k, .v, .o
            if name.endswith(f".{key}"):
                module_map[key].append(name)

    _MOMENT_MODULE_MAP = module_map
    return module_map


def get_target_module_names(
    module_map: Dict[str, List[str]],
    target_keys: List[str],
    layer_indices: List[int],
) -> List[str]:
    """Get full module paths for given target keys filtered by layer indices.

    Args:
        module_map: from discover_module_names()
        target_keys: e.g. ["q", "v"]
        layer_indices: e.g. [0, 1, 2] for first-half

    Returns:
        List of full module path strings
    """
    target_names = []
    for key in target_keys:
        modules = module_map[key]
        for idx in layer_indices:
            if idx < len(modules):
                target_names.append(modules[idx])
    return target_names


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: down_proj -> ReLU -> up_proj with residual."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.activation = nn.ReLU()
        # Initialize up_proj near zero for small initial perturbation
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x


def _patch_config_for_peft(model: nn.Module):
    """Patch MOMENT's NamespaceWithDefaults config to support dict-like .get() for peft."""
    cfg = getattr(model, "config", None)
    if cfg is not None and not hasattr(cfg, "get"):
        def _get(key, default=None):
            return getattr(cfg, key, default)
        cfg.get = _get


def attach_lora(model: nn.Module, adapter_cfg: AdapterConfig, module_map: Dict[str, List[str]]) -> nn.Module:
    """Attach LoRA adapter to MOMENT model using peft."""
    from peft import LoraConfig, get_peft_model

    _patch_config_for_peft(model)

    target_names = get_target_module_names(
        module_map,
        adapter_cfg.target_modules,
        adapter_cfg.layer_indices,
    )

    if not target_names:
        raise ValueError(f"No target modules found for config {adapter_cfg.config_id}")

    lora_config = LoraConfig(
        r=adapter_cfg.lora_rank,
        lora_alpha=adapter_cfg.lora_alpha,
        target_modules=target_names,
        init_lora_weights="gaussian",
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    # peft's gaussian init keeps lora_B=0, making adapter a no-op.
    # Randomize lora_B with small std so different configs produce different features.
    # Use config-derived seed for reproducibility across runs.
    seed = hash(adapter_cfg.config_id) % (2**31)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                device = param.device
                rng = torch.Generator(device=device).manual_seed(seed)
                param.normal_(0, 0.1, generator=rng)

    return peft_model


def attach_bottleneck(
    model: nn.Module,
    adapter_cfg: AdapterConfig,
    device: str = "cpu",
) -> Tuple[nn.Module, List[BottleneckAdapter]]:
    """Attach bottleneck adapters to specified encoder blocks via hooks.

    Returns the model and list of adapter modules (for parameter counting).
    """
    adapters = []
    hooks = []

    # Find encoder blocks
    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)

    for idx in adapter_cfg.layer_indices:
        if idx < len(encoder_blocks):
            adapter = BottleneckAdapter(hidden_dim, adapter_cfg.bottleneck_dim).to(device)
            adapters.append(adapter)

            def make_hook(adpt):
                def hook_fn(module, input, output):
                    # T5 block output is typically a tuple; modify the hidden states
                    if isinstance(output, tuple):
                        modified = adpt(output[0])
                        return (modified,) + output[1:]
                    return adpt(output)
                return hook_fn

            h = encoder_blocks[idx].register_forward_hook(make_hook(adapter))
            hooks.append(h)

    return model, adapters, hooks


def _get_encoder_blocks(model: nn.Module) -> list:
    """Navigate MOMENT model structure to find encoder blocks."""
    # Try common paths in MOMENT architecture
    for path_fn in [
        lambda m: m.encoder.block,
        lambda m: m.model.encoder.block,
        lambda m: m.backbone.encoder.block,
    ]:
        try:
            blocks = list(path_fn(model))
            return blocks
        except AttributeError:
            continue

    # Fallback: search for 'block' in named modules
    for name, module in model.named_modules():
        if hasattr(module, '__len__') and 'block' in name.lower():
            return list(module)

    raise RuntimeError("Could not find encoder blocks in model. Run discover_module_names() first to inspect architecture.")


def _get_hidden_dim(model: nn.Module) -> int:
    """Get hidden dimension from model config."""
    for attr in ['d_model', 'hidden_size']:
        # Check model config
        for cfg_attr in ['config', 'model_config']:
            cfg = getattr(model, cfg_attr, None)
            if cfg is not None:
                dim = getattr(cfg, attr, None)
                if dim is not None:
                    return dim
        # Check model directly
        dim = getattr(model, attr, None)
        if dim is not None:
            return dim

    # Fallback: inspect first linear layer
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return module.in_features

    raise RuntimeError("Could not determine hidden dimension")


def _disable_gradient_checkpointing(model: nn.Module):
    """Disable gradient checkpointing so forward hooks fire in train mode."""
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False


def _apply_unfreeze(model: nn.Module, unfreeze: str):
    """Selectively unfreeze encoder blocks based on strategy."""
    if unfreeze == "frozen":
        return
    encoder_blocks = _get_encoder_blocks(model)
    n_blocks = len(encoder_blocks)
    if unfreeze == "last2":
        unfreeze_from = max(0, n_blocks - 2)
    elif unfreeze == "last4":
        unfreeze_from = max(0, n_blocks - 4)
    elif unfreeze == "all":
        unfreeze_from = 0
    else:
        return
    for i in range(unfreeze_from, n_blocks):
        for param in encoder_blocks[i].parameters():
            param.requires_grad = True


def build_adapted_model(
    adapter_cfg: AdapterConfig,
    device: str = "cpu",
) -> Tuple[nn.Module, Optional[list]]:
    """Build MOMENT model with specified adapter config.

    Returns (model, cleanup_hooks) where cleanup_hooks should be called
    to remove bottleneck hooks when done.
    """
    model = load_moment(device)

    # Disable gradient checkpointing so forward hooks fire in train mode
    _disable_gradient_checkpointing(model)

    if adapter_cfg.adapter_type == "linear_probe":
        _apply_unfreeze(model, adapter_cfg.unfreeze)
        return model, None

    module_map = discover_module_names(model)

    if adapter_cfg.adapter_type == "lora":
        model = attach_lora(model, adapter_cfg, module_map)
        _disable_gradient_checkpointing(model)
        _apply_unfreeze(model, adapter_cfg.unfreeze)
        return model, None

    if adapter_cfg.adapter_type == "bottleneck":
        model, adapters, hooks = attach_bottleneck(model, adapter_cfg, device)
        _apply_unfreeze(model, adapter_cfg.unfreeze)
        return model, hooks

    raise ValueError(f"Unknown adapter type: {adapter_cfg.adapter_type}")


def register_feature_hooks(model: nn.Module) -> Tuple[Dict[str, list], list]:
    """Register forward hooks on first and last encoder blocks.

    Returns (feature_store, hook_handles).
    feature_store has keys 'first_layer' and 'last_layer', each a list
    that accumulates outputs per forward pass.
    """
    feature_store = {"first_layer": [], "last_layer": []}
    hook_handles = []

    encoder_blocks = _get_encoder_blocks(model)

    def capture(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output
            feature_store[name].append(feat.detach().cpu())
        return hook_fn

    h1 = encoder_blocks[0].register_forward_hook(capture("first_layer"))
    h2 = encoder_blocks[-1].register_forward_hook(capture("last_layer"))
    hook_handles.extend([h1, h2])

    return feature_store, hook_handles


def register_all_layer_hooks(model: nn.Module) -> Tuple[Dict[int, list], list]:
    """Register forward hooks on ALL encoder blocks.

    Returns (feature_store, hook_handles).
    feature_store maps layer index -> list of captured outputs.
    """
    feature_store = {}
    hook_handles = []

    encoder_blocks = _get_encoder_blocks(model)

    for idx, block in enumerate(encoder_blocks):
        feature_store[idx] = []

        def capture(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    feat = output[0]
                else:
                    feat = output
                feature_store[layer_idx].append(feat.detach().cpu())
            return hook_fn

        h = block.register_forward_hook(capture(idx))
        hook_handles.append(h)

    return feature_store, hook_handles
