"""Canonical adapter pool and code-string validator.

Provides the five hand-coded adapter architectures used as fixed baselines
across the experiment runners (linear / mlp2 / last-token-mlp / attention-pool
/ conv1d), plus a small validator that exec's an adapter code string and
checks shape, parameter count, and that it subclasses ``nn.Module``.

The runners pass these adapter code strings to ``train_adapter_from_code``
inside each script's training loop; the strings are exec'd into a sandboxed
namespace at call time so the same adapter pool can be shared without
duplicating ``nn.Module`` class definitions across runners.
"""

import torch
import torch.nn as nn


# --- Seed adapters (5 hand-coded baseline architectures) ---

SEED_ADAPTERS = [
    # 1. MeanPool + Linear (simplest baseline)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        return self.linear(pooled)""",

    # 2. MeanPool + MLP2
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid // 2, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)""",

    # 3. LastToken + MLP
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last = hidden_states[:, -1, :]
        return self.net(last)""",

    # 4. Attention pooling + Linear (learnable weighted pooling)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(d_model, 1)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.attn_weights(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return self.linear(pooled)""",

    # 5. Conv1d downsample + Linear (temporal pattern extraction)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        ch = 128
        self.proj = nn.Linear(d_model, ch)
        self.conv = nn.Conv1d(ch, ch, kernel_size=8, stride=4)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states)
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.linear(x)""",
]

MAX_ADAPTER_PARAMS = 500_000


# --- Validation ---

def validate_adapter_code(code: str, d_model: int = 768, output_dim: int = 96) -> dict:
    """Validate adapter code by exec + instantiate + dummy forward pass.

    Returns ``{"valid": bool, "error": str|None, "param_count": int}``.
    """
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "math": __import__("math"),
    }

    try:
        exec(code, namespace)
    except Exception as e:
        return {"valid": False, "error": f"exec error: {e}", "param_count": 0}

    if "Adapter" not in namespace:
        return {"valid": False, "error": "No 'Adapter' class defined", "param_count": 0}

    AdapterClass = namespace["Adapter"]
    if not (isinstance(AdapterClass, type) and issubclass(AdapterClass, nn.Module)):
        return {"valid": False, "error": "Adapter is not an nn.Module subclass", "param_count": 0}

    try:
        adapter = AdapterClass(d_model, output_dim)
    except Exception as e:
        return {"valid": False, "error": f"__init__ error: {e}", "param_count": 0}

    param_count = sum(p.numel() for p in adapter.parameters())
    if param_count > MAX_ADAPTER_PARAMS:
        return {
            "valid": False,
            "error": f"Too many params: {param_count} > {MAX_ADAPTER_PARAMS}",
            "param_count": param_count,
        }

    try:
        dummy = torch.randn(2, 512, d_model)
        with torch.no_grad():
            out = adapter(dummy)
    except Exception as e:
        return {"valid": False, "error": f"forward error: {e}", "param_count": param_count}

    if out.shape != (2, output_dim):
        return {
            "valid": False,
            "error": f"Wrong output shape: {out.shape}, expected (2, {output_dim})",
            "param_count": param_count,
        }

    return {"valid": True, "error": None, "param_count": param_count}
