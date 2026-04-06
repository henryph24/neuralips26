"""T3.A — AAS macro-expert pool for RR-MoA.

The canonical RR-MoA experts (mean, last, max, attn, conv1d in
``scripts/run_rr_moa.py``) are textbook pooling heads; they do not exercise
any cross-domain motif that AAS discovered in Section 3. The reviewer flagged
this as a contribution-integration gap (W1): the paper claims AAS populates
the RR-MoA expert pool, but the pool actually used throughout Tables 3--7 is
not populated from AAS.

This module closes that gap by exposing the top-5 distinct macro
architectures discovered by AAS on ETTh1 seed 42 (gpt-4o model,
``results/code_evolution/validated_ETTh1_42_gpt-4o.json``) as ``nn.Module``
expert classes that conform to RR-MoA's expert contract
``__init__(d_model, output_dim, hidden=None) -> forward(hidden_states)``.

The code topology of each class is a faithful transcription of the
corresponding AAS-evolved adapter; the only differences are cosmetic: a
shared two-arg constructor signature so that the RR-MoA factory can
instantiate them uniformly, and a ``hidden`` kwarg accepted (but ignored)
so that substitution with canonical heads is signature-compatible.

Source file: ``results/code_evolution/validated_ETTh1_42_gpt-4o.json``
Original reasoning strings are preserved in module docstrings so that the
provenance is unambiguous to a reviewer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNMeanLinearExpert(nn.Module):
    """AAS rank 0 -- BatchNorm over d_model then mean-pool and linear.

    Reasoning (LLM): 'Inspiration from the high performance of simple linear
    layers with normalization. This approach attempts batch normalization to
    stabilize learning.'

    Validated on ETTh1 seed 42 (gpt-4o): mse_15ep=0.5952, params=50272.
    """

    def __init__(self, d_model: int, output_dim: int, hidden: int = None):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, d_model) -> (B, d_model, T) for BN
        x = hidden_states.transpose(1, 2)
        x = self.bn(x)
        x = x.mean(dim=2)
        return self.linear(x)


class MultiScaleConvResidualExpert(nn.Module):
    """AAS rank 1 -- multi-scale convolutions (k=3,5,7) averaged with a 1x1 BN residual.

    Reasoning (LLM): 'Employ a more complex residual connection around a
    multi-scale convolutional stack for richer feature learning.'

    Validated on ETTh1 seed 42 (gpt-4o): mse_15ep=0.5915, params=104992.
    """

    def __init__(self, d_model: int, output_dim: int, hidden: int = None, ch: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(ch, ch, kernel_size=7, padding=3)
        self.residual = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1),
            nn.BatchNorm1d(ch),
        )
        self.fc = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states).transpose(1, 2)  # (B, ch, T)
        res = self.residual(x)
        x1 = F.gelu(self.conv1(x))
        x2 = F.gelu(self.conv2(x))
        x3 = F.gelu(self.conv3(x))
        x = (x1 + x2 + x3) / 3 + res
        return self.fc(x.mean(dim=2))


class Conv1dBNResidualExpert(nn.Module):
    """AAS rank 2 -- Conv1d + BatchNorm + 1x1 residual (ResNet-style stem).

    Reasoning (LLM): 'Applying residual connections and global average
    pooling right before linear mapping can enhance gradient flow and
    stability.'

    Validated on ETTh1 seed 42 (gpt-4o): mse_15ep=0.5640, params=38688.
    """

    def __init__(self, d_model: int, output_dim: int, hidden: int = None, ch: int = 48):
        super().__init__()
        self.proj = nn.Linear(d_model, ch)
        self.conv1d = nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(ch)
        self.residual = nn.Conv1d(ch, ch, kernel_size=1)
        self.fc = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states).transpose(1, 2)
        res = self.residual(x)
        x = F.relu(self.bn(self.conv1d(x)) + res)
        x = x.mean(dim=2)
        return self.fc(x)


class DepthwiseSeparableExpert(nn.Module):
    """AAS rank 3 -- depthwise separable conv (MobileNet motif), lowest param count.

    Reasoning (LLM): 'This design introduces depthwise separable
    convolutions to exploit the spatial structure of the 768 features. This
    should capture local patterns effectively while keeping parameter count
    low.'

    Validated on ETTh1 seed 42 (gpt-4o): mse_15ep=0.5817, params=20768.
    """

    def __init__(self, d_model: int, output_dim: int, hidden: int = None, ch: int = 32):
        super().__init__()
        self.proj = nn.Linear(d_model, ch)
        self.depthwise_conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pointwise_conv = nn.Conv1d(ch, ch, kernel_size=1)
        self.fc = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states).transpose(1, 2)
        x = F.relu(self.pointwise_conv(self.depthwise_conv(x)))
        x = x.mean(dim=2)
        return self.fc(x)


class GatedConvResidualExpert(nn.Module):
    """AAS rank 4 -- sigmoid-gated conv + residual (highway / GLU motif).

    Reasoning (LLM): 'Incorporating a simple gating mechanism to allow the
    model to modulate between skip connections and transformations from
    convolutions.'

    Validated on ETTh1 seed 42 (gpt-4o): mse_15ep=0.6155, params=55584.
    """

    def __init__(self, d_model: int, output_dim: int, hidden: int = None, ch: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, ch)
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states).transpose(1, 2)
        conv_x = F.relu(self.conv(x))
        gated_x = self.gate(x) * conv_x + x
        x = gated_x.mean(dim=2)
        return self.fc(x)


# Ordered list consumed by RR-MoA when ``--expert-pool macro`` is selected.
MACRO_EXPERT_CLASSES = [
    BNMeanLinearExpert,
    MultiScaleConvResidualExpert,
    Conv1dBNResidualExpert,
    DepthwiseSeparableExpert,
    GatedConvResidualExpert,
]
MACRO_EXPERT_NAMES = [
    "bn_mean_linear",
    "multiscale_conv_residual",
    "conv1d_bn_residual",
    "depthwise_separable",
    "gated_conv_residual",
]
