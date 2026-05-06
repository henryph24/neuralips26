"""Macro-expert pool for RR-MoA (auxiliary expert pool used by the macro-pool ablation).

The canonical RR-MoA experts (mean, last, max, attn, conv1d in
``scripts/run_rr_moa.py``) are textbook pooling heads. This module exposes
five additional macro architectures used as the alternative expert pool in
the macro-pool ablation, conforming to RR-MoA's expert contract:
``__init__(d_model, output_dim, hidden=None) -> forward(hidden_states)``.

Each class accepts a ``hidden`` kwarg (ignored if not used) so that
substitution between the canonical and macro pools is signature-compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNMeanLinearExpert(nn.Module):
    """Macro rank 0 -- BatchNorm over d_model then mean-pool and linear.

    Design rationale: 'Inspiration from the high performance of simple linear
    layers with normalization. This approach attempts batch normalization to
    stabilize learning.'

    Reference 15-epoch MSE on ETTh1 seed 42: 0.5952 (params=50272).
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
    """Macro rank 1 -- multi-scale convolutions (k=3,5,7) averaged with a 1x1 BN residual.

    Design rationale: 'Employ a more complex residual connection around a
    multi-scale convolutional stack for richer feature learning.'

    Reference 15-epoch MSE on ETTh1 seed 42: 0.5915 (params=104992).
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
    """Macro rank 2 -- Conv1d + BatchNorm + 1x1 residual (ResNet-style stem).

    Design rationale: 'Applying residual connections and global average
    pooling right before linear mapping can enhance gradient flow and
    stability.'

    Reference 15-epoch MSE on ETTh1 seed 42: 0.5640 (params=38688).
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
    """Macro rank 3 -- depthwise separable conv (MobileNet motif), lowest param count.

    Design rationale: 'This design introduces depthwise separable
    convolutions to exploit the spatial structure of the 768 features. This
    should capture local patterns effectively while keeping parameter count
    low.'

    Reference 15-epoch MSE on ETTh1 seed 42: 0.5817 (params=20768).
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
    """Macro rank 4 -- sigmoid-gated conv + residual (highway / GLU motif).

    Design rationale: 'Incorporating a simple gating mechanism to allow the
    model to modulate between skip connections and transformations from
    convolutions.'

    Reference 15-epoch MSE on ETTh1 seed 42: 0.6155 (params=55584).
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
