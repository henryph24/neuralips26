"""Batched feature extraction pipeline."""

import numpy as np
import torch
from typing import Dict, Tuple

from feasibility.config import AdapterConfig
from feasibility.data import extract_trend
from feasibility.model import (
    build_adapted_model,
    register_feature_hooks,
)


def extract_features(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Extract features for a single adapter config.

    Args:
        adapter_cfg: adapter configuration
        samples: (n_samples, seq_len) numpy array
        device: torch device string
        batch_size: inference batch size

    Returns:
        Dict with keys:
            'feature': (n_samples, hidden_dim) - last layer, original input
            'first_feature': (n_samples, hidden_dim) - first layer, original input
            'trend_feature': (n_samples, hidden_dim) - last layer, trend input
    """
    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)

    # Extract features for original input
    feature, first_feature = _extract_layer_features(
        model, samples, device, batch_size
    )

    # Extract features for trend input
    samples_tensor = torch.from_numpy(samples).float()
    trend_samples = extract_trend(samples_tensor).numpy()
    trend_feature, _ = _extract_layer_features(
        model, trend_samples, device, batch_size
    )

    # Cleanup
    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()
    del model
    torch.cuda.empty_cache() if device != "cpu" else None

    return {
        "feature": feature,
        "first_feature": first_feature,
        "trend_feature": trend_feature,
    }


def _extract_layer_features(
    model: torch.nn.Module,
    samples: np.ndarray,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run forward pass and collect first/last layer features.

    Returns (last_layer_features, first_layer_features) as numpy arrays,
    mean-pooled across the time/patch dimension.
    """
    feature_store, hook_handles = register_feature_hooks(model)

    n_samples = samples.shape[0]

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.from_numpy(samples[start:end]).float().to(device)

            # MOMENT expects (batch, n_channels, seq_len) for reconstruction
            if batch.dim() == 2:
                batch = batch.unsqueeze(1)

            # Create input mask (all ones = no masking)
            input_mask = torch.ones(batch.shape[0], batch.shape[2], device=device)

            try:
                model(x_enc=batch, input_mask=input_mask)
            except TypeError:
                # Fallback: try without input_mask
                try:
                    model(batch)
                except Exception:
                    model(x_enc=batch)

    # Concatenate all captured features
    first_feats = torch.cat(feature_store["first_layer"], dim=0)  # (n, seq, hidden)
    last_feats = torch.cat(feature_store["last_layer"], dim=0)

    # Mean pool across time/patch dimension
    first_features = first_feats.mean(dim=1).numpy()  # (n, hidden)
    last_features = last_feats.mean(dim=1).numpy()

    # Cleanup hooks
    for h in hook_handles:
        h.remove()

    # Clear feature store
    feature_store["first_layer"].clear()
    feature_store["last_layer"].clear()

    return last_features, first_features
