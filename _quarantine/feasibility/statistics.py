"""Compute scalar statistics from a single forward+backward pass for GP proxy search."""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from feasibility.config import AdapterConfig
from feasibility.model import (
    build_adapted_model,
    _get_encoder_blocks,
    _get_hidden_dim,
)
from feasibility.finetune import _build_head, _pool_features, _pooled_dim, _get_trainable_params
from feasibility.features import extract_features
from feasibility.scores import dl_score, pl_score, ta_score


def _safe_float(x: float) -> float:
    """Clip to [-1e6, 1e6], replace NaN/Inf with 0.0."""
    if not math.isfinite(x):
        return 0.0
    return max(-1e6, min(1e6, x))


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices (n, d)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def _register_cpu_hooks(encoder_blocks):
    """Register hooks that capture features detached to CPU (for activation stats)."""
    store = {}
    handles = []
    for idx, block in enumerate(encoder_blocks):
        store[idx] = []
        def capture(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    feat = output[0]
                else:
                    feat = output
                store[layer_idx].append(feat.detach().cpu())
            return hook_fn
        handles.append(block.register_forward_hook(capture(idx)))
    return store, handles


def _register_grad_hook(encoder_blocks, layer_idx):
    """Register a single hook that captures features ON-DEVICE with grad retained."""
    store = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            feat = output[0]
        else:
            feat = output
        store.append(feat)
    h = encoder_blocks[layer_idx].register_forward_hook(hook_fn)
    return store, h


def compute_all_statistics(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    labels: Optional[np.ndarray] = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute ~14 scalar statistics from one forward+backward pass.

    These are the atomic terminals for GP proxy formulas.
    """
    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)
    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)
    n_layers = len(encoder_blocks)

    # --- Prepare two mini-batches ---
    n = len(samples)
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    n_batch = min(batch_size, n // 2)
    batch1_idx = idx[:n_batch]
    batch2_idx = idx[n_batch : n_batch * 2]

    batch1 = torch.from_numpy(samples[batch1_idx]).float().unsqueeze(1).to(device)
    batch2 = torch.from_numpy(samples[batch2_idx]).float().unsqueeze(1).to(device)
    mask1 = torch.ones(batch1.shape[0], batch1.shape[2], device=device)
    mask2 = torch.ones(batch2.shape[0], batch2.shape[2], device=device)

    # =============================================
    # Phase 1: Forward pass (eval mode) for activation/geometry stats
    # Uses CPU-detached hooks for numpy analysis
    # =============================================
    cpu_store, cpu_hooks = _register_cpu_hooks(encoder_blocks)

    model.eval()
    with torch.no_grad():
        try:
            model(x_enc=batch1, input_mask=mask1)
        except TypeError:
            model(batch1)

    # Collect per-layer features (numpy, CPU)
    layer_features = {}
    for layer_idx, feats_list in cpu_store.items():
        if feats_list:
            feat = feats_list[0]  # (batch, seq, hidden)
            pooled = feat.mean(dim=1)  # (batch, hidden)
            layer_features[layer_idx] = pooled.numpy()

    # Remove CPU hooks — done with activation stats
    for h in cpu_hooks:
        h.remove()

    last_layer_feat = layer_features.get(n_layers - 1)
    first_layer_feat = layer_features.get(0)

    stats = {}

    # --- Activation statistics ---
    if last_layer_feat is not None:
        flat = last_layer_feat.flatten()
        stats["act_variance"] = float(np.var(flat))
        mu = np.mean(flat)
        centered = flat - mu
        var = np.var(flat)
        if var > 1e-12:
            stats["act_kurtosis"] = float(np.mean(centered ** 4) / (var ** 2) - 3.0)
        else:
            stats["act_kurtosis"] = 0.0
        stats["act_sparsity"] = float(np.mean(np.abs(flat) < 0.01))

        try:
            svs = np.linalg.svd(last_layer_feat, compute_uv=False)
            max_sv = svs[0] if len(svs) > 0 else 1.0
            nuclear = np.sum(svs)
            stats["feature_rank"] = float(nuclear / (max_sv + 1e-8))
        except np.linalg.LinAlgError:
            stats["feature_rank"] = 0.0
    else:
        stats["act_variance"] = 0.0
        stats["act_kurtosis"] = 0.0
        stats["act_sparsity"] = 0.0
        stats["feature_rank"] = 0.0

    if last_layer_feat is not None and first_layer_feat is not None:
        norm_last = np.linalg.norm(last_layer_feat)
        norm_first = np.linalg.norm(first_layer_feat)
        stats["feature_norm_ratio"] = float(norm_last / (norm_first + 1e-8))
    else:
        stats["feature_norm_ratio"] = 0.0

    # --- Layer CKA statistics ---
    sorted_layers = sorted(layer_features.keys())
    cka_values = []
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i + 1]
        cka = _linear_cka(layer_features[l1], layer_features[l2])
        cka_values.append(cka)

    if cka_values:
        stats["layer_cka_mean"] = float(np.mean(cka_values))
        stats["layer_cka_std"] = float(np.std(cka_values))
    else:
        stats["layer_cka_mean"] = 0.0
        stats["layer_cka_std"] = 0.0

    # =============================================
    # Phase 2: Forward+backward (train mode) for gradient stats
    # Uses on-device hook (keeps grad) on LAST layer only
    # =============================================
    pooled_dim = _pooled_dim(hidden_dim, adapter_cfg.pooling)
    is_classification = labels is not None
    forecast_horizon = 96

    if is_classification:
        n_classes = len(np.unique(labels))
        head = _build_head(adapter_cfg.head_type, pooled_dim, n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        batch1_labels = torch.from_numpy(labels[batch1_idx]).long().to(device)
    else:
        head = _build_head(adapter_cfg.head_type, pooled_dim, forecast_horizon).to(device)
        criterion = nn.MSELoss()
        batch1_targets = torch.from_numpy(
            samples[batch1_idx, -forecast_horizon:]
        ).float().to(device)

    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name or param.requires_grad:
            param.requires_grad_(True)

    # --- Batch 1 forward+backward ---
    grad_store1, grad_hook1 = _register_grad_hook(encoder_blocks, n_layers - 1)

    try:
        model(x_enc=batch1, input_mask=mask1)
    except TypeError:
        model(batch1)

    if grad_store1:
        feat_grad = grad_store1[0]  # on-device, with grad graph
        feat_pooled = _pool_features(feat_grad, adapter_cfg.pooling)
        logits = head(feat_pooled)

        if is_classification:
            loss = criterion(logits, batch1_labels)
        else:
            loss = criterion(logits, batch1_targets)

        loss.backward()

        # Gradient statistics
        grad_norms = []
        trainable_params = _get_trainable_params(model, adapter_cfg, head)
        for p in trainable_params:
            if p.grad is not None:
                grad_norms.append(p.grad.detach().cpu().flatten())

        if grad_norms:
            all_grads = torch.cat(grad_norms).numpy()
            stats["grad_norm"] = float(np.linalg.norm(all_grads))
            grad_abs = np.abs(all_grads)
            grad_std = np.std(grad_abs)
            stats["grad_snr"] = float(np.mean(grad_abs) / (grad_std + 1e-8))
        else:
            stats["grad_norm"] = 0.0
            stats["grad_snr"] = 0.0
    else:
        stats["grad_norm"] = 0.0
        stats["grad_snr"] = 0.0

    grad_hook1.remove()

    # --- Save batch 1 grads, then batch 2 for grad_conflict ---
    batch1_grad_vec = []
    trainable_params = _get_trainable_params(model, adapter_cfg, head)
    for p in trainable_params:
        if p.grad is not None:
            batch1_grad_vec.append(p.grad.detach().cpu().flatten())

    model.zero_grad()
    head.zero_grad()

    grad_store2, grad_hook2 = _register_grad_hook(encoder_blocks, n_layers - 1)

    try:
        model(x_enc=batch2, input_mask=mask2)
    except TypeError:
        model(batch2)

    if grad_store2:
        feat_grad2 = grad_store2[0]
        feat_pooled2 = _pool_features(feat_grad2, adapter_cfg.pooling)
        logits2 = head(feat_pooled2)

        if is_classification:
            batch2_labels = torch.from_numpy(labels[batch2_idx]).long().to(device)
            loss2 = criterion(logits2, batch2_labels)
        else:
            batch2_targets = torch.from_numpy(
                samples[batch2_idx, -forecast_horizon:]
            ).float().to(device)
            loss2 = criterion(logits2, batch2_targets)

        loss2.backward()

        batch2_grad_vec = []
        for p in trainable_params:
            if p.grad is not None:
                batch2_grad_vec.append(p.grad.detach().cpu().flatten())

        if batch1_grad_vec and batch2_grad_vec:
            g1 = torch.cat(batch1_grad_vec).numpy()
            g2 = torch.cat(batch2_grad_vec).numpy()
            dot = np.dot(g1, g2)
            norm1 = np.linalg.norm(g1)
            norm2 = np.linalg.norm(g2)
            stats["grad_conflict"] = float(dot / (norm1 * norm2 + 1e-8))
        else:
            stats["grad_conflict"] = 0.0
    else:
        stats["grad_conflict"] = 0.0

    grad_hook2.remove()

    # --- Parameter count ---
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_trainable += sum(p.numel() for p in head.parameters())
    stats["param_count_log"] = float(math.log10(max(1, total_trainable)))

    # --- TEMPLATE scores ---
    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()

    try:
        features = extract_features(adapter_cfg, samples[:min(500, n)], device=device, batch_size=batch_size)
        stats["dl_score"] = float(dl_score(features["feature"], features["trend_feature"]))
        stats["pl_score"] = float(pl_score(features["feature"]))
        stats["ta_score"] = float(ta_score(features["feature"], features["first_feature"], device=device))
    except Exception:
        stats["dl_score"] = 0.0
        stats["pl_score"] = 0.0
        stats["ta_score"] = 0.0

    return {k: _safe_float(v) for k, v in stats.items()}
