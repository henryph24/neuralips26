"""Fine-tuning for validation subset."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from feasibility.config import AdapterConfig
from feasibility.model import build_adapted_model, _get_encoder_blocks, _get_hidden_dim, _disable_gradient_checkpointing


# --- Head architectures ---

class LinearHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, n_hidden: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        mid_dim = hidden_dim // 2
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = mid_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_head(head_type: str, hidden_dim: int, output_dim: int) -> nn.Module:
    if head_type == "linear":
        return LinearHead(hidden_dim, output_dim)
    elif head_type == "mlp1":
        return MLPHead(hidden_dim, output_dim, n_hidden=1)
    elif head_type == "mlp2":
        return MLPHead(hidden_dim, output_dim, n_hidden=2)
    raise ValueError(f"Unknown head_type: {head_type}")


# --- Temporal pooling ---

def _pool_features(feat: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool (batch, seq, hidden) -> (batch, hidden_out).

    For cls_mean_max, output dim is 2*hidden.
    """
    if pooling == "mean":
        return feat.mean(dim=1)
    elif pooling == "max":
        return feat.max(dim=1).values
    elif pooling == "last":
        return feat[:, -1, :]
    elif pooling == "cls_mean_max":
        return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=-1)
    raise ValueError(f"Unknown pooling: {pooling}")


def _pooled_dim(hidden_dim: int, pooling: str) -> int:
    if pooling == "cls_mean_max":
        return hidden_dim * 2
    return hidden_dim


# --- Trainable param collection ---

def _get_trainable_params(model, adapter_cfg, head):
    """Collect trainable parameters: unfrozen backbone + adapter + head."""
    params = list(head.parameters())

    if adapter_cfg.adapter_type == "lora":
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                params.append(param)

    # Unfrozen backbone params (use id() to avoid tensor comparison)
    param_ids = {id(p) for p in params}
    for param in model.parameters():
        if param.requires_grad and id(param) not in param_ids:
            params.append(param)
            param_ids.add(id(param))

    return params


# --- Feature extraction helper ---

def _extract_features_batch(model, encoder_blocks, batch_x, input_mask, backbone_type="moment"):
    """Run forward pass and capture last encoder/decoder block output.

    Supports multiple backbone types:
    - moment: MOMENT pipeline (x_enc + input_mask)
    - chronos: Chronos T5 encoder (tokenize then encode)
    - moirai: Moirai encoder (patch projection then encode)
    - timer: Timer-XL decoder-only (raw input, internal patching)
    """
    features = []

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            features.append(output[0])
        else:
            features.append(output)

    h = encoder_blocks[-1].register_forward_hook(capture_hook)

    if backbone_type == "chronos":
        _forward_chronos(model, encoder_blocks, batch_x)
    elif backbone_type == "moirai":
        _forward_moirai(model, encoder_blocks, batch_x)
    elif backbone_type == "timer":
        _forward_timer(model, batch_x)
    else:
        # MOMENT and generic fallback
        try:
            model(x_enc=batch_x, input_mask=input_mask)
        except TypeError:
            try:
                model(batch_x)
            except TypeError:
                model(batch_x, input_mask)

    h.remove()
    return features[0]


def _forward_chronos(model, encoder_blocks, batch_x):
    """Forward pass for Chronos T5 encoder.

    Patches the raw time series into (B, n_patches, patch_size) chunks,
    then pads to d_model — same strategy as Moirai forward.
    The hook on encoder_blocks[-1] captures the hidden states.
    """
    import torch
    # batch_x is (B, 1, 512) — squeeze to (B, 512)
    x = batch_x.squeeze(1)

    encoder = model.encoder
    d_model = encoder.config.d_model
    patch_size = 32
    n_patches = x.shape[1] // patch_size  # 512 / 32 = 16

    # Reshape into patches: (B, n_patches, patch_size)
    patches = x[:, :n_patches * patch_size].reshape(x.shape[0], n_patches, patch_size)

    # Pad patches to d_model (same approach as Moirai forward)
    inputs_embeds = torch.nn.functional.pad(patches, (0, d_model - patch_size))

    # Forward through encoder — no torch.no_grad() so adapter gradients flow
    encoder(inputs_embeds=inputs_embeds)


def _forward_moirai(model, encoder_blocks, batch_x):
    """Forward pass for Moirai encoder.

    Discovers d_model dynamically (supports small/base/large variants).
    The hook on encoder_blocks[-1] captures the output automatically.
    """
    import torch
    from feasibility.model import _get_hidden_dim
    # batch_x is (B, 1, 512) — squeeze to (B, 512)
    x = batch_x.squeeze(1)

    # Discover d_model from the model itself (not hardcoded)
    d_model = _get_hidden_dim(model)
    patch_size = 32
    n_patches = x.shape[1] // patch_size  # 512 / 32 = 16

    # Reshape into patches: (B, n_patches, patch_size)
    patches = x[:, :n_patches * patch_size].reshape(x.shape[0], n_patches, patch_size)

    # Project patches to d_model using model's pre-trained input projection
    if hasattr(model, 'in_proj') and hasattr(model, 'patch_sizes'):
        # Moirai's MultiInSizeLinear expects (x_padded_to_max_feat, in_feat_size)
        max_feat = max(model.patch_sizes)
        patches_padded = torch.nn.functional.pad(patches, (0, max_feat - patch_size))
        in_feat_size = torch.full(patches.shape[:2], patch_size,
                                  dtype=torch.long, device=patches.device)
        projected = model.in_proj(patches_padded, in_feat_size)
    else:
        projected = torch.nn.functional.pad(patches, (0, d_model - patch_size))

    # Forward through encoder — hook on encoder_blocks[-1] captures output
    encoder = model.encoder
    encoder(projected)


def _forward_timer(model, batch_x):
    """Forward pass for Timer-XL decoder-only Transformer.

    Timer-XL internally patches input via input_token_len=96 and runs
    through 8 TimerDecoderLayer blocks. The hook on the last block
    captures hidden states of shape (B, n_tokens, 1024) where
    n_tokens = L // 96.  Input: batch_x (B, 1, L) or (B, L).
    """
    x = batch_x.squeeze(1) if batch_x.dim() == 3 else batch_x  # (B, L)
    model(x)  # Hook captures output from last decoder layer


# --- Forecasting ---

def finetune_forecasting(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 10,
    lr: float = 1e-3,
    forecast_horizon: int = 96,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Fine-tune adapter + head for forecasting."""
    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)

    for param in model.parameters():
        param.requires_grad = False

    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)
    pooled = _pooled_dim(hidden_dim, adapter_cfg.pooling)

    head = _build_head(adapter_cfg.head_type, pooled, forecast_horizon).to(device)
    trainable_params = _get_trainable_params(model, adapter_cfg, head)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()

    # Support both legacy (samples=512) and multi-horizon (samples=512+H)
    seq_len = samples.shape[1]
    if seq_len > 512:
        input_len = 512
        X = samples[:, :input_len]
        Y = samples[:, input_len:input_len + forecast_horizon]
    else:
        input_len = seq_len - forecast_horizon
        X = samples[:, :input_len]
        Y = samples[:, input_len:]

    X_padded = np.zeros((len(X), 512), dtype=samples.dtype)
    X_padded[:, :input_len] = X

    n = len(X_padded)
    split = int(0.8 * n)
    X_train, X_val = X_padded[:split], X_padded[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()),
        batch_size=batch_size,
    )

    has_trainable_backbone = adapter_cfg.unfreeze != "frozen" or adapter_cfg.adapter_type in ("lora", "bottleneck")

    for epoch in range(n_epochs):
        model.train() if has_trainable_backbone else model.eval()
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            feat = _pool_features(feat, adapter_cfg.pooling)

            pred = head(feat)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    head.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            feat = _pool_features(feat, adapter_cfg.pooling)
            pred = head(feat)
            val_losses.append(criterion(pred, batch_y).item())

    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()

    return {"mse": float(np.mean(val_losses))}


# --- Classification ---

def finetune_classification(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Fine-tune adapter + head for classification."""
    from sklearn.model_selection import train_test_split

    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)

    for param in model.parameters():
        param.requires_grad = False

    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)
    pooled = _pooled_dim(hidden_dim, adapter_cfg.pooling)

    n_classes = len(np.unique(labels))
    head = _build_head(adapter_cfg.head_type, pooled, n_classes).to(device)
    trainable_params = _get_trainable_params(model, adapter_cfg, head)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=batch_size,
    )

    has_trainable_backbone = adapter_cfg.unfreeze != "frozen" or adapter_cfg.adapter_type in ("lora", "bottleneck")

    for epoch in range(n_epochs):
        model.train() if has_trainable_backbone else model.eval()
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            feat = _pool_features(feat, adapter_cfg.pooling)

            pred = head(feat)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            feat = _pool_features(feat, adapter_cfg.pooling)
            pred = head(feat)
            correct += (pred.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)

    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()

    return {"accuracy": correct / total if total > 0 else 0.0}
