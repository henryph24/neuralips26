"""Fine-tuning for validation subset."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from feasibility.config import AdapterConfig
from feasibility.model import build_adapted_model, _get_encoder_blocks, _get_hidden_dim


class LinearHead(nn.Module):
    """Simple linear head for forecasting or classification."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def finetune_forecasting(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 10,
    lr: float = 1e-3,
    forecast_horizon: int = 96,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Fine-tune adapter + linear head for forecasting on ETTh1.

    Uses last `forecast_horizon` timesteps as target, rest as input.
    Returns dict with 'mse' metric.
    """
    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)

    head = LinearHead(hidden_dim, forecast_horizon).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Split samples: input = first (seq_len - horizon), target = last horizon
    seq_len = samples.shape[1]
    input_len = seq_len - forecast_horizon
    X = samples[:, :input_len]
    Y = samples[:, input_len:]

    # Pad input to seq_len for MOMENT
    X_padded = np.zeros_like(samples)
    X_padded[:, :input_len] = X

    # 80/20 train/val split
    n = len(X_padded)
    split = int(0.8 * n)
    X_train, X_val = X_padded[:split], X_padded[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Extract features + train head
    model.eval()
    for epoch in range(n_epochs):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            with torch.no_grad():
                # Get last encoder block output
                features = []
                def capture_hook(module, input, output):
                    if isinstance(output, tuple):
                        features.append(output[0])
                    else:
                        features.append(output)

                h = encoder_blocks[-1].register_forward_hook(capture_hook)
                try:
                    model(x_enc=batch_x, input_mask=input_mask)
                except TypeError:
                    model(batch_x)
                h.remove()

                feat = features[0].mean(dim=1)  # (batch, hidden)

            pred = head(feat)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    head.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            features = []
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    features.append(output[0])
                else:
                    features.append(output)

            h = encoder_blocks[-1].register_forward_hook(capture_hook)
            try:
                model(x_enc=batch_x, input_mask=input_mask)
            except TypeError:
                model(batch_x)
            h.remove()

            feat = features[0].mean(dim=1)
            pred = head(feat)
            val_losses.append(criterion(pred, batch_y).item())

    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()

    return {"mse": float(np.mean(val_losses))}


def finetune_classification(
    adapter_cfg: AdapterConfig,
    samples: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Fine-tune adapter + linear classifier for classification.

    Returns dict with 'accuracy' metric.
    """
    model, cleanup_hooks = build_adapted_model(adapter_cfg, device)

    for param in model.parameters():
        param.requires_grad = False

    encoder_blocks = _get_encoder_blocks(model)
    hidden_dim = _get_hidden_dim(model)

    n_classes = len(np.unique(labels))
    head = LinearHead(hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 80/20 split
    n = len(samples)
    split = int(0.8 * n)
    X_train, X_val = samples[:split], samples[split:]
    y_train, y_val = labels[:split], labels[split:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model.eval()
    for epoch in range(n_epochs):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            with torch.no_grad():
                features = []
                def capture_hook(module, input, output):
                    if isinstance(output, tuple):
                        features.append(output[0])
                    else:
                        features.append(output)

                h = encoder_blocks[-1].register_forward_hook(capture_hook)
                try:
                    model(x_enc=batch_x, input_mask=input_mask)
                except TypeError:
                    model(batch_x)
                h.remove()
                feat = features[0].mean(dim=1)

            pred = head(feat)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            features = []
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    features.append(output[0])
                else:
                    features.append(output)

            h = encoder_blocks[-1].register_forward_hook(capture_hook)
            try:
                model(x_enc=batch_x, input_mask=input_mask)
            except TypeError:
                model(batch_x)
            h.remove()

            feat = features[0].mean(dim=1)
            pred = head(feat)
            correct += (pred.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)

    if cleanup_hooks:
        for h in cleanup_hooks:
            h.remove()

    return {"accuracy": correct / total if total > 0 else 0.0}
