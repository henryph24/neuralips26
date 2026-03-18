"""Dataset download and preprocessing for feasibility experiment."""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen


ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
MOMENT_SEQ_LEN = 512


def extract_trend(X: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
    """Extract trend via 1D average pooling (paper Eq. 1).

    Args:
        X: (batch, seq_len) tensor
        kernel_size: smoothing window size

    Returns:
        Trend tensor of same shape as X
    """
    if X.dim() == 2:
        X = X.unsqueeze(1)  # (batch, 1, seq_len)
    pad = kernel_size // 2
    X_padded = F.pad(X, (pad, pad), mode="reflect")
    trend = F.avg_pool1d(X_padded, kernel_size, stride=1)
    return trend.squeeze(1)


def load_etth1(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    """Download ETTh1 and create sliding windows, channel-independent.

    Returns dict with 'samples' (n, seq_len) numpy array and 'name'.
    """
    response = urlopen(ETTH1_URL)
    df = pd.read_csv(io.BytesIO(response.read()))
    # Drop date column, keep 7 feature columns
    values = df.iloc[:, 1:].values.astype(np.float32)  # (T, 7)

    # Sliding windows
    n_timesteps, n_channels = values.shape
    windows = []
    for start in range(0, n_timesteps - seq_len + 1, stride):
        window = values[start : start + seq_len]  # (seq_len, 7)
        windows.append(window)
    windows = np.stack(windows)  # (n_windows, seq_len, 7)

    # Channel-independent: reshape to (n_windows * n_channels, seq_len)
    n_windows = windows.shape[0]
    samples = windows.transpose(0, 2, 1).reshape(n_windows * n_channels, seq_len)

    # Normalize per-sample
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples.T).T.astype(np.float32)

    return {"samples": samples, "name": "ETTh1", "task": "forecasting"}


def load_ethanol_concentration(seq_len: int = MOMENT_SEQ_LEN) -> dict:
    """Load EthanolConcentration dataset via aeon, pad/truncate to seq_len.

    Returns dict with 'samples' (n, seq_len), 'labels', and 'name'.
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification("EthanolConcentration", split="train")
    X_test, y_test = load_classification("EthanolConcentration", split="test")

    # X shape: (n_samples, n_channels, n_timesteps)
    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)

    n_samples, n_channels, n_timesteps = X.shape

    # Pad or truncate to seq_len
    if n_timesteps < seq_len:
        pad_width = seq_len - n_timesteps
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    elif n_timesteps > seq_len:
        X = X[:, :, :seq_len]

    # Channel-independent: (n_samples * n_channels, seq_len)
    samples = X.reshape(n_samples * n_channels, seq_len)

    # Normalize per-sample
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples.T).T.astype(np.float32)

    # Repeat labels for each channel
    labels = np.repeat(y, n_channels)
    # Encode labels to integers
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_map[l] for l in labels])

    return {
        "samples": samples,
        "labels": labels_int,
        "name": "EthanolConcentration",
        "task": "classification",
    }


def serialize_dataset(dataset: dict) -> bytes:
    """Serialize dataset dict to bytes for Modal transfer."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in dataset.items() if isinstance(v, np.ndarray)})
    meta = {k: v for k, v in dataset.items() if not isinstance(v, np.ndarray)}
    buf.seek(0)
    return buf.read(), meta


def deserialize_dataset(data_bytes: bytes, meta: dict) -> dict:
    """Deserialize dataset from bytes."""
    buf = io.BytesIO(data_bytes)
    arrays = dict(np.load(buf))
    arrays.update(meta)
    return arrays
