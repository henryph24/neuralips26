"""Dataset download and preprocessing for feasibility experiment."""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen


ETT_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
WEATHER_URL = "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/weather/weather.csv"
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


def _load_csv_forecasting(url: str, name: str, seq_len: int = MOMENT_SEQ_LEN,
                          stride: int = 64, max_samples: int = None) -> dict:
    """Generic loader for CSV forecasting datasets (ETT, Weather, etc.).

    Expects CSV with a date column followed by numeric feature columns.
    Returns dict with 'samples' (n, seq_len) numpy array and 'name'.
    """
    response = urlopen(url)
    df = pd.read_csv(io.BytesIO(response.read()))
    # Drop date column, keep numeric columns
    values = df.iloc[:, 1:].values.astype(np.float32)

    n_timesteps, n_channels = values.shape
    windows = []
    for start in range(0, n_timesteps - seq_len + 1, stride):
        window = values[start : start + seq_len]
        windows.append(window)
    windows = np.stack(windows)

    n_windows = windows.shape[0]
    samples = windows.transpose(0, 2, 1).reshape(n_windows * n_channels, seq_len)

    if max_samples and len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = samples[idx]

    scaler = StandardScaler()
    samples = scaler.fit_transform(samples.T).T.astype(np.float32)

    return {"samples": samples, "name": name, "task": "forecasting"}


def load_etth1(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    return _load_csv_forecasting(f"{ETT_BASE_URL}/ETTh1.csv", "ETTh1", seq_len, stride)


def load_etth2(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    return _load_csv_forecasting(f"{ETT_BASE_URL}/ETTh2.csv", "ETTh2", seq_len, stride)


def load_ettm1(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    return _load_csv_forecasting(f"{ETT_BASE_URL}/ETTm1.csv", "ETTm1", seq_len, stride)


def load_ettm2(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    return _load_csv_forecasting(f"{ETT_BASE_URL}/ETTm2.csv", "ETTm2", seq_len, stride)


def load_weather(seq_len: int = MOMENT_SEQ_LEN, stride: int = 64) -> dict:
    """Load Weather dataset. Requires local file at data/weather.csv.

    Download from: https://drive.google.com/drive/folders/1ohGYWWohJlJlB6l_IFxbKMiAt1grAfKQ
    """
    import os
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Weather dataset not found at {local_path}. "
            "Download weather.csv from the Time-Series-Library Google Drive."
        )
    df = pd.read_csv(local_path)
    values = df.iloc[:, 1:].values.astype(np.float32)

    n_timesteps, n_channels = values.shape
    windows = []
    for start in range(0, n_timesteps - seq_len + 1, stride):
        windows.append(values[start : start + seq_len])
    windows = np.stack(windows)

    n_windows = windows.shape[0]
    samples = windows.transpose(0, 2, 1).reshape(n_windows * n_channels, seq_len)

    # Subsample — Weather is very large (21 channels)
    if len(samples) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), 5000, replace=False)
        samples = samples[idx]

    scaler = StandardScaler()
    samples = scaler.fit_transform(samples.T).T.astype(np.float32)

    return {"samples": samples, "name": "Weather", "task": "forecasting"}


FORECASTING_DATASETS = {
    "ETTh1": load_etth1,
    "ETTh2": load_etth2,
    "ETTm1": load_ettm1,
    "ETTm2": load_ettm2,
}


def _load_uea_classification(dataset_name: str, seq_len: int = MOMENT_SEQ_LEN) -> dict:
    """Generic loader for UEA classification datasets via aeon.

    Returns dict with 'samples' (n, seq_len), 'labels', and 'name'.
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)

    n_samples, n_channels, n_timesteps = X.shape

    if n_timesteps < seq_len:
        pad_width = seq_len - n_timesteps
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    elif n_timesteps > seq_len:
        X = X[:, :, :seq_len]

    # Channel-independent: (n_samples * n_channels, seq_len)
    samples = X.reshape(n_samples * n_channels, seq_len)

    scaler = StandardScaler()
    samples = scaler.fit_transform(samples.T).T.astype(np.float32)

    labels = np.repeat(y, n_channels)
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_map[l] for l in labels])

    return {
        "samples": samples,
        "labels": labels_int,
        "name": dataset_name,
        "task": "classification",
    }


def load_ethanol_concentration(seq_len: int = MOMENT_SEQ_LEN) -> dict:
    return _load_uea_classification("EthanolConcentration", seq_len)


def load_japanese_vowels(seq_len: int = MOMENT_SEQ_LEN) -> dict:
    return _load_uea_classification("JapaneseVowels", seq_len)


def load_basic_motions(seq_len: int = MOMENT_SEQ_LEN) -> dict:
    return _load_uea_classification("BasicMotions", seq_len)


CLASSIFICATION_DATASETS = {
    "EthanolConcentration": load_ethanol_concentration,
    "JapaneseVowels": load_japanese_vowels,
    "BasicMotions": load_basic_motions,
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
