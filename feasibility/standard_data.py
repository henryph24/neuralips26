"""LTSF dataset loader and denormalization helpers (channel-wise StandardScaler).

This module is the single shared data-loading entry point used by every
experiment runner. It builds chronologically-split sliding windows of length
``INPUT_LEN`` over six LTSF benchmarks (ETTh1/2, ETTm1/2, Weather,
Electricity, plus optional Traffic / Exchange / Solar), fits a
``StandardScaler`` on the train segment, and returns per-channel-indexed
windows so that downstream code can convert MSE back to the original unit
space via :func:`compute_denorm_mse`.

Splits follow the standard LTSF convention used in Informer / PatchTST /
DLinear (Zeng et al. 2023): ``8640/2880/2880`` for ETTh*, ``34560/11520/11520``
for ETTm*, and a 60/20/20 fallback for Weather / Electricity / Traffic /
Exchange / Solar.
"""

import io
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from feasibility.model import _get_hidden_dim
from feasibility.finetune import _extract_features_batch


ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
INPUT_LEN = 512
SPLITS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}


def load_standard_data(dataset_name, forecast_horizon=96, max_samples=5000):
    """Load with standard chronological splits.

    Returns ``(splits, n_ch)`` where ``splits`` is a dict keyed by ``"train"``,
    ``"val"``, ``"test"`` whose values are ``(X, Y)`` numpy arrays. The dict
    also carries metadata used by denormalized-MSE reporting:

    - ``splits["_scaler"]`` : the fitted sklearn ``StandardScaler``
    - ``splits["<name>_ch"]`` : per-sample channel index array aligned with
      ``splits["<name>"]``. Since DataLoader iterates the test tensor with
      ``shuffle=False``, this can be used to denormalize predictions back to
      original units via
      ``pred * scaler.scale_[ch] + scaler.mean_[ch]``.
    """
    if dataset_name.startswith("ETT"):
        url = "%s/%s.csv" % (ETT_BASE, dataset_name)
        df = pd.read_csv(io.BytesIO(urlopen(url).read()))
    elif dataset_name == "Weather":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Electricity":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Traffic":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "traffic.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Exchange":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "exchange_rate.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Solar":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "solar.csv")
        df = pd.read_csv(local_path)
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    values = df.iloc[:, 1:].values.astype(np.float32)
    n_ch = values.shape[1]

    if dataset_name in SPLITS:
        n_train, n_val, n_test = SPLITS[dataset_name]
    else:
        n_total = len(values)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
    scaler = StandardScaler()
    scaler.fit(values[:n_train])

    splits = {"_scaler": scaler}
    for name, start, length in [
        ("train", 0, n_train),
        ("val", n_train, n_val),
        ("test", n_train + n_val, n_test),
    ]:
        data = scaler.transform(values[start:start + length]).astype(np.float32)
        total_len = INPUT_LEN + forecast_horizon
        X, Y, CH = [], [], []
        for ch in range(n_ch):
            s = data[:, ch]
            for i in range(0, len(s) - total_len + 1):
                X.append(s[i:i + INPUT_LEN])
                Y.append(s[i + INPUT_LEN:i + total_len])
                CH.append(ch)
        X = np.array(X, np.float32)
        Y = np.array(Y, np.float32)
        CH = np.array(CH, np.int32)
        if len(X) > max_samples:
            idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
            X, Y, CH = X[idx], Y[idx], CH[idx]
        splits[name] = (X, Y)
        splits[name + "_ch"] = CH

    return splits, n_ch


def compute_denorm_mse(preds, tgts, ch_idx, scaler):
    """Compute MSE and MAE in the original (un-standardized) unit space.

    Inverse-transforms normalized ``(B, H)`` predictions and targets using
    per-sample channel indices, then computes MSE and MAE on the original
    scale.
    """
    import torch as _torch
    if isinstance(preds, _torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(tgts, _torch.Tensor):
        tgts = tgts.detach().cpu().numpy()
    ch_idx = np.asarray(ch_idx, dtype=np.int64)
    scale = scaler.scale_[ch_idx][:, None].astype(np.float32)
    mean = scaler.mean_[ch_idx][:, None].astype(np.float32)
    preds_d = preds * scale + mean
    tgts_d = tgts * scale + mean
    err = preds_d - tgts_d
    mse_denorm = float(np.mean(err ** 2))
    mae_denorm = float(np.mean(np.abs(err)))
    return mse_denorm, mae_denorm


def _detect_backbone_type(backbone_name):
    """Detect backbone type from name string."""
    name = backbone_name.lower()
    if "chronos" in name:
        return "chronos"
    if "timer" in name:
        return "timer"
    if "moirai" in name:
        return "moirai"
    return "moment"


def train_adapter(code, model, blocks, X_train, Y_train, X_eval, Y_eval,
                  device="cuda", n_epochs=3, forecast_horizon=96, batch_size=128,
                  backbone_type="moment", eval_ch=None, scaler=None):
    """Train an adapter (defined as a code string) on the train set; evaluate on the eval set.

    Uses bf16 mixed precision and a larger batch size for throughput on
    modern GPUs. The ``code`` string must define a class named ``Adapter``
    that subclasses ``nn.Module`` and takes ``(d_model, output_dim)`` in its
    constructor (see :data:`feasibility.adapter_seeds.SEED_ADAPTERS`).
    """
    hdim = _get_hidden_dim(model)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)
    adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)
    param_count = sum(p.numel() for p in adapter.parameters())

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for _epoch in range(n_epochs):
        model.train()
        adapter.train()
        for bx, by in loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                loss = mse_fn(adapter(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    adapter.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_eval).float(), torch.from_numpy(Y_eval).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in eval_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            preds.append(adapter(_extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)).float().cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    out = {"mse": mse, "mae": mae, "param_count": param_count}
    if eval_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, eval_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d
    return out
