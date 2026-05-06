"""Standard LTSF benchmark on Modal GPU.

Runs the standard benchmark evaluation (chronological splits, test set MSE/MAE)
using Modal A10G GPUs. Each (dataset, horizon, adapter) combo is one remote call.

Usage:
    modal run scripts/run_standard_benchmark_modal.py --dataset ETTh1 --horizon 96
    modal run scripts/run_standard_benchmark_modal.py --dataset ETTh1 --horizon 96,192,336,720
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
import numpy as np

from feasibility.modal_app import app, image
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS

# Standard LTSF splits
ETT_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
SPLITS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}

ADAPTERS = {
    "linear": SEED_ADAPTERS[0],
    "mlp": SEED_ADAPTERS[1],
    "attention": SEED_ADAPTERS[3],
    "conv": SEED_ADAPTERS[4],
    "evolved_conv_bn": """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(128, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.transpose(1, 2)
        x = self.bn1(torch.relu(self.conv1(x)))
        x = x.mean(dim=2)
        return self.linear(x)""",
    "evolved_depthwise": """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding=1)
        self.fc = nn.Linear(d_model, output_dim)
        self.positional_weights = nn.Parameter(torch.randn(d_model))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x * self.positional_weights.view(1, -1, 1)
        x = x.mean(dim=2)
        return self.fc(x)""",
    "evolved_attention": """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(d_model))
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.linear2 = nn.Linear(d_model // 2, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        weights = F.softmax(self.attention_weights, dim=0)
        pooled = (hidden_states * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        x = torch.relu(self.linear1(pooled))
        return self.linear2(x)""",
}


@app.function(gpu="A10G", timeout=900, scaledown_window=2, max_containers=10)
def evaluate_standard(
    adapter_code: str,
    dataset_name: str,
    forecast_horizon: int,
    n_epochs: int = 15,
) -> dict:
    """Train adapter on standard split and evaluate on test set."""
    import sys
    sys.path.insert(0, "/root")

    import io
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from urllib.request import urlopen

    from feasibility.model import (
        load_moment, _get_encoder_blocks, _get_hidden_dim,
        _disable_gradient_checkpointing,
    )
    from feasibility.finetune import _extract_features_batch

    INPUT_LEN = 512
    ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
    SPLITS = {
        "ETTh1": (8640, 2880, 2880),
        "ETTh2": (8640, 2880, 2880),
        "ETTm1": (34560, 11520, 11520),
        "ETTm2": (34560, 11520, 11520),
    }

    try:
        # Load data
        url = "%s/%s.csv" % (ETT_BASE, dataset_name)
        response = urlopen(url)
        df = pd.read_csv(io.BytesIO(response.read()))
        values = df.iloc[:, 1:].values.astype(np.float32)
        n_ts, n_ch = values.shape

        n_train, n_val, n_test = SPLITS[dataset_name]
        train_raw = values[:n_train]
        test_raw = values[n_train + n_val:n_train + n_val + n_test]

        scaler = StandardScaler()
        scaler.fit(train_raw)
        train_data = scaler.transform(train_raw).astype(np.float32)
        test_data = scaler.transform(test_raw).astype(np.float32)

        total_len = INPUT_LEN + forecast_horizon

        def make_samples(data, max_n=10000):
            X, Y = [], []
            for ch in range(n_ch):
                s = data[:, ch]
                for start in range(0, len(s) - total_len + 1, 1):
                    X.append(s[start:start + INPUT_LEN])
                    Y.append(s[start + INPUT_LEN:start + total_len])
            X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
            if len(X) > max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), max_n, replace=False)
                X, Y = X[idx], Y[idx]
            return X, Y

        X_train, Y_train = make_samples(train_data)
        X_test, Y_test = make_samples(test_data)

        # Load model
        model = load_moment("cuda")
        _disable_gradient_checkpointing(model)
        blocks = _get_encoder_blocks(model)
        hdim = _get_hidden_dim(model)

        for p in model.parameters():
            p.requires_grad = False
        for i in range(max(0, len(blocks) - 4), len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True

        # Build adapter
        namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
        exec(adapter_code, namespace)
        adapter = namespace["Adapter"](hdim, forecast_horizon).to("cuda")
        param_count = sum(p.numel() for p in adapter.parameters())

        trainable = list(adapter.parameters())
        pids = {id(p) for p in trainable}
        for p in model.parameters():
            if p.requires_grad and id(p) not in pids:
                trainable.append(p)
                pids.add(id(p))

        optimizer = torch.optim.Adam(trainable, lr=1e-3)
        mse_fn = nn.MSELoss()

        # Train
        loader = DataLoader(TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float(),
        ), batch_size=64, shuffle=True)

        for epoch in range(n_epochs):
            model.train(); adapter.train()
            for bx, by in loader:
                bx = bx.to("cuda").unsqueeze(1)
                by = by.to("cuda")
                mask = torch.ones(bx.shape[0], bx.shape[2], device="cuda")
                feat = _extract_features_batch(model, blocks, bx, mask)
                loss = mse_fn(adapter(feat), by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Test
        model.eval(); adapter.eval()
        test_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(Y_test).float(),
        ), batch_size=64)

        preds, tgts = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to("cuda").unsqueeze(1)
                by = by.to("cuda")
                mask = torch.ones(bx.shape[0], bx.shape[2], device="cuda")
                feat = _extract_features_batch(model, blocks, bx, mask)
                preds.append(adapter(feat).cpu())
                tgts.append(by.cpu())

        preds = torch.cat(preds)
        tgts = torch.cat(tgts)
        return {
            "mse": nn.MSELoss()(preds, tgts).item(),
            "mae": nn.L1Loss()(preds, tgts).item(),
            "param_count": param_count,
        }

    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}


@app.local_entrypoint()
def benchmark(
    dataset: str = "ETTh1",
    horizon: str = "96",
    n_epochs: int = 15,
):
    os.makedirs("results/standard_benchmark", exist_ok=True)

    horizons = [int(h.strip()) for h in horizon.split(",")]
    results = {}

    for H in horizons:
        print("\n" + "=" * 60)
        print("STANDARD BENCHMARK: %s H=%d" % (dataset, H))
        print("=" * 60)

        # Validate all adapters locally first
        valid_adapters = {}
        for name, code in ADAPTERS.items():
            val = validate_adapter_code(code, output_dim=H)
            if val["valid"]:
                valid_adapters[name] = code
            else:
                print("  %s: INVALID for H=%d (%s)" % (name, H, val["error"]))

        # Run all valid adapters in parallel via starmap!
        print("  Running %d adapters in parallel..." % len(valid_adapters))
        eval_args = [
            (code, dataset, H, n_epochs)
            for name, code in valid_adapters.items()
        ]
        eval_results = list(evaluate_standard.starmap(eval_args))

        results[str(H)] = {}
        for (name, _), result in zip(valid_adapters.items(), eval_results):
            results[str(H)][name] = result
            if "error" in result:
                print("  %-20s: ERROR %s" % (name, result["error"]))
            else:
                print("  %-20s: MSE=%.4f  MAE=%.4f  params=%d" % (
                    name, result["mse"], result["mae"], result["param_count"]))

    # Summary
    print("\n" + "=" * 60)
    print("STANDARD BENCHMARK RESULTS — %s (MOMENT-large)" % dataset)
    print("=" * 60)
    for H in horizons:
        h_key = str(H)
        if h_key not in results:
            continue
        print("\nHorizon %d:" % H)
        print("  %-20s | %8s | %8s | %8s" % ("Adapter", "MSE", "MAE", "Params"))
        print("  " + "-" * 52)
        for name in ADAPTERS:
            r = results[h_key].get(name, {})
            if "error" in r:
                print("  %-20s | %8s | %8s | %8s" % (name, "ERR", "—", "—"))
            elif r:
                print("  %-20s | %8.4f | %8.4f | %8d" % (
                    name, r["mse"], r["mae"], r["param_count"]))

    save_path = "results/standard_benchmark/%s_%s.json" % (
        dataset, horizon.replace(",", "_"))
    with open(save_path, "w") as f:
        json.dump({"dataset": dataset, "results": results}, f, indent=2, default=str)
    print("\nSaved to %s" % save_path)
