"""Standard LTSF benchmark evaluation for evolved adapters.

Uses the standard protocol from PatchTST/iTransformer/MOMENT papers:
- Chronological 60/20/20 train/val/test split
- Global standardization (fit on train only)
- Multivariate: each channel is a separate univariate series
- Reports MSE and MAE on TEST set
- Multiple horizons: 96, 192, 336, 720

Usage:
    python scripts/run_standard_benchmark.py --dataset etth1 --horizon 96
    python scripts/run_standard_benchmark.py --dataset etth1 --horizon 96,192,336,720
    python scripts/run_standard_benchmark.py --dataset etth1 --horizon 96 --adapter evolved
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from feasibility.model import (
    load_moment, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from feasibility.finetune import _extract_features_batch


# --- Standard LTSF data loading ---

ETT_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
INPUT_LEN = 512  # MOMENT's sequence length


def load_standard_dataset(dataset_name, forecast_horizon=96):
    """Load dataset with standard LTSF chronological splits.

    Returns train/val/test arrays with proper normalization.
    Standard split: 60% train, 20% val, 20% test (chronological).
    ETT hourly: 12*30*24=8640 train, 12*30*24*4=2880 val, 2880 test
    ETT minutely: 12*30*24*4=34560 train, 11520 val, 11520 test
    """
    import io
    import pandas as pd

    # ETT datasets have standard split points
    SPLITS = {
        "ETTh1": (8640, 2880, 2880),
        "ETTh2": (8640, 2880, 2880),
        "ETTm1": (34560, 11520, 11520),
        "ETTm2": (34560, 11520, 11520),
    }

    if dataset_name.startswith("ETT"):
        from urllib.request import urlopen
        url = "%s/%s.csv" % (ETT_BASE_URL, dataset_name)
        response = urlopen(url)
        df = pd.read_csv(io.BytesIO(response.read()))
    elif dataset_name == "Weather":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Electricity":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity.csv")
        df = pd.read_csv(local_path)
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    values = df.iloc[:, 1:].values.astype(np.float32)
    n_timesteps, n_channels = values.shape

    if dataset_name in SPLITS:
        n_train, n_val, n_test = SPLITS[dataset_name]
    else:
        # Default 60/20/20 split
        n_train = int(0.6 * n_timesteps)
        n_val = int(0.2 * n_timesteps)
        n_test = n_timesteps - n_train - n_val

    train_data = values[:n_train]
    val_data = values[n_train:n_train + n_val]
    test_data = values[n_train + n_val:n_train + n_val + n_test]

    # Fit scaler on TRAIN only
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data).astype(np.float32)
    val_data = scaler.transform(val_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    # Create (input, target) pairs with sliding windows
    total_len = INPUT_LEN + forecast_horizon

    def make_samples(data):
        X_list, Y_list = [], []
        for ch in range(n_channels):
            series = data[:, ch]
            for start in range(0, len(series) - total_len + 1, 1):
                X_list.append(series[start:start + INPUT_LEN])
                Y_list.append(series[start + INPUT_LEN:start + total_len])
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

    X_train, Y_train = make_samples(train_data)
    X_val, Y_val = make_samples(val_data)
    X_test, Y_test = make_samples(test_data)

    # Subsample if too large (Weather/Electricity have many channels)
    max_samples = 10000
    for name, arrays in [("train", (X_train, Y_train)), ("val", (X_val, Y_val)), ("test", (X_test, Y_test))]:
        if len(arrays[0]) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(arrays[0]), max_samples, replace=False)
            if name == "train":
                X_train, Y_train = X_train[idx], Y_train[idx]
            elif name == "val":
                X_val, Y_val = X_val[idx], Y_val[idx]
            else:
                X_test, Y_test = X_test[idx], Y_test[idx]

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
        "n_channels": n_channels,
        "name": dataset_name,
    }


# --- Training and evaluation ---

def train_and_evaluate(code, data, model, encoder_blocks, device="cuda",
                       n_epochs=15, batch_size=64, forecast_horizon=96):
    """Train adapter on train set, evaluate on test set. Returns MSE and MAE."""
    namespace = {
        "torch": torch, "nn": nn,
        "F": torch.nn.functional, "math": __import__("math"),
    }
    exec(code, namespace)
    AdapterClass = namespace["Adapter"]

    hidden_dim = _get_hidden_dim(model)
    adapter = AdapterClass(hidden_dim, forecast_horizon).to(device)
    param_count = sum(p.numel() for p in adapter.parameters())

    trainable_params = list(adapter.parameters())
    param_ids = {id(p) for p in trainable_params}
    for param in model.parameters():
        if param.requires_grad and id(param) not in param_ids:
            trainable_params.append(param)
            param_ids.add(id(param))

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    mse_loss = nn.MSELoss()

    # Train
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X_train"]).float(),
            torch.from_numpy(data["Y_train"]).float(),
        ),
        batch_size=batch_size, shuffle=True,
    )

    for epoch in range(n_epochs):
        model.train()
        adapter.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)  # (B, 1, 512)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            pred = adapter(feat)
            loss = mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on TEST set
    model.eval()
    adapter.eval()
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X_test"]).float(),
            torch.from_numpy(data["Y_test"]).float(),
        ),
        batch_size=batch_size,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            pred = adapter(feat)
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    test_mse = nn.MSELoss()(preds, targets).item()
    test_mae = nn.L1Loss()(preds, targets).item()

    return {"mse": test_mse, "mae": test_mae, "param_count": param_count}


# --- Adapter definitions ---

ADAPTERS = {
    "linear": SEED_ADAPTERS[0],       # MeanPool + Linear
    "mlp": SEED_ADAPTERS[1],          # MeanPool + MLP2
    "attention": SEED_ADAPTERS[3],     # Attention pooling + Linear
    "conv": SEED_ADAPTERS[4],          # Conv1d + Linear
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", default="96", help="Comma-separated horizons")
    parser.add_argument("--adapters", default="all", help="Comma-separated adapter names or 'all'")
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizon.split(",")]
    adapter_names = list(ADAPTERS.keys()) if args.adapters == "all" else args.adapters.split(",")

    os.makedirs("results/standard_benchmark", exist_ok=True)

    # Load model once
    print("Loading MOMENT on %s..." % args.device)
    model = load_moment(args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    print("MOMENT: %d blocks, d_model=%d" % (len(blocks), hdim))

    # Freeze + unfreeze last 4
    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks) - 4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True

    results = {}

    for H in horizons:
        print("\n" + "=" * 60)
        print("Dataset: %s | Horizon: %d | Epochs: %d" % (args.dataset, H, args.n_epochs))
        print("=" * 60)

        data = load_standard_dataset(args.dataset, forecast_horizon=H)
        print("Train: %d, Val: %d, Test: %d samples" % (
            len(data["X_train"]), len(data["X_val"]), len(data["X_test"])))

        results[H] = {}

        for adapter_name in adapter_names:
            code = ADAPTERS.get(adapter_name)
            if code is None:
                print("  Unknown adapter: %s" % adapter_name)
                continue

            val = validate_adapter_code(code, d_model=hdim, output_dim=H)
            if not val["valid"]:
                print("  %s: INVALID for H=%d (%s)" % (adapter_name, H, val["error"]))
                results[H][adapter_name] = {"error": val["error"]}
                continue

            print("  Training %s (params=%d)..." % (adapter_name, val["param_count"]))
            try:
                # Reset backbone weights for fair comparison
                for i in range(max(0, len(blocks) - 4), len(blocks)):
                    for p in blocks[i].parameters():
                        p.requires_grad = True

                result = train_and_evaluate(
                    code, data, model, blocks,
                    device=args.device, n_epochs=args.n_epochs,
                    forecast_horizon=H,
                )
                results[H][adapter_name] = result
                print("    MSE=%.4f  MAE=%.4f  params=%d" % (
                    result["mse"], result["mae"], result["param_count"]))
            except Exception as e:
                print("    ERROR: %s" % e)
                results[H][adapter_name] = {"error": str(e)}

    # Print summary table
    print("\n" + "=" * 60)
    print("STANDARD BENCHMARK RESULTS — %s" % args.dataset)
    print("=" * 60)
    for H in horizons:
        print("\nHorizon %d:" % H)
        print("  %-20s | %8s | %8s | %8s" % ("Adapter", "MSE", "MAE", "Params"))
        print("  " + "-" * 52)
        for name in adapter_names:
            r = results.get(H, {}).get(name, {})
            if "error" in r:
                print("  %-20s | %8s | %8s | %8s" % (name, "ERR", "ERR", "—"))
            else:
                print("  %-20s | %8.4f | %8.4f | %8d" % (
                    name, r["mse"], r["mae"], r["param_count"]))

    # Save
    save_path = "results/standard_benchmark/%s_%s.json" % (
        args.dataset, args.horizon.replace(",", "_"))
    with open(save_path, "w") as f:
        json.dump({"dataset": args.dataset, "horizons": results}, f, indent=2, default=str)
    print("\nSaved to %s" % save_path)


if __name__ == "__main__":
    main()
