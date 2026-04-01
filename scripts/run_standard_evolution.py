"""Run code evolution on standard LTSF protocol.

Evolves adapters using standard chronological splits:
- Train on TRAIN set, evaluate fitness on VAL set
- After evolution, evaluate winners on TEST set

This gives each dataset its own optimal adapter architecture.

Usage:
    python scripts/run_standard_evolution.py --dataset ETTh1
    python scripts/run_standard_evolution.py --dataset ETTh1 --n-generations 12 --pop-size 20
"""

import argparse
import json
import os
import sys
import time
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from feasibility.code_evolution import (
    validate_adapter_code, SEED_ADAPTERS, CodeIndividual, CodeEvolutionLogger,
)
from feasibility.finetune import _extract_features_batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_ablations import run_random_code_ablation

ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
INPUT_LEN = 512
SPLITS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}


def load_standard_data(dataset_name, forecast_horizon=96, max_samples=5000):
    """Load with standard chronological splits."""
    if dataset_name.startswith("ETT"):
        url = "%s/%s.csv" % (ETT_BASE, dataset_name)
        df = pd.read_csv(io.BytesIO(urlopen(url).read()))
    elif dataset_name == "Weather":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")
        df = pd.read_csv(local_path)
    elif dataset_name == "Electricity":
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity.csv")
        df = pd.read_csv(local_path)
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    values = df.iloc[:, 1:].values.astype(np.float32)
    n_ch = values.shape[1]

    if dataset_name in SPLITS:
        n_train, n_val, n_test = SPLITS[dataset_name]
    else:
        # Default 60/20/20
        n_total = len(values)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
    scaler = StandardScaler()
    scaler.fit(values[:n_train])

    splits = {}
    for name, start, length in [
        ("train", 0, n_train),
        ("val", n_train, n_val),
        ("test", n_train + n_val, n_test),
    ]:
        data = scaler.transform(values[start:start+length]).astype(np.float32)
        total_len = INPUT_LEN + forecast_horizon
        X, Y = [], []
        for ch in range(n_ch):
            s = data[:, ch]
            for i in range(0, len(s) - total_len + 1):
                X.append(s[i:i+INPUT_LEN])
                Y.append(s[i+INPUT_LEN:i+total_len])
        X, Y = np.array(X, np.float32), np.array(Y, np.float32)
        if len(X) > max_samples:
            idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
            X, Y = X[idx], Y[idx]
        splits[name] = (X, Y)

    return splits, n_ch


def train_adapter(code, model, blocks, X_train, Y_train, X_eval, Y_eval,
                  device="cuda", n_epochs=3, forecast_horizon=96, batch_size=64):
    """Train adapter on train set, evaluate on eval set (val or test)."""
    hdim = _get_hidden_dim(model)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)
    adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)
    param_count = sum(p.numel() for p in adapter.parameters())

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p); pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx, mask)
            loss = mse_fn(adapter(feat), by)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Evaluate
    model.eval(); adapter.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_eval).float(), torch.from_numpy(Y_eval).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in eval_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            preds.append(adapter(_extract_features_batch(model, blocks, bx, mask)).cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    return {"mse": mse, "mae": mae, "param_count": param_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--n-generations", type=int, default=12)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/standard_evolution", exist_ok=True)

    # Load model
    print("Loading MOMENT...")
    model = load_moment(args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    print("MOMENT: %d blocks, d_model=%d" % (len(blocks), hdim))

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s: train=%d, val=%d, test=%d" % (
        args.dataset, len(X_train), len(X_val), len(X_test)))

    # Build evaluate_fn that trains on TRAIN, evaluates on VAL
    # This creates proper train/val separation for evolution fitness
    def evaluate_fn(code_strs):
        results = []
        for code in code_strs:
            try:
                result = train_adapter(
                    code, model, blocks, X_train, Y_train, X_val, Y_val,
                    device=args.device, n_epochs=3, forecast_horizon=args.horizon,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": "%s: %s" % (type(e).__name__, e)})
        return results

    # Run evolution
    print("\n" + "=" * 60)
    print("STANDARD PROTOCOL EVOLUTION: %s H=%d" % (args.dataset, args.horizon))
    print("Gens: %d | Pop: %d | Seed: %d" % (args.n_generations, args.pop_size, args.seed))
    print("Fitness = VAL MSE (train on TRAIN, evaluate on VAL)")
    print("=" * 60)

    start = time.time()
    result = run_random_code_ablation(
        evaluate_fn=evaluate_fn,
        n_generations=args.n_generations,
        pop_size=args.pop_size,
        seed=args.seed,
    )
    elapsed = time.time() - start
    print("\nEvolution complete: best_val_mse=%.4f, time=%.0fs" % (result["best_mse"], elapsed))

    # Evaluate top-5 on TEST set at 15 epochs
    top_n = 5
    valid_pop = [
        ind for ind in result["final_population"]
        if ind["error"] is None and ind["mse"] < float('inf')
    ][:top_n]

    print("\nEvaluating top-%d on TEST set (15 epochs)..." % len(valid_pop))
    test_results = []
    for ind in valid_pop:
        try:
            tr = train_adapter(
                ind["code"], model, blocks, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
            )
            test_results.append({
                "code": ind["code"],
                "val_mse": ind["mse"],
                "test_mse": tr["mse"],
                "test_mae": tr["mae"],
                "param_count": tr["param_count"],
            })
            print("  val=%.4f -> test_mse=%.4f test_mae=%.4f params=%d" % (
                ind["mse"], tr["mse"], tr["mae"], tr["param_count"]))
        except Exception as e:
            print("  ERROR: %s" % e)

    # Also evaluate hand-designed baselines on TEST set
    print("\nBaseline adapters on TEST set (15 epochs):")
    baselines = {
        "linear": SEED_ADAPTERS[0],
        "attention": SEED_ADAPTERS[3],
        "conv": SEED_ADAPTERS[4],
    }
    baseline_results = {}
    for name, code in baselines.items():
        val = validate_adapter_code(code, d_model=hdim, output_dim=args.horizon)
        if not val["valid"]:
            print("  %s: INVALID" % name)
            continue
        try:
            tr = train_adapter(
                code, model, blocks, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
            )
            baseline_results[name] = tr
            print("  %-15s test_mse=%.4f test_mae=%.4f params=%d" % (
                name, tr["mse"], tr["mae"], tr["param_count"]))
        except Exception as e:
            print("  %s: ERROR %s" % (name, e))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: %s H=%d (Standard Protocol)" % (args.dataset, args.horizon))
    print("=" * 60)
    best_evolved = min(test_results, key=lambda x: x["test_mse"]) if test_results else None
    best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"]) if baseline_results else None
    best_baseline = baseline_results.get(best_baseline_name, {}) if best_baseline_name else {}

    if best_evolved:
        print("Best evolved:  MSE=%.4f  MAE=%.4f  params=%d" % (
            best_evolved["test_mse"], best_evolved["test_mae"], best_evolved["param_count"]))
    if best_baseline:
        print("Best baseline: MSE=%.4f  MAE=%.4f  (%s)" % (
            best_baseline["mse"], best_baseline["mae"], best_baseline_name))
    if best_evolved and best_baseline:
        delta = (best_evolved["test_mse"] - best_baseline["mse"]) / best_baseline["mse"] * 100
        winner = "EVOLVED" if delta < 0 else "BASELINE"
        print("Delta: %+.1f%% -> Winner: %s" % (delta, winner))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "elapsed": elapsed,
        "evolved": test_results,
        "baselines": {k: v for k, v in baseline_results.items()},
        "generations": result["logger"].to_dict()["generations"],
    }
    path = "results/standard_evolution/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
