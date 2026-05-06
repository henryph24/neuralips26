"""DARTS-AAS: Differentiable Adapter Architecture Search.

Applies differentiable architecture search (DARTS) to the AAS problem.
Instead of evaluating hundreds of discrete architectures, defines a supernet
with continuous mixture weights over pooling, activation, and normalization
choices. Architecture weights are optimized on val, adapter weights on train.

Zero search overhead — architecture is found during a single training run.

Usage:
    python scripts/run_darts_aas.py --dataset ETTh1
    python scripts/run_darts_aas.py --dataset ETTh1 --epochs 20
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


# --- Pooling Operations (all output: B, d_model) ---

class MeanPool(nn.Module):
    def forward(self, h):
        return h.mean(dim=1)

class MaxPool(nn.Module):
    def forward(self, h):
        return h.max(dim=1).values

class LastTokenPool(nn.Module):
    def forward(self, h):
        return h[:, -1, :]

class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        return (h * w).sum(dim=1)  # (B, d_model)

class Conv1dPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=8, stride=4, padding=2)

    def forward(self, h):
        x = h.permute(0, 2, 1)  # (B, d, T)
        x = F.gelu(self.conv(x))  # (B, d, T')
        return x.mean(dim=2)  # (B, d)


# --- DARTS Supernet Adapter ---

class DARTSAdapter(nn.Module):
    """Differentiable adapter with architecture weights.

    Architecture params (alpha_*) are optimized on validation set.
    Adapter weights are optimized on training set.
    After training, discretize by argmax.
    """
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.d_model = d_model

        # === Architecture parameters (optimized on val) ===
        self.alpha_pool = nn.Parameter(torch.zeros(5))   # 5 pooling types
        self.alpha_norm = nn.Parameter(torch.zeros(2))   # no-norm vs LayerNorm
        self.alpha_act = nn.Parameter(torch.zeros(3))    # GELU, ReLU, SiLU
        self.alpha_hidden = nn.Parameter(torch.zeros(3)) # 64, 128, 256

        # === Pooling operations ===
        self.pool_ops = nn.ModuleList([
            MeanPool(),
            MaxPool(),
            LastTokenPool(),
            AttentionPool(d_model),
            Conv1dPool(d_model),
        ])
        self.pool_names = ["mean", "max", "last", "attention", "conv1d"]

        # === Pre-normalization ===
        self.norm = nn.LayerNorm(d_model)

        # === MLP heads for each hidden dim ===
        self.hidden_dims = [64, 128, 256]
        self.fc1_list = nn.ModuleList([nn.Linear(d_model, h) for h in self.hidden_dims])
        self.fc2_list = nn.ModuleList([nn.Linear(h, output_dim) for h in self.hidden_dims])

        # === Activations ===
        self.act_fns = [F.gelu, F.relu, F.silu]
        self.act_names = ["GELU", "ReLU", "SiLU"]

        self.dropout = nn.Dropout(0.1)

    def arch_parameters(self):
        """Return only architecture weights."""
        return [self.alpha_pool, self.alpha_norm, self.alpha_act, self.alpha_hidden]

    def weight_parameters(self):
        """Return only adapter weights (non-architecture)."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_ids]

    def forward(self, h):
        # h: (B, T, d_model)

        # Soft norm selection
        w_norm = F.softmax(self.alpha_norm, dim=0)
        h_input = w_norm[0] * h + w_norm[1] * self.norm(h)

        # Soft pooling selection
        w_pool = F.softmax(self.alpha_pool, dim=0)
        pooled = sum(w * op(h_input) for w, op in zip(w_pool, self.pool_ops))
        # pooled: (B, d_model)

        # Soft hidden dim + activation selection
        w_hidden = F.softmax(self.alpha_hidden, dim=0)
        w_act = F.softmax(self.alpha_act, dim=0)

        out = torch.zeros(pooled.shape[0], self.fc2_list[0].out_features,
                          device=pooled.device, dtype=pooled.dtype)

        for i, (fc1, fc2, w_h) in enumerate(zip(self.fc1_list, self.fc2_list, w_hidden)):
            x = fc1(pooled)
            # Soft activation
            x_act = sum(w_a * act(x) for w_a, act in zip(w_act, self.act_fns))
            x_act = self.dropout(x_act)
            out = out + w_h * fc2(x_act)

        return out

    def discretize(self):
        """Return the discrete architecture choices."""
        pool_idx = self.alpha_pool.argmax().item()
        norm_idx = self.alpha_norm.argmax().item()
        act_idx = self.alpha_act.argmax().item()
        hidden_idx = self.alpha_hidden.argmax().item()

        return {
            "pooling": self.pool_names[pool_idx],
            "norm": ["none", "LayerNorm"][norm_idx],
            "activation": self.act_names[act_idx],
            "hidden_dim": self.hidden_dims[hidden_idx],
            "alpha_pool": F.softmax(self.alpha_pool, dim=0).detach().cpu().tolist(),
            "alpha_norm": F.softmax(self.alpha_norm, dim=0).detach().cpu().tolist(),
            "alpha_act": F.softmax(self.alpha_act, dim=0).detach().cpu().tolist(),
            "alpha_hidden": F.softmax(self.alpha_hidden, dim=0).detach().cpu().tolist(),
        }

    def param_count_adapter(self):
        """Count non-architecture parameters."""
        return sum(p.numel() for p in self.weight_parameters())


def train_darts_aas(model, blocks, X_train, Y_train, X_val, Y_val,
                    device="cuda", n_epochs=15, warmup_epochs=3,
                    forecast_horizon=96, batch_size=128, backbone_type="moment",
                    lr_weights=1e-3, lr_arch=3e-3):
    """Train DARTS-AAS adapter with bilevel optimization.

    Phase 1 (warmup): Train adapter weights only (frozen architecture weights)
    Phase 2 (search): Alternate between weight update (train) and arch update (val)
    """
    hdim = _get_hidden_dim(model)
    adapter = DARTSAdapter(hdim, forecast_horizon).to(device)

    # Separate optimizers
    opt_weights = torch.optim.Adam(adapter.weight_parameters(), lr=lr_weights)
    opt_arch = torch.optim.Adam(adapter.arch_parameters(), lr=lr_arch, weight_decay=1e-3)

    # Also collect unfrozen backbone params
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    if backbone_params:
        opt_weights.add_param_group({"params": backbone_params, "lr": lr_weights * 0.1})

    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float(),
    ), batch_size=batch_size)

    arch_history = []

    for epoch in range(n_epochs):
        is_search_phase = epoch >= warmup_epochs

        # --- Train adapter weights on training set ---
        model.train()
        adapter.train()
        for bx, by in train_loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                loss = mse_fn(adapter(feat), by)
            opt_weights.zero_grad()
            loss.backward()
            opt_weights.step()

        # --- Update architecture weights on validation set (search phase only) ---
        if is_search_phase:
            model.eval()
            adapter.train()  # keep adapter in train mode for arch param gradients
            for bx, by in val_loader:
                bx, by = bx.to(device).unsqueeze(1), by.to(device)
                mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                    val_loss = mse_fn(adapter(feat), by)
                opt_arch.zero_grad()
                val_loss.backward()
                opt_arch.step()

        # Log architecture choices
        arch = adapter.discretize()
        arch_history.append(arch)

        if epoch % 3 == 0 or epoch == n_epochs - 1:
            print("  Epoch %2d: pool=%-10s norm=%-9s act=%-4s hidden=%d" % (
                epoch, arch["pooling"], arch["norm"], arch["activation"], arch["hidden_dim"]))

    return adapter, arch_history


def build_adapter_from_arch(arch, d_model=512, output_dim=96):
    """Build a discrete adapter code string from DARTS-discovered architecture."""
    pooling = arch["pooling"]
    norm = arch["norm"]
    activation = arch["activation"]
    hidden = arch["hidden_dim"]

    act_map = {"GELU": "nn.GELU()", "ReLU": "nn.ReLU()", "SiLU": "nn.SiLU()"}
    act_code = act_map.get(activation, "nn.GELU()")

    # Build norm
    if norm == "LayerNorm":
        norm_init = "self.pre_norm = nn.LayerNorm(d_model)"
        norm_fwd = "hidden_states = self.pre_norm(hidden_states)"
    else:
        norm_init = ""
        norm_fwd = ""

    # Build pooling init/forward
    if pooling == "mean":
        pool_init = ""
        pool_fwd = "x = hidden_states.mean(dim=1)"
    elif pooling == "max":
        pool_init = ""
        pool_fwd = "x = hidden_states.max(dim=1).values"
    elif pooling == "last":
        pool_init = ""
        pool_fwd = "x = hidden_states[:, -1, :]"
    elif pooling == "attention":
        pool_init = "self.attn = nn.Linear(d_model, 1)"
        pool_fwd = "w = torch.softmax(self.attn(hidden_states), dim=1)\n        x = (hidden_states * w).sum(dim=1)"
    elif pooling == "conv1d":
        conv_ch = min(hidden, 64)  # limit conv channels to fit 500K param budget
        pool_init = "self.conv = nn.Conv1d(d_model, %d, kernel_size=8, stride=4, padding=2)" % conv_ch
        pool_fwd = "x = hidden_states.permute(0, 2, 1)\n        x = F.gelu(self.conv(x))\n        x = x.mean(dim=2)"
        # Conv1d outputs hidden dim directly, skip fc1
        all_init = []
        all_fwd = []
        if norm_init:
            all_init.append(norm_init)
        all_init.append(pool_init)
        all_init.append("self.act = %s" % act_code)
        all_init.append("self.dropout = nn.Dropout(0.1)")
        all_init.append("self.fc2 = nn.Linear(%d, output_dim)" % conv_ch)
        if norm_fwd:
            all_fwd.append(norm_fwd)
        all_fwd.append(pool_fwd)
        all_fwd.append("x = self.act(x)")
        all_fwd.append("x = self.dropout(x)")
        all_fwd.append("x = self.fc2(x)")
        all_fwd.append("return x")

        init_body = "\n        ".join(all_init)
        fwd_body = "\n        ".join(all_fwd)

        return """import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        %s

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        %s""" % (init_body, fwd_body)
    else:
        pool_init = ""
        pool_fwd = "x = hidden_states.mean(dim=1)"

    # Assemble
    all_init = []
    all_fwd = []

    if norm_init:
        all_init.append(norm_init)
    if pool_init:
        all_init.append(pool_init)
    all_init.append("self.fc1 = nn.Linear(d_model, %d)" % hidden)
    all_init.append("self.act = %s" % act_code)
    all_init.append("self.dropout = nn.Dropout(0.1)")
    all_init.append("self.fc2 = nn.Linear(%d, output_dim)" % hidden)

    if norm_fwd:
        all_fwd.append(norm_fwd)
    all_fwd.append(pool_fwd)
    all_fwd.append("x = self.fc1(x)")
    all_fwd.append("x = self.act(x)")
    all_fwd.append("x = self.dropout(x)")
    all_fwd.append("x = self.fc2(x)")
    all_fwd.append("return x")

    init_body = "\n        ".join(all_init)
    fwd_body = "\n        ".join(all_fwd)

    code = """import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        %s

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        %s""" % (init_body, fwd_body)

    return code


def evaluate_adapter(adapter, model, blocks, X, Y, device, forecast_horizon, backbone_type, batch_size=128):
    """Evaluate adapter on a dataset, return MSE and MAE."""
    model.eval()
    adapter.eval()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(Y).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in loader:
            bx, by = bx.to(device).unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
            preds.append(adapter(feat).cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    return mse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain discrete adapter after DARTS search")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/darts_aas", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    print("d_model=%d, unfreezing last 4 blocks" % hdim)

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, val=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_val), len(X_test)))

    # === Run DARTS-AAS ===
    print("\n" + "=" * 60)
    print("DARTS-AAS: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)

    start = time.time()
    adapter, arch_history = train_darts_aas(
        model, blocks, X_train, Y_train, X_val, Y_val,
        device=args.device, n_epochs=args.epochs, warmup_epochs=args.warmup,
        forecast_horizon=args.horizon, backbone_type=bb_type,
    )
    elapsed = time.time() - start

    # Evaluate on test
    test_mse, test_mae = evaluate_adapter(
        adapter, model, blocks, X_test, Y_test,
        args.device, args.horizon, bb_type,
    )
    final_arch = adapter.discretize()
    param_count = adapter.param_count_adapter()

    print("\nDARTS-AAS supernet result:")
    print("  test_mse=%.4f  test_mae=%.4f  params=%d  time=%.0fs" % (
        test_mse, test_mae, param_count, elapsed))
    print("  Architecture: pool=%s norm=%s act=%s hidden=%d" % (
        final_arch["pooling"], final_arch["norm"],
        final_arch["activation"], final_arch["hidden_dim"]))

    # === Retrain discrete adapter (if --retrain) ===
    retrain_result = None
    if args.retrain:
        print("\nRetraining discrete adapter from DARTS-discovered architecture...")
        discrete_code = build_adapter_from_arch(final_arch)
        val = validate_adapter_code(discrete_code, d_model=hdim, output_dim=args.horizon)
        if val["valid"]:
            # Reload model fresh for clean retrain
            model_rt = load_backbone(args.backbone, args.device)
            _disable_gradient_checkpointing(model_rt)
            blocks_rt = _get_encoder_blocks(model_rt)
            for p in model_rt.parameters():
                p.requires_grad = False
            for i in range(max(0, len(blocks_rt)-4), len(blocks_rt)):
                for p in blocks_rt[i].parameters():
                    p.requires_grad = True

            start_rt = time.time()
            tr = train_adapter(
                discrete_code, model_rt, blocks_rt, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            elapsed_rt = time.time() - start_rt
            retrain_result = {
                "test_mse": tr["mse"],
                "test_mae": tr["mae"],
                "param_count": tr["param_count"],
                "elapsed_retrain": elapsed_rt,
                "elapsed_total": elapsed + elapsed_rt,
                "code": discrete_code,
            }
            print("  Retrain MSE=%.4f  MAE=%.4f  params=%d  retrain_time=%.0fs  total=%.0fs" % (
                tr["mse"], tr["mae"], tr["param_count"], elapsed_rt, elapsed + elapsed_rt))

            # Use retrain result as the DARTS result
            test_mse = tr["mse"]
            test_mae = tr["mae"]
            param_count = tr["param_count"]
            elapsed = elapsed + elapsed_rt
        else:
            print("  ERROR: discrete adapter invalid: %s" % val["error"])

    # === Run fixed baselines for comparison ===
    print("\nFixed baselines (15 epochs):")
    baselines = {
        "linear": SEED_ADAPTERS[0],
        "attention": SEED_ADAPTERS[3],
        "conv": SEED_ADAPTERS[4],
    }
    baseline_results = {}

    # Need to reload model to reset unfrozen weights
    model2 = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model2)
    blocks2 = _get_encoder_blocks(model2)
    for p in model2.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks2)-4), len(blocks2)):
        for p in blocks2[i].parameters():
            p.requires_grad = True

    for name, code in baselines.items():
        try:
            tr = train_adapter(
                code, model2, blocks2, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            baseline_results[name] = tr
            print("  %-15s MSE=%.4f  MAE=%.4f  params=%d" % (
                name, tr["mse"], tr["mae"], tr["param_count"]))
        except Exception as e:
            print("  %-15s ERROR: %s" % (name, e))

    # Summary
    best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"])
    best_baseline_mse = baseline_results[best_baseline_name]["mse"]
    delta = (test_mse - best_baseline_mse) / best_baseline_mse * 100
    winner = "DARTS-AAS" if test_mse < best_baseline_mse else "BASELINE"

    print("\n" + "=" * 60)
    print("SUMMARY: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    print("DARTS-AAS:     MSE=%.4f  (%s, %s, %s, hidden=%d)" % (
        test_mse, final_arch["pooling"], final_arch["norm"],
        final_arch["activation"], final_arch["hidden_dim"]))
    print("Best baseline: MSE=%.4f  (%s)" % (best_baseline_mse, best_baseline_name))
    print("Delta: %+.1f%% -> Winner: %s" % (delta, winner))
    print("DARTS-AAS time: %.0fs (%.1f min) — single training run, zero search overhead" % (
        elapsed, elapsed / 60))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "retrain": args.retrain,
        "darts_aas": {
            "test_mse": test_mse,
            "test_mae": test_mae,
            "param_count": param_count,
            "elapsed": elapsed,
            "final_architecture": final_arch,
            "arch_history": arch_history,
        },
        "retrain_result": retrain_result,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner,
        "delta_pct": delta,
    }
    path = "results/darts_aas/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
