"""Input-length ablation for RR-MoA (rebuttal Q1).

Reviewer Pm4m asks whether the method and conclusions hold when the input
length changes from the standard 512. MOMENT-small is pretrained at a fixed
512 context, so a shorter observed length ``L`` is realised the MOMENT-native
way: only the most-recent ``L`` steps are observed (``input_mask=1``); the
older ``512-L`` steps are masked out (``input_mask=0``); and the raw router
reads only the ``L`` observed steps.  ``L=512`` (all-ones mask) reduces
exactly to the published pipeline, so it doubles as a correctness check
(ETTh1 frozen Top-2 must reproduce ~0.69, Weather ~0.29).

All ``L`` in {96,192,336,512} are multiples of MOMENT's patch size (8), so the
observed/masked boundary is patch-aligned.

Per (dataset, L, seed) on a strictly frozen MOMENT-small we train:
  * RR-MoA (raw routing)            -> MSE, routing entropy   (the method)
  * RR-MoA (revin routing) control  -> MSE, routing entropy   (the mechanism)
  * best fixed adapter {linear,attention,conv} -> best MSE     (the 54/54 ref)
and report RR-MoA-raw's delta vs the best fixed adapter and vs the revin
control.  A persisting raw>revin gap + healthy entropy + a win over the best
fixed adapter at every L answers Q1.

This module only *forks* the two proven training loops (train_rr_moa in
scripts/run_rr_moa.py and train_adapter in feasibility/standard_data.py); the
only changes are the input mask (last-L observed) and slicing the raw router
input to the last L steps.  It never edits the archived scripts that
evidence_vm/verify.py depends on.
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.adapter_seeds import SEED_ADAPTERS
from feasibility.standard_data import load_standard_data, _detect_backbone_type
from run_rr_moa import RawRoutedMoA


def _mask_last_l(batch, seq_len, input_len, device):
    """input_mask marking only the most-recent ``input_len`` steps as observed."""
    m = torch.zeros(batch, seq_len, device=device)
    m[:, seq_len - input_len:] = 1.0
    return m


def train_rr_moa_masked(model, blocks, X_train, Y_train, X_test, Y_test, input_len,
                        device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                        backbone_type="moment", K=5, top_k=2, router_input_mode="raw",
                        expert_pool="canonical"):
    """Fork of scripts/run_rr_moa.train_rr_moa with a last-L input mask and a
    raw router that reads only the L observed steps.  Backbone strictly frozen."""
    hdim = _get_hidden_dim(model)
    adapter = RawRoutedMoA(
        hdim, forecast_horizon, input_len=input_len, K=K, hidden=64, top_k=top_k,
        router_input_mode=router_input_mode, expert_pool=expert_pool,
    ).to(device)

    trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for _epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            bx_full = bx.to(device)                 # (B, 512)
            bx_enc = bx_full.unsqueeze(1)           # (B, 1, 512) for backbone
            by = by.to(device)
            raw_l = bx_full[:, -input_len:]         # (B, L) observed router input
            mask = _mask_last_l(bx_enc.shape[0], bx_enc.shape[2], input_len, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat, raw_l)
                loss = mse_fn(pred, by)
                loss = loss + adapter.load_balance_coeff * adapter.load_balance_loss(raw_l, hidden_states=feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts, routing = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_full = bx.to(device)
            bx_enc = bx_full.unsqueeze(1)
            by = by.to(device)
            raw_l = bx_full[:, -input_len:]
            mask = _mask_last_l(bx_enc.shape[0], bx_enc.shape[2], input_len, device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(adapter(feat, raw_l).float().cpu())
            tgts.append(by.cpu())
            routing.append(adapter.get_routing_stats(raw_l, hidden_states=feat).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(routing)
    mse = nn.MSELoss()(preds, tgts).item()
    entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    max_w = routing.max(dim=-1).values.mean().item()
    return {"mse": mse, "routing_entropy": entropy, "routing_max_weight": max_w,
            "param_count": adapter.param_count()}


def train_fixed_masked(code, model, blocks, X_train, Y_train, X_test, Y_test, input_len,
                       device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                       backbone_type="moment"):
    """Fork of feasibility.standard_data.train_adapter with a last-L input mask.
    The fixed head reads only backbone features, so only the mask changes."""
    hdim = _get_hidden_dim(model)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": __import__("math")}
    exec(code, namespace)
    adapter = namespace["Adapter"](hdim, forecast_horizon).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)
    for _epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in loader:
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = _mask_last_l(bx_enc.shape[0], bx_enc.shape[2], input_len, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                loss = mse_fn(adapter(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval(); adapter.eval()
    eval_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)
    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in eval_loader:
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = _mask_last_l(bx_enc.shape[0], bx_enc.shape[2], input_len, device)
            preds.append(adapter(_extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)).float().cpu())
            tgts.append(by.cpu())
    preds, tgts = torch.cat(preds), torch.cat(tgts)
    return nn.MSELoss()(preds, tgts).item()


def run_cell(dataset, input_len, seed, n_epochs=15, backbone="AutonLab/MOMENT-1-small",
             device="cuda", with_revin=True, with_baselines=True):
    """Train raw RR-MoA (+ revin control + best fixed adapter) at observed
    length ``input_len`` on a strictly frozen MOMENT-small.  Returns a dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_backbone(backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False  # strictly frozen
    bb_type = _detect_backbone_type(backbone)

    splits, _ = load_standard_data(dataset, 96)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    out = {"dataset": dataset, "input_len": int(input_len), "seed": int(seed),
           "horizon": 96, "backbone": backbone, "n_epochs": n_epochs}

    t0 = time.time()
    raw = train_rr_moa_masked(model, blocks, X_train, Y_train, X_test, Y_test, input_len,
                              device=device, n_epochs=n_epochs, backbone_type=bb_type,
                              router_input_mode="raw")
    out["rrmoa_raw_mse"] = raw["mse"]
    out["rrmoa_raw_entropy"] = raw["routing_entropy"]
    out["rrmoa_raw_max_weight"] = raw["routing_max_weight"]
    out["param_count"] = raw["param_count"]

    if with_revin:
        rev = train_rr_moa_masked(model, blocks, X_train, Y_train, X_test, Y_test, input_len,
                                  device=device, n_epochs=n_epochs, backbone_type=bb_type,
                                  router_input_mode="revin")
        out["rrmoa_revin_mse"] = rev["mse"]
        out["rrmoa_revin_entropy"] = rev["routing_entropy"]
        out["delta_vs_revin_pct"] = 100.0 * (raw["mse"] - rev["mse"]) / rev["mse"]

    if with_baselines:
        baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
        bl = {}
        for name, code in baselines.items():
            try:
                bl[name] = train_fixed_masked(code, model, blocks, X_train, Y_train, X_test, Y_test,
                                              input_len, device=device, n_epochs=n_epochs,
                                              backbone_type=bb_type)
            except Exception as exc:  # noqa: BLE001 - record and continue
                bl[name] = None
                out.setdefault("baseline_errors", {})[name] = str(exc)
        valid = {k: v for k, v in bl.items() if v is not None}
        out["baselines"] = bl
        if valid:
            best_name = min(valid, key=valid.get)
            out["best_fixed_mse"] = valid[best_name]
            out["best_fixed_name"] = best_name
            out["delta_vs_fixed_pct"] = 100.0 * (raw["mse"] - valid[best_name]) / valid[best_name]
            out["rrmoa_wins_vs_fixed"] = bool(raw["mse"] < valid[best_name])

    out["elapsed"] = time.time() - t0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-revin", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()

    res = run_cell(args.dataset, args.input_len, args.seed, n_epochs=args.epochs,
                   device=args.device, with_revin=not args.no_revin,
                   with_baselines=not args.no_baselines)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
