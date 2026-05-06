"""AdaMix: Learned Adapter Mixture for TSFM Adaptation.

Instead of searching for one adapter architecture, trains a fixed mixture
of K canonical adapter heads with a learned per-sample router.

Key insight from DARTS experiments: the soft mixture (undiscretized supernet)
beats any single discretized architecture. AdaMix makes this permanent —
instance-level routing lets different time series windows get different
adapter combinations.

Usage:
    python scripts/run_adamix.py --dataset ETTh1
    python scripts/run_adamix.py --dataset ETTh1 --K 3
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
from feasibility.adapter_seeds import SEED_ADAPTERS, validate_adapter_code
from feasibility.standard_data import (
    load_standard_data, train_adapter, _detect_backbone_type,
    compute_denorm_mse,
)


# --- Individual adapter heads (simple, canonical designs) ---

class MeanPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.mean(dim=1)
        return self.out(F.gelu(self.fc(x)))


class LastTokenHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h[:, -1, :]
        return self.out(F.gelu(self.fc(x)))


class MaxPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.max(dim=1).values
        return self.out(F.gelu(self.fc(x)))


class AttentionPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        x = (h * w).sum(dim=1)
        return self.out(F.gelu(self.fc(x)))


class Conv1dPoolHead(nn.Module):
    def __init__(self, d_model, output_dim, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, hidden, kernel_size=8, stride=4, padding=2)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, h):
        x = h.permute(0, 2, 1)  # (B, d, T)
        x = F.gelu(self.conv(x))  # (B, hidden, T')
        x = x.mean(dim=2)  # (B, hidden)
        return self.out(x)


# --- AdaMix: Learned Adapter Mixture ---

HEAD_CLASSES = [MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead]
HEAD_NAMES = ["mean", "last", "max", "attention", "conv1d"]


class AdaMix(nn.Module):
    """Learned mixture of K canonical adapter heads with per-sample routing.

    The router produces instance-level weights over K adapter heads.
    Different time series windows get different adapter combinations.

    Supports four MoE-rescue variants evaluated in the RR-MoA rescue sub-table:
      - router_type="softmax": standard Switch Transformer softmax gating
      - router_type="relu":     ReMoE ReLU gating (Wang et al. ICLR 2025)
      - router_type="expert-choice": Zhou et al. 2022 per-expert top-k
      - load_balance_variant="mean-prob": legacy AdaMix (f_i = P_i; preserved
        for reproducibility of prior paper Table 2 numbers)
      - load_balance_variant="argmax":    correct Switch Transformer Eq. 4-6
        (f_i uses argmax indicator, P_i the mean softmax probability)

    router_input="hidden" (default): routes on mean-pooled hidden states.
    router_input="raw": routes on raw pre-normalization input (Exp D ablation).
    """
    def __init__(self, d_model, output_dim, K=5, hidden=64, router_hidden=32,
                 router_type="softmax",
                 load_balance_coef=0.01,
                 load_balance_variant="mean-prob",
                 entropy_reg_coef=0.0,
                 z_loss_coef=0.0,
                 relu_l1_coef=0.0,
                 capacity_factor=2.0,
                 router_input="hidden",
                 input_len=512):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.router_type = router_type
        self.router_input = router_input
        self.load_balance_coef = load_balance_coef
        self.load_balance_variant = load_balance_variant
        self.entropy_reg_coef = entropy_reg_coef
        self.z_loss_coef = z_loss_coef
        self.relu_l1_coef = relu_l1_coef
        self.capacity_factor = capacity_factor

        # Back-compat attribute (older scripts reference .load_balance_coeff)
        self.load_balance_coeff = load_balance_coef

        # K adapter heads
        self.adapters = nn.ModuleList([
            HEAD_CLASSES[i % len(HEAD_CLASSES)](d_model, output_dim, hidden)
            for i in range(K)
        ])

        # Instance-level router
        if router_input == "raw":
            # Exp D: Conv1d raw-input router (same architecture as RR-MoA)
            self.raw_router = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(4),
            )
            self.router = nn.Linear(64, K)
        else:
            self.router = nn.Sequential(
                nn.Linear(d_model, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, K),
            )

    # ------------------------------------------------------------------ #
    #  Routing primitives                                                #
    # ------------------------------------------------------------------ #

    def _compute_logits(self, hidden_states, raw_input=None):
        """Shared: mean-pool to (B, d_model) and project to (B, K).
        If router_input='raw', uses raw_input instead of hidden_states."""
        if self.router_input == "raw" and raw_input is not None:
            x = raw_input.unsqueeze(1)  # (B, 1, L)
            feat = self.raw_router(x).flatten(1)  # (B, 64)
            return self.router(feat)
        h_summary = hidden_states.mean(dim=1)
        return self.router(h_summary)  # (B, K)

    def _compute_weights(self, logits):
        """Dispatch to the requested routing scheme.  Returns (B, K) per-sample
        mixture weights that sum to 1 along dim=-1 (or close to it, for
        expert-choice fallback samples)."""
        if self.router_type == "softmax":
            return F.softmax(logits, dim=-1)

        if self.router_type == "relu":
            # ReMoE: unnormalized ReLU gate.  For a mixture-of-adapters we
            # renormalize per sample so the output magnitude matches softmax;
            # L1 sparsity pressure is applied on the raw ReLU values via
            # relu_l1_loss() below.
            raw = F.relu(logits)
            denom = raw.sum(dim=-1, keepdim=True)
            # Fallback to softmax for any sample where all logits are <=0.
            fallback = F.softmax(logits, dim=-1)
            weights = torch.where(
                denom > 1e-8,
                raw / denom.clamp_min(1e-8),
                fallback,
            )
            return weights

        if self.router_type == "expert-choice":
            # Zhou et al. 2022 expert-choice: each expert picks top-k samples,
            # with k = B * c / K (c = capacity factor).  Simplified for the
            # mixture-of-adapters context where we cannot drop samples.
            B = logits.shape[0]
            S = F.softmax(logits, dim=-1)           # (B, K) token->expert probs
            k = max(1, min(B, int(round(B * self.capacity_factor / self.K))))
            S_T = S.transpose(0, 1)                 # (K, B)
            _, top_idx = torch.topk(S_T, k, dim=-1)  # (K, k)
            mask = torch.zeros_like(S_T)
            mask.scatter_(1, top_idx, 1.0)          # (K, B) 1 where expert picked
            gate = S_T * mask                       # (K, B)
            weights = gate.transpose(0, 1)          # (B, K)
            denom = weights.sum(dim=-1, keepdim=True)
            # Samples not selected by any expert fall back to softmax (very
            # rare when capacity_factor>=1).
            weights = torch.where(
                denom > 1e-8,
                weights / denom.clamp_min(1e-8),
                S,
            )
            return weights

        raise ValueError("Unknown router_type: %s" % self.router_type)

    def forward(self, hidden_states, raw_input=None):
        self._last_raw_input = raw_input  # cache for aux losses
        logits = self._compute_logits(hidden_states, raw_input)
        weights = self._compute_weights(logits)  # (B, K)

        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)  # (B, K, H)
        mixed = (weights.unsqueeze(-1) * outputs).sum(dim=1)  # (B, H)
        return mixed

    def get_routing_stats(self, hidden_states, raw_input=None):
        """Return (B, K) mixture weights for analysis (no grad)."""
        with torch.no_grad():
            logits = self._compute_logits(hidden_states, raw_input)
            weights = self._compute_weights(logits)
        return weights

    # ------------------------------------------------------------------ #
    #  Auxiliary loss terms (all return scalars, pre-coefficient)        #
    # ------------------------------------------------------------------ #

    def load_balance_loss(self, hidden_states):
        """Switch Transformer auxiliary loss (Fedus et al. 2022, Eq. 4-6).

        L_B = N * Σ_i f_i * P_i

        - load_balance_variant == "mean-prob": legacy formulation where f_i = P_i
          (equivalent to N * Σ P_i^2).  Preserved so prior Table 2 numbers in
          the paper reproduce exactly.
        - load_balance_variant == "argmax":    correct Switch Transformer form
          where f_i uses the non-differentiable argmax indicator while P_i
          remains the differentiable mean probability.
        """
        raw = getattr(self, '_last_raw_input', None)
        logits = self._compute_logits(hidden_states, raw)
        probs = F.softmax(logits, dim=-1)          # (B, K)
        P_i = probs.mean(dim=0)                    # (K,)
        if self.load_balance_variant == "argmax":
            argmax_idx = probs.argmax(dim=-1)      # (B,)
            f_i = F.one_hot(argmax_idx, self.K).float().mean(dim=0)  # (K,)
        else:
            f_i = P_i
        return self.K * (f_i * P_i).sum()

    def entropy_regularization(self, hidden_states):
        """Negative entropy of the router distribution: −H(p).

        Added with a POSITIVE coefficient, this encourages HIGH routing
        entropy (forces diversity).  Returning −H rather than +H means the
        training-loop combination line stays the simple pattern
            loss = mse + coef * term.
        """
        raw = getattr(self, '_last_raw_input', None)
        logits = self._compute_logits(hidden_states, raw)
        probs = F.softmax(logits, dim=-1)
        H = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1).mean()
        return -H

    def z_loss(self, hidden_states):
        """ST-MoE router z-loss (Zoph et al. 2022, Eq. 5):
            L_z = (1/B) * Σ_i (log Σ_j exp(x_j^(i)))^2

        Penalizes large router logit magnitudes; empirically stabilizes
        bfloat16 training and the routing distribution."""
        raw = getattr(self, '_last_raw_input', None)
        logits = self._compute_logits(hidden_states, raw)
        lse = torch.logsumexp(logits, dim=-1)      # (B,)
        return (lse ** 2).mean()

    def relu_l1_loss(self, hidden_states):
        """ReMoE L1 sparsity regularization (Wang et al. 2025, Eq. 9).

        Only meaningful when router_type == 'relu'; we return 0 otherwise so
        the training loop can unconditionally add `relu_l1_coef * relu_l1_loss()`.
        Uses the load-balanced variant (Eq. 10) which weights over-active
        experts more aggressively, identical to Switch load balancing up to a
        constant once experts are used."""
        if self.router_type != "relu":
            return hidden_states.new_zeros(())
        raw = getattr(self, '_last_raw_input', None)
        logits = self._compute_logits(hidden_states, raw)
        raw = F.relu(logits)                       # (B, K)
        # Non-differentiable f_{l,e} = (K / k*B) * count(raw > 0) [k=1 default].
        active = (raw > 0).float()
        f_e = active.mean(dim=0) * self.K           # (K,) (k/E normalization)
        # Avoid zero gradient when everyone is firing
        weight = f_e.detach().clamp_min(1.0)
        return (weight * raw.mean(dim=0)).sum()

    def total_aux_loss(self, hidden_states):
        """Sum of all enabled auxiliary losses, each pre-multiplied by its
        coefficient.  Called from the training loop as a single line."""
        total = hidden_states.new_zeros(())
        if self.load_balance_coef != 0.0:
            total = total + self.load_balance_coef * self.load_balance_loss(hidden_states)
        if self.entropy_reg_coef != 0.0:
            total = total + self.entropy_reg_coef * self.entropy_regularization(hidden_states)
        if self.z_loss_coef != 0.0:
            total = total + self.z_loss_coef * self.z_loss(hidden_states)
        if self.relu_l1_coef != 0.0 and self.router_type == "relu":
            total = total + self.relu_l1_coef * self.relu_l1_loss(hidden_states)
        return total

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def train_adamix(model, blocks, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", K=5, hidden=64, test_ch=None, scaler=None,
                 trajectory_path=None, trajectory_max_steps=400,
                 router_type="softmax",
                 load_balance_coef=0.01,
                 load_balance_variant="mean-prob",
                 entropy_reg_coef=0.0,
                 z_loss_coef=0.0,
                 relu_l1_coef=0.0,
                 capacity_factor=2.0,
                 router_input="hidden"):
    """Train AdaMix adapter.

    If ``trajectory_path`` is not None, writes a JSONL file with per-step
    routing / gradient diagnostics for the first ``trajectory_max_steps``
    optimizer steps. Used by T2.A to provide mechanistic evidence for the
    gradient co-adaptation hypothesis (see review W5): under an unfrozen
    backbone, we expect to see (a) the router entropy collapse toward 0 within
    a few dozen steps, (b) one expert's gradient norm grow relative to the
    others, and (c) the backbone encoder-block gradient norm remain active
    (confirming joint updates to $theta_active$). Under the frozen control,
    (a)--(c) should all be absent.
    """
    hdim = _get_hidden_dim(model)
    adapter = AdaMix(
        hdim, forecast_horizon, K=K, hidden=hidden,
        router_type=router_type,
        load_balance_coef=load_balance_coef,
        load_balance_variant=load_balance_variant,
        entropy_reg_coef=entropy_reg_coef,
        z_loss_coef=z_loss_coef,
        relu_l1_coef=relu_l1_coef,
        capacity_factor=capacity_factor,
        router_input=router_input,
    ).to(device)

    # Collect trainable params (adapter + unfrozen backbone)
    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p)
            pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float(),
    ), batch_size=batch_size)

    # T2.A trajectory logging: open file and take a snapshot of the
    # unfrozen encoder-block parameters for grad-norm tracking.
    trajectory_file = None
    if trajectory_path is not None:
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)
        trajectory_file = open(trajectory_path, "w")
    unfrozen_block_params = [p for b in blocks for p in b.parameters() if p.requires_grad]
    step_counter = [0]  # mutable closure

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            raw_flat = bx.to(device)  # (B, L) — raw input before unsqueeze
            bx, by = raw_flat.unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
                raw_for_router = raw_flat if router_input == "raw" else None
                pred = adapter(feat, raw_input=raw_for_router)
                loss = mse_fn(pred, by) + adapter.total_aux_loss(feat)
            optimizer.zero_grad()
            loss.backward()

            # T2.A: snapshot gradient / routing state BEFORE optimizer.step().
            if trajectory_file is not None and step_counter[0] < trajectory_max_steps:
                with torch.no_grad():
                    h_summary = feat.mean(dim=1)
                    logits = adapter.router(h_summary)
                    weights = torch.softmax(logits, dim=-1).float()
                    mean_weights = weights.mean(dim=0)
                    entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
                    max_w = weights.max(dim=-1).values.mean().item()

                    expert_grad_norms = []
                    for e_idx, expert in enumerate(adapter.adapters):
                        g_sq = 0.0
                        for p in expert.parameters():
                            if p.grad is not None:
                                g_sq += float(p.grad.detach().float().pow(2).sum().item())
                        expert_grad_norms.append(g_sq ** 0.5)

                    router_grad_sq = 0.0
                    for p in adapter.router.parameters():
                        if p.grad is not None:
                            router_grad_sq += float(p.grad.detach().float().pow(2).sum().item())
                    router_grad_norm = router_grad_sq ** 0.5

                    backbone_grad_sq = 0.0
                    for p in unfrozen_block_params:
                        if p.grad is not None:
                            backbone_grad_sq += float(p.grad.detach().float().pow(2).sum().item())
                    backbone_grad_norm = backbone_grad_sq ** 0.5

                    trajectory_file.write(json.dumps({
                        "step": step_counter[0],
                        "epoch": epoch,
                        "loss": float(loss.item()),
                        "routing_entropy": float(entropy),
                        "routing_max_weight": float(max_w),
                        "mean_routing_weights": [float(w) for w in mean_weights.tolist()],
                        "expert_grad_norms": expert_grad_norms,
                        "router_grad_norm": router_grad_norm,
                        "backbone_unfrozen_grad_norm": backbone_grad_norm,
                    }) + "\n")
                    trajectory_file.flush()
                step_counter[0] += 1

            optimizer.step()

    if trajectory_file is not None:
        trajectory_file.close()
        print("Trajectory saved: %s (%d steps)" % (trajectory_path, step_counter[0]))

    # Evaluate on test
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    all_routing_weights = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            raw_flat = bx.to(device)
            bx, by = raw_flat.unsqueeze(1), by.to(device)
            mask = torch.ones(bx.shape[0], bx.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=backbone_type)
            raw_for_router = raw_flat if router_input == "raw" else None
            preds.append(adapter(feat, raw_input=raw_for_router).float().cpu())
            tgts.append(by.cpu())
            all_routing_weights.append(adapter.get_routing_stats(feat, raw_input=raw_for_router).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing_weights)

    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()

    # Routing analysis
    mean_routing = routing.mean(dim=0).tolist()
    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()

    out = {
        "mse": mse,
        "mae": mae,
        "param_count": adapter.param_count(),
        "mean_routing_weights": {HEAD_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(HEAD_NAMES)])},
        "routing_entropy": routing_entropy,
    }
    if test_ch is not None and scaler is not None:
        mse_d, mae_d = compute_denorm_mse(preds, tgts, test_ch, scaler)
        out["mse_denorm"] = mse_d
        out["mae_denorm"] = mae_d
    return out


def _apply_unfreeze(blocks, unfreeze):
    """Selectively unfreeze encoder blocks."""
    n = len(blocks)
    if unfreeze == "frozen":
        return
    elif unfreeze == "last2":
        start = max(0, n - 2)
    elif unfreeze == "last4":
        start = max(0, n - 4)
    elif unfreeze == "all":
        start = 0
    else:
        raise ValueError("Unknown unfreeze: %s" % unfreeze)
    for i in range(start, n):
        for p in blocks[i].parameters():
            p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--unfreeze", default="last4", choices=["frozen", "last2", "last4", "all"],
                        help="Backbone unfreezing strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectory", default=None,
                        help="Path to write per-step routing/gradient trajectory JSONL "
                             "(T2.A mechanistic verification). If omitted, no trajectory is "
                             "recorded. Example: --trajectory results/adamix/trajectory_ETTh1_last4_42.jsonl")
    parser.add_argument("--trajectory-max-steps", type=int, default=400)
    parser.add_argument("--disable-revin", action="store_true",
                        help="Controlled ablation: disable RevIN inside MOMENT backbone")
    parser.add_argument("--norm-type", default="revin",
                        choices=["revin", "batchnorm", "groupnorm"],
                        help="Normalization type inside MOMENT (generalization test)")
    # --- MoE rescue-baseline flags (Section 3 rescue sub-table) -----------
    parser.add_argument("--router-type", default="softmax",
                        choices=["softmax", "relu", "expert-choice"],
                        help="Routing mechanism. softmax=Switch Transformer "
                             "(default/legacy), relu=ReMoE ReLU gate (Wang et "
                             "al. ICLR 2025), expert-choice=Zhou et al. 2022 "
                             "per-expert top-k.")
    parser.add_argument("--load-balance-coef", type=float, default=0.01,
                        help="Coefficient on Switch Transformer load-balance "
                             "auxiliary loss (alpha in Fedus et al. Eq. 4). "
                             "Default 0.01 matches paper Table 2.")
    parser.add_argument("--load-balance-variant", default="mean-prob",
                        choices=["mean-prob", "argmax"],
                        help="mean-prob: legacy AdaMix form f_i=P_i (reproduces "
                             "existing paper numbers). argmax: correct Switch "
                             "Transformer form where f_i is the argmax indicator.")
    parser.add_argument("--entropy-reg-coef", type=float, default=0.0,
                        help="Coefficient on -H(router) regularizer. Positive "
                             "values encourage uniform routing. Rescue sweep: "
                             "{0, 0.01, 0.1, 1.0}.")
    parser.add_argument("--z-loss-coef", type=float, default=0.0,
                        help="Coefficient on ST-MoE router z-loss (Zoph et al. "
                             "2022 Eq. 5). Recommended 0.001; rescue sweep "
                             "{0, 0.001, 0.01, 0.1}.")
    parser.add_argument("--relu-l1-coef", type=float, default=0.0,
                        help="Coefficient on ReMoE L1 sparsity regularizer. "
                             "Only used when --router-type=relu. Rescue sweep "
                             "uses fixed value {0.001, 0.01, 0.1}.")
    parser.add_argument("--capacity-factor", type=float, default=2.0,
                        help="Expert-choice routing capacity factor c. "
                             "k = B*c/K tokens per expert (Zhou et al. 2022).")
    parser.add_argument("--router-input", default="hidden",
                        choices=["hidden", "raw"],
                        help="Router input source. hidden=mean-pooled hidden states "
                             "(default AdaMix). raw=pre-normalization raw input "
                             "(Exp D: proves diagnosis is architecture-agnostic).")
    parser.add_argument("--results-dir", default="results/adamix",
                        help="Output directory. Rescue sweep writes to "
                             "results/adamix_rescue/.")
    parser.add_argument("--run-baselines", default="auto",
                        choices=["auto", "yes", "no"],
                        help="Whether to also run fixed-head baselines. "
                             "'auto' skips baselines when rescue flags are set "
                             "(to keep the rescue sweep cheap), runs them "
                             "otherwise.")
    args = parser.parse_args()

    # Detect whether we're running a rescue-sweep config (any non-default
    # rescue flag set).  Default baseline configurations are written to
    # results/adamix/ to avoid churning existing experiments; rescue configs
    # go to results/adamix_rescue/ for easy analysis.
    rescue_active = (
        args.router_type != "softmax"
        or args.entropy_reg_coef != 0.0
        or args.z_loss_coef != 0.0
        or args.relu_l1_coef != 0.0
        or args.load_balance_variant != "mean-prob"
        or args.load_balance_coef != 0.01
    )
    results_dir = args.results_dir
    if rescue_active and results_dir == "results/adamix":
        results_dir = "results/adamix_rescue"

    os.makedirs(results_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device,
                          disable_revin=args.disable_revin,
                          norm_type=args.norm_type)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, args.unfreeze)
    backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("d_model=%d, K=%d, hidden=%d, unfreeze=%s, backbone_trainable=%d" % (
        hdim, args.K, args.hidden, args.unfreeze, backbone_trainable))

    # Load data
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")
    print("%s H=%d: train=%d, val=%d, test=%d" % (
        args.dataset, args.horizon, len(X_train), len(X_val), len(X_test)))

    # === Run AdaMix ===
    print("\n" + "=" * 60)
    print("AdaMix: %s H=%d K=%d seed=%d" % (args.dataset, args.horizon, args.K, args.seed))
    print("=" * 60)

    start = time.time()
    result = train_adamix(
        model, blocks, X_train, Y_train, X_val, Y_val, X_test, Y_test,
        device=args.device, n_epochs=args.epochs, forecast_horizon=args.horizon,
        backbone_type=bb_type, K=args.K, hidden=args.hidden,
        test_ch=test_ch, scaler=scaler,
        trajectory_path=args.trajectory,
        trajectory_max_steps=args.trajectory_max_steps,
        router_type=args.router_type,
        load_balance_coef=args.load_balance_coef,
        load_balance_variant=args.load_balance_variant,
        entropy_reg_coef=args.entropy_reg_coef,
        z_loss_coef=args.z_loss_coef,
        relu_l1_coef=args.relu_l1_coef,
        capacity_factor=args.capacity_factor,
        router_input=args.router_input,
    )
    elapsed = time.time() - start

    print("AdaMix: MSE=%.4f  MAE=%.4f  params=%d  time=%.0fs" % (
        result["mse"], result["mae"], result["param_count"], elapsed))
    if "mse_denorm" in result:
        print("AdaMix: MSE_denorm=%.4f  MAE_denorm=%.4f (original units)" % (
            result["mse_denorm"], result["mae_denorm"]))
    print("Routing: %s" % result["mean_routing_weights"])
    print("Routing entropy: %.3f (max=%.3f for K=%d)" % (
        result["routing_entropy"], np.log(args.K), args.K))

    # === Run fixed baselines (skipped during rescue sweeps) ===
    run_baselines = (args.run_baselines == "yes" or
                     (args.run_baselines == "auto" and not rescue_active))
    baseline_results = {}
    if run_baselines:
        print("\nFixed baselines (%d epochs, unfreeze=%s):" % (args.epochs, args.unfreeze))
        # Reload model for fair baseline comparison
        model2 = load_backbone(args.backbone, args.device,
                               disable_revin=args.disable_revin)
        _disable_gradient_checkpointing(model2)
        blocks2 = _get_encoder_blocks(model2)
        for p in model2.parameters():
            p.requires_grad = False
        _apply_unfreeze(blocks2, args.unfreeze)

        baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
        for name, code in baselines.items():
            try:
                tr = train_adapter(code, model2, blocks2, X_train, Y_train, X_test, Y_test,
                                   device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                                   backbone_type=bb_type, eval_ch=test_ch, scaler=scaler)
                baseline_results[name] = tr
                if "mse_denorm" in tr:
                    print("  %-15s MSE=%.4f  MSE_denorm=%.4f  params=%d" % (
                        name, tr["mse"], tr["mse_denorm"], tr["param_count"]))
                else:
                    print("  %-15s MSE=%.4f  params=%d" % (name, tr["mse"], tr["param_count"]))
            except Exception as e:
                print("  %-15s ERROR: %s" % (name, e))
    else:
        print("\nFixed baselines SKIPPED (rescue sweep mode)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    print("AdaMix (K=%d):  MSE=%.4f  params=%d  time=%.0fs" % (
        args.K, result["mse"], result["param_count"], elapsed))
    if baseline_results:
        best_bl_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"])
        best_bl_mse = baseline_results[best_bl_name]["mse"]
        delta = (result["mse"] - best_bl_mse) / best_bl_mse * 100
        winner = "AdaMix" if result["mse"] < best_bl_mse else "BASELINE"
        print("Best baseline: MSE=%.4f  (%s)" % (best_bl_mse, best_bl_name))
        print("Delta: %+.1f%% -> Winner: %s" % (delta, winner))
    else:
        delta = None
        winner = None

    # Save
    scaler_info = None
    if scaler is not None:
        scaler_info = {
            "scale_": [float(x) for x in scaler.scale_],
            "mean_": [float(x) for x in scaler.mean_],
            "mean_scale_sq": float(np.mean(scaler.scale_ ** 2)),
        }
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "K": args.K,
        "hidden": args.hidden,
        "unfreeze": args.unfreeze,
        "disable_revin": args.disable_revin,
        "backbone_trainable_params": backbone_trainable,
        "adamix": result,
        "elapsed": elapsed,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner,
        "delta_pct": delta,
        "scaler": scaler_info,
        # Rescue-baseline configuration (present on every run so the summary
        # scripts can slice by these fields):
        "router_type": args.router_type,
        "load_balance_coef": args.load_balance_coef,
        "load_balance_variant": args.load_balance_variant,
        "entropy_reg_coef": args.entropy_reg_coef,
        "z_loss_coef": args.z_loss_coef,
        "relu_l1_coef": args.relu_l1_coef,
        "capacity_factor": args.capacity_factor,
        "rescue_active": rescue_active,
    }
    revin_suffix = "_no_revin" if args.disable_revin else ""
    norm_suffix = "_%s" % args.norm_type if args.norm_type != "revin" else ""
    raw_suffix = "_rawrouter" if args.router_input == "raw" else ""
    # Encode the rescue configuration in the filename so concurrent sweeps
    # never collide on the same output file.
    if rescue_active:
        rescue_tag = "_rtr%s_lb%g_lv%s_ent%g_z%g_l1%g_cf%g" % (
            args.router_type, args.load_balance_coef, args.load_balance_variant,
            args.entropy_reg_coef, args.z_loss_coef, args.relu_l1_coef,
            args.capacity_factor,
        )
    else:
        rescue_tag = ""
    path = "%s/%s_H%d_K%d_%s_%d%s%s%s%s.json" % (
        results_dir, args.dataset, args.horizon, args.K, args.unfreeze,
        args.seed, revin_suffix, norm_suffix, raw_suffix, rescue_tag,
    )
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
