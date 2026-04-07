"""Synthetic validation: Per-instance normalization destroys MoE routing.

Proves the normalization-routing incompatibility in a controlled setting:
1. Train a GOOD backbone (no normalization) on clustered regression data
2. Freeze backbone — hidden states encode cluster identity
3. Test routing under three conditions:
   A: Route on per-instance-normalized hidden states → collapses
   B: Route on raw hidden states → works
   C: Route on raw input → works

The key insight: per-instance normalization (zero-mean, unit-variance per sample)
strips the location-scale statistics that distinguish clusters, killing routing.
This is the same mechanism as RevIN in TSFMs but in a domain-agnostic setting.

Runs on CPU in <2 minutes. Produces figures/synthetic_routing_collapse.pdf.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

K = 3
SEQ_LEN = 32
D_HIDDEN = 64
D_OUT = 8
N = 3000
BACKBONE_EPOCHS = 200
MOE_EPOCHS = 300
BATCH_SIZE = 128
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]

CLUSTER_MEANS = [0.0, 5.0, -3.0]
CLUSTER_SCALES = [1.0, 3.0, 0.5]


def per_instance_norm(x):
    """Normalize each sample to zero mean, unit variance (like RevIN)."""
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
    return (x - mu) / sigma


def generate_data(seed=0):
    rng = np.random.default_rng(seed)
    cluster_ids = rng.integers(0, K, size=N)
    Ws = [rng.normal(0, 0.3, (D_OUT, SEQ_LEN)).astype(np.float32) for _ in range(K)]

    X = np.zeros((N, SEQ_LEN), dtype=np.float32)
    Y = np.zeros((N, D_OUT), dtype=np.float32)
    for i in range(N):
        k = cluster_ids[i]
        z = rng.normal(0, 1, SEQ_LEN).astype(np.float32)
        X[i] = CLUSTER_MEANS[k] + CLUSTER_SCALES[k] * z
        Y[i] = Ws[k] @ X[i] + rng.normal(0, 0.1, D_OUT).astype(np.float32)

    n_train = int(0.8 * N)
    return (torch.tensor(X[:n_train]), torch.tensor(Y[:n_train]),
            torch.tensor(cluster_ids[:n_train]),
            torch.tensor(X[n_train:]), torch.tensor(Y[n_train:]),
            torch.tensor(cluster_ids[n_train:]))


class Backbone(nn.Module):
    """Good backbone without normalization — learns cluster-aware representations."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN, D_HIDDEN), nn.ReLU(),
            nn.Linear(D_HIDDEN, D_HIDDEN), nn.ReLU(),
            nn.Linear(D_HIDDEN, D_HIDDEN),
        )

    def forward(self, x):
        return self.net(x)


class SimpleMoE(nn.Module):
    def __init__(self, d_router_in, d_expert_in):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_expert_in, D_OUT) for _ in range(K)])
        self.router = nn.Linear(d_router_in, K)

    def forward(self, router_input, expert_input):
        logits = self.router(router_input)
        weights = F.softmax(logits, dim=-1)
        outputs = torch.stack([e(expert_input) for e in self.experts], dim=1)
        y = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        return y, weights


def train_backbone(backbone, X_train, Y_train):
    head = nn.Linear(D_HIDDEN, D_OUT)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=LR)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    backbone.train()
    for _ in range(BACKBONE_EPOCHS):
        for xb, yb in loader:
            pred = head(backbone(xb))
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return loss.item()


def train_moe(moe, backbone, X_train, Y_train, X_test, Y_test, cluster_test,
              normalize_router_input=False, use_raw_for_router=False):
    """Train MoE with frozen backbone.

    Args:
        normalize_router_input: if True, apply per-instance norm to hidden states
                                before feeding to router (simulates RevIN effect)
        use_raw_for_router: if True, router reads raw input x instead of hidden states
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(moe.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(MOE_EPOCHS):
        for xb, yb in loader:
            with torch.no_grad():
                hb = backbone(xb)

            if use_raw_for_router:
                router_in = xb
            elif normalize_router_input:
                router_in = per_instance_norm(hb)
            else:
                router_in = hb

            pred, weights = moe(router_in, hb)
            mse = F.mse_loss(pred, yb)
            avg_w = weights.mean(dim=0)
            load_loss = K * (avg_w * avg_w).sum()
            loss = mse + 0.01 * load_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    backbone.eval()
    moe.eval()
    with torch.no_grad():
        h_test = backbone(X_test)
        if use_raw_for_router:
            router_in = X_test
        elif normalize_router_input:
            router_in = per_instance_norm(h_test)
        else:
            router_in = h_test
        pred, weights = moe(router_in, h_test)
        test_mse = F.mse_loss(pred, Y_test).item()

    avg_probs = weights.mean(dim=0)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()
    routing_acc = (weights.argmax(dim=1) == cluster_test).float().mean().item()

    return {"entropy": entropy, "accuracy": routing_acc, "mse": test_mse}


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, Y_tr, C_tr, X_te, Y_te, C_te = generate_data(seed)

    # Train ONE good backbone (no normalization)
    backbone = Backbone()
    bb_loss = train_backbone(backbone, X_tr, Y_tr)

    results = {}

    # A: Route on NORMALIZED hidden states (per-instance norm applied to h)
    #    Simulates routing on post-RevIN representations
    moe_a = SimpleMoE(D_HIDDEN, D_HIDDEN)
    results["A: Norm hidden"] = train_moe(
        moe_a, backbone, X_tr, Y_tr, X_te, Y_te, C_te,
        normalize_router_input=True)

    # B: Route on RAW hidden states (no normalization)
    moe_b = SimpleMoE(D_HIDDEN, D_HIDDEN)
    results["B: Raw hidden"] = train_moe(
        moe_b, backbone, X_tr, Y_tr, X_te, Y_te, C_te,
        normalize_router_input=False)

    # C: Route on raw input (bypasses backbone entirely for routing)
    moe_c = SimpleMoE(SEQ_LEN, D_HIDDEN)
    results["C: Raw input"] = train_moe(
        moe_c, backbone, X_tr, Y_tr, X_te, Y_te, C_te,
        use_raw_for_router=True)

    return results, bb_loss


def main():
    import os
    conditions = ["A: Norm hidden", "B: Raw hidden", "C: Raw input"]
    all_results = {c: {"entropy": [], "accuracy": [], "mse": []} for c in conditions}

    print("=" * 70)
    print("SYNTHETIC: PER-INSTANCE NORMALIZATION DESTROYS MoE ROUTING")
    print("=" * 70)
    print(f"K={K} clusters, means={CLUSTER_MEANS}, scales={CLUSTER_SCALES}")
    print(f"Same frozen backbone for all conditions (no normalization in backbone)")
    print(f"Condition A normalizes hidden states before router (simulates RevIN)")
    print()

    for seed in SEEDS:
        results, bb_loss = run_seed(seed)
        print(f"Seed {seed} (backbone loss={bb_loss:.4f}):")
        for cond, metrics in results.items():
            for k, v in metrics.items():
                all_results[cond][k].append(v)
            print(f"  {cond:<18} entropy={metrics['entropy']:.3f}  "
                  f"acc={metrics['accuracy']:.1%}  mse={metrics['mse']:.4f}")

    print("\n" + "=" * 70)
    print(f"SUMMARY (mean ± std, {len(SEEDS)} seeds)")
    print("=" * 70)
    print(f"{'Condition':<20} {'Entropy':<16} {'Routing Acc':<16} {'Test MSE'}")
    print("-" * 68)
    for cond in conditions:
        e = np.array(all_results[cond]["entropy"])
        a = np.array(all_results[cond]["accuracy"])
        m = np.array(all_results[cond]["mse"])
        print(f"{cond:<20} {e.mean():.3f}±{e.std():.3f}     "
              f"{a.mean():.1%}±{a.std():.1%}     {m.mean():.4f}±{m.std():.4f}")
    print(f"\nMax entropy = log({K}) = {np.log(K):.3f}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        labels = ["Norm\nhidden", "Raw\nhidden", "Raw\ninput"]
        colors = ["#d62728", "#1f77b4", "#2ca02c"]

        for ax, metric, ylabel, title in zip(
            axes,
            ["entropy", "accuracy", "mse"],
            ["Routing entropy (nats)", "Routing accuracy", "Test MSE"],
            ["(a) Routing Entropy", "(b) Routing Accuracy", "(c) Test MSE"],
        ):
            means = [np.mean(all_results[c][metric]) for c in conditions]
            stds = [np.std(all_results[c][metric]) for c in conditions]
            ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            if metric == "entropy":
                ax.axhline(np.log(K), color="gray", ls="--", lw=1, label=f"max=log({K})")
                ax.legend(fontsize=7)
            if metric == "accuracy":
                ax.axhline(1.0 / K, color="gray", ls="--", lw=1, label=f"chance=1/{K}")
                ax.legend(fontsize=7)
                ax.set_ylim(0, 1.05)

        plt.tight_layout()
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/synthetic_routing_collapse.pdf", bbox_inches="tight", dpi=300)
        print("\nFigure saved to figures/synthetic_routing_collapse.pdf")
    except ImportError:
        print("\nMatplotlib not available — skipping figure.")

    import json
    os.makedirs("results/analysis", exist_ok=True)
    with open("results/analysis/synthetic_routing_collapse.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
