"""
Vision MoE Routing Collapse Experiment
=======================================
Proves normalization-induced routing collapse is cross-modal (not just time-series).

Uses pretrained ResNet-18 on CIFAR-10 with K=5 expert classifiers.
Tests 4 conditions:
  A) Unfrozen + InstanceNorm → expect collapse
  B) Frozen   + InstanceNorm → expect no collapse (frozen breaks feedback loop)
  C) Unfrozen + No norm      → expect no collapse
  D) Unfrozen + BatchNorm    → expect collapse

Usage:
  python scripts/run_vision_moe_collapse.py --seed 42 --device cuda
  python scripts/run_vision_moe_collapse.py --seed 42 --device cpu --epochs 2  # smoke test
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ============================================================
# Model components
# ============================================================

def load_resnet18_cifar(pretrained=True):
    """Load ResNet-18 adapted for CIFAR-10 (32x32 input)."""
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    # Adapt for 32x32: replace 7x7 stride-2 conv with 3x3 stride-1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Remove classification head — we use MoE
    model.fc = nn.Identity()
    return model  # outputs (B, 512)


class VisionMoERouter(nn.Module):
    """Lightweight router: features → K expert logits."""

    def __init__(self, d_feat: int, K: int):
        super().__init__()
        self.gate = nn.Linear(d_feat, K)

    def forward(self, x):
        return self.gate(x)  # (B, K)


class VisionMoE(nn.Module):
    """K-expert MoE classifier with optional normalization before routing."""

    def __init__(self, d_feat: int, K: int, num_classes: int, norm_type: str = "none"):
        super().__init__()
        self.K = K
        self.norm_type = norm_type

        # Normalization before router (the variable under test)
        # "instancenorm" = per-sample zero-mean unit-variance (exact RevIN analog)
        # "batchnorm" = BatchNorm1d across the batch
        self.norm_type = norm_type
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(d_feat)
        else:
            self.norm = None  # instancenorm handled manually in forward

        self.router = VisionMoERouter(d_feat, K)
        self.experts = nn.ModuleList([nn.Linear(d_feat, num_classes) for _ in range(K)])

    def forward(self, features):
        """
        Args:
            features: (B, d_feat) from backbone
        Returns:
            logits: (B, num_classes)
            routing_weights: (B, K) softmax probabilities
            routing_entropy: scalar (aggregate entropy)
        """
        # Apply normalization to router input
        if self.norm_type == "instancenorm":
            # Per-sample zero-mean unit-variance (exact RevIN analog)
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True) + 1e-5
            normed = (features - mean) / std
        elif self.norm is not None:
            normed = self.norm(features)
        else:
            normed = features

        # Route
        gate_logits = self.router(normed)
        weights = F.softmax(gate_logits, dim=-1)  # (B, K)

        # Mixture of expert outputs
        expert_outputs = torch.stack([exp(features) for exp in self.experts], dim=1)  # (B, K, C)
        logits = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, C)

        # Aggregate entropy: mean probs across batch, then entropy
        with torch.no_grad():
            avg_probs = weights.mean(dim=0)  # (K,)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()

        return logits, weights, entropy

    def load_balance_loss(self, weights):
        """Encourage uniform expert utilization."""
        # f_i = fraction of samples routed to expert i
        f = weights.mean(dim=0)  # (K,)
        # p_i = mean routing probability for expert i
        p = weights.mean(dim=0)
        return self.K * (f * p).sum()


# ============================================================
# Data
# ============================================================

def get_cifar10(batch_size=128, data_dir="./data"):
    """CIFAR-10 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


# ============================================================
# Training
# ============================================================

def train_condition(backbone, moe, train_loader, test_loader, epochs, device,
                    frozen, lr=1e-3, lb_coeff=0.01):
    """Train one condition, return trajectory and final metrics."""
    backbone.to(device)
    moe.to(device)

    if frozen:
        backbone.eval()
        params = list(moe.parameters())
    else:
        backbone.train()
        params = list(backbone.parameters()) + list(moe.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    trajectory = []
    global_step = 0

    for epoch in range(epochs):
        if not frozen:
            backbone.train()
        moe.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            if frozen:
                with torch.no_grad():
                    features = backbone(images)
            else:
                features = backbone(images)

            logits, weights, entropy = moe(features)

            # Loss
            ce_loss = criterion(logits, labels)
            lb_loss = moe.load_balance_loss(weights)
            loss = ce_loss + lb_coeff * lb_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Log gradient norms
            with torch.no_grad():
                expert_grad_norms = []
                for exp in moe.experts:
                    gnorm = sum(p.grad.norm().item() ** 2 for p in exp.parameters() if p.grad is not None) ** 0.5
                    expert_grad_norms.append(round(gnorm, 6))

                router_grad_norm = sum(
                    p.grad.norm().item() ** 2 for p in moe.router.parameters() if p.grad is not None
                ) ** 0.5

                if not frozen:
                    backbone_grad_norm = sum(
                        p.grad.norm().item() ** 2 for p in backbone.parameters() if p.grad is not None
                    ) ** 0.5
                else:
                    backbone_grad_norm = 0.0

                # Accuracy
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()

            optimizer.step()

            # Log trajectory
            trajectory.append({
                "step": global_step,
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "ce_loss": round(ce_loss.item(), 6),
                "routing_entropy": round(entropy, 6),
                "routing_max_weight": round(weights.mean(dim=0).max().item(), 6),
                "mean_routing_weights": [round(w, 6) for w in weights.mean(dim=0).tolist()],
                "expert_grad_norms": expert_grad_norms,
                "router_grad_norm": round(router_grad_norm, 6),
                "backbone_grad_norm": round(backbone_grad_norm, 6),
                "train_acc": round(acc, 4),
            })
            global_step += 1

    # Test evaluation
    backbone.eval()
    moe.eval()
    correct = 0
    total = 0
    test_entropies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            logits, weights, entropy = moe(features)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            test_entropies.append(entropy)

    test_acc = correct / total
    test_entropy = np.mean(test_entropies)

    return trajectory, test_acc, test_entropy


# ============================================================
# Main
# ============================================================

CONDITIONS = {
    "A_unfrozen_instancenorm": {"frozen": False, "norm_type": "instancenorm"},
    "B_frozen_instancenorm":   {"frozen": True,  "norm_type": "instancenorm"},
    "C_unfrozen_nonorm":       {"frozen": False, "norm_type": "none"},
    "D_unfrozen_batchnorm":    {"frozen": False, "norm_type": "batchnorm"},
}


def main():
    parser = argparse.ArgumentParser(description="Vision MoE Routing Collapse Experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="results/vision_moe")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}, Seed: {args.seed}, Epochs: {args.epochs}")

    # Load data once
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, data_dir=args.data_dir)

    summary = {"seed": args.seed, "epochs": args.epochs, "K": args.K, "conditions": {}}

    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (frozen={cond_cfg['frozen']}, norm={cond_cfg['norm_type']})")
        print(f"{'='*60}")

        # Fix seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Fresh backbone + MoE for each condition
        backbone = load_resnet18_cifar(pretrained=True)
        moe = VisionMoE(d_feat=512, K=args.K, num_classes=10, norm_type=cond_cfg["norm_type"])

        n_backbone = sum(p.numel() for p in backbone.parameters())
        n_moe = sum(p.numel() for p in moe.parameters())
        n_trainable = sum(p.numel() for p in moe.parameters() if p.requires_grad)
        if not cond_cfg["frozen"]:
            n_trainable += sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Backbone params: {n_backbone:,}, MoE params: {n_moe:,}, Trainable: {n_trainable:,}")

        t0 = time.time()
        trajectory, test_acc, test_entropy = train_condition(
            backbone, moe, train_loader, test_loader,
            epochs=args.epochs, device=device,
            frozen=cond_cfg["frozen"], lr=args.lr,
        )
        elapsed = time.time() - t0

        # Extract key trajectory points
        entropies = [t["routing_entropy"] for t in trajectory]
        entropy_step50 = entropies[min(50, len(entropies) - 1)]
        entropy_step200 = entropies[min(200, len(entropies) - 1)]
        entropy_final = entropies[-1]
        max_entropy = math.log(args.K)

        print(f"  Entropy: step0={entropies[0]:.4f}, step50={entropy_step50:.4f}, "
              f"step200={entropy_step200:.4f}, final={entropy_final:.4f} (max={max_entropy:.3f})")
        print(f"  Test acc: {test_acc:.4f}, Elapsed: {elapsed:.1f}s")

        collapsed = entropy_final < 0.1
        print(f"  Collapsed: {'YES' if collapsed else 'NO'}")

        # Save trajectory
        traj_path = os.path.join(args.output_dir, f"trajectory_{cond_name}_{args.seed}.jsonl")
        with open(traj_path, "w") as f:
            for t in trajectory:
                f.write(json.dumps(t) + "\n")

        # Store summary
        summary["conditions"][cond_name] = {
            "frozen": cond_cfg["frozen"],
            "norm_type": cond_cfg["norm_type"],
            "test_acc": round(test_acc, 4),
            "test_entropy": round(test_entropy, 4),
            "entropy_step0": round(entropies[0], 4),
            "entropy_step50": round(entropy_step50, 4),
            "entropy_step200": round(entropy_step200, 4),
            "entropy_final": round(entropy_final, 4),
            "collapsed": collapsed,
            "elapsed_sec": round(elapsed, 1),
            "n_trainable_params": n_trainable,
        }

    # Save summary
    summary_path = os.path.join(args.output_dir, f"collapse_summary_{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print final comparison table
    print(f"\n{'='*80}")
    print(f"SUMMARY (seed={args.seed})")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Entropy(final)':>15} {'Collapsed?':>12} {'Test Acc':>10}")
    print(f"{'-'*70}")
    for cond_name, res in summary["conditions"].items():
        print(f"{cond_name:<30} {res['entropy_final']:>15.4f} "
              f"{'YES' if res['collapsed'] else 'NO':>12} {res['test_acc']:>10.4f}")


if __name__ == "__main__":
    main()
