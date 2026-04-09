"""
Vision MoE Routing Collapse Experiment — V2
=============================================
Proves normalization-induced routing collapse is cross-modal.

Key changes from V1:
  - Architecturally diverse experts (mean-pool, max-pool, attention-pool,
    conv-pool, last-patch) — matching the time-series RR-MoA setup
  - InstanceNorm2d on feature maps BEFORE pooling (closer RevIN analog)
  - No load-balancing loss (allows natural specialization/collapse)
  - Routes on POOLED features from normalized feature maps (AdaMix analog)

Tests 4 conditions:
  A) Unfrozen + InstanceNorm2d on feature maps → expect collapse
  B) Frozen   + InstanceNorm2d on feature maps → expect no collapse
  C) Unfrozen + No norm → expect no collapse
  D) Unfrozen + BatchNorm2d on feature maps → expect collapse

Usage:
  python scripts/run_vision_moe_collapse_v2.py --seed 42 --device cuda
  python scripts/run_vision_moe_collapse_v2.py --seed 42 --device cpu --epochs 2
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
# Backbone: ResNet-18 adapted for CIFAR-10, returns feature maps
# ============================================================

class ResNet18FeatureExtractor(nn.Module):
    """ResNet-18 that returns feature maps (B, 512, H, W) instead of logits."""

    def __init__(self, pretrained=True):
        super().__init__()
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Adapt for 32x32 CIFAR input
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        # Keep everything except avgpool and fc
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 512, 4, 4) for CIFAR-10


# ============================================================
# Architecturally diverse expert heads
# ============================================================

class MeanPoolExpert(nn.Module):
    """Global average pooling → linear."""
    def __init__(self, d_feat, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_feat, num_classes)

    def forward(self, feat_maps):
        x = feat_maps.mean(dim=(-2, -1))  # (B, C)
        return self.fc(x)


class MaxPoolExpert(nn.Module):
    """Global max pooling → linear."""
    def __init__(self, d_feat, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_feat, num_classes)

    def forward(self, feat_maps):
        x = feat_maps.amax(dim=(-2, -1))  # (B, C)
        return self.fc(x)


class AttentionPoolExpert(nn.Module):
    """Learned attention over spatial positions → weighted pool → linear."""
    def __init__(self, d_feat, num_classes):
        super().__init__()
        self.attn = nn.Linear(d_feat, 1)
        self.fc = nn.Linear(d_feat, num_classes)

    def forward(self, feat_maps):
        B, C, H, W = feat_maps.shape
        x = feat_maps.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        weights = F.softmax(self.attn(x), dim=1)  # (B, HW, 1)
        pooled = (x * weights).sum(dim=1)  # (B, C)
        return self.fc(pooled)


class ConvPoolExpert(nn.Module):
    """1x1 conv → ReLU → global avg pool → linear."""
    def __init__(self, d_feat, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(d_feat, d_feat // 4, kernel_size=1)
        self.fc = nn.Linear(d_feat // 4, num_classes)

    def forward(self, feat_maps):
        x = F.relu(self.conv(feat_maps))
        x = x.mean(dim=(-2, -1))  # (B, C//4)
        return self.fc(x)


class LastPatchExpert(nn.Module):
    """Take bottom-right patch features → linear (analog of last-token)."""
    def __init__(self, d_feat, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_feat, num_classes)

    def forward(self, feat_maps):
        x = feat_maps[:, :, -1, -1]  # (B, C) — last spatial position
        return self.fc(x)


# ============================================================
# Vision MoE with diverse experts
# ============================================================

class VisionMoEv2(nn.Module):
    """MoE with diverse expert architectures and optional normalization on feature maps."""

    def __init__(self, d_feat: int, num_classes: int, norm_type: str = "none"):
        super().__init__()
        self.norm_type = norm_type

        # Normalization on feature maps BEFORE pooling (RevIN analog)
        if norm_type == "instancenorm":
            self.norm = nn.InstanceNorm2d(d_feat, affine=False)
        elif norm_type == "batchnorm":
            self.norm = nn.BatchNorm2d(d_feat)
        else:
            self.norm = None

        # Router: pool normalized features → gate logits
        self.router_pool = nn.AdaptiveAvgPool2d(1)
        self.router_gate = nn.Linear(d_feat, 5)

        # 5 architecturally distinct experts (matching time-series RR-MoA)
        self.experts = nn.ModuleList([
            MeanPoolExpert(d_feat, num_classes),
            MaxPoolExpert(d_feat, num_classes),
            AttentionPoolExpert(d_feat, num_classes),
            ConvPoolExpert(d_feat, num_classes),
            LastPatchExpert(d_feat, num_classes),
        ])
        self.K = len(self.experts)

    def forward(self, feat_maps):
        """
        Args:
            feat_maps: (B, C, H, W) from backbone
        Returns:
            logits: (B, num_classes)
            routing_weights: (B, K)
            routing_entropy: scalar
        """
        # Apply normalization to feature maps (strips per-channel spatial statistics)
        if self.norm is not None:
            normed = self.norm(feat_maps)
        else:
            normed = feat_maps

        # Route on pooled normalized features
        pooled = self.router_pool(normed).flatten(1)  # (B, C)
        gate_logits = self.router_gate(pooled)
        weights = F.softmax(gate_logits, dim=-1)  # (B, K)

        # Experts operate on normalized features (same as AdaMix)
        expert_outputs = torch.stack(
            [exp(normed) for exp in self.experts], dim=1
        )  # (B, K, num_classes)
        logits = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, num_classes)

        # Aggregate entropy
        with torch.no_grad():
            avg_probs = weights.mean(dim=0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()

        return logits, weights, entropy


# ============================================================
# Data
# ============================================================

def get_cifar10(batch_size=128, data_dir="./data"):
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

def train_condition(backbone, moe, train_loader, test_loader, epochs, device, frozen, lr=1e-3):
    """Train one condition. No load-balancing loss — let routing specialize naturally."""
    backbone.to(device)
    moe.to(device)

    if frozen:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
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
                    feat_maps = backbone(images)
            else:
                feat_maps = backbone(images)

            logits, weights, entropy = moe(feat_maps)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Log gradient norms
            with torch.no_grad():
                expert_grad_norms = []
                for exp in moe.experts:
                    gnorm = sum(
                        p.grad.norm().item() ** 2
                        for p in exp.parameters() if p.grad is not None
                    ) ** 0.5
                    expert_grad_norms.append(round(gnorm, 6))

                router_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in moe.router_gate.parameters() if p.grad is not None
                ) ** 0.5

                if not frozen:
                    backbone_grad_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in backbone.parameters() if p.grad is not None
                    ) ** 0.5
                else:
                    backbone_grad_norm = 0.0

                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()

                # Per-expert usage (fraction of samples where each expert has max weight)
                max_expert = weights.argmax(dim=-1)
                usage = [(max_expert == k).float().mean().item() for k in range(moe.K)]

            optimizer.step()

            trajectory.append({
                "step": global_step,
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "routing_entropy": round(entropy, 6),
                "routing_max_weight": round(weights.mean(dim=0).max().item(), 6),
                "mean_routing_weights": [round(w, 6) for w in weights.mean(dim=0).tolist()],
                "expert_usage": [round(u, 4) for u in usage],
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
            feat_maps = backbone(images)
            logits, weights, entropy = moe(feat_maps)
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
    parser = argparse.ArgumentParser(description="Vision MoE Routing Collapse V2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="results/vision_moe_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}, Seed: {args.seed}, Epochs: {args.epochs}")

    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, data_dir=args.data_dir)

    summary = {"seed": args.seed, "epochs": args.epochs, "K": 5, "conditions": {}}

    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (frozen={cond_cfg['frozen']}, norm={cond_cfg['norm_type']})")
        print(f"{'='*60}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        backbone = ResNet18FeatureExtractor(pretrained=True)
        moe = VisionMoEv2(d_feat=512, num_classes=10, norm_type=cond_cfg["norm_type"])

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

        entropies = [t["routing_entropy"] for t in trajectory]
        entropy_step50 = entropies[min(50, len(entropies) - 1)]
        entropy_step200 = entropies[min(200, len(entropies) - 1)]
        entropy_final = entropies[-1]
        max_entropy = math.log(5)

        # Expert usage at end
        final_usage = trajectory[-1]["expert_usage"]
        final_weights = trajectory[-1]["mean_routing_weights"]

        print(f"  Entropy: step0={entropies[0]:.4f}, step50={entropy_step50:.4f}, "
              f"step200={entropy_step200:.4f}, final={entropy_final:.4f} (max={max_entropy:.3f})")
        print(f"  Final weights: {final_weights}")
        print(f"  Expert usage: {final_usage}")
        print(f"  Test acc: {test_acc:.4f}, Elapsed: {elapsed:.1f}s")

        collapsed = entropy_final < 0.3
        print(f"  Collapsed: {'YES' if collapsed else 'NO'}")

        # Save trajectory
        traj_path = os.path.join(args.output_dir, f"trajectory_{cond_name}_{args.seed}.jsonl")
        with open(traj_path, "w") as f:
            for t in trajectory:
                f.write(json.dumps(t) + "\n")

        summary["conditions"][cond_name] = {
            "frozen": cond_cfg["frozen"],
            "norm_type": cond_cfg["norm_type"],
            "test_acc": round(test_acc, 4),
            "test_entropy": round(test_entropy, 4),
            "entropy_step0": round(entropies[0], 4),
            "entropy_step50": round(entropy_step50, 4),
            "entropy_step200": round(entropy_step200, 4),
            "entropy_final": round(entropy_final, 4),
            "final_weights": final_weights,
            "final_usage": final_usage,
            "collapsed": collapsed,
            "elapsed_sec": round(elapsed, 1),
            "n_trainable_params": n_trainable,
        }

    summary_path = os.path.join(args.output_dir, f"collapse_summary_{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

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
