"""Cross-modal MoE routing collapse experiment on Vision Transformer (ViT-B/16).

Tests whether normalization-induced routing collapse generalizes beyond time
series to vision transformers. ViT patch embeddings are structurally analogous
to time-series windows: a sequence of tokens where instance normalization can
strip per-token statistics that may carry routing signal.

6 conditions:
  E: Frozen ViT + InstanceNorm1d on features, route on normed features
     → Expected: COLLAPSE (InstanceNorm strips patch-level routing stats)
  F: Frozen ViT + no norm, route on hidden features
     → Expected: No collapse (features retain routing info)
  G: Frozen ViT + InstanceNorm1d on features, route on RAW PATCHES
     → Expected: No collapse (RR-MoA analog: raw-input routing bypasses norm)
  H: Unfrozen ViT + InstanceNorm1d, route on normed features
     → Expected: Collapse (co-adaptation + norm stripping)
  I: Unfrozen ViT + no norm, route on hidden features
     → Expected: Collapse (co-adaptation, like ResNet condition C)
  J: Frozen ViT + LayerNorm (ViT-native), route on normed features
     → Expected: No collapse (LayerNorm preserves spatial/semantic routing info)

The critical comparison is E vs G: same backbone, same normalization on experts,
but E routes on normed features (should collapse) while G routes on raw input
patches (should NOT collapse). This is the ViT analog of the TS RR-MoA story.

Usage:
    python scripts/run_vision_moe_vit_collapse.py --seed 42 --device cuda
    python scripts/run_vision_moe_vit_collapse.py --seed 42 --conditions E G --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ============================================================
# ViT Feature Extractor
# ============================================================

class ViTFeatureExtractor(nn.Module):
    """ViT-B/16 that returns patch embeddings (B, N, D) instead of logits.

    N = 197 (196 patches + 1 CLS token), D = 768 for ViT-B/16.
    Uses torchvision's native ViT, extracts features before the final
    LayerNorm + classification head.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        base = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
        )
        # Keep everything except the classification head
        self.conv_proj = base.conv_proj        # Patch embedding: (B,3,224,224) -> (B,768,14,14)
        self.class_token = base.class_token    # (1, 1, 768)
        self.encoder = base.encoder            # Transformer encoder
        self.seq_length = base.seq_length      # 197
        self.hidden_dim = base.hidden_dim      # 768
        # We intentionally skip base.heads (classification) and base.ln (final LN)

    def forward(self, x):
        """Returns patch embeddings (B, 197, 768) from the last encoder layer."""
        B = x.shape[0]
        # Patch embedding
        x = self.conv_proj(x)  # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        # Prepend CLS token
        cls = self.class_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        # Positional embedding + encoder
        x = self.encoder(x)  # (B, 197, 768)
        return x


# ============================================================
# Expert pool for sequence features (ViT)
# ============================================================

class CLSExpert(nn.Module):
    """Use CLS token for classification."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, seq):  # (B, N, D)
        return self.fc(seq[:, 0])  # CLS token


class SeqMeanPoolExpert(nn.Module):
    """Mean-pool over all patch tokens (exclude CLS)."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, seq):
        return self.fc(seq[:, 1:].mean(dim=1))


class SeqMaxPoolExpert(nn.Module):
    """Max-pool over patch tokens."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, seq):
        return self.fc(seq[:, 1:].max(dim=1).values)


class SeqAttentionPoolExpert(nn.Module):
    """Learned attention-weighted pool over patch tokens."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, seq):
        patches = seq[:, 1:]  # (B, 196, D)
        w = F.softmax(self.attn(patches), dim=1)  # (B, 196, 1)
        pooled = (w * patches).sum(dim=1)  # (B, D)
        return self.fc(pooled)


class SeqLastTokenExpert(nn.Module):
    """Use last patch token (analog of LastPatchExpert in ResNet version)."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, seq):
        return self.fc(seq[:, -1])


# ============================================================
# ViT MoE with normalization + raw-input routing options
# ============================================================

class ViTMoE(nn.Module):
    """MoE for ViT with optional normalization and raw-input routing.

    Normalization options on patch embeddings (before expert execution):
    - 'instancenorm': InstanceNorm1d(d_model) strips per-channel stats across patches
    - 'layernorm': LayerNorm(d_model) — ViT-native normalization
    - 'none': no normalization

    Routing options:
    - 'features': route on (possibly normalized) backbone features
    - 'raw': route on raw input patches (RR-MoA analog)
    """

    def __init__(self, d_model, num_classes, norm_type="none", router_input="features",
                 image_size=224, patch_size=16):
        super().__init__()
        self.norm_type = norm_type
        self.router_input_mode = router_input
        self.d_model = d_model

        # Normalization on patch embeddings
        if norm_type == "instancenorm":
            self.norm = nn.InstanceNorm1d(d_model, affine=False)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None

        # Router: pool features → gate logits
        K = 5
        if router_input == "raw":
            # Raw-input router: Conv2d on raw image (matches TS Conv1d router)
            self.raw_router = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=16, stride=16, padding=0),  # match patch size
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.router_gate = nn.Linear(32, K)
        else:
            self.raw_router = None
            self.router_gate = nn.Linear(d_model, K)

        # 5 architecturally distinct experts
        self.experts = nn.ModuleList([
            CLSExpert(d_model, num_classes),
            SeqMeanPoolExpert(d_model, num_classes),
            SeqMaxPoolExpert(d_model, num_classes),
            SeqAttentionPoolExpert(d_model, num_classes),
            SeqLastTokenExpert(d_model, num_classes),
        ])
        self.K = K

    def forward(self, patch_embeddings, raw_images=None):
        """
        Args:
            patch_embeddings: (B, N, D) from ViT backbone
            raw_images: (B, 3, H, W) original images (needed for raw routing)
        Returns:
            logits, routing_weights, routing_entropy
        """
        # Normalize patch embeddings
        if self.norm is not None:
            if self.norm_type == "instancenorm":
                # InstanceNorm1d expects (B, C, L): transpose to (B, D, N)
                normed = self.norm(patch_embeddings.transpose(1, 2)).transpose(1, 2)
            else:
                normed = self.norm(patch_embeddings)
        else:
            normed = patch_embeddings

        # Routing
        if self.router_input_mode == "raw" and raw_images is not None:
            router_feat = self.raw_router(raw_images)  # (B, 32)
        else:
            # Route on CLS token of (possibly normalized) features
            router_feat = normed[:, 0]  # (B, D) — CLS token
        gate_logits = self.router_gate(router_feat)
        weights = F.softmax(gate_logits, dim=-1)  # (B, K)

        # Experts operate on normalized features
        expert_outputs = torch.stack(
            [exp(normed) for exp in self.experts], dim=1
        )  # (B, K, num_classes)
        logits = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # Entropy
        with torch.no_grad():
            avg_probs = weights.mean(dim=0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()

        return logits, weights, entropy


# ============================================================
# Data (CIFAR-10 at 224x224 for ViT)
# ============================================================

def get_cifar10_vit(batch_size=64, data_dir="./data"):
    """CIFAR-10 with 224x224 resize for ViT-B/16."""
    transform_train = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
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
# Training (adapted from run_vision_moe_collapse_v2.py)
# ============================================================

def train_condition(backbone, moe, train_loader, test_loader, epochs, device,
                    frozen, lr=1e-3, needs_raw_images=False):
    """Train one condition."""
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

            if frozen:
                with torch.no_grad():
                    patch_emb = backbone(images)
            else:
                patch_emb = backbone(images)

            raw_imgs = images if needs_raw_images else None
            logits, weights, entropy = moe(patch_emb, raw_images=raw_imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                expert_grad_norms = []
                for exp in moe.experts:
                    gnorm = sum(
                        p.grad.norm().item() ** 2
                        for p in exp.parameters() if p.grad is not None
                    ) ** 0.5
                    expert_grad_norms.append(round(gnorm, 6))

                router_params = list(moe.router_gate.parameters())
                if moe.raw_router is not None:
                    router_params += list(moe.raw_router.parameters())
                router_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in router_params if p.grad is not None
                ) ** 0.5

                backbone_grad_norm = 0.0
                if not frozen:
                    backbone_grad_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in backbone.parameters() if p.grad is not None
                    ) ** 0.5

                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                max_expert = weights.argmax(dim=-1)
                usage = [(max_expert == k).float().mean().item() for k in range(moe.K)]

            optimizer.step()

            trajectory.append({
                "step": global_step, "epoch": epoch,
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
            patch_emb = backbone(images)
            raw_imgs = images if needs_raw_images else None
            logits, weights, entropy = moe(patch_emb, raw_images=raw_imgs)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            test_entropies.append(entropy)

    test_acc = correct / total
    test_entropy = np.mean(test_entropies)
    return trajectory, test_acc, test_entropy


# ============================================================
# Conditions
# ============================================================

CONDITIONS = {
    "E_frozen_instancenorm": {
        "frozen": True, "norm_type": "instancenorm",
        "router_input": "features",
        "desc": "Frozen ViT + InstanceNorm1d, route on normed CLS",
    },
    "F_frozen_nonorm": {
        "frozen": True, "norm_type": "none",
        "router_input": "features",
        "desc": "Frozen ViT + no norm, route on hidden CLS",
    },
    "G_frozen_instancenorm_rawroute": {
        "frozen": True, "norm_type": "instancenorm",
        "router_input": "raw",
        "desc": "Frozen ViT + InstanceNorm1d, route on RAW image (RR-MoA)",
    },
    "H_unfrozen_instancenorm": {
        "frozen": False, "norm_type": "instancenorm",
        "router_input": "features",
        "desc": "Unfrozen ViT + InstanceNorm1d, route on normed CLS",
    },
    "I_unfrozen_nonorm": {
        "frozen": False, "norm_type": "none",
        "router_input": "features",
        "desc": "Unfrozen ViT + no norm, route on hidden CLS (co-adaptation test)",
    },
    "J_frozen_layernorm": {
        "frozen": True, "norm_type": "layernorm",
        "router_input": "features",
        "desc": "Frozen ViT + LayerNorm (ViT-native), route on normed CLS",
    },
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ViT Cross-Modal MoE Routing Collapse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="results/vision_moe_vit")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run specific conditions (e.g. E G). Default: all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print("Device: %s, Seed: %d" % (device, args.seed))

    print("Loading CIFAR-10 (224x224 for ViT)...")
    train_loader, test_loader = get_cifar10_vit(args.batch_size, args.data_dir)

    # Select conditions
    if args.conditions:
        cond_names = []
        for c in args.conditions:
            matches = [k for k in CONDITIONS if k.startswith(c)]
            cond_names.extend(matches)
        if not cond_names:
            print("No conditions matched: %s" % args.conditions)
            return
    else:
        cond_names = list(CONDITIONS.keys())

    results = {}

    for cond_name in cond_names:
        cfg = CONDITIONS[cond_name]
        print("\n" + "=" * 60)
        print("Condition: %s" % cond_name)
        print("  %s" % cfg["desc"])
        print("=" * 60)

        t0 = time.time()

        # Fresh backbone for each condition
        backbone = ViTFeatureExtractor(pretrained=True)
        moe = ViTMoE(
            d_model=768, num_classes=10,
            norm_type=cfg["norm_type"],
            router_input=cfg["router_input"],
        )

        n_backbone = sum(p.numel() for p in backbone.parameters())
        n_moe = sum(p.numel() for p in moe.parameters())
        n_trainable = n_moe if cfg["frozen"] else n_backbone + n_moe
        print("  Backbone: %.1fM, MoE: %.1fK, Trainable: %.1fM" % (
            n_backbone / 1e6, n_moe / 1e3, n_trainable / 1e6))

        needs_raw = (cfg["router_input"] == "raw")
        trajectory, test_acc, test_entropy = train_condition(
            backbone, moe, train_loader, test_loader,
            epochs=args.epochs, device=device,
            frozen=cfg["frozen"], lr=args.lr,
            needs_raw_images=needs_raw,
        )

        elapsed = time.time() - t0
        collapsed = test_entropy < 0.3

        print("  Test acc: %.3f, Entropy: %.4f, Collapsed: %s (%.1fs)" % (
            test_acc, test_entropy, collapsed, elapsed))

        # Save trajectory
        traj_path = os.path.join(args.output_dir,
                                 "trajectory_%s_%d.jsonl" % (cond_name, args.seed))
        with open(traj_path, "w") as f:
            for entry in trajectory:
                f.write(json.dumps(entry) + "\n")

        results[cond_name] = {
            "frozen": cfg["frozen"],
            "norm_type": cfg["norm_type"],
            "router_input": cfg["router_input"],
            "desc": cfg["desc"],
            "test_acc": round(test_acc, 4),
            "test_entropy": round(test_entropy, 4),
            "entropy_step0": trajectory[0]["routing_entropy"] if trajectory else None,
            "entropy_final": trajectory[-1]["routing_entropy"] if trajectory else None,
            "final_weights": trajectory[-1]["mean_routing_weights"] if trajectory else None,
            "final_usage": trajectory[-1]["expert_usage"] if trajectory else None,
            "collapsed": bool(collapsed),
            "elapsed_sec": round(elapsed, 1),
            "n_trainable_params": n_trainable,
        }

        # Free GPU memory
        del backbone, moe
        torch.cuda.empty_cache()

    # Save summary
    summary = {"seed": args.seed, "epochs": args.epochs, "conditions": results}
    summary_path = os.path.join(args.output_dir,
                                "vit_collapse_summary_%d.json" % args.seed)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to %s" % summary_path)

    # Print comparison table
    print("\n" + "=" * 70)
    print("%-35s  %7s  %5s  %7s  %s" % (
        "Condition", "Entropy", "Acc", "Collap", "Router"))
    print("-" * 70)
    for name, r in results.items():
        print("%-35s  %7.4f  %5.3f  %7s  %s" % (
            name, r["test_entropy"], r["test_acc"],
            "YES" if r["collapsed"] else "no",
            r["router_input"]))


if __name__ == "__main__":
    main()
