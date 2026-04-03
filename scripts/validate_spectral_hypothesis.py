"""Validate whether spectral features of backbone hidden states predict adapter success.

Quick validation: compute spectral fingerprints for each dataset (via frozen MOMENT),
then check if winning adapter families correlate with spectral profiles.

No training needed — uses existing validated adapter results.

Usage:
    python scripts/validate_spectral_hypothesis.py
"""

import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from scripts.run_standard_evolution import load_standard_data, _detect_backbone_type


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]


def compute_spectral_fingerprint(hidden_states):
    """Compute spectral features from backbone hidden states.

    hidden_states: (N, T, d_model) — batch of encoder outputs
    Returns dict of spectral features.
    """
    # Average over batch and feature dims to get temporal profile
    # Shape: (T,) — mean activation over time
    temporal_profile = hidden_states.mean(dim=(0, 2)).numpy()

    # FFT
    fft = np.fft.rfft(temporal_profile)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(temporal_profile))

    # Normalize PSD
    psd_norm = psd / (psd.sum() + 1e-10)

    # Features
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    spectral_centroid = np.sum(freqs * psd_norm)

    # Energy bands (low/mid/high thirds of frequency range)
    n_freqs = len(freqs)
    third = n_freqs // 3
    energy_low = psd[:third].sum() / (psd.sum() + 1e-10)
    energy_mid = psd[third:2*third].sum() / (psd.sum() + 1e-10)
    energy_high = psd[2*third:].sum() / (psd.sum() + 1e-10)

    # Dominant peaks (above 2 sigma)
    threshold = psd.mean() + 2 * psd.std()
    n_peaks = int((psd > threshold).sum())

    # Also compute per-feature spectral diversity
    # hidden_states: (N, T, d)
    per_feature_psd = np.abs(np.fft.rfft(hidden_states.mean(dim=0).numpy(), axis=0)) ** 2
    feature_spectral_std = per_feature_psd.std(axis=1).mean()  # how much spectral content varies across features

    return {
        "spectral_entropy": float(spectral_entropy),
        "spectral_centroid": float(spectral_centroid),
        "energy_low": float(energy_low),
        "energy_mid": float(energy_mid),
        "energy_high": float(energy_high),
        "n_dominant_peaks": n_peaks,
        "feature_spectral_std": float(feature_spectral_std),
        "psd_max": float(psd.max()),
        "psd_concentration": float(psd.max() / (psd.sum() + 1e-10)),  # how concentrated is the spectrum
    }


def classify_adapter(code):
    """Classify adapter architecture into families based on code."""
    code_lower = code.lower()
    if "conv1d" in code_lower or "conv2d" in code_lower:
        return "conv"
    elif "softmax" in code_lower or "attn" in code_lower or "attention" in code_lower:
        return "attention"
    elif "[:, -1" in code or "last" in code_lower:
        return "last_token"
    elif ".max(" in code:
        return "max_pool"
    else:
        return "mean_pool"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    args = parser.parse_args()

    # Load model once
    print("Loading backbone...")
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False

    # === Step 1: Compute spectral fingerprints ===
    print("\n" + "=" * 60)
    print("STEP 1: Spectral Fingerprints")
    print("=" * 60)

    fingerprints = {}
    for dataset in DATASETS:
        try:
            splits, _ = load_standard_data(dataset, 96, max_samples=500)
            X = splits["train"][0][:200]  # just 200 samples for speed

            model.eval()
            batch = torch.from_numpy(X).float().to(args.device).unsqueeze(1)
            mask = torch.ones(batch.shape[0], batch.shape[2], device=args.device)

            with torch.no_grad():
                feat = _extract_features_batch(model, blocks, batch, mask, backbone_type=bb_type)

            fp = compute_spectral_fingerprint(feat.cpu())
            fingerprints[dataset] = fp
            print("%-12s entropy=%.2f centroid=%.4f low=%.2f mid=%.2f high=%.2f peaks=%d conc=%.4f" % (
                dataset, fp["spectral_entropy"], fp["spectral_centroid"],
                fp["energy_low"], fp["energy_mid"], fp["energy_high"],
                fp["n_dominant_peaks"], fp["psd_concentration"]))
        except Exception as e:
            print("%-12s ERROR: %s" % (dataset, e))

    # === Step 2: Analyze winning adapter families per dataset ===
    print("\n" + "=" * 60)
    print("STEP 2: Winning Adapter Families")
    print("=" * 60)

    winning_families = {}
    for dataset in DATASETS:
        evo_path = "results/standard_evolution/%s_H96_42.json" % dataset
        if not os.path.exists(evo_path):
            print("%-12s: no results" % dataset)
            continue

        d = json.load(open(evo_path))
        evolved = d.get("evolved", [])
        if not evolved:
            continue

        # Classify each adapter and its MSE
        adapter_data = []
        for e in evolved:
            family = classify_adapter(e.get("code", ""))
            adapter_data.append({
                "family": family,
                "test_mse": e["test_mse"],
                "param_count": e.get("param_count", 0),
            })

        # Best adapter family
        best = min(adapter_data, key=lambda x: x["test_mse"])
        families = [a["family"] for a in adapter_data]
        family_counts = {}
        for f in families:
            family_counts[f] = family_counts.get(f, 0) + 1

        winning_families[dataset] = best["family"]
        print("%-12s best_family=%-10s MSE=%.4f  all_families=%s" % (
            dataset, best["family"], best["test_mse"], family_counts))

    # === Step 3: Correlate spectral features with winning families ===
    print("\n" + "=" * 60)
    print("STEP 3: Spectral → Family Correlation")
    print("=" * 60)

    # Group datasets by spectral characteristics
    datasets_with_both = [d for d in DATASETS if d in fingerprints and d in winning_families]

    print("\nDataset spectral profiles vs winning adapter family:")
    print("%-12s %-10s %8s %8s %8s %8s %6s" % (
        "Dataset", "Winner", "Entropy", "Centroid", "Low%", "Conc", "Peaks"))
    for d in datasets_with_both:
        fp = fingerprints[d]
        print("%-12s %-10s %8.2f %8.4f %8.2f %8.4f %6d" % (
            d, winning_families[d],
            fp["spectral_entropy"], fp["spectral_centroid"],
            fp["energy_low"], fp["psd_concentration"], fp["n_dominant_peaks"]))

    # Check: do datasets with similar spectral profiles have similar winning families?
    print("\n--- Transfer matrix correlation ---")
    transfer_path = "results/transferability/transfer_H96.json"
    if os.path.exists(transfer_path):
        transfer = json.load(open(transfer_path))
        matrix = transfer.get("transfer_matrix", {})

        print("\nSpectral distance vs transfer success:")
        for src in datasets_with_both:
            for tgt in datasets_with_both:
                if src == tgt:
                    continue
                fp_src = fingerprints[src]
                fp_tgt = fingerprints[tgt]

                # Spectral distance (L2 over key features)
                dist = np.sqrt(
                    (fp_src["spectral_entropy"] - fp_tgt["spectral_entropy"])**2 +
                    (fp_src["energy_low"] - fp_tgt["energy_low"])**2 +
                    (fp_src["psd_concentration"] - fp_tgt["psd_concentration"])**2
                )

                entry = matrix.get(src, {}).get(tgt, {})
                beats = entry.get("beats_baseline", False)
                delta = entry.get("delta_vs_baseline", 0)

                print("  %s → %-10s dist=%.3f  transfer=%s  Δ=%+.1f%%" % (
                    src, tgt, dist, "WIN" if beats else "lose", delta))

    # === Step 4: Verdict ===
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Check if spectral features separate datasets meaningfully
    if len(datasets_with_both) >= 4:
        entropies = [fingerprints[d]["spectral_entropy"] for d in datasets_with_both]
        concentrations = [fingerprints[d]["psd_concentration"] for d in datasets_with_both]
        entropy_range = max(entropies) - min(entropies)
        conc_range = max(concentrations) - min(concentrations)

        print("Spectral entropy range: %.2f (min=%.2f, max=%.2f)" % (
            entropy_range, min(entropies), max(entropies)))
        print("PSD concentration range: %.4f (min=%.4f, max=%.4f)" % (
            conc_range, min(concentrations), max(concentrations)))

        if entropy_range > 0.5 or conc_range > 0.01:
            print("\n→ Spectral features SEPARATE datasets. Proceed with spectral-guided AAS.")
        else:
            print("\n→ Spectral features do NOT separate datasets well. Consider Direction 2.")

    # Save
    save_data = {
        "fingerprints": fingerprints,
        "winning_families": winning_families,
    }
    os.makedirs("results/spectral_validation", exist_ok=True)
    with open("results/spectral_validation/validation.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved to results/spectral_validation/validation.json")


if __name__ == "__main__":
    main()
