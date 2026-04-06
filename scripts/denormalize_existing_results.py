"""Post-hoc denormalized MSE injector for existing result JSONs.

Walks results/{rr_moa,adamix,lora_baseline,dlinear}/ and rewrites each JSON
with two additional fields where possible:

    rr_moa / adamix:
        <method>["mse_denorm_approx"] = <method>["mse"] * mean(scaler.scale_ ** 2)
        baselines[name]["mse_denorm_approx"] = baselines[name]["mse"] * mean(scaler.scale_ ** 2)

    lora_baseline:
        lora_mse_denorm_approx = lora_mse * mean(scaler.scale_ ** 2)

    dlinear:
        dlinear_mse_denorm_approx = dlinear_mse * mean(scaler.scale_ ** 2)

The "_approx" suffix is important: this is the variance-weighted average
approximation, not the exact per-sample-channel denormalization. The exact
version requires the per-sample channel index which was not tracked before
today's refactor to ``load_standard_data``, so historical JSONs can only be
patched with the variance-weighted approximation. New runs (after today)
record the exact ``mse_denorm`` directly in ``train_rr_moa`` / ``train_adamix``
/ ``train_lora_baseline`` / ``run_dlinear_baseline.py`` using
``compute_denorm_mse`` with the true ch-indexed scaler.

The approximation is an unbiased estimator of the true original-scale MSE
under the assumption that per-channel squared-errors are equal in normalized
space, which is approximately true because the model is trained to minimize
normalized MSE uniformly across samples. See T1.A in
.claude/plans/cozy-puzzling-lampson.md for rationale.

Usage::

    python scripts/denormalize_existing_results.py            # all methods
    python scripts/denormalize_existing_results.py --dry-run  # report only
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.run_standard_evolution import load_standard_data


# Cache of (dataset, horizon) -> scaler.
_SCALER_CACHE = {}


def get_scaler(dataset, horizon):
    key = (dataset, horizon)
    if key not in _SCALER_CACHE:
        try:
            splits, _ = load_standard_data(dataset, horizon)
            _SCALER_CACHE[key] = splits["_scaler"]
        except Exception as e:
            print("  ! could not load %s H=%d: %s" % (dataset, horizon, e))
            _SCALER_CACHE[key] = None
    return _SCALER_CACHE[key]


def scale_factor(scaler):
    """mean(scale_ ** 2): multiply normalized MSE to get approx original-scale MSE."""
    if scaler is None:
        return None
    return float(np.mean(np.asarray(scaler.scale_) ** 2))


def patch_rr_moa_json(path, dry_run=False):
    with open(path) as f:
        data = json.load(f)
    dataset = data.get("dataset")
    horizon = data.get("horizon", 96)
    scaler = get_scaler(dataset, horizon)
    if scaler is None:
        return False
    k = scale_factor(scaler)

    changed = False
    rr = data.get("rr_moa")
    if rr is not None and "mse" in rr and "mse_denorm_approx" not in rr:
        rr["mse_denorm_approx"] = rr["mse"] * k
        changed = True
    baselines = data.get("baselines") or {}
    for name, entry in baselines.items():
        if isinstance(entry, dict) and "mse" in entry and "mse_denorm_approx" not in entry:
            entry["mse_denorm_approx"] = entry["mse"] * k
            changed = True
    if changed and not dry_run:
        data["scaler_mean_scale_sq"] = k
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    return changed


def patch_adamix_json(path, dry_run=False):
    with open(path) as f:
        data = json.load(f)
    dataset = data.get("dataset")
    horizon = data.get("horizon", 96)
    scaler = get_scaler(dataset, horizon)
    if scaler is None:
        return False
    k = scale_factor(scaler)

    changed = False
    ad = data.get("adamix")
    if ad is not None and "mse" in ad and "mse_denorm_approx" not in ad:
        ad["mse_denorm_approx"] = ad["mse"] * k
        changed = True
    baselines = data.get("baselines") or {}
    for name, entry in baselines.items():
        if isinstance(entry, dict) and "mse" in entry and "mse_denorm_approx" not in entry:
            entry["mse_denorm_approx"] = entry["mse"] * k
            changed = True
    if changed and not dry_run:
        data["scaler_mean_scale_sq"] = k
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    return changed


def patch_lora_json(path, dry_run=False):
    with open(path) as f:
        data = json.load(f)
    dataset = data.get("dataset")
    horizon = data.get("horizon", 96)
    scaler = get_scaler(dataset, horizon)
    if scaler is None:
        return False
    k = scale_factor(scaler)

    if "lora_mse" not in data or "lora_mse_denorm_approx" in data:
        return False
    data["lora_mse_denorm_approx"] = data["lora_mse"] * k
    data["scaler_mean_scale_sq"] = k
    if not dry_run:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    return True


def patch_dlinear_json(path, dry_run=False):
    with open(path) as f:
        data = json.load(f)
    dataset = data.get("dataset")
    horizon = data.get("horizon", 96)
    scaler = get_scaler(dataset, horizon)
    if scaler is None:
        return False
    k = scale_factor(scaler)

    if "dlinear_mse" not in data or "dlinear_mse_denorm_approx" in data:
        return False
    data["dlinear_mse_denorm_approx"] = data["dlinear_mse"] * k
    data["scaler_mean_scale_sq"] = k
    if not dry_run:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    return True


PATCHERS = {
    "rr_moa":        ("results/rr_moa/*.json",        patch_rr_moa_json),
    "adamix":        ("results/adamix/*.json",        patch_adamix_json),
    "lora_baseline": ("results/lora_baseline/*.json", patch_lora_json),
    "dlinear":       ("results/dlinear/*.json",       patch_dlinear_json),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of: " + ",".join(PATCHERS.keys()))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    os.chdir(args.root)

    selected = set(PATCHERS.keys())
    if args.only:
        selected = set(args.only.split(","))

    total_changed, total_scanned = 0, 0
    for name, (pattern, fn) in PATCHERS.items():
        if name not in selected:
            continue
        paths = sorted(glob.glob(pattern))
        print("\n# %s: %d files" % (name, len(paths)))
        n_changed = 0
        for p in paths:
            try:
                ok = fn(p, dry_run=args.dry_run)
            except Exception as e:
                print("  ! %s: %s" % (p, e))
                continue
            total_scanned += 1
            if ok:
                n_changed += 1
                total_changed += 1
        print("  patched %d / %d" % (n_changed, len(paths)))

    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print("\n[%s] %d / %d JSONs updated." % (mode, total_changed, total_scanned))


if __name__ == "__main__":
    main()
