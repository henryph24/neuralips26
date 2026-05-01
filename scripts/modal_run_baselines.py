"""Modal.com entrypoint to run `run_baselines_only.py` for the 10 missing
(cell, seed) baselines that block a clean 54/54 paired-seed claim.

Usage (from repo root, with Modal authenticated):
    modal run scripts/modal_run_baselines.py            # smoke test (1 cell)
    modal run scripts/modal_run_baselines.py --all      # run all 10 missing cells

Outputs land in `results/baselines_only/` via Modal volume + post-run download.
"""
import json
import os
import sys
import time

import modal

# Cells that need backfill: 9 main-table frozen + 1 extended ETTm2 frozen seed-42.
MISSING_CELLS = [
    ("ETTh1", "frozen", 42), ("ETTh1", "frozen", 43), ("ETTh1", "frozen", 44),
    ("ETTm1", "frozen", 42), ("ETTm1", "frozen", 43), ("ETTm1", "frozen", 44),
    ("Weather", "frozen", 42), ("Weather", "frozen", 43), ("Weather", "frozen", 44),
    ("ETTm2", "frozen", 42),
]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "momentfm",
        "peft>=0.7.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
    )
    .add_local_dir(os.path.join(REPO, "feasibility"), "/root/feasibility")
    .add_local_dir(os.path.join(REPO, "scripts"), "/root/scripts")
    .add_local_dir(os.path.join(REPO, "data"), "/root/data")
)

app = modal.App("rrmoa-baselines-backfill", image=image)


@app.function(gpu="A10G", timeout=900)
def run_one(dataset: str, unfreeze: str, seed: int) -> dict:
    """Train linear/attention/conv heads on frozen MOMENT-small at this seed."""
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")
    import numpy as np
    import torch

    from feasibility.model import (
        load_backbone, _get_encoder_blocks, _apply_unfreeze,
        _disable_gradient_checkpointing,
    )
    from feasibility.code_evolution import SEED_ADAPTERS
    from scripts.run_standard_evolution import (
        load_standard_data, train_adapter, _detect_backbone_type,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    bb_type = _detect_backbone_type("AutonLab/MOMENT-1-small")
    splits, _ = load_standard_data(dataset, 96)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    test_ch = splits.get("test_ch")
    scaler = splits.get("_scaler")

    model = load_backbone("AutonLab/MOMENT-1-small", "cuda")
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    _apply_unfreeze(blocks, unfreeze)

    baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
    results = {}
    t0 = time.time()
    for name, code in baselines.items():
        tr = train_adapter(code, model, blocks, X_train, Y_train, X_test, Y_test,
                           device="cuda", n_epochs=15, forecast_horizon=96,
                           backbone_type=bb_type, eval_ch=test_ch, scaler=scaler)
        results[name] = tr
        print(f"  {name:10s} MSE={tr['mse']:.4f}", flush=True)
    elapsed = time.time() - t0

    return {
        "dataset": dataset, "horizon": 96, "seed": seed, "unfreeze": unfreeze,
        "K": 5, "top_k": 2, "backbone": "AutonLab/MOMENT-1-small",
        "baselines": results, "elapsed": elapsed,
        "source": "modal_run_baselines.py",
    }


@app.local_entrypoint()
def main(all: bool = False, smoke: bool = True):
    """Run smoke test (1 cell) by default; pass --all for the full backfill."""
    out_dir = os.path.join(REPO, "results", "baselines_only")
    os.makedirs(out_dir, exist_ok=True)

    if all:
        cells = MISSING_CELLS
        print(f"Running ALL {len(cells)} missing-baseline cells in parallel...", flush=True)
        results = list(run_one.starmap(cells))
    else:
        cells = MISSING_CELLS[:1]
        print(f"Smoke test: running 1 cell {cells[0]}...", flush=True)
        results = [run_one.remote(*cells[0])]

    for r in results:
        ds, fz, sd = r["dataset"], r["unfreeze"], r["seed"]
        fname = f"{ds}_H96_K5_top2_{fz}_{sd}_baselines.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Wrote {path}", flush=True)
