"""Modal A10G driver for the learned / dynamically-generated expert-pool probe
(rebuttal 8b2Z: "learned, dynamically generated ... adapter pools", which the
paper lists as open, L619).

Unlike scripts/modal_pool.py (which swaps in richer/larger *hand-designed*
pools), this replaces the hand-designed experts with a `hyper-gen` pool: each
expert's transform is generated per sample by a small hypernetwork (FiLM),
rather than being a fixed pooling op. The raw router (the collapse fix) is
unchanged. This is an initial feasibility check, not a rigorous learned-pool
method; NAS-style search / larger generators remain future work.

Usage (run from repo root):

    # preflight: canonical regression check (must reproduce ~0.646) + hyper-gen smoke test
    ~/.venv-modal/bin/modal run scripts/modal_learned_pool.py --mode preflight

    # full: hyper-gen on ETTh1 + Weather x 3 seeds = 6 cells
    ~/.venv-modal/bin/modal run scripts/modal_learned_pool.py --mode all

Results are written to results/learned_pool/.
"""
import json
import os

import modal

CACHE_DIR = "/root/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.1.0", "momentfm", "numpy", "scipy", "scikit-learn", "pandas")
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
)

app = modal.App("rrmoa-learned-pool", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=2400, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, pool: str, seed: int, n_epochs: int = 15) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from pool_core import run_pool_cell

    res = run_pool_cell(dataset, pool, seed, n_epochs=n_epochs, device="cuda")
    hf_cache.commit()
    return res


def _build_cells(mode, n_epochs):
    SEEDS = [42, 43, 44]
    if mode == "preflight":
        # canonical = same-image reference; hyper-gen = smoke test
        return [("ETTh1", "canonical", 42, n_epochs), ("ETTh1", "hyper-gen", 42, n_epochs)]
    # both pools in the SAME image so canonical is a fair same-image baseline
    return [(d, p, s, n_epochs)
            for d in ("ETTh1", "Weather")
            for p in ("canonical", "hyper-gen")
            for s in SEEDS]


@app.local_entrypoint()
def main(mode: str = "all", n_epochs: int = 15, out_dir: str = "results/learned_pool"):
    cells = _build_cells(mode, n_epochs)
    print(f"[{mode}] launching {len(cells)} cells on A10G")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        tag = f"lp_{res['dataset']}_{res['pool']}_s{res['seed']}"
        with open(os.path.join(out_dir, tag + ".json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        mse = res.get("rrmoa_raw_mse")
        ent = res.get("rrmoa_raw_entropy")
        dvf = res.get("delta_vs_fixed_pct")
        print(f"  {tag:<30} mse={mse:.4f} ent={ent:.3f} vs_fixed={dvf:+.1f}%"
              if mse is not None and ent is not None and dvf is not None else f"  {tag}: {res}")

    with open(os.path.join(out_dir, f"_all_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
