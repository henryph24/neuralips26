"""Modal A10G driver for the expert-pool composition experiment (rebuttal 8b2Z:
"fixed set of relatively simple adapters (linear, attention-based,
convolutional)").

Extends the submitted diversity ablation (Table I.2, canonical vs identical)
with richer / larger / deeper hand-designed pools, to show RR-MoA is robust to
the adapter pool. This does NOT address learned / dynamically-generated pools,
which remain future work (paper open-problems line).

Pools (per scripts/pool_core.py):
  - canonical    : 5 simple pooling heads (mean/last/max/attention/conv1d) -- the method
  - macro        : 5 richer conv/residual/gated/depthwise experts
  - large-diverse: 10 heterogeneous experts (canonical + macro)
  - deep-mlp     : 5 deeper 2-hidden-layer MLP experts

Usage (run from repo root):

    # preflight: 1 cell (canonical ETTh1 s42), warms the weight cache +
    # validates the harness (must reproduce ~0.646, matching horizon_k/paper)
    ~/.venv-modal/bin/modal run scripts/modal_pool.py --mode preflight

    # full experiment: 4 pools x ETTh1 x 3 seeds = 12 cells
    ~/.venv-modal/bin/modal run scripts/modal_pool.py --mode all

Results are written to results/pool/.
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
)

app = modal.App("rrmoa-pool", image=image)
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
    POOLS = ["canonical", "macro", "large-diverse", "deep-mlp"]
    if mode == "preflight":
        return [("ETTh1", "canonical", 42, n_epochs)]
    return [("ETTh1", p, s, n_epochs) for p in POOLS for s in SEEDS]


@app.local_entrypoint()
def main(mode: str = "all", n_epochs: int = 15, out_dir: str = "results/pool"):
    cells = _build_cells(mode, n_epochs)
    print(f"[{mode}] launching {len(cells)} cells on A10G")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        tag = f"pool_{res['dataset']}_{res['pool']}_s{res['seed']}"
        with open(os.path.join(out_dir, tag + ".json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        mse = res.get("rrmoa_raw_mse")
        ent = res.get("rrmoa_raw_entropy")
        dvf = res.get("delta_vs_fixed_pct")
        k = res.get("K")
        print(f"  {tag:<34} K={k:<2} mse={mse:.4f} ent={ent:.3f} vs_fixed={dvf:+.1f}%"
              if mse is not None and ent is not None and dvf is not None else f"  {tag}: {res}")

    with open(os.path.join(out_dir, f"_all_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
