"""Modal A10G driver: LayerNorm-swap-on-MOMENT collapse test (rebuttal 8b2Z W4).

Each cell runs the full normalizer family {revin, none, batchnorm, groupnorm, layernorm}
swapped in at MOMENT's RevIN position, with a hidden-state AdaMix router, and reports
final routing entropy per normalizer. batchnorm/groupnorm are self-validation positive
controls (must reproduce App H collapse); 'none' is the healthy negative control.

    # preflight (self-validate the harness on ETTh1)
    modal run scripts/modal_norm_swap.py --datasets ETTh1 --seeds 42
    # full: ETTh1/ETTm1/Weather x 3 seeds
    modal run scripts/modal_norm_swap.py

Results return to results/norm_swap/.
"""

import json
import os

import modal

CACHE_DIR = "/root/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "momentfm",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
    )
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
)

app = modal.App("rrmoa-norm-swap", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=3600, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, unfreeze: str = "last4") -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from norm_swap_core import run_cell

    result = run_cell(dataset, seed, unfreeze=unfreeze, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTm1,Weather",
         seeds: str = "42,43,44",
         unfreeze: str = "last4",
         out_dir: str = "results/norm_swap"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, unfreeze) for d in ds for s in sd]

    print(f"Launching {len(cells)} norm-swap cells on A10G (unfreeze={unfreeze})")

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_s{res['seed']}_{res['unfreeze']}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        ent = {k: round(v.get("routing_entropy"), 3) if isinstance(v.get("routing_entropy"), float) else v
               for k, v in res["norms"].items()}
        print(f"  {res['dataset']:<10} s{res['seed']}  entropy per norm: {ent}")

    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
    print("Self-check: batchnorm/groupnorm should be ~0 (collapse), 'none' ~0.8 (healthy).")
