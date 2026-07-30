"""Modal A10G driver: single-variable backbone toggle (Pm4m round 2).

    modal run scripts/modal_backbone_toggle.py --datasets ETTh1 --seeds 42   # preflight
    modal run scripts/modal_backbone_toggle.py                               # H=96 grid
    modal run scripts/modal_backbone_toggle.py --horizon 192 --out-dir results/backbone_toggle_h192
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
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
)

app = modal.App("rrmoa-backbone-toggle", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=5400, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, horizon: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from backbone_toggle_core import run_cell
    result = run_cell(dataset, seed, horizon=horizon, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity",
         seeds: str = "42,43,44", horizon: int = 96,
         out_dir: str = "results/backbone_toggle"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, horizon) for d in ds for s in sd]
    print(f"Backbone toggle: {len(cells)} paired cells on A10G (H={horizon})")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        with open(os.path.join(out_dir, f"{res['dataset']}_H{res['horizon']}_s{res['seed']}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<12} s{res['seed']}  on={res['mse_backbone_on']:.4f}  "
              f"off={res['mse_backbone_off']:.4f}  gain={res['backbone_gain_pct']:+.1f}%  "
              f"{'HELPS' if res['backbone_helps'] else 'no'}")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} to {out_dir}/")
