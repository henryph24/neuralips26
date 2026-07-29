"""Modal A10G driver: RIA+ vs DLinear vs from-scratch NLinear (Pm4m round 2).

    # preflight (1 cell): ETTh1 seed 42, H=96 -- DLinear should land near 0.416
    modal run scripts/modal_nlinear_anchor.py --datasets ETTh1 --seeds 42
    # full H=96 grid
    modal run scripts/modal_nlinear_anchor.py
    # H=192 (where the paper claims the TSFM contributes maximally)
    modal run scripts/modal_nlinear_anchor.py --horizon 192 --out-dir results/nlinear_anchor_h192
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
    .add_local_file("data/exchange_rate.csv", "/root/data/exchange_rate.csv")
    .add_local_file("data/solar.csv", "/root/data/solar.csv")
    .add_local_file("data/traffic.csv", "/root/data/traffic.csv")
)

app = modal.App("rrmoa-nlinear-anchor", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=3600, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, horizon: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from nlinear_anchor_core import run_cell
    result = run_cell(dataset, seed, horizon=horizon, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity",
         seeds: str = "42,43,44", horizon: int = 96,
         out_dir: str = "results/nlinear_anchor"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, horizon) for d in ds for s in sd]
    print(f"Launching {len(cells)} RIA+/DLinear/NLinear cells on A10G (H={horizon})")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_H{res['horizon']}_s{res['seed']}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<12} s{res['seed']}  RIA+={res['ria_mse']:.4f}  "
              f"DL={res['dlinear_mse']:.4f}  NL={res['nlinear_mse']:.4f}  | "
              f"NL vs DL {res['nlinear_vs_dlinear_pct']:+.1f}%  "
              f"RIA+ vs NL {res['ria_vs_nlinear_pct']:+.1f}%  "
              f"{'WIN' if res['ria_beats_nlinear'] else 'loss'}")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
