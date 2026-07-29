"""Modal driver: DLinear vs NLinear anchors at n=10 seeds (Pm4m round 2).

    modal run scripts/modal_linear_anchors.py                    # H=96
    modal run scripts/modal_linear_anchors.py --horizon 192
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

app = modal.App("rrmoa-linear-anchors", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=3600, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, horizon: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from linear_anchors_core import run_cell
    result = run_cell(dataset, horizon=horizon, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Exchange",
         horizon: int = 96, out_dir: str = "results/linear_anchors"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    cells = [(d, horizon) for d in ds]
    print(f"DLinear vs NLinear, n=10 seeds, H={horizon}: {len(cells)} datasets")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        with open(os.path.join(out_dir, f"{res['dataset']}_H{res['horizon']}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<12} DL={res['dlinear_mean']:.4f}+/-{res['dlinear_std']:.4f}  "
              f"NL={res['nlinear_mean']:.4f}+/-{res['nlinear_std']:.4f}  "
              f"NL vs DL {res['nl_vs_dl_pct_mean']:+6.1f}%  NL wins {res['nl_wins']}/{res['n_seeds']}")
    with open(os.path.join(out_dir, f"_all_H{horizon}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} to {out_dir}/")
