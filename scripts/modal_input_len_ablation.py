"""Modal A10G driver for the RR-MoA input-length ablation (rebuttal Q1).

Preflight one validation cell first (L=512 must reproduce the published
number and warm the weight cache), then fan out the full grid:

    # 1) validation + cache warm (ETTh1 L=512 seed42 -> expect ~0.66-0.69)
    modal run scripts/modal_input_len_ablation.py \
        --datasets ETTh1 --lengths 512 --seeds 42

    # 1b) confirm the mask actually shortens context (L=96 must differ)
    modal run scripts/modal_input_len_ablation.py \
        --datasets ETTh1 --lengths 96 --seeds 42

    # 2) full grid: 6 datasets x {96,192,336,512} x 3 seeds = 72 cells
    modal run scripts/modal_input_len_ablation.py

Results are returned to the local machine and written to
results/input_len_ablation/.
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

app = modal.App("rrmoa-inputlen-ablation", image=image)

# Shared HF weight cache so MOMENT downloads once (warmed by the preflight cell).
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=1800, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, input_len: int, seed: int, n_epochs: int = 15) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from input_len_ablation import run_cell

    result = run_cell(dataset, input_len, seed, n_epochs=n_epochs, device="cuda")
    hf_cache.commit()  # persist any freshly downloaded weights for later cells
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity",
         lengths: str = "96,192,336,512",
         seeds: str = "42,43,44",
         n_epochs: int = 15,
         out_dir: str = "results/input_len_ablation"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    ls = [int(x) for x in lengths.split(",") if x.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, L, s, n_epochs) for d in ds for L in ls for s in sd]

    print(f"Launching {len(cells)} cells on A10G: "
          f"{len(ds)} datasets x {len(ls)} lengths x {len(sd)} seeds")

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_L{res['input_len']}_s{res['seed']}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        raw = res.get("rrmoa_raw_mse")
        ent = res.get("rrmoa_raw_entropy")
        dvf = res.get("delta_vs_fixed_pct")
        dvr = res.get("delta_vs_revin_pct")
        print(f"  {res['dataset']:<12} L={res['input_len']:<3} s={res['seed']}  "
              f"raw_mse={raw:.4f}  ent={ent:.3f}  "
              f"vs_fixed={dvf:+.1f}%  vs_revin={dvr:+.1f}%"
              if raw is not None and dvf is not None and dvr is not None
              else f"  {res['dataset']} L={res['input_len']} s={res['seed']}  {res}")

    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
