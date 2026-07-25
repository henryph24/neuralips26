"""Modal A10G driver: RR-MoA 200-epoch convergence check incl. Electricity (rebuttal 8b2Z W6/Q3).

Reviewer 8b2Z asks for a longer training regime (200 epochs) and worries the
15-epoch numbers are under-converged, esp. on Electricity (the one dataset the
paper flags as under-converged at 15). This re-uses the proven input-length harness
at full context L=512 with n_epochs=200 on ETTh1/ETTm1/Weather (the 100-epoch set)
PLUS Electricity, reporting routing entropy (late-collapse check) and MSE
(convergence check) vs the best fixed adapter. The revin routing control is skipped
here (not needed for convergence) to halve compute.

    # preflight one cell (validate + warm cache)
    modal run scripts/modal_convergence.py --datasets Electricity --seeds 42
    # full: 4 datasets x 3 seeds @ 200 epochs
    modal run scripts/modal_convergence.py

Results return to results/convergence_200ep/.
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

app = modal.App("rrmoa-convergence-200ep", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=7200, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, n_epochs: int = 200) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from input_len_ablation import run_cell

    result = run_cell(dataset, 512, seed, n_epochs=n_epochs, device="cuda",
                      with_revin=False, with_baselines=False)
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTm1,Weather,Electricity",
         seeds: str = "42,43,44",
         n_epochs: int = 200,
         out_dir: str = "results/convergence_200ep"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, n_epochs) for d in ds for s in sd]

    print(f"Launching {len(cells)} convergence cells on A10G @ {n_epochs} epochs "
          f"({len(ds)} datasets x {len(sd)} seeds)")

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_s{res['seed']}_e{res.get('n_epochs')}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        mse = res.get("rrmoa_raw_mse")
        ent = res.get("rrmoa_raw_entropy")
        dvf = res.get("delta_vs_fixed_pct")
        print(f"  {res['dataset']:<12} s{res['seed']} e{res.get('n_epochs')}  "
              f"mse={mse:.4f}  ent={ent:.3f}  vs_fixed={dvf:+.1f}%"
              if mse is not None and ent is not None and dvf is not None
              else f"  {res['dataset']} s{res['seed']}  {res}")

    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
