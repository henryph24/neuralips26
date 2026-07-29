"""Modal A10G driver: is (M,Sigma) decodable from the router's hidden-state input?

    # preflight: the two MOMENT arms (RevIN on vs off)
    modal run scripts/modal_stat_decodability.py --arms moment,moment_norevin
    # full six-arm sweep
    modal run scripts/modal_stat_decodability.py
"""

import json
import os

import modal

CACHE_DIR = "/root/hf_cache"

# Base image: byte-identical to the proven modal_ria_offsuite/nlinear_anchor image.
# Do NOT add uni2ts here -- it forces a scipy source build (no BLAS) and the
# whole build fails. Moirai gets its own image below.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.1.0", "momentfm", "numpy", "scipy", "scikit-learn", "pandas")
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
)

# Moirai needs uni2ts. Install it in a later layer so scipy is already satisfied
# from a wheel and pip does not try to rebuild it from source.
moirai_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1.0", "numpy", "scipy", "scikit-learn", "pandas")
    .pip_install("uni2ts")
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
)

app = modal.App("rrmoa-stat-decodability", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)

# label -> (backbone, disable_revin)
ARMS = {
    "moment":         ("AutonLab/MOMENT-1-small", False),   # regime (i), RevIN
    "moment_norevin": ("AutonLab/MOMENT-1-small", True),    # positive control
    "moirai":         ("Salesforce/moirai-1.1-R-small", False),   # regime (ii)
    "moirai_moe":     ("Salesforce/moirai-moe-1.0-R-small", False),  # regime (ii)
    "chronos":        ("amazon/chronos-t5-small", False),   # regime (iii)
    "timer":          ("thuml/timer-base-84m", False),      # regime (iii)
}


def _run(label: str, dataset: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from stat_decodability_core import run_cell
    backbone, disable_revin = ARMS[label]
    return run_cell(backbone, dataset=dataset, disable_revin=disable_revin,
                    device="cuda", label=label)


@app.function(gpu="A10G", timeout=1800, volumes={CACHE_DIR: hf_cache})
def run_arm_remote(label: str, dataset: str) -> dict:
    result = _run(label, dataset)
    hf_cache.commit()
    return result


@app.function(gpu="A10G", timeout=1800, image=moirai_image,
              volumes={CACHE_DIR: hf_cache})
def run_moirai_arm_remote(label: str, dataset: str) -> dict:
    result = _run(label, dataset)
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(arms: str = "moment,moment_norevin,chronos,timer",
         dataset: str = "ETTh1", out_dir: str = "results/stat_decodability"):
    labels = [a.strip() for a in arms.split(",") if a.strip()]
    cells = [(a, dataset) for a in labels]
    print(f"Probing (M,Sigma) decodability on {dataset}: {len(cells)} arms")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    moirai_cells = [c for c in cells if c[0].startswith("moirai")]
    base_cells = [c for c in cells if not c[0].startswith("moirai")]
    streams = []
    if base_cells:
        streams.append(run_arm_remote.starmap(base_cells, return_exceptions=True))
    if moirai_cells:
        streams.append(run_moirai_arm_remote.starmap(moirai_cells, return_exceptions=True))
    for res in [r for st in streams for r in st]:
        if isinstance(res, Exception):
            print(f"  ARM FAILED: {res}")
            continue
        results.append(res)
        with open(os.path.join(out_dir, f"{res['label']}_{res['dataset']}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['label']:<16} d={res['hidden_dim']:<5} "
              f"R2(mu)={res['r2_mu_hidden']:+.3f}  "
              f"R2(log sigma)={res['r2_logsigma_hidden']:+.3f}   "
              f"[raw ceiling {res['r2_mu_raw_ceiling']:.3f} / "
              f"{res['r2_logsigma_raw_ceiling']:.3f}]")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} arms to {out_dir}/")
