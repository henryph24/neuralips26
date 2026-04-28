"""Generate a talk-slide visualization of normalization-induced routing collapse
using Gemini 3 Pro Image (Nano Banana Pro).

Reads GEMINI_API_KEY from .env at repo root. Writes PNGs to figures/talk/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
OUT_DIR = REPO_ROOT / "figures" / "talk"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(ENV_PATH)
if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set; check .env")

from google import genai
from google.genai import types

MODEL = "gemini-3-pro-image-preview"

# Prompts: two conceptual visualizations of the collapse phenomenon for talk slides.
PROMPTS = {
    "problem_overview": (
        "An academic-paper-quality scientific figure for a NeurIPS submission, "
        "rendered as a clean two-panel diagram comparing the standard practice "
        "(panel A) with its natural extension that catastrophically fails "
        "(panel B). The two panels share the same underlying architecture so "
        "the reader can see exactly what changes. "
        ""
        "PANEL A (top, labeled '(a) Standard practice: one static head'): "
        "A horizontal pipeline drawn left-to-right. On the left, three distinct "
        "univariate time-series input windows of length 512 are stacked "
        "vertically, each rendered as a small inline waveform plot in a thin "
        "gray frame: window 1 is a smooth slow sinusoid, window 2 is a noisy "
        "high-volatility signal, window 3 is a sharp step pattern. A single "
        "small caption beneath the windows reads 'Input windows X (different "
        "per-window statistics)'. Each window feeds into the same large "
        "rounded box labeled 'Frozen TSFM Backbone (e.g., MOMENT, Moirai)' "
        "with a small lock icon, and a tiny tag underneath reading 'Instance "
        "Normalization (RevIN)'. The backbone produces hidden states H, drawn "
        "as a stack of three small matrices, which all flow into ONE single "
        "head box labeled 'Single static adapter head'. The head's output is "
        "labelled y_hat. A header banner above panel A reads in clean sans "
        "serif: 'TSFM adapters today use a single static head for every input "
        "window' --- treat this as a real headline that must appear verbatim "
        "in the figure. "
        ""
        "PANEL B (bottom, labeled '(b) Natural extension: mixture-of-experts "
        "adapter --- collapses'): Same backbone, same three input windows, "
        "same hidden states, but now the H states feed into a 'Router' box, "
        "which connects to five expert heads laid out in a vertical column "
        "(E1, E2, E3, E4, E5). Only E1 is drawn in solid orange, with a "
        "thick bold orange arrow from the router into E1. E2 through E5 are "
        "drawn faded in gray, with thin dashed inactive arrows and a small "
        "label 'unused' next to each. To the right of the expert column a "
        "small inline annotation reads 'routing entropy = 0.000'. A header "
        "banner above panel B reads in clean sans serif: 'A natural mixture-"
        "of-experts adapter catastrophically collapses on instance-normalized "
        "TSFMs'. "
        ""
        "Visual style: monochrome scientific figure with minimal accent "
        "colors --- white background, navy blue lines for the architecture, "
        "muted orange only for the active expert in panel B and for "
        "highlighting key terms, light gray for dimmed/inactive components. "
        "Sans-serif typography (similar to Helvetica or Inter), thin "
        "vector-style strokes, NeurIPS / ICML paper-figure look (not a slide, "
        "not infographic, not 3D, not glossy). All text labels must be "
        "perfectly legible, no decorative emoji, no shadows, no gradients. "
        "16:9 aspect ratio, single composed figure, panels stacked vertically "
        "with a thin horizontal divider line between them."
    ),
}


def generate(name: str, prompt: str) -> Path:
    client = genai.Client()
    print(f"[gen] {name}: requesting {MODEL}...", flush=True)
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="2K",
            )
        ),
    )
    out = OUT_DIR / f"{name}.png"
    saved = False
    for part in response.parts:
        if getattr(part, "text", None):
            print(f"[gen] {name}: model text -> {part.text[:200]}", flush=True)
        if getattr(part, "inline_data", None):
            img = part.as_image()
            img.save(out)
            print(f"[gen] {name}: saved {out} ({out.stat().st_size} bytes)", flush=True)
            saved = True
            break
    if not saved:
        print(f"[gen] {name}: NO IMAGE returned", flush=True)
    return out


if __name__ == "__main__":
    targets = sys.argv[1:] or list(PROMPTS.keys())
    for name in targets:
        if name not in PROMPTS:
            print(f"unknown prompt: {name}; choices: {list(PROMPTS)}")
            continue
        try:
            generate(name, PROMPTS[name])
        except Exception as e:  # noqa: BLE001
            print(f"[gen] {name}: FAILED -- {type(e).__name__}: {e}", flush=True)
