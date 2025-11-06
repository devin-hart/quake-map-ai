# Quake 1 AI Map Generator

## Overview

This project aims to create a data-driven system that **learns to generate playable single-player Quake 1 maps**.
The goal is to train a generative model that can produce `.map` files—complete with brush geometry, entities, and lighting—that compile successfully under modern Quake 1 tools (ericw-tools suite) and yield maps that play like authentic id-era levels.

---

## Objectives

1. **Collect and normalize** a large dataset of single-player `.map` source files (≥ 1000 maps).
2. **Analyze and rasterize** maps into fixed-size numeric representations suitable for machine learning.
3. **Train generative models** capable of creating new Quake-style layouts and entity distributions.
4. **Convert model output back** into `.map` brush geometry and compile to `.bsp`.
5. **Automate evaluation** using compile logs, reachability checks, and basic playability metrics.
6. **Iterate and improve** through automated scoring, human curation, and model fine-tuning.

---

## Architecture

### Directory Layout

```
quake1-map-ai/
├── data/
│   ├── raw_maps/       # original .map files
│   ├── processed/      # rasterized 512×512×4 numpy tensors
│   └── derived/        # manifest.csv, stats, metrics
├── src/
│   ├── preprocess/     # build_manifest.py, rasterize_maps.py
│   ├── models/         # layout + entity generators (future)
│   └── postprocess/    # brushify, compile, validation
├── tools/              # qbsp/vis/light wrappers, map cleaner
├── results/
│   ├── generated_maps/ # AI-generated .map outputs
│   └── logs/           # compile/test logs
└── README.md
```

---

## Current Progress

✅ 1. Collected 1001 single-player `.map` files.
✅ 2. Manifest (`manifest.csv`) built with brush counts, entities, themes, and bounding boxes.
🟡 3. Rasterizer (`rasterize_maps.py`) produces normalized 512×512×4 grids from each map.
⬜ 4. Upcoming: Graybox “brushify” reconstruction to convert grids back to `.map`.
⬜ 5. AI training pipeline (stages DP02–DP04).
⬜ 6. Automated evaluation and scoring suite.

---

## Development Plan (DP Milestones)

| ID       | Stage                    | Description                                                           | Output                        |
| -------- | ------------------------ | --------------------------------------------------------------------- | ----------------------------- |
| **DP00** | Procedural Baseline      | Build deterministic generator + compile validator                     | Auto-compiling graybox `.map` |
| **DP01** | Dataset Prep             | Rasterize and normalize dataset (complete)                            | 512×512×4 tensors + metadata  |
| **DP02** | Layout Model             | Train diffusion/transformer on layout grids to produce new 2D layouts | Generated layout tensors      |
| **DP03** | Entity Placement         | Train small policy net for monsters, items, secrets placement         | Entity layer tensors          |
| **DP04** | Map Synthesis            | Combine layout + entities → brushify → `.map` → compile               | Playable maps in results/     |
| **DP05** | Evaluation Loop          | Automatic compile + metric scoring + curation                         | Ranked and filtered outputs   |
| **DP06** | Refinement & Theme Packs | Texture sets + lighting presets per theme                             | Themed final maps             |

---

## Tools & Dependencies

* **Python 3.10+**
* **NumPy**, **Pandas**, **PyTorch / TensorFlow** (for ML stages)
* **ericw-tools** (`qbsp`, `vis`, `light`) for compilation
* **TrenchBroom** (optional manual inspection)
* **Matplotlib / Pillow** (for visualizing raster grids)

---

## Data Representation

Each map becomes a 512×512 tensor with 4 channels:

1. **solid** – geometry occupancy
2. **walkable** – floor regions
3. **height** – normalized floor Z
4. **special** – liquid/teleport/hazard markers

Normalization scales every map’s XY bounds to fit the grid, retaining relative geometry and vertical proportions.

---

## AI Pipeline Outline

1. **Layout Diffusion Model** → produces new layout grids.
2. **Entity Policy Network** → populates monsters, items, secrets.
3. **Brushify Converter** → transforms grids into brush geometry on grid snaps.
4. **Compilation Pass** → `.map` → `.bsp` (via ericw-tools).
5. **Evaluation Engine** → checks connectivity, compilation success, path length, item reachability.
6. **Selection / Ranking** → keeps top-K maps for release.

---

## Long-Term Vision

* Produce new, playable, fun Quake 1 SP maps autonomously.
* Offer a dataset and trained models to the community for AI-assisted map creation.
* Extend pipeline to other id Tech games (Quake II, III).

