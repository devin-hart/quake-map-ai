#!/usr/bin/env python3
"""
Rasterize Quake 1 .map (worldspawn brushes) into 512x512 tensors.

Output per map:
  data/processed/<name>.npy     # float32 HxWxC tensor (C=4: solid, walkable, height, special)
  data/processed/<name>.json    # metadata (bbox, scale, offsets)

Notes (MVP, coarse but useful for ML):
- Treat each brush by its face vertices; approximate as an XY AABB when rasterizing.
- 'solid': union of brush XY AABBs (worldspawn only).
- 'walkable': union of horizontal faces’ XY AABBs (faces with ~constant Z).
- 'height': normalized floor Z for walkable cells in [0..1]; 0 elsewhere.
- 'special': crude liquid/hazard mask if any face texture contains {lava,slime,water}.
- All maps are normalized to fit a 512x512 grid (keeping XY aspect); Z is normalized using map Z-bounds.
- This is a coarse rasterizer (no exact CSG); good enough for layout learning. Later passes can refine.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # /projects/quake1-map-ai
RAW = ROOT / "data" / "raw_maps"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

H = W = 512
EPS_Z = 0.5  # z-equality tolerance for "horizontal face"

# Face line regex: 3 points + texture token (classic Q1 .map)
FACE_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"([^\s]+)"
)

LIQUID_KEYS = ("lava", "slime", "water")


def read_worldspawn_blocks(text: str):
    """Return list of brush blocks (each list of face lines) inside the first entity (worldspawn)."""
    # Strip // comments; keep braces to track nesting
    lines = []
    for raw in text.splitlines():
        lines.append(raw.split("//", 1)[0])

    entity_depth = 0
    brush_depth = 0
    in_worldspawn = False
    cur_brush = []
    brushes = []

    def maybe_push_brush():
        nonlocal cur_brush
        if cur_brush:
            brushes.append(cur_brush)
            cur_brush = []

    # Detect worldspawn crudely: first entity in file
    for line in lines:
        opens = line.count("{")
        closes = line.count("}")

        # enter entity
        for ch in line:
            if ch == "{":
                if entity_depth == 0:
                    in_worldspawn = True  # first entity
                else:
                    # possible start of a brush
                    if in_worldspawn and brush_depth == 0:
                        cur_brush = []
                    brush_depth += 1
                entity_depth += 1
            elif ch == "}":
                if brush_depth > 0:
                    brush_depth -= 1
                    if in_worldspawn and brush_depth == 0:
                        maybe_push_brush()
                else:
                    entity_depth -= 1
                    if entity_depth == 0 and in_worldspawn:
                        in_worldspawn = False

        # collect face lines if inside a brush
        if in_worldspawn and brush_depth > 0:
            if FACE_RE.search(line):
                cur_brush.append(line.strip())

    return brushes


def brush_stats(face_lines):
    """Return dict: bbox, is_liquid, horizontal_faces[ (z, xmin,xmax,ymin,ymax) ... ], textures."""
    xs = []
    ys = []
    zs = []
    textures = []
    horiz_boxes = []

    is_liquid = False

    for ln in face_lines:
        m = FACE_RE.search(ln)
        if not m:
            continue
        pts = [
            (float(m.group(1)), float(m.group(2)), float(m.group(3))),
            (float(m.group(4)), float(m.group(5)), float(m.group(6))),
            (float(m.group(7)), float(m.group(8)), float(m.group(9))),
        ]
        tex = m.group(10)
        textures.append(tex)
        if any(k in tex.lower() for k in LIQUID_KEYS):
            is_liquid = True

        for (x, y, z) in pts:
            xs.append(x)
            ys.append(y)
            zs.append(z)

        # horizontal face if all z nearly equal
        z0, z1, z2 = pts[0][2], pts[1][2], pts[2][2]
        if max(abs(z0 - z1), abs(z0 - z2), abs(z1 - z2)) <= EPS_Z:
            xmin = min(p[0] for p in pts)
            xmax = max(p[0] for p in pts)
            ymin = min(p[1] for p in pts)
            ymax = max(p[1] for p in pts)
            zmean = (z0 + z1 + z2) / 3.0
            horiz_boxes.append((zmean, xmin, xmax, ymin, ymax))

    if not xs:
        return None

    bbox = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    return {
        "bbox": bbox,
        "is_liquid": is_liquid,
        "horiz_boxes": horiz_boxes,
        "textures": textures,
    }


def to_grid_mapper(xmin, ymin, xmax, ymax):
    """Return function mapping world (x,y) → grid (i,j) with normalization to 512x512 keeping aspect."""
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        sx = sy = 1.0
    else:
        scale = (W - 2) / max(width, height)
        sx = sy = scale
    # center inside 0..W-1/H-1 with padding
    pad_x = (W - (width * sx)) * 0.5
    pad_y = (H - (height * sy)) * 0.5

    def map_xy(x, y):
        j = int(round((x - xmin) * sx + pad_x))  # col
        i = int(round((y - ymin) * sy + pad_y))  # row
        # clamp
        if i < 0: i = 0
        if j < 0: j = 0
        if i >= H: i = H - 1
        if j >= W: j = W - 1
        return i, j

    return map_xy, sx, sy, pad_x, pad_y


def rasterize_map(path: Path):
    text = path.read_text(errors="ignore")
    brushes = read_worldspawn_blocks(text)
    if not brushes:
        return None

    # Collect brush stats
    bstats = []
    wx_min = wy_min = wz_min = math.inf
    wx_max = wy_max = wz_max = -math.inf
    for b in brushes:
        st = brush_stats(b)
        if not st:
            continue
        bstats.append(st)
        xmin, ymin, zmin, xmax, ymax, zmax = st["bbox"]
        wx_min = min(wx_min, xmin); wy_min = min(wy_min, ymin); wz_min = min(wz_min, zmin)
        wx_max = max(wx_max, xmax); wy_max = max(wy_max, ymax); wz_max = max(wz_max, zmax)

    if not bstats or not math.isfinite(wx_min):
        return None

    map_xy, sx, sy, pad_x, pad_y = to_grid_mapper(wx_min, wy_min, wx_max, wy_max)
    z_span = max(wz_max - wz_min, 1e-6)

    solid = np.zeros((H, W), dtype=np.float32)
    walkable = np.zeros((H, W), dtype=np.float32)
    height = np.zeros((H, W), dtype=np.float32)
    special = np.zeros((H, W), dtype=np.float32)

    # Rasterize solids (XY AABBs of brushes)
    for st in bstats:
        xmin, ymin, _, xmax, ymax, _ = st["bbox"]
        i0, j0 = map_xy(ymin, xmin)  # careful: we mapped (y,x) earlier? keep consistent
        # we defined map_xy(x,y) -> (i,j); don't swap
        i00, j00 = map_xy(xmin, ymin)  # WRONG order; fix:
        # Correct mapping:
        i_lo, j_lo = map_xy(xmin, ymin)
        i_hi, j_hi = map_xy(xmax, ymax)
        r0, r1 = sorted((i_lo, i_hi))
        c0, c1 = sorted((j_lo, j_hi))
        solid[r0:r1+1, c0:c1+1] = 1.0
        if st["is_liquid"]:
            special[r0:r1+1, c0:c1+1] = 1.0

    # Rasterize horizontal faces as walkable; height = normalized z
    # Keep the top-most floor per cell (max z)
    zbuf = np.full((H, W), -1e9, dtype=np.float32)

    for st in bstats:
        for z, xmin, xmax, ymin, ymax in st["horiz_boxes"]:
            i_lo, j_lo = map_xy(xmin, ymin)
            i_hi, j_hi = map_xy(xmax, ymax)
            r0, r1 = sorted((i_lo, i_hi))
            c0, c1 = sorted((j_lo, j_hi))
            # update walkable + height where this z is higher
            z_norm = (z - wz_min) / z_span
            # slice
            zs = zbuf[r0:r1+1, c0:c1+1]
            mask = z_norm > zs
            if mask.any():
                zbuf[r0:r1+1, c0:c1+1][mask] = z_norm
                walkable[r0:r1+1, c0:c1+1][mask] = 1.0
                height[r0:r1+1, c0:c1+1][mask] = z_norm

    tensor = np.stack([solid, walkable, height, special], axis=-1).astype(np.float32)

    meta = {
        "source": path.name,
        "grid_size": [H, W],
        "channels": ["solid", "walkable", "height", "special"],
        "world_bbox": {
            "xmin": wx_min, "ymin": wy_min, "zmin": wz_min,
            "xmax": wx_max, "ymax": wy_max, "zmax": wz_max
        },
        "scale": {"sx": float(sx), "sy": float(sy), "pad_x": float(pad_x), "pad_y": float(pad_y)},
        "notes": "Coarse rasterization using brush XY AABBs and horizontal face boxes (normalized).",
    }
    return tensor, meta


def main():
    maps = sorted(RAW.glob("*.map"))
    total = len(maps)
    done = 0
    for p in maps:
        try:
            res = rasterize_map(p)
        except Exception as e:
            res = None
        if res is None:
            done += 1
            continue
        tensor, meta = res
        stem = p.stem
        np.save(OUT / f"{stem}.npy", tensor)
        with (OUT / f"{stem}.json").open("w") as f:
            json.dump(meta, f, indent=2)
        done += 1
        if done % 50 == 0:
            print(f"[{done}/{total}] {stem}")

    print(f"Finished rasterization: {done}/{total} maps processed → {OUT}")


if __name__ == "__main__":
    main()
