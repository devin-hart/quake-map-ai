#!/usr/bin/env python3
# src/postprocess/brushify_from_grid.py
import argparse, json
from pathlib import Path
import numpy as np

# --- CONFIG (from you) ---
WAD_PATH     = "/home/wizardbeard/projects/quake1-map-ai/wads/base.wad;/home/wizardbeard/projects/quake1-map-ai/wads/misc.wad"
FLOOR_TEX    = "sfloor1_5"
CEIL_TEX     = "tech01_1"
WALL_TEX     = "twall5_3"
TRIGGER_TEX  = "trigger"
DEFAULT_TEX  = "crate1_side"

# --- Tunables (safe) ---
GRID_SIZE    = 512
CELL         = 4.0          # world units per grid cell
WALL_THICK   = 32.0
FLOOR_THICK  = 16.0
CEIL_CLEAR   = 192.0
WALK_THR     = 0.5
MIN_REGION_CELLS = 100
MERGE_PAD    = 1            # fuse near pixels
OUT_FUDGE    = 0.0          # push walls slightly outward
SNAP         = 16.0          # snap to 8u grid

def quantize(v, q=SNAP): return float(np.floor(v / q + 0.5) * q)

def load_tensor(path: Path):
    t = np.load(path)
    if t.shape != (GRID_SIZE, GRID_SIZE, 4):
        raise SystemExit(f"Unexpected tensor shape {t.shape}")
    return t

def dilate(mask, iters=1):
    if iters <= 0: return mask
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=iters)
    except Exception:
        # fallback 3x3 dilation repeated
        out = mask.copy()
        for _ in range(iters):
            pad = np.pad(out, 1)
            acc = np.zeros_like(out, dtype=bool)
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    acc |= pad[1+dr:1+dr+out.shape[0], 1+dc:1+dc+out.shape[1]]
            out = acc
        return out

def components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    from collections import deque
    for r in range(H):
        for c in range(W):
            if mask[r,c] and not seen[r,c]:
                q = deque([(r,c)]); seen[r,c] = True; cells=[]
                while q:
                    rr, cc = q.popleft(); cells.append((rr,cc))
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and mask[nr,nc] and not seen[nr,nc]:
                            seen[nr,nc] = True; q.append((nr,nc))
                comps.append(cells)
    return comps

def bbox_cells(cells):
    rs = [r for r,_ in cells]; cs = [c for _,c in cells]
    return min(rs), min(cs), max(rs), max(cs)

def grid_bbox_to_xy(r0,c0,r1,c1):
    x0 = quantize(c0 * CELL); y0 = quantize(r0 * CELL)
    x1 = quantize((c1+1) * CELL); y1 = quantize((r1+1) * CELL)
    return x0,y0,x1,y1

def emit_plane(f, p1, p2, p3, tex):
    # One face = one plane: 3 points, CCW when viewed from OUTSIDE the solid.
    f.write(f"( {p1[0]:.1f} {p1[1]:.1f} {p1[2]:.1f} ) ( {p2[0]:.1f} {p2[1]:.1f} {p2[2]:.1f} ) ( {p3[0]:.1f} {p3[1]:.1f} {p3[2]:.1f} ) {tex} 0 0 0 1 1\n")

def write_box_brush(f, mins, maxs, tex):
    # Build a convex axis-aligned box with correct outward-facing windings.
    x0,y0,z0 = mins; x1,y1,z1 = maxs
    # Ensure proper ordering and add a tiny epsilon to avoid z-fighting/coplanars.
    if x0 > x1: x0,x1 = x1,x0
    if y0 > y1: y0,y1 = y1,y0
    if z0 > z1: z0,z1 = z1,z0
    eps = 0.01
    x0 -= eps; y0 -= eps; z0 -= eps
    x1 += eps; y1 += eps; z1 += eps

    f.write("{\n")
    # Bottom (-Z), seen from below → CCW from below; but we want normal DOWN, so from OUTSIDE (below):
    emit_plane(f, (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), tex)
    # Top (+Z), normal UP, view from above:
    emit_plane(f, (x0,y0,z1), (x0,y1,z1), (x1,y1,z1), tex)
    # West (-X), normal toward -X, view from -X:
    emit_plane(f, (x0,y0,z0), (x0,y0,z1), (x0,y1,z1), tex)
    # East (+X), normal toward +X, view from +X:
    emit_plane(f, (x1,y0,z0), (x1,y1,z0), (x1,y1,z1), tex)
    # South (-Y), normal toward -Y, view from -Y:
    emit_plane(f, (x0,y0,z0), (x1,y0,z0), (x1,y0,z1), tex)
    # North (+Y), normal toward +Y, view from +Y:
    emit_plane(f, (x0,y1,z0), (x0,y1,z1), (x1,y1,z1), tex)
    f.write("}\n")


def add_room(f, x0,y0,x1,y1, floor_z):
    floor_z = quantize(floor_z)
    ceil_z  = floor_z + CEIL_CLEAR
    # slabs
    write_box_brush(f, (x0,y0,floor_z - FLOOR_THICK), (x1,y1,floor_z), FLOOR_TEX)
    write_box_brush(f, (x0,y0,ceil_z), (x1,y1,ceil_z + FLOOR_THICK), CEIL_TEX)
    # perimeter (slight outward offset to avoid coplanar)
    o = OUT_FUDGE
    write_box_brush(f, (x0 - WALL_THICK - o, y0 - WALL_THICK - o, floor_z - FLOOR_THICK),
                       (x0 + o,               y1 + WALL_THICK + o, ceil_z + FLOOR_THICK), WALL_TEX)
    write_box_brush(f, (x1 - o,               y0 - WALL_THICK - o, floor_z - FLOOR_THICK),
                       (x1 + WALL_THICK + o,  y1 + WALL_THICK + o, ceil_z + FLOOR_THICK), WALL_TEX)
    write_box_brush(f, (x0 - o,               y0 - WALL_THICK - o, floor_z - FLOOR_THICK),
                       (x1 + o,               y0 + o,               ceil_z + FLOOR_THICK), WALL_TEX)
    write_box_brush(f, (x0 - o,               y1 - o,               floor_z - FLOOR_THICK),
                       (x1 + o,               y1 + WALL_THICK + o,  ceil_z + FLOOR_THICK), WALL_TEX)
    return ((x0+x1)/2.0, (y0+y1)/2.0, floor_z + 32.0), ceil_z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--meta", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = load_tensor(Path(args.tensor))
    walk = t[...,1]
    hch  = t[...,2]

    mask = (walk >= WALK_THR)
    mask = dilate(mask, MERGE_PAD)

    comps = [c for c in components(mask) if len(c) >= MIN_REGION_CELLS]
    comps.sort(key=len, reverse=True)
    if not comps: raise SystemExit("No significant walkable components found.")

    rooms = []
    for cells in comps:
        r0,c0,r1,c1 = bbox_cells(cells)
        x0,y0,x1,y1 = grid_bbox_to_xy(r0,c0,r1,c1)
        sub = hch[r0:r1+1, c0:c1+1]
        hz  = float(np.median(sub[sub>0])) if np.any(sub>0) else 0.0
        rooms.append((x0,y0,x1,y1,hz))

    # center near origin
    minx=min(r[0] for r in rooms); maxx=max(r[2] for r in rooms)
    miny=min(r[1] for r in rooms); maxy=max(r[3] for r in rooms)
    cx=(minx+maxx)*0.5; cy=(miny+maxy)*0.5
    rooms=[(x0-cx,y0-cy,x1-cx,y1-cy,hz) for (x0,y0,x1,y1,hz) in rooms]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("{\n")
        f.write('"classname" "worldspawn"\n')
        f.write(f'"_wad" "{WAD_PATH}"\n')

        spawn = None; last_center=None
        for i,(x0,y0,x1,y1,hz) in enumerate(rooms):
            floor_z = hz * CEIL_CLEAR
            origin, _ = add_room(f, x0,y0,x1,y1, floor_z)
            if i==0: spawn = origin
            last_center = origin

        f.write("}\n")
        sx,sy,sz = spawn if spawn else (0.0,0.0,64.0)
        f.write("{\n\"classname\" \"info_player_start\"\n")
        f.write(f"\"origin\" \"{sx:.1f} {sy:.1f} {sz:.1f}\"\n}}\n")

        ex,ey,ez = last_center if last_center else (sx,sy,sz)
        # simple exit trigger brush
        bx0,by0,bz0 = ex-24.0, ey-24.0, ez+16.0
        bx1,by1,bz1 = ex+24.0, ey+24.0, ez+48.0
        f.write("{\n\"classname\" \"trigger_changelevel\"\n\"map\" \"start\"\n")
        write_box_brush(f, (bx0,by0,bz0), (bx1,by1,bz1), TRIGGER_TEX)
        f.write("}\n")

    print(f"Wrote graybox map: {out}")

if __name__ == "__main__":
    main()
