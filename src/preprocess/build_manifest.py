#!/usr/bin/env python3
"""
Builds a manifest for Quake 1 .map files.

Scans quake1-map-ai/data/raw_maps/*.map and writes quake1-map-ai/data/derived/manifest.csv
Fields:
  filename, entities, brushes, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z, theme_guess, type
Notes:
  - Brush counting: counts `{ <plane libuildnes> }` blocks inside an entity.
  - Entity counting: counts top-level `{ ... }` blocks (each is an entity).
  - BBox: aggregated from all plane-line 3D points in all brushes.
  - Theme guess: very rough heuristic from texture tokens.
  - Type: fixed to "SP" per project scope.
"""

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # quake1-map-ai/
RAW = ROOT / "data" / "raw_maps"
OUT_DIR = ROOT / "data" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "manifest.csv"

# Face line pattern (classic Quake MAP): three points then texture and mapping
# e.g. ( -64 0 64 ) ( -64 0 0 ) ( -64 128 0 ) metal5_2 0 0 0 1 1
FACE_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*"
    r"([^\s]+)"
)

THEME_KEYS = {
    "base": {"base", "tech", "comp", "metal", "start", "computer", "pipe", "truss"},
    "medieval": {"med", "wood", "tim", "castle"},
    "runic": {"runic", "rune"},
    "wizard": {"wiz", "wizard"},
    "id1": {"id", "e1", "e2", "e3", "e4", "start"},
    "zombie": {"zombie", "crypt"},
    "lava": {"lava", "hell", "slime"},  # not a theme, but useful tag
}

def guess_theme(texture_tokens):
    tx = " ".join(texture_tokens).lower()
    score = {k: 0 for k in THEME_KEYS}
    for k, keys in THEME_KEYS.items():
        for s in keys:
            if s in tx:
                score[k] += 1
    if not any(score.values()):
        return ""
    return max(score, key=score.get)

def parse_map(path: Path):
    """
    Returns: dict with entities, brushes, bbox (min/max), theme_guess
    """
    text = path.read_text(errors="ignore")

    # Normalize line endings and remove comments beginning with //
    lines = []
    for raw in text.splitlines():
        # strip inline // comments (not perfect but helps)
        line = raw.split("//", 1)[0].rstrip()
        if line:
            lines.append(line)
    text = "\n".join(lines)

    i = 0
    n = len(text)
    # Tokenize only braces to manage nesting while scanning lines
    # We'll iterate by lines for face parsing and brace structure for counts.

    entities = 0
    brushes = 0
    theme_textures = []

    # Bounding box (aggregate from all face points)
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf

    # Simple state machine:
    # top-level: expect entity starting with '{'
    # inside_entity: read lines; when encountering '{', start brush; collect until matching '}' -> brush++
    # key/value lines are ignored
    # We rely on brace counts to detect entity vs brush:
    #  - entity opens: '{' when not currently inside a brush and entity_brace_depth == 0
    #  - brush opens: '{' when inside entity and brush_depth == 0
    entity_depth = 0
    brush_depth = 0

    # Pre-split to lines for regex; keep braces per line
    for line in text.splitlines():
        # Count braces on this line
        opens = line.count("{")
        closes = line.count("}")

        # Enter entity?
        # Heuristic: an entity begins when entity_depth == 0 and opens > 0
        if opens and entity_depth == 0:
            entities += 1

        # If we're inside a brush (brush_depth > 0), try to parse face lines
        if entity_depth > 0 and brush_depth > 0:
            m = FACE_RE.search(line)
            if m:
                # Extract 3 points and texture token
                pts = [
                    (float(m.group(1)), float(m.group(2)), float(m.group(3))),
                    (float(m.group(4)), float(m.group(5)), float(m.group(6))),
                    (float(m.group(7)), float(m.group(8)), float(m.group(9))),
                ]
                tex = m.group(10)
                theme_textures.append(tex)
                for (x, y, z) in pts:
                    if x < min_x: min_x = x
                    if y < min_y: min_y = y
                    if z < min_z: min_z = z
                    if x > max_x: max_x = x
                    if y > max_y: max_y = y
                    if z > max_z: max_z = z

        # Update depths; each '{' inside an entity that increases brush_depth from 0→1 marks a new brush
        # But the first '{' that created the entity should not count as a brush.
        # We detect new brush starts when entity_depth >= 1 and brush_depth == 0 and opens > 0 AND NOT starting the entity itself on this line.
        # Simpler approach: count brush when brush_depth transitions from 0 to 1 AND entity_depth >= 1, but ignore the entity's first open.
        # We'll update incrementally per character to catch multiple braces on a line.
        prev_entity_depth = entity_depth
        prev_brush_depth = brush_depth
        for ch in line:
            if ch == "{":
                if entity_depth == 0:
                    # starting a new entity
                    entity_depth = 1
                else:
                    # either starting a brush or nested brace inside brush
                    if brush_depth == 0:
                        brushes += 1
                    brush_depth += 1
            elif ch == "}":
                if brush_depth > 0:
                    brush_depth -= 1
                else:
                    if entity_depth > 0:
                        entity_depth -= 1
                # (We assume no stray closing braces)
        # End per-line

    # If bbox never updated (no faces parsed), set to zeros
    if not math.isfinite(min_x):
        min_x = min_y = min_z = 0.0
        max_x = max_y = max_z = 0.0

    theme = guess_theme(theme_textures)
    return {
        "entities": entities,
        "brushes": brushes,
        "bbox_min_x": int(min_x),
        "bbox_min_y": int(min_y),
        "bbox_min_z": int(min_z),
        "bbox_max_x": int(max_x),
        "bbox_max_y": int(max_y),
        "bbox_max_z": int(max_z),
        "theme_guess": theme,
    }

def main():
    paths = sorted(RAW.glob("*.map"))
    if not paths:
        print(f"No .map files found in {RAW}")
        return

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "entities",
            "brushes",
            "bbox_min_x","bbox_min_y","bbox_min_z",
            "bbox_max_x","bbox_max_y","bbox_max_z",
            "theme_guess",
            "type",
        ])
        for p in paths:
            try:
                info = parse_map(p)
            except Exception as e:
                # On parse error, write minimal row and continue
                info = {
                    "entities": 0,
                    "brushes": 0,
                    "bbox_min_x": 0, "bbox_min_y": 0, "bbox_min_z": 0,
                    "bbox_max_x": 0, "bbox_max_y": 0, "bbox_max_z": 0,
                    "theme_guess": "",
                }
            writer.writerow([
                p.name,
                info["entities"],
                info["brushes"],
                info["bbox_min_x"], info["bbox_min_y"], info["bbox_min_z"],
                info["bbox_max_x"], info["bbox_max_y"], info["bbox_max_z"],
                info["theme_guess"],
                "SP",
            ])

    print(f"Wrote manifest: {OUT_CSV} ({len(paths)} maps)")

if __name__ == "__main__":
    main()
