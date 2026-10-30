"""Poster-size charts, drawn as SVG from the repository's CSVs.

Each chart is sized in points at about its printed width, so 20 px text prints at about 20 pt.
Colours are the first three categorical slots of the dataviz reference palette (blue, orange,
aqua), which validate on all pairs, plus violet for SFT-v3 in Marta's panel. SFT-v3 as the
reference and the base model are neutral ink. Text is always ink, never a series colour.
"""

from __future__ import annotations

import csv
import math
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

FONT = "Liberation Sans, sans-serif"
INK, INK2, MUTED, GRID, BAND = "#2C2C2A", "#52514e", "#888780", "#e4e3df", "#efeee9"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num(v: float | str, digits: int = 2) -> str:
    """Signed, half to even on the value as written (as build_poster.py prints it)."""
    d = Decimal(str(v)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
    return f"{d:+}".replace("-", "−")


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=20, anchor="start", fill=INK, bold=False):
        weight = ' font-weight="bold"' if bold else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}"{weight}>{_esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1.5, dash=None, opacity=1.0, clip=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' stroke-opacity="{opacity}"' if opacity < 1 else ""
        extra += f' clip-path="url(#{clip})"' if clip else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def path(self, pts, stroke, width=3.0, dash=None, opacity=1.0, clip=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' stroke-opacity="{opacity}"' if opacity < 1 else ""
        extra += f' clip-path="url(#{clip})"' if clip else ""
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def circle(self, x, y, r, fill, stroke="#ffffff", width=3.0):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"/>')

    def rect(self, x, y, w, h, fill, opacity=1.0):
        op = f' fill-opacity="{opacity}"' if opacity < 1 else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'fill="{fill}"{op}/>')

    def clip(self, cid, x, y, w, h):
        self.add(f'<defs><clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}"/>'
                 f'</clipPath></defs>')

    def write(self, path: Path) -> None:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">')
        path.write_text("\n".join([head, *self.parts, "</svg>"]) + "\n")


def _scale(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) * (r1 - r0) / (d1 - d0)


def _spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Move label positions apart by at least gap, keeping their order, inside [lo, hi]."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k, i in enumerate(order):
        if k and out[i] < out[order[k - 1]] + gap:
            out[i] = out[order[k - 1]] + gap
    over = max(0.0, out[order[-1]] - hi)
    for i in order:
        out[i] = max(lo, out[i] - over)
    return out


def _seeds(table: list[dict], name: str) -> list[dict]:
    pattern = rf"{re.escape(name)} s\d+"
    return [r for r in table if r["label"] == name or re.fullmatch(pattern, r["label"])]


# Recipes on the poster: recipes.csv name, label, colour, exploratory.
RECIPES = [
    ("Distill", "Distillation", AQUA, False),
    ("DPO_V3_4880", "DPO", BLUE, False),
    ("GRPO+decay R4", "GRPO + lr decay", ORANGE, False),
    ("DAPO+decay D1", "DAPO + lr decay", ORANGE, False),
    ("GRPO-v3-all ckpt 525", "GRPO ckpt 525", ORANGE, True),
]


def cost_quality(root: Path, out: Path) -> None:
    table = _rows(root / "reports/final/final_table.csv")
    rec = {r["name"]: r for r in _rows(root / "reports/final/recipes.csv")}
    ref = next(r for r in table if r["label"] == "SFT-v3")
    common = float(ref["common_hire_rate_pct"])

    w, h = 950, 600
    x0, x1, y0, y1 = 96, 600, 66, 462
    seeds = {name: _seeds(table, name) for name, *_ in RECIPES}
    lows = [float(s["ci_common_low"]) for ss in seeds.values() for s in ss]
    highs = [float(s["ci_common_high"]) for ss in seeds.values() for s in ss]
    ylo, yhi = math.floor(min(lows) * 2) / 2, math.ceil(max(highs) * 2) / 2
    sx, sy = _scale(0, 40, x0, x1), _scale(ylo, yhi, y1, y0)

    g = Svg(w, h)
    title = f"Change in unstable sets against SFT-v3 at its hire rate ({common:.2f}%), pp"
    g.text(x0 - 70, 30, title, size=21, fill=INK2)
    v = ylo
    while v <= yhi + 1e-9:
        g.line(x0, sy(v), x1, sy(v), stroke=GRID, width=1.2)
        g.text(x0 - 12, sy(v) + 7, _num(v, 1) if v else "0", size=19, anchor="end", fill=INK2)
        v += 0.5
    for t in range(0, 41, 10):
        g.text(sx(t), y1 + 32, str(t), size=19, anchor="middle", fill=INK2)
    g.text((x0 + x1) / 2, y1 + 64, "extra GPU-hours per model over SFT-v3 (its data and training)",
           size=20, anchor="middle", fill=INK2)
    g.line(x0, sy(0), x1, sy(0), stroke=INK, width=2.5)
    g.text(x0 + 10, sy(0) - 10, "SFT-v3", size=19, fill=INK)

    labels = []
    for name, label, colour, exploratory in RECIPES:
        ss = seeds[name]
        k = len(ss)
        offsets = [0.0] if k == 1 else [-0.7 + 1.4 * i / (k - 1) for i in range(k)]
        for s, dx in zip(ss, offsets):
            x = sx(float(s["extra_gpu_h"]) + dx)
            g.line(x, sy(float(s["ci_common_low"])), x, sy(float(s["ci_common_high"])),
                   stroke=colour, width=3, opacity=0.45)
            g.circle(x, sy(float(s["delta_common_pp"])), 6, colour, width=1.5)
        mx, my = sx(float(rec[name]["extra_gpu_h"])), sy(float(rec[name]["delta_common_pp"]))
        if exploratory:
            g.circle(mx, my, 12, "#ffffff", stroke=colour, width=5)
        else:
            g.circle(mx, my, 13, colour, width=3.5)
        n = rec[name]["seeds"]
        text = f"{label}, {n} seed{'s' if n != '1' else ''}: {_num(rec[name]['delta_common_pp'])}"
        labels.append((my, mx, text, colour))

    ys = _spread([lab[0] for lab in labels], 34, y0 + 10, y1 - 4)
    lx = x1 + 34
    for (my, mx, text, colour), ly in zip(labels, ys):
        g.line(mx + 15, my, lx - 22, ly - 7, stroke=MUTED, width=1.2)
        g.circle(lx - 12, ly - 7, 7, colour, width=0)
        g.text(lx, ly, text, size=19)

    ly = h - 14
    g.circle(x0 - 60, ly - 7, 11, INK2, width=3)
    g.text(x0 - 42, ly, "mean over training seeds", size=18, fill=INK2)
    g.circle(x0 + 230, ly - 7, 5, INK2, width=1.5)
    g.line(x0 + 230, ly - 22, x0 + 230, ly + 8, stroke=INK2, width=3, opacity=0.45)
    g.text(x0 + 246, ly, "one seed, 95% CI", size=18, fill=INK2)
    g.circle(x0 + 430, ly - 7, 10, "#ffffff", stroke=INK2, width=4)
    g.text(x0 + 448, ly, "checkpoint picked with audit in view", size=18, fill=INK2)
    g.write(out)


CURVE_RECIPES = [("DPO_V3_4880", "DPO", BLUE), ("Distill", "Distillation", AQUA),
                 ("GRPO+decay R4", "GRPO + decay", ORANGE)]


def curves(root: Path, out: Path) -> None:
    rows = _rows(root / "reports/final/curves.csv")
    table = _rows(root / "reports/final/final_table.csv")
    ref_row = next(r for r in table if r["label"] == "SFT-v3")
    common = float(ref_row["common_hire_rate_pct"])
    by = {}
    for r in rows:
        by.setdefault(r["label"], {})[float(r["hire_rate_pct"])] = float(r["unstable_pct"])
    ref = by["SFT-v3"]
    grid = sorted(ref)

    series = []
    for name, label, colour in CURVE_RECIPES:
        seeds = [lab for lab in by if re.fullmatch(rf"{re.escape(name)} s\d+", lab)]
        deltas = [[by[s][x] - ref[x] for x in grid] for s in seeds]
        means = [sum(col) / len(col) for col in zip(*deltas)]
        series.append((label, colour, deltas, means))

    w, h = 680, 480
    x0, x1, y0, y1 = 78, 486, 58, 360
    allm = [v for *_, m in series for v in m]
    ylo, yhi = math.floor(min(allm)) - 0.5, math.ceil(max(allm)) + 0.5
    sx, sy = _scale(grid[0], grid[-1], x0, x1), _scale(ylo, yhi, y1, y0)

    g = Svg(w, h)
    g.clip("plot", x0, y0, x1 - x0, y1 - y0)
    g.text(x0 - 64, 28, "Change in unstable sets against SFT-v3 (the 0 line), pp", size=21,
           fill=INK2)
    v = math.ceil(ylo)
    while v <= yhi:
        g.line(x0, sy(v), x1, sy(v), stroke=GRID, width=1.2)
        g.text(x0 - 10, sy(v) + 6, _num(v, 0) if v else "0", size=19, anchor="end", fill=INK2)
        v += 1
    for t in range(10, 36, 5):
        g.text(sx(t), y1 + 30, f"{t}%", size=19, anchor="middle", fill=INK2)
    g.text((x0 + x1) / 2, y1 + 62, "hire rate the arm is moved to", size=20, anchor="middle",
           fill=INK2)
    g.line(sx(common), y0, sx(common), y1, stroke=INK2, width=1.5, dash="6 5")
    g.text(sx(common) + 6, y0 + 16, f"SFT-v3's {common:.1f}%", size=18, fill=INK2)
    g.line(x0, sy(0), x1, sy(0), stroke=INK, width=2.5)

    ends = []
    for label, colour, deltas, means in series:
        for d in deltas:
            pts = [(sx(x), sy(v)) for x, v in zip(grid, d)]
            g.path(pts, colour, width=1.5, opacity=0.3, clip="plot")
        g.path([(sx(x), sy(v)) for x, v in zip(grid, means)], colour, width=4, clip="plot")
        ends.append((sy(means[-1]), label, colour))
    ys = _spread([e[0] for e in ends], 30, y0 + 10, y1)
    for (ey, label, colour), ly in zip(ends, ys):
        g.line(x1 + 4, ey, x1 + 18, ly - 6, stroke=MUTED, width=1.2)
        g.circle(x1 + 28, ly - 6, 6, colour, width=0)
        g.text(x1 + 40, ly, label, size=19)
    note = "Bold: mean of three training seeds; thin: each seed."
    g.text(x0 - 60, h - 8, note, size=18, fill=INK2)
    g.write(out)


def residual(root: Path, out: Path, contrast: str = "mil_combat_civ_impl") -> None:
    def load(spec):
        rows = _rows(root / f"reports/interp/patching/{spec}-grid-{contrast}-n64-s42/real.csv")
        res = {}
        for r in rows:
            if r["kind"] == "resid_pre":
                res.setdefault(r["cls"], {})[int(r["layer"])] = float(r["removed_logits_per_pair"])
            elif r["kind"] == "final_prenorm":
                res.setdefault(r["cls"], {})[32] = float(r["removed_logits_per_pair"])
        return res

    data = {"Base": (load("base"), MUTED), "SFT-v3": (load("sft"), VIOLET)}
    w, h = 640, 450
    x0, x1, y0, y1 = 74, 540, 62, 340
    top = max(v for d, _ in data.values() for c in d.values() for v in c.values())
    yhi = math.ceil(top * 4) / 4
    sx, sy = _scale(0, 32, x0, x1), _scale(0, yhi, y1, y0)

    g = Svg(w, h)
    for layer in range(3, 32, 4):
        a, b = max(0, layer + 0.5), min(32, layer + 1.5)
        g.rect(sx(a), y0, sx(b) - sx(a), y1 - y0, BAND)
    g.text(x0 - 60, 26, "Attribute effect removed, logits per pair", size=19, fill=INK2)
    v = 0.0
    while v <= yhi + 1e-9:
        g.line(x0, sy(v), x1, sy(v), stroke=GRID, width=1.0)
        g.text(x0 - 10, sy(v) + 6, f"{v:.1f}", size=17, anchor="end", fill=INK2)
        v += 0.5
    for t in range(0, 33, 5):
        g.text(sx(t), y1 + 28, str(t), size=17, anchor="middle", fill=INK2)
    g.text((x0 + x1) / 2, y1 + 58, "residual stream patched at the input to layer L", size=18,
           anchor="middle", fill=INK2)
    g.line(x0, y1, x1, y1, stroke=INK2, width=1.5)

    for label, (d, colour) in data.items():
        for cls, dash in (("suffix", None), ("late", "10 7")):
            pts = sorted(d[cls].items())
            g.path([(sx(x), sy(v)) for x, v in pts], colour, width=4, dash=dash)
        g.text(x1 + 12, sy(d["suffix"][32]) + 6, label, size=18, bold=True)

    ly = h - 10
    g.line(x0 - 60, ly - 6, x0 - 24, ly - 6, stroke=INK2, width=4)
    g.text(x0 - 14, ly, "all tokens after the attribute", size=17, fill=INK2)
    g.line(x0 + 262, ly - 6, x0 + 298, ly - 6, stroke=INK2, width=4, dash="10 7")
    g.text(x0 + 308, ly, "last tokens only", size=17, fill=INK2)
    g.write(out)


def build(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    cost_quality(root, out_dir / "cost_quality.svg")
    curves(root, out_dir / "curves.svg")
    residual(root, out_dir / "residual.svg")
