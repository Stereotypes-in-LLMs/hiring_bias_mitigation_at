"""A self-contained HTML report of line charts and tables, with no plotting dependency.

The analysis scripts on the cluster run without matplotlib, so the charts are drawn here, as
inline SVG, and a drawn chart needs nothing at view time. That matters because the viewers this
report is read in sandbox the page and run none of its scripts, and some also drop its
stylesheet: every mark therefore carries its light-surface colour as an attribute, which the
stylesheet's role tokens (one set for a light surface, one for a dark) override wherever it
applies. Where scripts do run, a small one adds the hover layer to the drawn chart -- a
crosshair that snaps to the nearest step and one tooltip listing every series, the same on
keyboard focus with the arrow keys -- and draws nothing itself. Without it each marker still
names its value in a native tooltip, and every chart has a table view.

Series "a" and "b" are the first two slots of the reference categorical palette (validated for
colour-vision deficiency on both surfaces), "neutral" is muted ink. Primary series are 2px lines
with 8px markers and a direct label at their end, spread apart where two ends would collide;
secondary series are 1px de-emphasised lines.
"""

from __future__ import annotations

import html
import json
import math

#: Role tokens for a light and a dark surface. The light values double as the marks' colour
#: attributes, which is what a viewer without the stylesheet shows.
LIGHT = {"page": "#f9f9f7", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink-2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "ring": "rgba(11,11,11,0.10)", "a": "#2a78d6", "b": "#eb6834", "neutral": "#898781"}
DARK = {"page": "#0d0d0d", "surface": "#1a1a19", "ink": "#f0efec", "ink-2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "ring": "rgba(255,255,255,0.10)", "a": "#3987e5", "b": "#d95926", "neutral": "#898781"}
ROLES = ("a", "b", "neutral")

#: Chart geometry in viewBox units; the drawn chart scales to its card.
WIDTH, HEIGHT = 680, 300
MARGIN = {"l": 46, "r": 16, "t": 12, "b": 40}
#: The right margin when primary series carry end labels, and the labels' minimum spacing.
LABEL_ROOM, LABEL_GAP = 112, 14
FONT = "system-ui, -apple-system, 'Segoe UI', 'Noto Sans', sans-serif"


def _tokens(theme: dict) -> str:
    return " ".join(f"--{k}: {v};" for k, v in theme.items())


_STYLE = (
    f":root {{ color-scheme: light; {_tokens(LIGHT)} }}\n"
    '@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { '
    f"color-scheme: dark; {_tokens(DARK)} }} }}\n"
    f':root[data-theme="dark"] {{ color-scheme: dark; {_tokens(DARK)} }}\n'
    + "".join(f".s-{r} {{ stroke: var(--{r}); }} .f-{r} {{ fill: var(--{r}); }}\n" for r in ROLES)
    + """
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 0 0 2px; }
.lead, .note { color: var(--ink-2); } .note { font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
  padding: 16px; margin: 18px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 10px 0 4px; font-size: 13px;
  color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { flex: none; }
.plot { position: relative; overflow-x: auto; }
.plot:focus-visible { outline: 2px solid var(--a); outline-offset: 2px; border-radius: 6px; }
.plot > svg { display: block; width: 100%; height: auto; min-width: 480px; }
.plot text { font-variant-numeric: tabular-nums; }
.c-grid { stroke: var(--grid); } .c-axis, .c-cross { stroke: var(--axis); }
.c-tick { fill: var(--muted); } .c-lead { stroke: var(--muted); }
.c-end { fill: var(--ink-2); } .c-ring { stroke: var(--surface); }
.key { display: inline-block; width: 18px; height: 0; border-top-style: solid; }
.tip { position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12.5px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); min-width: 190px; display: none; z-index: 2; }
.tip .row { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.tip b { font-variant-numeric: tabular-nums; min-width: 44px; display: inline-block; }
.tip .name { color: var(--ink-2); } .tip .head { color: var(--muted); margin-bottom: 4px; }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12.5px; margin-top: 8px; width: 100%; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
""")

#: The hover layer. It reads each chart's geometry from the page data and adds a crosshair and
#: a tooltip to the chart already drawn; a page whose scripts do not run loses only this.
_SCRIPT = r"""
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const NS = 'http://www.w3.org/2000/svg';
function fmt(v, spec) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return spec.percent ? (100 * v).toFixed(1) + '%' : v.toFixed(spec.digits ?? 2);
}
function attach(spec) {
  const box = document.getElementById('plot-' + spec.id);
  const svg = box && box.querySelector('svg');
  if (!svg) return;
  const g = spec.geom;
  const X = v => g.l + (v - g.x0) / (g.x1 - g.x0) * (g.W - g.l - g.r);
  for (const t of svg.querySelectorAll('circle > title')) t.remove();
  const cross = document.createElementNS(NS, 'line');
  for (const [k, v] of Object.entries({class: 'c-cross', y1: g.t, y2: g.H - g.b,
                                       stroke: '#c3c2b7', visibility: 'hidden'}))
    cross.setAttribute(k, v);
  svg.appendChild(cross);
  const tip = document.createElement('div'); tip.className = 'tip'; box.appendChild(tip);
  const steps = [...new Set(spec.series.flatMap(s => s.points.map(p => p[0])))]
    .sort((a, b) => a - b);
  let at = null;
  function show(xv) {
    at = xv;
    cross.setAttribute('x1', X(xv)); cross.setAttribute('x2', X(xv));
    cross.setAttribute('visibility', 'visible');
    tip.textContent = '';
    const head = document.createElement('div'); head.className = 'head';
    head.textContent = `${spec.x_label} ${xv}`; tip.appendChild(head);
    for (const s of spec.series) {
      let best = null;
      for (const p of s.points)
        if (best === null || Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      if (!best) continue;
      const row = document.createElement('div'); row.className = 'row';
      const key = document.createElement('span'); key.className = 'key';
      key.style.borderTopWidth = s.weight === 'primary' ? '2px' : '1px';
      key.style.borderTopColor = `var(--${s.role})`;
      const val = document.createElement('b'); val.textContent = fmt(best[1], spec);
      const name = document.createElement('span'); name.className = 'name';
      name.textContent = s.name + (best[0] !== xv ? `（${spec.x_label} ${best[0]}）` : '');
      row.append(key, val, name); tip.appendChild(row);
    }
    tip.style.display = 'block';
    const r = svg.getBoundingClientRect(), b = box.getBoundingClientRect();
    const px = r.left - b.left + box.scrollLeft + X(xv) / g.W * r.width;
    const w = tip.offsetWidth, room = box.scrollLeft + box.clientWidth;
    tip.style.left = (px + 12 + w <= room ? px + 12 : Math.max(box.scrollLeft, px - 12 - w)) + 'px';
    tip.style.top = '8px';
  }
  function hide() { cross.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; }
  svg.addEventListener('pointermove', ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * g.W;
    const xv = g.x0 + (px - g.l) / (g.W - g.l - g.r) * (g.x1 - g.x0);
    show(steps.reduce((a, c) => Math.abs(c - xv) < Math.abs(a - xv) ? c : a));
  });
  svg.addEventListener('pointerleave', hide);
  box.tabIndex = 0;
  box.setAttribute('aria-label', spec.title + '：方向键移动读数');
  box.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    const i = at === null ? 0 : steps.indexOf(at) + (ev.key === 'ArrowRight' ? 1 : -1);
    show(steps[Math.max(0, Math.min(steps.length - 1, i))]); ev.preventDefault();
  });
  box.addEventListener('focus', () => show(at ?? steps[0]));
  box.addEventListener('blur', hide);
}
for (const spec of DATA.charts) attach(spec);
"""


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _fmt(v, spec: dict) -> str:
    if not _finite(v):
        return "–"
    return f"{100 * v:.1f}%" if spec.get("percent") else f"{v:.{spec.get('digits', 2)}f}"


def _segments(points: list) -> list[list]:
    """The runs of consecutive finite points; a missing value breaks the line."""
    runs, run = [], []
    for x, y in points:
        if _finite(y):
            run.append((float(x), float(y)))
        elif run:
            runs.append(run)
            run = []
    return [*runs, run] if run else runs


def dodge(ys: list[float], gap: float, low: float, high: float) -> list[float]:
    """Label heights, in input order: each as near its own y as `gap` spacing allows (labels
    that would collide spread evenly about their mean), the whole stack kept in [low, high]."""
    if not ys:
        return []
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    clusters = []  # [mean of the members' own heights, member count]
    for i in order:
        clusters.append([ys[i], 1])
        while len(clusters) > 1:
            (m1, n1), (m2, n2) = clusters[-2], clusters[-1]
            if (m2 - (n2 - 1) * gap / 2) - (m1 + (n1 - 1) * gap / 2) >= gap:
                break
            clusters[-2:] = [[(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2]]
    placed = [m - (n - 1) * gap / 2 + k * gap for m, n in clusters for k in range(n)]
    shift = max(0.0, low - placed[0]) - max(0.0, placed[-1] - high)
    out = [0.0] * len(ys)
    for rank, i in enumerate(order):
        out[i] = placed[rank] + shift
    return out


def geometry(spec: dict) -> dict:
    """The chart's frame: viewBox size, margins and the data ranges the axes span."""
    xs = [float(p[0]) for s in spec["series"] for p in s["points"]]
    labelled = any(s.get("weight") == "primary" and s.get("label")
                   and any(_finite(p[1]) for p in s["points"]) for s in spec["series"])
    return {"W": WIDTH, "H": HEIGHT, **MARGIN, "r": LABEL_ROOM if labelled else MARGIN["r"],
            "x0": float(spec["x_min"] if spec.get("x_min") is not None else min(xs)),
            "x1": float(spec["x_max"] if spec.get("x_max") is not None else max(xs)),
            "y0": float(spec["y_min"]), "y1": float(spec["y_max"])}


def chart_svg(spec: dict) -> str:
    """One chart as an inline SVG element (see `render_report` for the spec)."""
    g = geometry(spec)
    left, right, top, bottom = g["l"], g["W"] - g["r"], g["t"], g["H"] - g["b"]

    def X(v):
        return left + (float(v) - g["x0"]) / (g["x1"] - g["x0"]) * (right - left)

    def Y(v):
        return top + (1 - (float(v) - g["y0"]) / (g["y1"] - g["y0"])) * (bottom - top)

    esc = html.escape
    tick = (lambda t: f"{round(100 * t)}%") if spec.get("percent") else (lambda t: f"{t:g}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {g["W"]} {g["H"]}" '
           f'width="100%" role="img" aria-label="{esc(spec["title"])}" '
           f'font-family="{FONT}">']
    for t in spec["y_ticks"]:
        y = Y(t)
        out.append(f'<line class="c-grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="{LIGHT["grid"]}" stroke-width="1"/>')
        out.append(f'<text class="c-tick" x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="{LIGHT["muted"]}">{tick(t)}</text>')
    out.append(f'<line class="c-axis" x1="{left}" x2="{right}" y1="{Y(g["y0"]):.1f}" '
               f'y2="{Y(g["y0"]):.1f}" stroke="{LIGHT["axis"]}" stroke-width="1"/>')
    for t in spec["x_ticks"]:
        out.append(f'<text class="c-tick" x="{X(t):.1f}" y="{bottom + 16}" text-anchor="middle" '
                   f'font-size="11" fill="{LIGHT["muted"]}">{t:g}</text>')
    out.append(f'<text class="c-tick" x="{(left + right) / 2:.1f}" y="{g["H"] - 6}" '
               f'text-anchor="middle" font-size="11" fill="{LIGHT["muted"]}">'
               f'{esc(spec["x_label"])}</text>')
    # secondary series underneath, each primary series' markers right after its own line
    for s in sorted(spec["series"], key=lambda s: s.get("weight") == "primary"):
        runs = _segments(s["points"])
        if not runs:
            continue
        primary, role = s.get("weight") == "primary", s["role"]
        fade = "" if primary else ' opacity="0.55"'
        d = " ".join("M" + " L".join(f"{X(x):.1f} {Y(y):.1f}" for x, y in run) for run in runs)
        out.append(f'<path class="s-{role}" d="{d}" fill="none" stroke="{LIGHT[role]}" '
                   f'stroke-width="{2 if primary else 1}" stroke-linejoin="round" '
                   f'stroke-linecap="round"{fade}/>')
        if primary:
            for x, y in (p for run in runs for p in run):
                out.append(f'<circle class="f-{role} c-ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" '
                           f'r="4" fill="{LIGHT[role]}" stroke="{LIGHT["surface"]}" '
                           f'stroke-width="2"><title>{esc(s["name"])}，{esc(spec["x_label"])} '
                           f'{x:g}：{_fmt(y, spec)}</title></circle>')
    ends = []
    for s in spec["series"]:
        if s.get("weight") == "primary" and s.get("label"):
            runs = _segments(s["points"])
            if runs:
                ends.append((s, runs[-1][-1]))
    heights = dodge([Y(p[1]) for _, p in ends], LABEL_GAP, top + 4, bottom)
    for (s, (x, y)), h in zip(ends, heights):
        role, ex, ey = s["role"], X(x), Y(y)
        # a thin leader from the series' end to its label, then the series' key and value
        out.append(f'<line class="c-lead" x1="{ex + 6:.1f}" y1="{ey:.1f}" x2="{ex + 12:.1f}" '
                   f'y2="{h:.1f}" stroke="{LIGHT["muted"]}" stroke-width="1"/>')
        out.append(f'<line class="s-{role}" x1="{ex + 13:.1f}" y1="{h:.1f}" x2="{ex + 22:.1f}" '
                   f'y2="{h:.1f}" stroke="{LIGHT[role]}" stroke-width="2" '
                   'stroke-linecap="round"/>')
        out.append(f'<text class="c-end" x="{ex + 26:.1f}" y="{h + 4:.1f}" font-size="12" '
                   f'fill="{LIGHT["ink-2"]}">{esc(s["label"])} {_fmt(y, spec)}</text>')
    out.append("</svg>")
    return "".join(out)


def _key(role: str, primary: bool) -> str:
    colour, fade = LIGHT[role], "" if primary else ' opacity="0.55"'
    line = (f'<line class="s-{role}" x1="1" x2="21" y1="6" y2="6" stroke="{colour}" '
            f'stroke-width="{2 if primary else 1}"{fade}/>')
    dot = (f'<circle class="f-{role} c-ring" cx="11" cy="6" r="4" fill="{colour}" '
           f'stroke="{LIGHT["surface"]}" stroke-width="2"/>') if primary else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="22" height="12" '
            f'viewBox="0 0 22 12" aria-hidden="true">{line}{dot}</svg>')


def _legend(series: list[dict]) -> str:
    items = [f'<span>{_key(s["role"], s.get("weight") == "primary")}{html.escape(s["name"])}'
             "</span>" for s in series]
    return '<div class="legend">' + "".join(items) + "</div>"


def _table(columns: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in rows)
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def render_report(title: str, intro: list[str], charts: list[dict], tables: list[dict]) -> str:
    """The page. `charts`: id, title, subtitle, x_label, x_ticks, x_min, x_max, y_min, y_max,
    y_ticks, percent, digits, series [{name, label, role a|b|neutral, weight
    primary|secondary, points [[x, y], ...]}], table {columns, rows}. `tables`: title, note,
    columns, rows."""
    parts = [f"<h1>{html.escape(title)}</h1>"]
    parts += [f'<p class="lead">{html.escape(p)}</p>' for p in intro]
    for c in charts:
        parts.append(
            f'<section class="card"><h2>{html.escape(c["title"])}</h2>'
            f'<div class="note">{html.escape(c["subtitle"])}</div>{_legend(c["series"])}'
            f'<div class="plot" id="plot-{c["id"]}">{chart_svg(c)}</div>'
            f'<details><summary>表格视图</summary>{_table(**c["table"])}</details></section>')
    for t in tables:
        parts.append(f'<section class="card"><h2>{html.escape(t["title"])}</h2>'
                     f'<div class="note">{html.escape(t.get("note", ""))}</div>'
                     f'{_table(t["columns"], t["rows"])}</section>')

    def clean(v):
        return float(v) if _finite(v) else None

    payload = [{"id": c["id"], "title": c["title"], "x_label": c["x_label"],
                "percent": bool(c.get("percent")), "digits": c.get("digits", 2),
                "geom": geometry(c),
                "series": [{"name": s["name"], "role": s["role"], "weight": s.get("weight"),
                            "points": [[float(p[0]), clean(p[1])] for p in s["points"]]}
                           for s in c["series"]]}
               for c in charts]
    data = json.dumps({"charts": payload}, allow_nan=False, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return ("<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
            + "".join(parts)
            + f'</main><script type="application/json" id="report-data">{data}</script>'
            + f"<script>{_SCRIPT}</script></body></html>")
