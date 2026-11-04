"""Vector figures for the revised poster, using completed audit tables only.

Run with a Python environment containing matplotlib and numpy. No model is run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
WIDTH = 19.375
INK, MUTED, GRID = "#203C4A", "#526873", "#DCE4E7"
METHODS = (
    ("DPO", "DPO_V3_4880", "dpo4880", "#203C4A", "-"),
    ("GRPO + lr decay", "GRPO+decay R4", "grpo_r4", "#708087", (0, (5, 2.5))),
    ("Self-distillation", "Distill", "distill43", "#547F2F", (0, (6, 2, 1.5, 2))),
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 28,
    "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
    "axes.labelsize": 28, "axes.titlesize": 28, "xtick.labelsize": 25,
    "ytick.labelsize": 28, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": "#A9B9C0",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "savefig.facecolor": "white", "figure.facecolor": "white",
    "svg.fonttype": "none", "legend.frameon": False,
})


def read(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="") as stream:
        return list(csv.DictReader(stream))


def method_rows(rows: list[dict], prefix: str) -> list[dict]:
    selected = [row for row in rows if row["label"] in [f"{prefix} s{seed}" for seed in (42, 43, 44)]]
    selected.sort(key=lambda row: int(row["seed"]))
    assert len(selected) == 3 and [int(row["seed"]) for row in selected] == [42, 43, 44], prefix
    return selected


def style(ax, *, ygrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x", color=GRID, lw=.65)
    ax.tick_params(axis="both", length=4.5, width=1.1, pad=7)


def save(fig, name: str, preview_dir: Path | None):
    # Check at the actual figure size: a poster render can hide clipped labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                raise ValueError(f"Clipped text in {name}: {label.get_text()!r} at {box.bounds}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": f"Poster: {name}",
                "Subject": "Exploratory development-benchmark audit; existing measurements only"})
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=130)
    print(f"{name}.pdf: {fig.get_size_inches()[0]:.3f} x {fig.get_size_inches()[1]:.3f} inches")
    plt.close(fig)


def behaviour(preview_dir: Path | None):
    final = read("reports/final/final_table.csv")
    curves = read("reports/final/curves.csv")
    reference = {float(row["hire_rate_pct"]): float(row["unstable_pct"])
                 for row in curves if row["label"] == "SFT-v3"}
    assert len(reference) == 51
    fig = plt.figure(figsize=(WIDTH, 5.0))
    forest = fig.add_axes([.185, .305, .17, .555])
    curve = fig.add_axes([.465, .305, .52, .555])
    forest.set_title("At 22.51% hire rate", loc="left", pad=16, fontsize=26)
    forest.axvline(0, color="#6E818A", lw=1.8, ls=(0, (3, 3)), zorder=1)
    forest.set_xlim(-1.85, 1.3)
    forest.set_xticks([-1, 0, 1])
    forest.set_ylim(-.55, 2.55)
    forest.set_yticks([2, 1, 0], [method[0] for method in METHODS])
    forest.tick_params(axis="y", length=0, pad=12)
    forest.set_xlabel("Δ unstable sets vs SFT (pp)", labelpad=8, fontsize=26)
    style(forest)
    all_delta = []
    handles = []
    for group, (label, prefix, _, colour, line) in enumerate(METHODS):
        selected = method_rows(final, prefix)
        for offset, row in zip((.24, 0, -.24), selected):
            point, low, high = (float(row[key]) for key in
                                ("delta_common_pp", "ci_common_low", "ci_common_high"))
            assert low <= point <= high
            forest.errorbar(point, 2-group+offset, xerr=[[point-low], [high-point]],
                            color=colour, lw=3.2, capsize=5, capthick=2.2,
                            fmt="o", markersize=11, mfc="white", mec=colour, mew=2.5, zorder=3)
        means = []
        for seed in (42, 43, 44):
            series = sorted((float(row["hire_rate_pct"]), float(row["unstable_pct"]))
                            for row in curves if row["label"] == f"{prefix} s{seed}")
            assert len(series) == 51, (prefix, seed)
            x = np.array([row[0] for row in series])
            delta = np.array([row[1]-reference[row[0]] for row in series])
            assert np.all(np.diff(x) > 0) and x[0] == 10 and x[-1] == 35
            means.append(delta)
            all_delta.extend(delta)
            curve.plot(x, delta, color=colour, lw=1.6, alpha=.40, ls=line, zorder=2)
        marker = ("o", "s", "^")[group]
        mean_delta = np.mean(means, axis=0)
        curve.plot(x, mean_delta, color=colour, lw=5.0, ls=line,
                   marker=marker, markevery=10, markersize=7.5, markeredgewidth=1.5,
                   markerfacecolor="white", clip_on=False, zorder=4+group)
        if label == "DPO":
            endpoint = float(mean_delta[-1])
            curve.annotate(f"DPO: +{endpoint:.1f} pp more\nunstable at {x[-1]:.0f}% hire",
                           xy=(x[-1], endpoint), xytext=(18.6, 3.3),
                           ha="left", va="center", fontsize=24, fontweight="bold",
                           color=colour,
                           arrowprops={"arrowstyle": "->", "color": colour,
                                       "lw": 1.8, "shrinkA": 8, "shrinkB": 9},
                           zorder=10)
        legend_label = ("DPO", "GRPO + decay", "Self-distillation")[group]
        handles.append(Line2D([], [], color=colour, lw=5.0, ls=line, label=legend_label,
                              marker=marker, markersize=7.5, markerfacecolor="white"))
    curve.axhline(0, color="#6E818A", lw=1.8, zorder=1)
    curve.axvline(22.512, color="#6E818A", lw=1.8, ls=(0, (3, 3)), zorder=1)
    curve.set_xlim(10, 35)
    curve.set_xticks([10, 15, 20, 25, 30, 35])
    curve.set_ylim(min(all_delta)-.16, max(all_delta)+.23)
    curve.set_yticks([-1, 0, 1, 2, 3, 4])
    curve.set_xlabel("Matched hire rate (%)", labelpad=11)
    curve.set_ylabel("Δ unstable sets (pp)\nvs SFT", labelpad=11, fontsize=26)
    curve.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(-.005, 1.04),
                 borderaxespad=0, fontsize=26, handlelength=1.7, columnspacing=.8,
                 handletextpad=.6)
    style(curve, ygrid=True)
    fig.text(.012, .035, "Paired 95% CIs over candidates, seeds 42/43/44 top to bottom; 1 of 9 excludes zero. "
             "Thick curves: seed means; thin: single seeds.", ha="left", va="bottom", fontsize=20,
             color=MUTED)
    save(fig, "behaviour", preview_dir)


def residual(preview_dir: Path | None):
    overlap = read("reports/seed_overlap/overlap.csv")
    fig = plt.figure(figsize=(WIDTH, 3.8))
    jac = fig.add_axes([.185, .29, .26, .47])
    tail = fig.add_axes([.60, .29, .385, .47])
    fig.text(.185, .985, "Unstable-set overlap", ha="left", va="top", fontsize=26, fontweight="bold")
    fig.text(.60, .985, "Attribute-effect distribution", ha="left", va="top", fontsize=26, fontweight="bold")
    jac.set_yticks([2, 1, 0], [method[0] for method in METHODS])
    jac.set_ylim(-.9, 2.5)
    jac.set_xlim(0, 1)
    jac.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    jac.tick_params(axis="y", length=0, pad=12)
    jac.set_xlabel("Jaccard, seeds 43 / 44", labelpad=9)
    jac.legend(handles=[Line2D([], [], marker="o", color=MUTED, mfc=MUTED, ls="", markersize=11,
                              label="Actual"),
                        Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", markersize=11,
                               markeredgewidth=2.5, label="SFT threshold control")],
               ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.05), borderaxespad=0,
               borderpad=0, fontsize=26, handlelength=1, handletextpad=.4, columnspacing=.7)
    tail.set_xscale("log")
    tail.set_xlim(.1, 16)
    tail.set_ylim(-.9, 2.5)
    tail.set_yticks([])
    tail.xaxis.set_major_locator(FixedLocator([.1, .3, 1, 3, 10]))
    tail.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    tail.xaxis.set_minor_locator(FixedLocator([]))
    tail.axvline(1, color="#6E818A", lw=1.8, ls=(0, (3, 3)))
    tail.set_xlabel("Absolute effect / SFT · log scale", labelpad=9)
    tail.legend(handles=[Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", markersize=11,
                               markeredgewidth=2.5, label="p50"),
                         Line2D([], [], marker="o", color=MUTED, mfc=MUTED, ls="", markersize=11,
                                label="p99")],
                ncol=2, loc="lower left", bbox_to_anchor=(0, 1.05), borderaxespad=0,
                borderpad=0, fontsize=26, handletextpad=.5, columnspacing=1.3)
    style(jac)
    style(tail)
    for index, (label, prefix, spec, colour, _) in enumerate(METHODS):
        y = 2-index
        row = next(row for row in overlap if row["recipe"] == prefix and row["seeds"] == "43/44")
        actual, control = float(row["jaccard"]), float(row["sft_v3_jaccard"])
        jac.plot([actual, control], [y, y], color=colour, alpha=.9, lw=3.2)
        jac.plot(actual, y, "o", color=colour, markersize=11.5)
        jac.plot(control, y, "o", mec=colour, mfc="white", mew=2.5, markersize=11.5)
        jac.text(actual, y-.20, f"{actual:.2f}", ha="center", va="top", fontsize=26, color=colour)
        jac.text(control, y-.20, f"{control:.2f}", ha="center", va="top", fontsize=26, color=colour)
        sensitivity = read(f"reports/interp/distill/{spec}/sensitivity.csv")
        assert len(sensitivity) == 4 and all(int(r["pairs"]) == 450 for r in sensitivity)
        for key, offset, fill in (("r_attr_median", .20, "white"), ("r_attr_p99", -.20, colour)):
            values = np.array([float(r[key]) for r in sensitivity])
            assert np.all(values > 0)
            tail.plot([min(values), max(values)], [y+offset]*2,
                      color=colour, alpha=.9, lw=2.4)
            # Four contrasts are points with deterministic vertical offsets to keep
            # near-coincident values visible. The line spans their range, never a CI.
            for value, jitter in zip(values, (-.07, -.023, .023, .07)):
                tail.plot(value, y+offset+jitter, marker="o", ls="", mec=colour, mfc=fill,
                          mew=2.2, markersize=10)
    save(fig, "residual", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    behaviour(args.preview_dir)
    residual(args.preview_dir)
