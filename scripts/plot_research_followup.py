"""Render the completed CPU follow-up; no model inference or new statistics."""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("DPO", "GRPO + lr decay", "Self-distillation")
COLORS = ("#183b4c", "#82949b", "#4f812b")


def read(root, name):
    with (root / (name + ".csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def selected(rows, **where):
    return [r for r in rows if all(r[k] == str(v) for k, v in where.items())]


def main(source, out):
    out.mkdir(parents=True, exist_ok=True)
    curve, geo, overlaps, effects = [read(source, x) for x in
        ("global_curve", "threshold_geometry", "common_rate_overlap", "effect_summary")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelcolor": COLORS[0], "text.color": COLORS[0],
                         "axes.edgecolor": "#bac6ca", "xtick.color": COLORS[0],
                         "ytick.color": COLORS[0], "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    fig.subplots_adjust(left=.075, right=.97, bottom=.18, top=.87, hspace=.61, wspace=.27)
    fig.suptitle("Residual instability: magnitude, thresholds and affected CVs", x=.075,
                 y=.975, ha="left", fontsize=23, fontweight="bold")
    fig.text(.075, .935, "Development benchmark · three post-training seeds · shared SFT-v3 reference",
             fontsize=13)
    rates = sorted({float(r["rate_label"]) for r in curve if r["rate_label"] != "sft_common"})
    dpo_geo = selected(geo, recipe="DPO")
    assert all(int(r["top1pct_pairs_including_ties"]) == 5 for r in dpo_geo)
    assert all(float(r["top1pct_crossing_weight"]) == 0 for r in dpo_geo
               if r["rate_label"] != "sft_common" and float(r["rate_label"]) >= 31)
    for backend, color, style, label in (("vllm", COLORS[0], "-", "Served / vLLM"),
                                        ("hf", COLORS[1], "--", "Interpretation / HF")):
        curves, shares = [], []
        for seed in (42, 43, 44):
            c = {float(r["rate_label"]): float(r["delta_unstable_set_pp"]) for r in
                 selected(curve, backend=backend, recipe="DPO", seed=seed)
                 if r["rate_label"] != "sft_common"}
            vals = [c[x] for x in rates]
            curves.append(vals)
            ax[0, 0].plot(rates, vals, color=color, alpha=.28, lw=1, ls=style)
            vals = []
            for rate in rates:
                rr = [r for r in selected(geo, backend=backend, recipe="DPO", seed=seed)
                      if r["rate_label"] != "sft_common" and float(r["rate_label"]) == rate]
                vals.append(100 * sum(float(r["top1pct_crossing_weight"]) for r in rr)
                            / sum(float(r["top1pct_pairs_including_ties"]) for r in rr))
            shares.append(vals)
        ax[0, 0].plot(rates, np.mean(curves, axis=0), color=color, lw=2.8, ls=style, label=label)
        ax[0, 1].plot(rates, np.mean(shares, axis=0), color=color, lw=2.8, ls=style, label=label)
    for a in ax[0]:
        a.axvspan(31, 35, color=COLORS[2], alpha=.10)
        a.axvline(100 * 6990 / 31050, color=COLORS[1], lw=1, ls=":")
        a.set_xlim(10, 35)
        a.set_xlabel("Matched hire rate (%)")
        a.grid(axis="y", color="#e4e9eb")
    ax[0, 0].set_title("A  DPO deteriorates at high hire rates", loc="left", fontweight="bold", pad=17)
    ax[0, 0].set_ylabel("Δ unstable sets vs SFT-v3 (pp)")
    ax[0, 0].axhline(0, color=COLORS[1], lw=.8)
    ax[0, 0].legend(fontsize=10, frameon=False, loc="upper left")
    ax[0, 0].text(0, -.32, "Thick: three-seed means. Thin: individual seeds, not CIs.",
                  transform=ax[0, 0].transAxes, fontsize=10)
    ax[0, 1].set_title("B  The largest effects do not cross there", loc="left", fontweight="bold", pad=17)
    ax[0, 1].set_ylabel("Top-1% contrast pairs crossing the cut (%)")
    ax[0, 1].set_ylim(-2, 85)
    ax[0, 1].text(32.8, 16, "0% at\n31–35% hire", ha="center", fontsize=12, fontweight="bold", color=COLORS[2])
    ax[0, 1].text(0, -.32, "Top 5 pairs per contrast × 4 contrasts; descriptive means.\nThese binary contrasts are a subset of the full set-level audit.",
                  transform=ax[0, 1].transAxes, fontsize=10)
    units = ("set", "pair", "cv")
    for method, color in zip(METHODS, COLORS):
        vals = [[float(r["weighted_jaccard"]) for r in selected(overlaps, backend="vllm", recipe=method, unit=u)] for u in units]
        means = np.mean(vals, axis=1)
        err = np.array([means - np.min(vals, axis=1), np.max(vals, axis=1) - means])
        ax[1, 0].errorbar(range(3), means, yerr=err, color=color, marker="o", ms=6,
                         lw=2, elinewidth=1.5, capsize=4, label=method.replace(" + lr decay", " + decay"))
    ax[1, 0].set_title("C  CV overlap exceeds set overlap", loc="left", fontweight="bold", pad=17)
    ax[1, 0].set_xticks(range(3), ("Attribute set", "CV–job pair", "CV"))
    ax[1, 0].set_xlim(-.15, 2.15)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_ylabel("Weighted Jaccard at 22.512% hire")
    ax[1, 0].legend(fontsize=10, frameon=False, loc="upper left")
    ax[1, 0].grid(axis="y", color="#e4e9eb")
    ax[1, 0].text(0, -.32, "Served / vLLM scores. Bars: min–max across three seed pairs,\nnot confidence intervals. Membership = any unstable set in the unit.",
                  transform=ax[1, 0].transAxes, fontsize=10)
    for j, (method, color) in enumerate(zip(METHODS, COLORS)):
        rows = selected(effects, backend="hf", recipe=method)
        for k, column in enumerate(("absolute_effect_p99_ratio_vs_sft", "rank_effect_p99_ratio_vs_sft")):
            values = np.array([float(r[column]) for r in rows])
            x = j * 3 + k
            ax[1, 1].scatter(x + np.linspace(-.13, .13, len(values)), values, s=27,
                             facecolors=color if k == 0 else "white", edgecolors=color, linewidths=1.2)
    ax[1, 1].axhline(1, color=COLORS[1], lw=1, ls="--")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(.25, 17)
    ax[1, 1].set_yticks((.5, 1, 2, 4, 8, 16), ("0.5", "1", "2", "4", "8", "16"))
    ax[1, 1].set_xticks((0, 1, 3, 4, 6, 7), ("Logit", "Rank", "Logit", "Rank", "Logit", "Rank"))
    for x, label in ((.5, "DPO"), (3.5, "GRPO"), (6.5, "Distillation")):
        ax[1, 1].text(x, -.16, label, ha="center", transform=ax[1, 1].get_xaxis_transform(), fontsize=11)
    ax[1, 1].set_title("D  Tail expansion persists in rank space", loc="left", fontweight="bold", pad=17)
    ax[1, 1].set_ylabel("p99 effect / SFT-v3 p99 effect · log scale")
    ax[1, 1].grid(axis="y", color="#e4e9eb")
    ax[1, 1].text(0, -.32, "HF interpretation scores. Each column: 3 seeds × 4 contrasts.\nRank effect = within-model empirical percentile difference.",
                  transform=ax[1, 1].transAxes, fontsize=10)
    fig.text(.075, .025, "HF uses the interpretation PEFT path; vLLM serves sequential bf16 merges. Differences between paths are not pure backend effects.\nAll findings are descriptive on the development benchmark. Model scores are not hiring-quality ground truth.", fontsize=10, color="#526b77")
    fig.savefig(out / "research-followup.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "reports/research_followup/decision_geometry")
    p.add_argument("--out", type=Path, default=ROOT / "reports/research_followup/figures")
    a = p.parse_args()
    main(a.source, a.out)
