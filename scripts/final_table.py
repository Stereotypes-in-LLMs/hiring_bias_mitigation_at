#!/usr/bin/env python
"""The report's final table: every arm trained after SFT-v3, its quality on one footing and its
cost by kind, from files the standard already writes. CPU only, no new measurement.

Quality, three ways, all from the op-margin-v2 margin files:
  own point     the standard's decision: SFT-v3 moved to the arm's hire rate
                (reports/operating_point/v2/<run>--vs--<SFT-v3>.json, read as written);
  common point  the arm moved to SFT-v3's own hire rate (user, 2026-10-29), with the same
                matching and candidate bootstrap (`matched_comparison` with the roles swapped),
                so every arm is read at one operating point;
  curve         unstable within-group sets at each hire rate on a grid, each arm moved there
                by the same matching; curve_bands.csv averages (arm - SFT-v3) over hire-rate
                bands, descriptively, since grid points are not independent.
TPR where GPT-4o hires is given at the common point from the margins, beside the standard's
greedy guardrails at the arm's own point (reports/reference_metrics.csv). How far the margins
sit from the cut is given as the share within 1 nat of 0, the median |margin|, and the range of
the unstable share within 0.5 pp of the arm's own hire rate.

Cost, from reports/cost/ (scripts/cost_report.py), measured GPU-h, never summed across rows:
  data      the recipe steps of the jobs that built the arm's training data (`data_jobs` in
            reports/final/arms.csv; a job shared by several arms counts in full in each);
  train     GPU-h to the audited checkpoint (checkpoints.csv), plus the parent's when an arm
            starts from another arm;
  extra     data + train: what one more such model costs on top of SFT-v3;
  eval      the fixed evaluation, the same work for every arm, reported once;
  explore   every step of a method family (reports/final/families.csv), failed and diagnostic
            ones included, patching analysis left out.

    python scripts/final_table.py
    # -> reports/final/{final_table,recipes,curves,curve_bands,family_cost,eval_cost}.csv,
    #    curves.svg, cost_quality.svg
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import operating_point_control as opc  # noqa: E402

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

FINAL = REPO / "reports" / "final"
REFERENCE = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"
HIRE_GRID = np.round(np.arange(0.10, 0.3501, 0.005), 3)
BANDS = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]   # pp; the last one is closed
LOCAL = np.round(np.arange(-0.005, 0.00501, 0.001), 3)        # around the arm's own hire rate
# The series drawn in curves.svg, one per recipe, seed 42 where there are several (its other
# seeds are drawn thin, unlabelled); categorical slots of the reference palette (dataviz
# skill), SFT-v3 in neutral ink.
CURVE_SERIES = [
    ("SFT-v3", "#52514e"), ("DPO_V3_4880 s42", "#2a78d6"), ("GRPO-v3-all ckpt 525", "#eb6834"),
    ("GRPO+decay R3 s42", "#1baf7a"), ("DAPO+decay D1 s42", "#eda100"), ("Distill s42", "#6250d6"),
]
FAMILY_COLOR = {"dpo": "#2a78d6", "grpo": "#eb6834", "dapo": "#eb6834", "distill": "#1baf7a"}


def margins(run: str) -> pd.DataFrame:
    frame, _ = opc.load_scores(run, "adapter")
    return frame


def adapter_dir_and_step(run: str) -> tuple[str, int | None]:
    _, meta = opc.load_scores(run, "adapter")
    path = Path(meta["model"]["adapters"][-1]["path"])
    if path.name.startswith("checkpoint-"):
        return str(path.parent), int(path.name.split("-")[1])
    return str(path), None


def tpr_at_count(frame: pd.DataFrame, reference: pd.Series, k: int) -> float:
    """Expected hire rate on prompts GPT-4o hires when the arm hires exactly `k` prompts (the bin
    tied at the cut hired with weight w, as matched_flags does), attribute-free rows included as
    in reference_metrics.hire_by_reference."""
    m = opc.snap(frame["margin"])
    level, w = opc.cut_for_count(m, k)
    hire = np.where(m > level, 1.0, np.where(m == level, w, 0.0))
    on_hire = reference.to_numpy() == "hire"
    return float(100 * hire[on_hire].mean())


def reference_verdicts(run: str, frame: pd.DataFrame) -> pd.Series:
    raw = pd.read_parquet(resolve_output_path(f"outputs/raw/{run}.parquet"),
                          columns=[*opc.KEY, "reference_decision"])
    ref = frame[opc.KEY].merge(raw, on=opc.KEY, how="left")["reference_decision"]
    return ref.astype(str).str.strip().str.lower()


def curve(frame: pd.DataFrame, grid: np.ndarray = HIRE_GRID) -> list[float]:
    n = len(frame)
    return [100 * opc.matched_flags(frame, "margin", round(h * n))[1]["unstable"]
            for h in grid]


def spread(frame: pd.DataFrame, hire: float) -> dict:
    """How far an arm's margins sit from its cut, on the snapped margins the matching cuts."""
    m = pd.Series(np.abs(np.asarray(opc.snap(frame["margin"]))))
    local = curve(frame, hire + LOCAL)
    return {"near_cut_pct": 100 * float((m < 1).mean()), "median_abs_margin": float(m.median()),
            "unstable_local_min_pct": min(local), "unstable_local_max_pct": max(local)}


def curve_bands(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean of (arm - SFT-v3) over the grid points in each hire-rate band, and the share of all
    grid points where the arm is below SFT-v3."""
    curves = curves.assign(hire_rate_pct=curves["hire_rate_pct"].round(3))
    ref = curves[curves["run"] == REFERENCE].set_index("hire_rate_pct")["unstable_pct"]
    rows = []
    for label, part in curves[curves["run"] != REFERENCE].groupby("label", sort=False):
        diff = part.set_index("hire_rate_pct")["unstable_pct"] - ref
        row = {"label": label, "run": part["run"].iloc[0]}
        for lo, hi in BANDS:
            inside = (diff.index >= lo) & ((diff.index < hi) | (hi == BANDS[-1][1]))
            row[f"band_{lo}_{hi}_pp"] = float(diff[inside].mean())
        row["points_below_pct"] = 100 * float((diff < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def costs(arms: pd.DataFrame) -> pd.DataFrame:
    steps = pd.read_csv(REPO / "reports/cost/steps.csv", dtype={"job_id": str})
    points = pd.read_csv(REPO / "reports/cost/checkpoints.csv", dtype={"job_id": str})

    def train_to(run: str) -> float:
        out_dir, step = adapter_dir_and_step(run)
        rows = points[points["output_dir"] == out_dir]
        if rows.empty:
            return float("nan")
        row = rows[rows["checkpoint_step"] == (step or rows["checkpoint_step"].max())]
        return float(row["gpu_h_to_checkpoint"].iloc[0])

    rows = []
    for arm in arms.itertuples():
        jobs = str(arm.data_jobs).split() if isinstance(arm.data_jobs, str) else []
        data = steps[steps["job_id"].isin(jobs) & (steps["stage"] == "datagen")
                     & (steps["recipe"] == True)]  # noqa: E712 (a CSV bool column)
        train = train_to(arm.run)
        if isinstance(arm.parent, str):
            train += train_to(arm.parent)
        rows.append({"run": arm.run, "data_gpu_h": round(float(data["gpu_h"].sum()), 2),
                     "data_jobs_found": data["job_id"].nunique(),
                     "train_gpu_h": round(train, 2)})
    out = pd.DataFrame(rows)
    out["extra_gpu_h"] = (out["data_gpu_h"] + out["train_gpu_h"]).round(2)
    return out


def family_cost(families: pd.DataFrame) -> pd.DataFrame:
    steps = pd.read_csv(REPO / "reports/cost/steps.csv", dtype={"job_id": str})
    steps = steps[steps["stage"] != "patching"]
    arms = steps["arm"].fillna("").str.split("+")
    rows = []
    for fam in families.itertuples():
        mine = set(fam.cost_arms.split())
        part = steps[arms.map(lambda a, mine=mine: bool(mine & set(a)))]
        ok = part["status"].fillna("ok") == "ok"
        rows.append({"family": fam.family, "gpu_h": round(float(part["gpu_h"].sum()), 2),
                     "gpu_h_not_ok": round(float(part.loc[~ok, "gpu_h"].sum()), 2),
                     "by_stage": "; ".join(f"{s} {g:.2f}" for s, g in
                                           part.groupby("stage")["gpu_h"].sum().items()),
                     "jobs": part["job_id"].nunique(), "steps": len(part),
                     "cost_arms": fam.cost_arms, "note": fam.note})
    return pd.DataFrame(rows)


def eval_cost() -> pd.DataFrame:
    steps = pd.read_csv(REPO / "reports/cost/steps.csv", dtype={"job_id": str})
    jobs = pd.read_csv(REPO / "reports/cost/jobs.csv", dtype={"job_id": str})
    ev = steps[(steps["job_name"] == "xw-eval-arm") & (steps["source"] == "record")]
    per = ev.groupby("job_id")["gpu_h"].sum().rename("gpu_h_measured").reset_index()
    per = per.merge(jobs[["job_id", "gpu_h_alloc"]], on="job_id", how="left")
    return pd.DataFrame([{
        "evaluate_arm_jobs": len(per),
        "gpu_h_measured_median": round(float(per["gpu_h_measured"].median()), 3),
        "gpu_h_alloc_median": round(float(per["gpu_h_alloc"].median()), 3),
        "gpu_h_alloc_min": round(float(per["gpu_h_alloc"].min()), 3),
        "gpu_h_alloc_max": round(float(per["gpu_h_alloc"].max()), 3),
    }])


def build(arms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(REPO / "reports/reference_metrics.csv").set_index("run")
    ref = margins(REFERENCE)
    ref_hire, ref_unstable = opc.unstable_share(ref, "margin", 0.0)
    ref_tpr = tpr_at_count(ref, reference_verdicts(REFERENCE, ref),
                           int((opc.snap(ref["margin"]) > 0).sum()))
    rows, curves = [], []
    for arm in arms.itertuples():
        try:
            frame = margins(arm.run)
        except SystemExit as exc:
            print(f"skip {arm.label}: {exc}")
            continue
        m = metrics.loc[arm.run] if arm.run in metrics.index else pd.Series(dtype=object)
        row = {"label": arm.label, "run": arm.run, "family": arm.family, "recipe": arm.recipe,
               "seed": arm.seed, "checkpoint_rule": arm.checkpoint_rule,
               "headline": arm.headline}
        if arm.run == REFERENCE:
            row.update({"hire_rate_own_pct": 100 * ref_hire, "unstable_own_pct": 100 * ref_unstable,
                        "delta_common_pp": 0.0, "tpr_common_pct": ref_tpr,
                        "tpr_common_vs_ref_pp": 0.0, **spread(frame, ref_hire)})
        else:
            own = opc.matched_comparison(ref, frame)            # SFT-v3 moved to the arm
            common = opc.matched_comparison(frame, ref)         # the arm moved to SFT-v3
            tpr = tpr_at_count(frame, reference_verdicts(arm.run, frame),
                               round(common["hire_rate_run"] * len(frame)))
            row.update({
                "hire_rate_own_pct": 100 * own["hire_rate_run"],
                "unstable_own_pct": 100 * own["unstable_run"],
                "delta_own_pp": own["delta_matched_pp"], "ci_own_low": own["ci_low"],
                "ci_own_high": own["ci_high"],
                "hire_rate_common_pct": 100 * common["hire_rate_ref_matched"],
                "unstable_common_pct": 100 * common["unstable_ref_matched"],
                "delta_common_pp": -common["delta_matched_pp"],
                "ci_common_low": -common["ci_high"], "ci_common_high": -common["ci_low"],
                "tpr_common_pct": tpr, "tpr_common_vs_ref_pp": tpr - ref_tpr,
                "std_delta_pp": m.get("primary_delta_pp"),
                "std_tpr_vs_ref_pp": m.get("std_tpr_vs_ref_pp"),
                "std_agree_vs_ref_pp": m.get("std_agree_vs_ref_pp"),
                "merge_kept_pct": m.get("merge_kept_pct"),
                "guardrails": "/".join(str(m.get(g, "")) for g in
                                       ("guard_merge", "guard_tpr", "guard_agree",
                                        "guard_decided")),
                "std_verdict": m.get("std_verdict"),
                **spread(frame, own["hire_rate_run"]),
            })
        rows.append(row)
        curves += [{"label": arm.label, "run": arm.run, "hire_rate_pct": 100 * h,
                    "unstable_pct": u} for h, u in zip(HIRE_GRID, curve(frame))]
    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("hire_rate_own_pct"), "common_hire_rate_pct",
                 round(100 * ref_hire, 3))
    return table, pd.DataFrame(curves)


# ---- figures ----------------------------------------------------------------------------------

def _svg(w: int, h: int, body: list[str]) -> str:
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'font-family="sans-serif" font-size="11" fill="#0b0b0b">',
                      f'<rect width="{w}" height="{h}" fill="#fcfcfb"/>', *body, "</svg>"]) + "\n"


def other_seeds(curves: pd.DataFrame, label: str) -> pd.DataFrame:
    """The curves of a charted series' other seeds of the same recipe."""
    recipe = label.removesuffix(" s42")
    return curves[curves["label"].str.fullmatch(rf"{re.escape(recipe)} s\d+")
                  & (curves["label"] != label)]


def curves_svg(curves: pd.DataFrame, common: float) -> str:
    w, h, left, right, top, bottom = 760, 440, 56, 170, 52, 44
    data = curves[curves["label"].isin([s for s, _ in CURVE_SERIES])]
    thin = [(other_seeds(curves, label), color) for label, color in CURVE_SERIES[1:]]
    y_hi = float(np.ceil(max([data["unstable_pct"].max()]
                             + [o["unstable_pct"].max() for o, _ in thin if not o.empty])))

    def X(v):
        span = 100 * (HIRE_GRID[-1] - HIRE_GRID[0])
        return left + (v - 100 * HIRE_GRID[0]) / span * (w - left - right)

    def Y(v):
        return top + (1 - v / y_hi) * (h - top - bottom)

    out = [f'<text x="{left}" y="18" font-weight="bold">Unstable within-group sets against hire '
           'rate, each arm moved along its own margins (op-margin-v2)</text>',
           f'<text x="{left}" y="34" fill="#52514e">labelled: one seed per recipe (s42); thin: '
           'the same recipe\'s other training seeds</text>']
    for t in range(0, int(y_hi) + 1):
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                   'stroke="#e4e3df"/>')
        out.append(f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="#52514e">'
                   f'{t}%</text>')
    for t in range(10, 36, 5):
        out.append(f'<text x="{X(t):.1f}" y="{h - bottom + 16}" text-anchor="middle" '
                   f'fill="#52514e">{t}%</text>')
    out.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - 8}" text-anchor="middle" '
               'fill="#52514e">hire rate (margin pass)</text>')
    out.append(f'<line x1="{X(common):.1f}" x2="{X(common):.1f}" y1="{top}" y2="{h - bottom}" '
               'stroke="#8a8984" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{X(common) + 4:.1f}" y="{top + 10}" fill="#52514e">SFT-v3 '
               f'{common:.2f}%</text>')
    for others, color in thin:
        for _, part in others.groupby("label"):
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in
                           zip(part["hire_rate_pct"], part["unstable_pct"]))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                       'stroke-width="1" stroke-opacity="0.45"/>')
    ends = []
    for label, color in CURVE_SERIES:
        part = data[data["label"] == label]
        if part.empty:
            continue
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in
                       zip(part["hire_rate_pct"], part["unstable_pct"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        ends.append([Y(part["unstable_pct"].iloc[-1]), label, color])
    ends.sort()
    for i in range(1, len(ends)):             # keep the direct labels 13 px apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, label, color in ends:
        out.append(f'<rect x="{w - right + 8}" y="{y - 4:.1f}" width="10" height="3" '
                   f'fill="{color}"/>')
        out.append(f'<text x="{w - right + 22}" y="{y + 3:.1f}">{html.escape(label)}</text>')
    return _svg(w, h, out)


def recipes(table: pd.DataFrame) -> pd.DataFrame:
    """One row per headline recipe: seeds averaged (sd across seeds, ddof 1), guardrails passed
    by every seed."""
    rows = table[(table["run"] != REFERENCE) & (table["headline"] == "yes")].copy()
    rows["name"] = rows["label"].str.replace(r" s\d+$", "", regex=True)
    rows["passes"] = rows["guardrails"].astype(str).str.split("/").map(
        lambda g: g.count("pass") == 4)
    return rows.groupby(["name", "family"], sort=False).agg(
        seeds=("seed", "size"), extra_gpu_h=("extra_gpu_h", "mean"),
        delta_common_pp=("delta_common_pp", "mean"), delta_own_pp=("delta_own_pp", "mean"),
        delta_common_sd=("delta_common_pp", "std"), delta_own_sd=("delta_own_pp", "std"),
        delta_own_max=("delta_own_pp", "max"),
        tpr_common_vs_ref_pp=("tpr_common_vs_ref_pp", "mean"),
        passes=("passes", "all")).reset_index()


def cost_quality_svg(table: pd.DataFrame) -> str:
    rows = table[(table["run"] != REFERENCE) & (table["headline"] == "yes")].copy()
    groups = recipes(table)
    w, h, left, right, top, bottom = 840, 470, 60, 250, 62, 64
    x_hi = float(np.ceil(rows["extra_gpu_h"].max() / 5) * 5)
    y_lo = float(np.floor(min(rows["ci_common_low"].min(), -2)))
    y_hi = float(np.ceil(max(rows["ci_common_high"].max(), 1)))

    def X(v):
        return left + v / x_hi * (w - left - right)

    def Y(v):
        return top + (y_hi - v) / (y_hi - y_lo) * (h - top - bottom)

    out = [f'<text x="{left}" y="18" font-weight="bold">Change in unstable sets at SFT-v3\'s hire '
           'rate (pp) against extra GPU-h per model over SFT-v3</text>',
           f'<text x="{left}" y="34" fill="#52514e">large: mean over seeds, filled when every '
           'seed passes every guardrail at its own point; small: each seed with its 95% CI</text>',
           f'<text x="{left}" y="48" fill="#52514e">label: seeds, and the TPR change where GPT-4o '
           'hires at the common point (pp)</text>']
    for t in range(int(y_lo), int(y_hi) + 1):
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                   f'stroke="{"#8a8984" if t == 0 else "#e4e3df"}"/>')
        out.append(f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="#52514e">'
                   f'{t:+d}</text>'.replace(">+0<", ">0<"))
    for t in range(0, int(x_hi) + 1, 5):
        out.append(f'<text x="{X(t):.1f}" y="{h - bottom + 16}" text-anchor="middle" '
                   f'fill="#52514e">{t}</text>')
    out.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - bottom + 34}" '
               'text-anchor="middle" fill="#52514e">extra GPU-h per model: its training data + '
               'training to the audited checkpoint (measured)</text>')
    legend = [("DPO", "#2a78d6"), ("GRPO / DAPO", "#eb6834"), ("decision distillation", "#1baf7a")]
    for i, (name, color) in enumerate(legend):
        x = left + 10 + 170 * i
        out.append(f'<circle cx="{x}" cy="{h - 12}" r="4" fill="{color}"/>')
        out.append(f'<text x="{x + 8}" y="{h - 8}">{name}</text>')
    for r in rows.itertuples():
        color = FAMILY_COLOR[r.family]
        x = X(r.extra_gpu_h)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(r.ci_common_low):.1f}" '
                   f'y2="{Y(r.ci_common_high):.1f}" stroke="{color}" stroke-width="1.5" '
                   'stroke-opacity="0.35"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{Y(r.delta_common_pp):.1f}" r="2.5" '
                   f'fill="{color}" fill-opacity="0.6"/>')
    labels = []
    for g in groups.itertuples():
        color = FAMILY_COLOR[g.family]
        x, y = X(g.extra_gpu_h), Y(g.delta_common_pp)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" stroke-width="2" '
                   + (f'fill="{color}" stroke="#fcfcfb"/>' if g.passes else
                      f'fill="#fcfcfb" stroke="{color}"/>'))
        labels.append([y, x, g])
    labels.sort(key=lambda item: item[0])
    for i in range(1, len(labels)):           # the label column keeps 15 px between rows
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 15)
    for y_label, x, g in labels:
        y = Y(g.delta_common_pp)
        out.append(f'<polyline points="{x + 7:.1f},{y:.1f} {w - right + 4},{y_label:.1f}" '
                   'fill="none" stroke="#c3c2b7" stroke-width="1"/>')
        out.append(f'<text x="{w - right + 8}" y="{y_label + 4:.1f}">{html.escape(g.name)} '
                   f'<tspan fill="#52514e">· {g.seeds} seed{"s" if g.seeds > 1 else ""} · '
                   f'TPR {g.tpr_common_vs_ref_pp:+.1f}</tspan></text>')
    return _svg(w, h, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=FINAL)
    args = ap.parse_args()
    arms = pd.read_csv(FINAL / "arms.csv")
    table, curves = build(arms)
    table = table.merge(costs(arms), on="run", how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out / "final_table.csv", index=False)
    curves.round(4).to_csv(args.out / "curves.csv", index=False)
    curve_bands(curves).round(3).to_csv(args.out / "curve_bands.csv", index=False)
    family_cost(pd.read_csv(FINAL / "families.csv")).to_csv(args.out / "family_cost.csv",
                                                            index=False)
    eval_cost().to_csv(args.out / "eval_cost.csv", index=False)
    recipes(table).round(3).to_csv(args.out / "recipes.csv", index=False)
    common = float(table["common_hire_rate_pct"].iloc[0])
    (args.out / "curves.svg").write_text(curves_svg(curves, common))
    (args.out / "cost_quality.svg").write_text(cost_quality_svg(table))
    cols = ["label", "hire_rate_own_pct", "delta_own_pp", "ci_own_low", "ci_own_high",
            "delta_common_pp", "ci_common_low", "ci_common_high", "tpr_common_vs_ref_pp",
            "guardrails", "data_gpu_h", "train_gpu_h", "extra_gpu_h"]
    print(table[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
