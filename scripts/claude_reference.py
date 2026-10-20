# ruff: noqa: E501  (command lines in the docstring)
"""Every Qwen3.5-9B English arm scored against a second reference: Claude's blind review.

Why. Utility on the benchmark is read as agreement with one GPT-4o verdict per pair, and the
training anchors are a third verdict again (the v1 teacher's single attribute-free sample at
temperature 0.7, 85.8% of pool pairs reject, against GPT-4o's 66.7% on the benchmark). When an
arm "loses agreement", that is model error only if the reference is right. Claude graded 153
benchmark pairs from the attribute-free CV alone, blind to every model's and GPT-4o's decision
(`reports/claude_blind_review/`), so the same arms can be read against an independent
reviewer, and the pairs it calls borderline can be told apart from the ones it calls clear.

The 153 pairs are a stratified sample of the 450, not a random one, and every population
number here is weighted accordingly:

  disagree      every pair on which any of SFT, r1-300, r2-102, r2em-100 disagrees with
                GPT-4o on the attr_free row (`DEFINING`); fully enumerated (n = N = 113).
  agree-hire    the other pairs, where all five agree, and GPT-4o hires; 20 graded.
  agree-reject  the same, GPT-4o rejects; 20 graded.

  estimate = sum_s N_s * mean_s / 450, N_s counted from the raw parquets, mean_s over the
  graded pairs of stratum s. The interval resamples graded pairs within each stratum (1,000
  draws, seed 42), the same draws for every arm, so differences between arms are paired. The
  disagree stratum is a census of the benchmark, so resampling it treats its 113 pairs as
  draws from a population of such pairs; the interval is wider than a finite-population one.

The README beside the grades extrapolated instead with one pooled rate for the 337 agree
pairs (its 40 controls, 82.5%), which weights agree-hire and agree-reject 20:20 where the
benchmark holds them 67:270. Both methods are reported side by side (`README_EXTRAPOLATED`).
For the four DEFINING arms and GPT-4o, the decision on an agree pair is GPT-4o's by
construction; for every other arm, its agreement with GPT-4o is also estimated from the
graded pairs, and the estimate is printed next to the exact value as a check of the sample.

Outputs, under reports/claude_reference/:

  agreement.csv          per arm: exact attr_free hire rate and agreement with GPT-4o, the
                         stratified estimates of both, the stratified agreement with Claude
                         (CI, delta vs SFT with paired CI), the README method, per-stratum counts
  flips.csv              attr_free decisions changed against SFT, per arm and direction, and
                         what Claude (decision, fit bucket) and GPT-4o say on the graded ones;
                         ungraded changed pairs are counted, not dropped
  stability_by_fit.csv   unstable counterfactual sets (greedy audit decisions) on the graded
                         pairs by Claude fit bucket, by Claude-vs-GPT-4o agreement and by
                         stratum (raw, and weighted by N_s / n_s), the same within the
                         disagree census, plus where each arm's unstable sets lie over all
                         2,700
  summary.json           everything above, the consistency checks, the reference-noise
                         tables (Claude vs GPT-4o per stratum and fit bucket), the training
                         anchors' reject share (a different population) and the code state

A set here is (pair_id, protected_group, condition), as in scripts/operating_point_control.py
(2,700 sets on the 34 scoped variants), not the (pair_id, condition) of analysis/stability.py.
Instability is read from each arm's greedy decisions at its own operating point; the
equal-hire-rate comparison is operating_point_control.py's, not this script's.

    HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw python scripts/claude_reference.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

#: Display label -> audit run name, in reading order. Missing runs are skipped and listed.
ARMS = {
    "base": "Qwen3.5-9B--en--baseline--cscs",
    "SFT": "Qwen3.5-9B--en--sft--adapter--en_only_cscs",
    "r1-100": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r1_step100",
    "r1-300": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r1_step300",
    "r2-50": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2_step50",
    "r2-102": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2_step102",
    "r2em-50": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step50",
    "r2em-100": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100",
    "r2em-148": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step148",
    "GRPO": "Qwen3.5-9B--en--grpo--adapter--en_only_grpo_pilot",
    "RFT": "Qwen3.5-9B--en--sft--adapter--en_only_rft",
}
#: The arms whose disagreement with GPT-4o defined the 'disagree' stratum (README step 1).
DEFINING = ("SFT", "r1-300", "r2-102", "r2em-100")
#: Column holding each DEFINING arm's attr_free decision in the grades CSV.
GRADE_COLUMNS = {arm: f"dec_{arm}" for arm in DEFINING}
REFERENCE = "gpt4o"
CLAUDE = "claude"
BASELINE_ARM = "SFT"
STRATA = ("disagree", "agree-hire", "agree-reject")
BUCKETS = ("1-2", "3", "4-5")
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
SET = ["pair_id", "protected_group", "condition"]
N_BOOT = 1000
BOOT_SEED = 42
#: The README's "与我一致（外推）" column (reports/claude_blind_review/README.md), in %, which
#: the pooled method must reproduce to its one decimal.
README_EXTRAPOLATED = {"gpt4o": 78.2, "SFT": 77.6, "r2em-100": 76.9, "r2-102": 71.6,
                       "r1-300": 70.4}
DEFAULT_GRADES = "reports/claude_blind_review/claude_grades_153.csv"
DEFAULT_ANCHORS = "artifacts/semisynthetic-v2/onpolicy-full/rows.parquet"


# ---- inputs -------------------------------------------------------------------------------

def _single_threaded() -> None:
    """The login node has few cores per user; pyarrow would otherwise take all of them."""
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def read_audit(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, use_threads=False,
                           columns=[*KEY, "decision", "reference_decision"])


def free_decisions(audits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per pair: every arm's attr_free decision and GPT-4o's reference.

    Refuses audits whose references differ, or that hold other than one attr_free row a pair.
    """
    columns, reference = {}, None
    for arm, audit in audits.items():
        free = audit[audit["condition"] == "attr_free"]
        if not free["pair_id"].is_unique:
            raise SystemExit(f"{arm}: more than one attr_free row for a pair")
        free = free.set_index("pair_id")
        columns[arm] = free["decision"]
        if reference is None:
            reference = free["reference_decision"]
        elif not free["reference_decision"].reindex(reference.index).equals(reference):
            raise SystemExit(f"{arm}: reference_decision differs from the first audit's")
    frame = pd.DataFrame(columns)
    frame[REFERENCE] = reference
    return frame


def assign_strata(free: pd.DataFrame, defining=DEFINING) -> pd.Series:
    """'disagree' where any defining arm departs from GPT-4o, else agree-<GPT-4o decision>."""
    disagree = free[list(defining)].ne(free[REFERENCE], axis=0).any(axis=1)
    agree = "agree-" + free[REFERENCE].astype(str)
    return agree.where(~disagree, "disagree").rename("stratum")


def fit_bucket(score) -> str:
    """Claude's 1-5 fit score as the README reads it: 1-2 clear reject, 3 borderline, 4-5 hire."""
    score = int(score)
    return "1-2" if score <= 2 else "3" if score == 3 else "4-5"


def check_grades(grades: pd.DataFrame, free: pd.DataFrame, strata: pd.Series) -> dict:
    """The grades CSV against the raw parquets. Raises when the sample design does not hold."""
    if not grades["pair_id"].is_unique:
        raise SystemExit("grades: a pair is graded twice")
    missing = sorted(set(grades["pair_id"]) - set(free.index))
    if missing:
        raise SystemExit(f"grades: {len(missing)} pairs not in the audits, e.g. {missing[:3]}")
    g = grades.set_index("pair_id")
    graded_disagree = set(g.index[g["set"] == "disagree"])
    population_disagree = set(strata.index[strata == "disagree"])
    if graded_disagree != population_disagree:
        raise SystemExit(
            f"the 'disagree' set in the grades ({len(graded_disagree)}) is not the raw parquets' "
            f"({len(population_disagree)}): {len(graded_disagree ^ population_disagree)} differ")
    controls = g.index[g["set"] == "agree"]
    if (strata.reindex(controls) == "disagree").any():
        raise SystemExit("grades: an 'agree' control is a disagree pair in the raw parquets")
    checks = {
        "disagree_N_raw": len(population_disagree),
        "disagree_set_matches_grades": True,
        "agree_controls_in_agree_strata": True,
        "gpt4o_matches_raw": bool((g["decision_gpt4o"] == free[REFERENCE].reindex(g.index)).all()),
    }
    for arm, column in GRADE_COLUMNS.items():
        checks[f"{column}_matches_raw"] = bool((g[column] == free[arm].reindex(g.index)).all())
    buckets = g["fit_score_claude"].map(fit_bucket)
    inconsistent = (((buckets == "1-2") & (g["decision_claude"] != "reject"))
                    | ((buckets == "4-5") & (g["decision_claude"] != "hire")))
    checks["claude_decision_contradicts_fit_bucket"] = int(inconsistent.sum())
    checks["claude_decisions"] = sorted(g["decision_claude"].unique().tolist())
    bad = [k for k, v in checks.items() if v is False]
    if bad:
        raise SystemExit(f"grades do not match the raw parquets: {bad}")
    return checks


def graded_frame(grades: pd.DataFrame, free: pd.DataFrame, strata: pd.Series) -> pd.DataFrame:
    """The graded pairs with every arm's attr_free decision, GPT-4o's, Claude's and the stratum."""
    g = grades.set_index("pair_id")
    out = free.reindex(g.index).copy()
    out[CLAUDE] = g["decision_claude"]
    out["fit_score"] = g["fit_score_claude"].astype(int)
    out["fit"] = out["fit_score"].map(fit_bucket)
    out["stratum"] = strata.reindex(g.index)
    out["claude_vs_gpt4o"] = np.where(out[CLAUDE] == out[REFERENCE], "agree", "disagree")
    return out


# ---- 1. agreement with Claude, stratified -------------------------------------------------

def stratum_sizes(strata: pd.Series, graded: pd.DataFrame) -> pd.DataFrame:
    sizes = pd.DataFrame({"N": strata.value_counts(), "n": graded["stratum"].value_counts()})
    return sizes.reindex(list(STRATA)).fillna(0).astype(int)


def stratified_estimate(values: pd.DataFrame, stratum: pd.Series, sizes: pd.DataFrame) -> pd.Series:
    """sum_s N_s * mean_s / sum_s N_s for every column of `values` (graded pairs only)."""
    means = values.astype(float).groupby(stratum).mean().reindex(sizes.index)
    weights = sizes["N"] / sizes["N"].sum()
    return means.mul(weights, axis=0).sum(axis=0)


def stratified_bootstrap(values: pd.DataFrame, stratum: pd.Series, sizes: pd.DataFrame,
                         n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, columns) stratified estimates, graded pairs resampled within their stratum.

    The same draws are applied to every column, so a difference of two columns is paired.
    """
    rng = np.random.default_rng(seed)
    total = sizes["N"].sum()
    out = np.zeros((n_boot, values.shape[1]))
    for s in sizes.index:
        block = values[stratum == s].to_numpy(dtype=float)
        if len(block) == 0:
            if sizes.loc[s, "N"]:
                raise SystemExit(f"stratum {s} has N = {sizes.loc[s, 'N']} and no graded pair")
            continue
        idx = rng.integers(0, len(block), size=(n_boot, len(block)))
        out += sizes.loc[s, "N"] / total * block[idx].mean(axis=1)
    return out


def pooled_estimate(values: pd.DataFrame, stratum: pd.Series, sizes: pd.DataFrame) -> pd.Series:
    """The README's method: the disagree stratum as measured, every agree pair at the pooled
    rate of the agree controls (agree-hire and agree-reject weighted by their n, not their N)."""
    disagree = stratum == "disagree"
    n_disagree = sizes.loc["disagree", "N"]
    total = sizes["N"].sum()
    v = values.astype(float)
    return (n_disagree * v[disagree].mean() + (total - n_disagree) * v[~disagree].mean()) / total


def _ci(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def agreement_table(free: pd.DataFrame, graded: pd.DataFrame, sizes: pd.DataFrame,
                    arms: list[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """One row per arm (GPT-4o and Claude included): exact and stratified numbers side by side."""
    targets = [REFERENCE, *arms]
    stratum = graded["stratum"]
    agree_claude = pd.DataFrame({a: graded[a] == graded[CLAUDE] for a in targets})
    agree_ref = pd.DataFrame({a: graded[a] == graded[REFERENCE] for a in targets})
    hires = pd.DataFrame({a: graded[a] == "hire" for a in [*targets, CLAUDE]})
    # one set of draws for every quantity, so all intervals and deltas share them
    values = pd.concat([agree_claude.add_prefix("c:"), hires.add_prefix("h:")], axis=1)
    point = stratified_estimate(values, stratum, sizes)
    draws = stratified_bootstrap(values, stratum, sizes, n_boot, seed)
    low, high = _ci(draws)
    col = {c: i for i, c in enumerate(values.columns)}
    pooled = pooled_estimate(agree_claude, stratum, sizes)
    ref_est = stratified_estimate(agree_ref, stratum, sizes)
    base = f"c:{BASELINE_ARM}"
    rows = []
    for a in [*targets, CLAUDE]:
        h = f"h:{a}"
        row = {"arm": a, "run": ARMS.get(a, ""),
               "hire_rate_attr_free": float((free[a] == "hire").mean()) if a in free else None,
               "hire_rate_stratified": point[h],
               "hire_ci_low": low[col[h]], "hire_ci_high": high[col[h]]}
        if a != CLAUDE:
            c = f"c:{a}"
            delta = 100 * (draws[:, col[c]] - draws[:, col[base]]) if base in col else None
            row.update({
                "agree_gpt4o_exact": float((free[a] == free[REFERENCE]).mean()),
                "agree_gpt4o_stratified": ref_est[a],
                "agree_claude_stratified": point[c],
                "agree_claude_ci_low": low[col[c]], "agree_claude_ci_high": high[col[c]],
                "delta_vs_sft_pp": (100 * (point[c] - point[base])) if base in col else None,
                "delta_ci_low": float(np.percentile(delta, 2.5)) if delta is not None else None,
                "delta_ci_high": float(np.percentile(delta, 97.5)) if delta is not None else None,
                "agree_claude_readme_method": pooled[a],
            })
            for s in STRATA:
                part = agree_claude.loc[stratum == s, a]
                row[f"agree_claude_{s}"] = f"{int(part.sum())}/{len(part)}"
            # how much of this arm's disagreement with GPT-4o the graded pairs can see
            off = free[a] != free[REFERENCE]
            row["disagree_gpt4o_pairs"] = int(off.sum())
            row["disagree_gpt4o_graded"] = int(off.reindex(graded.index).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def defining_exactness(table: pd.DataFrame) -> float:
    """For GPT-4o and the DEFINING arms the stratified hire rate and GPT-4o agreement are
    exact by construction (their decision on an agree pair is GPT-4o's); the largest gap."""
    rows = table[table["arm"].isin([REFERENCE, *DEFINING])]
    gaps = [(rows["hire_rate_stratified"] - rows["hire_rate_attr_free"]).abs().max(),
            (rows["agree_gpt4o_stratified"] - rows["agree_gpt4o_exact"]).abs().max()]
    return float(max(gaps))


def readme_check(table: pd.DataFrame) -> dict:
    """The pooled method against the README's published extrapolation, per arm."""
    pooled = table.set_index("arm")["agree_claude_readme_method"]
    out = {}
    for arm, published in README_EXTRAPOLATED.items():
        value = round(100 * float(pooled[arm]), 1) if arm in pooled.index else None
        out[arm] = {"published_pct": published, "reproduced_pct": value,
                    "matches": value is not None and abs(value - published) < 0.05}
    return out


# ---- 2. decisions changed against SFT -----------------------------------------------------

def flip_table(free: pd.DataFrame, graded: pd.DataFrame, strata: pd.Series,
               arms: list[str], against: str = BASELINE_ARM) -> pd.DataFrame:
    """attr_free decisions an arm changed against `against`, and what the references say.

    `claude_with_arm` counts the graded changed pairs on which Claude takes the arm's new
    decision. Every disagree pair is graded, so an ungraded changed pair lies in an agree
    stratum, where SFT's decision is GPT-4o's; for a DEFINING arm none can (it would make the
    pair a disagree pair). Ungraded pairs are counted per stratum.
    """
    rows = []
    for arm in arms:
        if arm == against:
            continue
        for start, toward in (("hire", "reject"), ("reject", "hire")):
            changed = free.index[(free[against] == start) & (free[arm] == toward)]
            seen = graded.reindex(graded.index.intersection(changed))
            unseen = changed.difference(graded.index)
            row = {"arm": arm, "direction": f"{against} {start} -> {toward}",
                   "changed": len(changed), "graded": len(seen), "ungraded": len(unseen)}
            for s in STRATA:
                row[f"graded_{s}"] = int((seen["stratum"] == s).sum())
            for s in STRATA:
                row[f"ungraded_{s}"] = int((strata.reindex(unseen) == s).sum())
            row["claude_with_arm"] = int((seen[CLAUDE] == toward).sum())
            row["claude_with_sft"] = int((seen[CLAUDE] == start).sum())
            row["claude_with_arm_share"] = (row["claude_with_arm"] / len(seen)) if len(seen) else None
            for b in BUCKETS:
                row[f"fit_{b}"] = int((seen["fit"] == b).sum())
            row["gpt4o_with_arm"] = int((seen[REFERENCE] == toward).sum())
            row["gpt4o_with_sft"] = int((seen[REFERENCE] == start).sum())
            row["both_with_arm"] = int(((seen[CLAUDE] == toward) & (seen[REFERENCE] == toward)).sum())
            row["both_with_sft"] = int(((seen[CLAUDE] == start) & (seen[REFERENCE] == start)).sum())
            row["gpt4o_with_arm_all"] = int((free.loc[changed, REFERENCE] == toward).sum())
            rows.append(row)
    return pd.DataFrame(rows)


# ---- 3. instability against Claude's view of the CV ---------------------------------------

def common_rows(audits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every arm restricted to the attributed prompts all arms hold (matched rows)."""
    keys = None
    for audit in audits.values():
        k = pd.MultiIndex.from_frame(audit.loc[audit["condition"] != "attr_free", KEY])
        keys = k if keys is None else keys.intersection(k)
    out, dropped = {}, {}
    for arm, audit in audits.items():
        attributed = audit[audit["condition"] != "attr_free"].set_index(KEY)
        if not attributed.index.is_unique:
            raise SystemExit(f"{arm}: a prompt key {KEY} appears twice")
        dropped[arm] = int(len(attributed) - len(keys))
        out[arm] = attributed.loc[keys].reset_index()
    return out, dropped


def set_flags(audit: pd.DataFrame) -> pd.DataFrame:
    """Per (pair_id, protected_group, condition): variants, decided variants, and instability.

    A set counts only if every variant was decided (analysis/stability.py: dropping unparsed
    rows makes a set of failures look stable).
    """
    decided = audit.assign(d=audit["decision"].where(audit["decision"].isin(["hire", "reject"])))
    grouped = decided.groupby(SET)["d"]
    return pd.DataFrame({"variants": grouped.size(), "decided": grouped.count(),
                         "unstable": grouped.nunique() > 1})


def complete_flags(flags: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """One boolean column per arm over the sets every arm decided completely."""
    complete = None
    for f in flags.values():
        ok = f.index[f["decided"] == f["variants"]]
        complete = ok if complete is None else complete.intersection(ok)
    total = len(next(iter(flags.values())))
    frame = pd.DataFrame({arm: f.loc[complete, "unstable"] for arm, f in flags.items()})
    return frame, total - len(complete)


def _pair_level(index: pd.MultiIndex, column: pd.Series) -> np.ndarray:
    return column.reindex(index.get_level_values("pair_id")).to_numpy()


def stability_table(unstable: pd.DataFrame, graded: pd.DataFrame, strata: pd.Series,
                    sizes: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Unstable sets per arm, on the graded pairs by Claude's view, and over all sets.

    `share_of_arm_unstable` is the share of the arm's unstable sets (over every complete set)
    that falls in the row, so the 'population' rows show how much of the instability the
    graded pairs cover. `unstable_share_weighted` weights each graded pair by N_s / n_s of its
    stratum: the instability the slice would show over the 450 pairs, if the 20 controls stand
    for their stratum.
    """
    idx = unstable.index
    pair = idx.get_level_values("pair_id")
    on_graded = pair.isin(graded.index)
    stratum = _pair_level(idx, strata)
    fit = _pair_level(idx, graded["fit"])
    vs = _pair_level(idx, graded["claude_vs_gpt4o"])
    weight_of = (sizes["N"] / sizes["n"].where(sizes["n"] > 0)).to_dict()
    weight = np.where(on_graded, pd.Series(stratum).map(weight_of).to_numpy(), 0.0)
    slices = [("population", "all sets", np.ones(len(idx), bool)),
              ("population", "graded pairs", on_graded),
              ("population", "ungraded pairs", ~on_graded)]
    slices += [("population", f"{s} (all)", stratum == s) for s in STRATA]
    slices += [("population", f"{s} (ungraded)", (stratum == s) & ~on_graded)
               for s in STRATA if s != "disagree"]
    slices += [("graded: all", "all", on_graded)]
    slices += [("graded: stratum", s, on_graded & (stratum == s)) for s in STRATA]
    slices += [("graded: fit", b, on_graded & (fit == b)) for b in BUCKETS]
    slices += [("graded: claude_vs_gpt4o", v, on_graded & (vs == v)) for v in ("agree", "disagree")]
    slices += [("graded: fit x claude_vs_gpt4o", f"{b} / {v}", on_graded & (fit == b) & (vs == v))
               for b in BUCKETS for v in ("agree", "disagree")]
    # the disagree stratum is a census, so these slices need no weighting and carry no
    # sample-design confound (the agree controls are both stable and mostly Claude = GPT-4o)
    census = stratum == "disagree"
    slices += [("disagree stratum: fit", b, census & (fit == b)) for b in BUCKETS]
    slices += [("disagree stratum: claude_vs_gpt4o", v, census & (vs == v))
               for v in ("agree", "disagree")]
    totals = unstable.sum()
    rows = []
    for breakdown, level, mask in slices:
        mask = np.asarray(mask, bool)
        sets = int(mask.sum())
        w = weight[mask]
        for arm in arms:
            flags = unstable[arm].to_numpy()[mask]
            base = unstable[BASELINE_ARM].to_numpy()[mask]
            share = float(flags.mean()) if sets else None
            row = {"arm": arm, "breakdown": breakdown, "level": level,
                   "pairs": int(pd.Index(pair[mask]).nunique()), "sets": sets,
                   "unstable_sets": int(flags.sum()), "unstable_share": share,
                   "share_of_arm_unstable": (float(flags.sum() / totals[arm])
                                             if totals[arm] else None),
                   "delta_vs_sft_pp": (100 * (share - float(base.mean()))) if sets else None}
            if breakdown.startswith("graded"):
                row["unstable_share_weighted"] = (float((w * flags).sum() / w.sum())
                                                  if w.sum() else None)
                row["pairs_weighted"] = float(pd.Series(w).groupby(pair[mask]).first().sum()
                                              if sets else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


# ---- 4. reference noise on the graded pairs -----------------------------------------------

def cohen_kappa(a: pd.Series, b: pd.Series) -> float | None:
    """Two-rater kappa for hire/reject. None when either rater is constant on the slice (an
    agree stratum is constant for GPT-4o by construction): kappa is then 0 whatever the raters
    do, which reads as chance agreement and is not."""
    pa, pb = float((a == "hire").mean()), float((b == "hire").mean())
    if pa in (0.0, 1.0) or pb in (0.0, 1.0):
        return None
    observed = float((a == b).mean())
    chance = pa * pb + (1 - pa) * (1 - pb)
    return (observed - chance) / (1 - chance)


def reference_noise(graded: pd.DataFrame, sizes: pd.DataFrame) -> dict:
    """Claude against GPT-4o on the graded pairs, per stratum, fit bucket and score."""
    def describe(part: pd.DataFrame) -> dict:
        if part.empty:
            return {"n": 0}
        both = pd.crosstab(part[CLAUDE], part[REFERENCE]).reindex(
            index=["hire", "reject"], columns=["hire", "reject"], fill_value=0)
        return {"n": len(part), "agree": int((part[CLAUDE] == part[REFERENCE]).sum()),
                "agree_share": float((part[CLAUDE] == part[REFERENCE]).mean()) if len(part) else None,
                "kappa": cohen_kappa(part[CLAUDE], part[REFERENCE]) if len(part) else None,
                "claude_hire_share": float((part[CLAUDE] == "hire").mean()) if len(part) else None,
                "gpt4o_hire_share": float((part[REFERENCE] == "hire").mean()) if len(part) else None,
                "claude_hire_gpt4o_hire": int(both.loc["hire", "hire"]),
                "claude_hire_gpt4o_reject": int(both.loc["hire", "reject"]),
                "claude_reject_gpt4o_hire": int(both.loc["reject", "hire"]),
                "claude_reject_gpt4o_reject": int(both.loc["reject", "reject"])}

    agree = (graded[CLAUDE] == graded[REFERENCE]).to_frame("a")
    cross = {}
    for s in STRATA:
        for b in BUCKETS:
            part = graded[(graded["stratum"] == s) & (graded["fit"] == b)]
            if len(part):
                cross[f"{s} / {b}"] = describe(part)
    return {
        "all_graded": describe(graded),
        "by_stratum": {s: describe(graded[graded["stratum"] == s]) for s in STRATA},
        "by_fit": {b: describe(graded[graded["fit"] == b]) for b in BUCKETS},
        "by_fit_score": {int(k): describe(p) for k, p in graded.groupby("fit_score")},
        "by_stratum_and_fit": cross,
        "agree_stratified_450": float(stratified_estimate(agree, graded["stratum"], sizes)["a"]),
    }


# ---- context: the training anchors --------------------------------------------------------

def anchor_context(path: Path) -> dict | None:
    """The training anchors' attribute-free verdict per pool pair. A different population
    (the training pool, not the benchmark), so it is context, never a matched comparison."""
    if not path.exists():
        return None
    rows = pd.read_parquet(path, use_threads=False, columns=["pair_id", "reference_decision"])
    per_pair = rows.groupby("pair_id")["reference_decision"]
    first = per_pair.first()
    return {"path": str(path), "rows": len(rows), "pairs": int(first.size),
            "pairs_with_one_verdict": int((per_pair.nunique() == 1).sum()),
            "pair_reject_share": float((first == "reject").mean()),
            "row_reject_share": float((rows["reference_decision"] == "reject").mean()),
            "note": "training pool pairs; not the benchmark's 450, not matched"}


# ---- report -------------------------------------------------------------------------------

def code_state() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted": status.splitlines() if status else []}


def _pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def render(sizes: pd.DataFrame, agreement: pd.DataFrame, readme: dict, flips: pd.DataFrame,
           stability: pd.DataFrame, noise: dict, anchors: dict | None, checks: dict,
           skipped: list[str]) -> str:
    """The readable summary printed to stdout (every number also in the CSVs and JSON)."""
    lines = [f"strata (N in the 450, n graded): {sizes.to_dict('index')}",
             f"checks: {checks}"]
    if skipped:
        lines.append(f"arms skipped (no parquet): {skipped}")
    a = agreement.copy()
    table = pd.DataFrame({
        "arm": a["arm"],
        "hire% exact": a["hire_rate_attr_free"].map(_pct),
        "hire% strat": a["hire_rate_stratified"].map(_pct),
        "hire CI": [f"[{_pct(lo)}, {_pct(hi)}]" for lo, hi in zip(a["hire_ci_low"], a["hire_ci_high"])],
        "=4o exact": a["agree_gpt4o_exact"].map(_pct),
        "=4o strat": a["agree_gpt4o_stratified"].map(_pct),
        "=Claude strat": a["agree_claude_stratified"].map(_pct),
        "CI": [f"[{_pct(lo)}, {_pct(hi)}]" if lo == lo else ""
               for lo, hi in zip(a["agree_claude_ci_low"], a["agree_claude_ci_high"])],
        "d vs SFT pp": [f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" if d == d else ""
                        for d, lo, hi in zip(a["delta_vs_sft_pp"], a["delta_ci_low"], a["delta_ci_high"])],
        "=Claude README": a["agree_claude_readme_method"].map(_pct),
        "disagree": a["agree_claude_disagree"].fillna(""),
        "agree-hire": a["agree_claude_agree-hire"].fillna(""),
        "agree-reject": a["agree_claude_agree-reject"].fillna(""),
        # how much of the arm's disagreement with GPT-4o lies on graded pairs
        "off-4o graded": [f"{int(g)}/{int(t)}" if t == t else "" for g, t in
                          zip(a["disagree_gpt4o_graded"], a["disagree_gpt4o_pairs"])],
    })
    lines += ["", "1. agreement over the 450 attr_free pairs (%; strat = stratified estimate)",
              table.to_string(index=False),
              f"README extrapolation reproduced: {readme}"]
    f = flips[["arm", "direction", "changed", "graded", "ungraded", "claude_with_arm",
               "claude_with_sft", "fit_1-2", "fit_3", "fit_4-5", "gpt4o_with_arm",
               "both_with_arm", "both_with_sft"]]
    lines += ["", "2. attr_free decisions changed against SFT (Claude/GPT-4o on the graded ones)",
              f.to_string(index=False)]
    s = stability[stability["breakdown"].isin(["population", "graded: fit",
                                               "graded: claude_vs_gpt4o", "graded: stratum",
                                               "disagree stratum: fit",
                                               "disagree stratum: claude_vs_gpt4o"])]
    pivot = s.assign(cell=[f"{u}/{n} {_pct(v)}" for u, n, v in
                           zip(s["unstable_sets"], s["sets"], s["unstable_share"])]
                     ).pivot_table(index=["breakdown", "level"], columns="arm", values="cell",
                                   aggfunc="first", sort=False)
    pivot = pivot[[c for c in agreement["arm"] if c in pivot.columns]]
    lines += ["", "3. unstable sets (pair, group, condition), greedy decisions: unstable/sets %",
              pivot.to_string()]
    w = stability[stability["breakdown"].isin(["graded: all", "graded: fit",
                                               "graded: claude_vs_gpt4o"])]
    weighted = w.pivot_table(index=["breakdown", "level"], columns="arm",
                             values="unstable_share_weighted", aggfunc="first", sort=False)
    weighted = (100 * weighted[[c for c in agreement["arm"] if c in weighted.columns]]).round(1)
    lines += ["the same graded slices, each pair weighted by N_s / n_s of its stratum (%)",
              weighted.to_string()]
    cover = stability[(stability["breakdown"] == "population")
                      & (stability["level"] == "graded pairs")]
    lines += ["share of each arm's unstable sets on graded pairs: "
              + ", ".join(f"{r.arm} {_pct(r.share_of_arm_unstable)}%" for r in cover.itertuples())]
    rows = [{"slice": k, **{c: v[c] for c in ("n", "agree", "agree_share", "kappa",
                                               "claude_hire_share", "gpt4o_hire_share")}}
            for group in ("by_stratum", "by_fit") for k, v in noise[group].items()]
    lines += ["", "4. Claude vs GPT-4o on the graded pairs",
              pd.DataFrame([{"slice": "all graded", **{c: noise["all_graded"][c] for c in
                             ("n", "agree", "agree_share", "kappa", "claude_hire_share",
                              "gpt4o_hire_share")}}, *rows]).round(3).to_string(index=False),
              f"stratified Claude-GPT-4o agreement over the 450: {_pct(noise['agree_stratified_450'])}%"]
    if anchors:
        lines += ["", f"context, training anchors (different population): {anchors}"]
    return "\n".join(lines)


def _native(value):
    return value.item() if hasattr(value, "item") else str(value)


def _jsonable(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grades", default=str(REPO_ROOT / DEFAULT_GRADES))
    parser.add_argument("--raw-dir", default=None,
                        help="audit parquets (default: $HBM_OUTPUT_ROOT/outputs/raw)")
    parser.add_argument("--anchors", default=None,
                        help=f"training pool rows.parquet (default: $HBM_OUTPUT_ROOT/{DEFAULT_ANCHORS})")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "claude_reference"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args()
    _single_threaded()

    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve_output_path("outputs/raw")
    present = {a: r for a, r in ARMS.items() if (raw_dir / f"{r}.parquet").exists()}
    skipped = [a for a in ARMS if a not in present]
    if missing := [a for a in DEFINING if a not in present]:
        raise SystemExit(f"the arms that define the strata are missing: {missing}")
    audits = {a: read_audit(raw_dir / f"{r}.parquet") for a, r in present.items()}
    arms = list(present)

    free = free_decisions(audits)
    strata = assign_strata(free)
    grades_path = Path(args.grades)
    grades = pd.read_csv(grades_path)
    checks = check_grades(grades, free, strata)
    graded = graded_frame(grades, free, strata)
    sizes = stratum_sizes(strata, graded)

    agreement = agreement_table(free, graded, sizes, arms, args.n_boot, args.seed)
    readme = readme_check(agreement)
    checks["defining_stratified_minus_exact_max"] = defining_exactness(agreement)
    flips = flip_table(free, graded, strata, arms)
    matched, dropped_rows = common_rows(audits)
    unstable, incomplete = complete_flags({a: set_flags(f) for a, f in matched.items()})
    stability = stability_table(unstable, graded, strata, sizes, arms)
    noise = reference_noise(graded, sizes)
    anchor_path = Path(args.anchors) if args.anchors else resolve_output_path(DEFAULT_ANCHORS)
    anchors = anchor_context(anchor_path)
    checks.update({"sets": len(unstable), "sets_incomplete_dropped": incomplete,
                   "attributed_rows_not_in_every_arm": dropped_rows})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(out / "agreement.csv", index=False)
    flips.to_csv(out / "flips.csv", index=False)
    stability.to_csv(out / "stability_by_fit.csv", index=False)
    summary = {
        "grades": {"path": str(grades_path),
                   "sha256": hashlib.sha256(grades_path.read_bytes()).hexdigest()},
        "runs": present, "skipped": skipped, "defining_arms": list(DEFINING),
        "strata": sizes.to_dict("index"), "checks": checks,
        "method": {"estimate": "sum_s N_s * mean_s / 450 over strata " + ", ".join(STRATA),
                   "bootstrap": {"draws": args.n_boot, "seed": args.seed,
                                 "unit": "graded pair, within stratum; same draws for every arm"},
                   "set": SET, "set_decisions": "greedy audit decision at each arm's own "
                   "operating point; a set counts only if every variant was decided"},
        "agreement": _jsonable(agreement), "readme_extrapolation": readme,
        "flips": _jsonable(flips), "stability": _jsonable(stability),
        "reference_noise": noise, "training_anchors": anchors, "code": code_state(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 default=_native))
    print(render(sizes, agreement, readme, flips, stability, noise, anchors, checks, skipped))
    print(f"\nwrote {out}/agreement.csv, flips.csv, stability_by_fit.csv, summary.json")


if __name__ == "__main__":
    main()
