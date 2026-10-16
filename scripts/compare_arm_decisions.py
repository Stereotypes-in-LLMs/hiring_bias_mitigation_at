"""Where one audited arm's benchmark decisions differ from another's: per prompt, per
counterfactual set and per CV/job pair. CPU only.

Why. A matched-hire-rate delta says how much an arm moved, not which decisions moved.
r2em-100 is -0.96 pp against SFT at matched hire rate (reports/operating_point/v2/), yet
read prompt by prompt it changes several hundred of SFT's 31,050 decisions, about as many
into errors as out of them, mostly on a few borderline pairs. This script writes those
differences out so they can be read, with the second arm as the reference:

  sets.csv             analysis/stability.compare on every set (pair x condition, all
                       groups), then per group and condition: unstable shares, the delta with
                       its candidate-clustered CI, fixed and broken sets, the sign test
  prompts_changed.csv  every prompt whose decision differs, with both feedbacks, the
                       reference verdict, and whether the change fixes an error of the
                       reference arm or makes a new one
  set_status.csv       every attributed set: stable or unstable under each arm
  pairs.csv            per CV/job pair with a change: how many variants moved, which way
  summary.md           the tables above as markdown; regenerate it, never edit it
  meta.json            the two raw parquets' sha256 and the code commit

"Error" means disagreeing with the benchmark's attribute-free reference verdict.

    python scripts/compare_arm_decisions.py \\
        --run Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100 \\
        --against Qwen3.5-9B--en--sft--adapter--en_only_cscs

Reads $HBM_OUTPUT_ROOT/outputs/raw/<run>.parquet for both arms and writes to
reports/decision_diff/<run>--vs--<against>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiring_bias_mitigation.analysis import stability as S  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

KEY = ["pair_id", "condition", "protected_group", "protected_attr"]
COLUMNS = [*KEY, "candidate_id", "job_position", "reference_decision", "decision", "feedback"]
#: Share of a set's variants that changed decision. A few changed variants point at
#: particular attributes; most of the set moving points at the pair itself.
SHARE_BINS = [-0.001, 0.0, 0.1, 0.5, 0.9, 1.0]
SHARE_LABELS = ["none", "up to 10%", "10-50%", "50-90%", "over 90%"]


def load(path: str | Path) -> pd.DataFrame:
    """One arm's prompts with its decision and the reference, both normalised."""
    frame = pd.read_parquet(path, columns=COLUMNS)
    return normalise(frame)


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["d"] = frame["decision"].map(S._decision)
    frame["ref"] = frame["reference_decision"].map(S._decision)
    return frame


def prompt_diff(against: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """Both arms' decisions on every shared prompt, the attribute-free ones included."""
    a = against.drop(columns=["decision"]).rename(
        columns={"d": "d_against", "feedback": "feedback_against"})
    r = run[[*KEY, "d", "feedback"]].rename(columns={"d": "d_run", "feedback": "feedback_run"})
    m = a.merge(r, on=KEY, how="inner")
    m["changed"] = m["d_against"] != m["d_run"]
    m["flip"] = m["d_against"].fillna("none") + "->" + m["d_run"].fillna("none")
    right_a, right_r = m["d_against"] == m["ref"], m["d_run"] == m["ref"]
    m["effect"] = "same"
    m.loc[m["changed"] & ~right_a & right_r, "effect"] = "fixes_error"
    m.loc[m["changed"] & right_a & ~right_r, "effect"] = "new_error"
    m.loc[m["changed"] & ~right_a & ~right_r, "effect"] = "other"
    return m


def set_status(diff: pd.DataFrame) -> pd.DataFrame:
    """One row per attributed set (pair x condition): instability under each arm."""
    att = diff[diff["condition"] != "attr_free"]
    g = att.groupby(["pair_id", "condition"])
    out = pd.DataFrame({
        "job_position": g["job_position"].first(),
        "reference": g["ref"].first(),
        "variants": g.size(),
        "changed": g["changed"].sum(),
        "unstable_against": g["d_against"].nunique() > 1,
        "unstable_run": g["d_run"].nunique() > 1,
        "hire_share_against": g["d_against"].apply(lambda s: round((s == "hire").mean(), 3)),
        "hire_share_run": g["d_run"].apply(lambda s: round((s == "hire").mean(), 3)),
    })
    out["status"] = "stable_both"
    out.loc[out["unstable_against"] & out["unstable_run"], "status"] = "unstable_both"
    out.loc[out["unstable_against"] & ~out["unstable_run"], "status"] = "fixed"
    out.loc[~out["unstable_against"] & out["unstable_run"], "status"] = "broken"
    out["changed_share"] = (out["changed"] / out["variants"]).round(3)
    return out.reset_index()


def pair_summary(diff: pd.DataFrame) -> pd.DataFrame:
    """Per CV/job pair with any change: how many attributed variants moved, which way."""
    att = diff[diff["condition"] != "attr_free"]
    free = (diff[diff["condition"] == "attr_free"].drop_duplicates("pair_id")
            .set_index("pair_id")[["d_against", "d_run"]]
            .rename(columns={"d_against": "attr_free_against", "d_run": "attr_free_run"}))
    g = att.groupby("pair_id")
    out = pd.DataFrame({
        "job_position": g["job_position"].first(),
        "reference": g["ref"].first(),
        "variants": g.size(),
        "changed": g["changed"].sum(),
        "hire_to_reject": g["flip"].apply(lambda s: int((s == "hire->reject").sum())),
        "reject_to_hire": g["flip"].apply(lambda s: int((s == "reject->hire").sum())),
        "fixes_error": g["effect"].apply(lambda s: int((s == "fixes_error").sum())),
        "new_error": g["effect"].apply(lambda s: int((s == "new_error").sum())),
        "hire_share_against": g["d_against"].apply(lambda s: round((s == "hire").mean(), 3)),
        "hire_share_run": g["d_run"].apply(lambda s: round((s == "hire").mean(), 3)),
    }).join(free)
    out = out[out["changed"] > 0].reset_index()
    return out.sort_values(["changed", "pair_id"], ascending=[False, True]).reset_index(drop=True)


def concentration(status: pd.DataFrame) -> pd.DataFrame:
    """Sets and changed prompts by the share of a set's variants that changed."""
    bins = pd.cut(status["changed_share"], SHARE_BINS, labels=SHARE_LABELS)
    out = status.groupby(bins, observed=False).agg(sets=("variants", "size"),
                                                   changed_prompts=("changed", "sum"))
    total = out["changed_prompts"].sum()
    out["share_of_changes"] = (out["changed_prompts"] / total).round(3) if total else 0.0
    return out.reset_index().rename(columns={"changed_share": "variants_changed"})


def set_comparison(against_rows: pd.DataFrame, run_rows: pd.DataFrame) -> pd.DataFrame:
    """stability.compare with the second arm as the reference: all sets, then per cell.

    Takes `stability.load_rows` frames. A per-cell set holds one group's variants under one
    condition, which is the set definition of the operating-point control.
    """
    rows = [{"scope": "all groups (pair x condition)", **S.compare(against_rows, run_rows)}]
    for group in sorted(run_rows["protected_group"].unique()):
        for cond in sorted(run_rows["condition"].unique()):
            a = against_rows[(against_rows["protected_group"] == group)
                             & (against_rows["condition"] == cond)]
            r = run_rows[(run_rows["protected_group"] == group) & (run_rows["condition"] == cond)]
            res = S.compare(a, r, [*S.SET, "protected_group"])
            if res.get("sets"):
                rows.append({"scope": f"{group} / {cond}", **res})
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(run: str, against: str, sets: pd.DataFrame, diff: pd.DataFrame,
           conc: pd.DataFrame, pairs: pd.DataFrame, status: pd.DataFrame) -> str:
    s = sets[["scope", "sets", "unstable_base_pct", "unstable_run_pct", "delta_pp", "ci_low",
              "ci_high", "fixed", "broken", "p_sign"]].rename(
        columns={"unstable_base_pct": "unstable_against_pct"})
    kind = diff["condition"].eq("attr_free").map({True: "attr_free", False: "attributed"})
    flips = (diff[diff["changed"]].assign(prompts=kind[diff["changed"]])
             .groupby(["prompts", "flip", "effect"]).size().rename("n").reset_index())
    top = pairs.head(15).assign(pair_id=lambda f: f["pair_id"].str.slice(0, 8))
    counts = status["status"].value_counts()
    status_line = ", ".join(f"{k} {int(counts.get(k, 0))}"
                            for k in ("stable_both", "fixed", "broken", "unstable_both"))
    return "\n".join([
        f"# {run} against {against}",
        "",
        "Generated by `scripts/compare_arm_decisions.py`; regenerate it, do not edit it. The "
        "inputs' sha256 and the code commit are in `meta.json`. Greedy benchmark decisions. "
        "\"Error\" means disagreeing with the benchmark's attribute-free reference verdict.",
        "",
        "## Counterfactual sets, the second arm as the reference",
        "",
        "`all groups` is the set of `reports/set_stability.csv` (one pair under one condition, "
        "every group's variants); the per-cell rows hold one group's variants, as in the "
        "operating-point control. Raw instability at each arm's own hire rate, not matched.",
        "",
        _table(s),
        "",
        f"## Decisions that differ: {int(diff['changed'].sum())} of {len(diff)} prompts",
        "",
        _table(flips),
        "",
        "## How concentrated the changes are",
        "",
        "Attributed sets by the share of their variants whose decision changed.",
        "",
        _table(conc),
        "",
        f"Sets: {status_line}.",
        "",
        "## Pairs with the most changed variants",
        "",
        _table(top),
        "",
        "Every changed prompt with both feedbacks is in `prompts_changed.csv`, every pair with "
        "a change in `pairs.csv`, every set in `set_status.csv`.",
        "",
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_state() -> dict:
    """The commit this code runs at, and its uncommitted changes to tracked code."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=no", "--", "scripts", "src")
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted": [line[3:] for line in status.splitlines() if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the arm read against the reference")
    parser.add_argument("--against", required=True, help="the reference arm, e.g. SFT")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "decision_diff"))
    args = parser.parse_args()

    raw = Path(resolve_output_path("outputs/raw"))
    run_path, against_path = raw / f"{args.run}.parquet", raw / f"{args.against}.parquet"
    out = Path(args.out) / f"{args.run}--vs--{args.against}"
    out.mkdir(parents=True, exist_ok=True)

    sets = set_comparison(S.load_rows(str(against_path)), S.load_rows(str(run_path)))
    diff = prompt_diff(load(against_path), load(run_path))
    status = set_status(diff)
    pairs = pair_summary(diff)
    conc = concentration(status)

    sets.to_csv(out / "sets.csv", index=False)
    keep = [*KEY, "candidate_id", "job_position", "ref", "d_against", "d_run", "flip", "effect",
            "feedback_against", "feedback_run"]
    (diff[diff["changed"]][keep].rename(columns={"ref": "reference"})
     .sort_values(KEY).to_csv(out / "prompts_changed.csv", index=False))
    status.to_csv(out / "set_status.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    (out / "summary.md").write_text(render(args.run, args.against, sets, diff, conc, pairs,
                                           status), encoding="utf-8")
    meta = {
        "run": args.run, "against": args.against,
        "inputs": {name: {"path": f"outputs/raw/{name}.parquet", "sha256": _sha256(p)}
                   for name, p in ((args.run, run_path), (args.against, against_path))},
        "shared_prompts": len(diff), "changed_prompts": int(diff["changed"].sum()),
        "code": code_state(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {meta['changed_prompts']} of {meta['shared_prompts']} decisions differ")


if __name__ == "__main__":
    main()
