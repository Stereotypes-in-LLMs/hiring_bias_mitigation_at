"""Does the v2 judge favour its own text, or a correct verdict? Pre-registered in
docs/JUDGE_AUTHOR_CONTROL.md.

Why. On hire-reference rows the 122B teacher judge passes v1's teacher-written `chosen` 95-99% of
the time and the SFT student's wrong `rejected` 53-71% (reports/teacher_leakage/judge_by_cell.csv
at cc38e2e). Two things differ between those sides: who wrote the text (the judge itself, or the
student) and the verdict (right, or wrong). This control holds the author at "student" and the
verdict at "right": the same judge on the student's own CORRECT attributed sample from the same
rows. If it passes like `chosen`, the gap is the verdict; if it passes like `rejected`, the judge
penalises student text.

Rows: the selected, judged rows of each round (selected.parquet, judge_strategies only, the rows
behind judge_by_cell.csv). Control sample: per row, the first attributed sample in sample order
whose decision is the reference and that passes the same hard filters as the negatives
(negatives_v2.filter_negatives: parsable, right language, length within max_length_delta_words
of chosen, no attribute or group named, no meta-commentary). The filter is reused unchanged by
handing it the row with its reference verdict flipped, so "flipped" means "correct".

Stages (one round per config; both rounds by default):

  select     CPU. The control sample per row and why rows have none -> <round>/judge_control/
  judge      GPU, 4 x GH200, TP4. The teacher judges the control samples with the round's own
             judge_generation, and re-judges a seeded subsample of hire-reference rows' chosen
             and rejected (retest: does the judge still return what it returned in October?)
  summarize  CPU. Pass rates, failure reasons, Wilson CIs and paired comparisons on matched
             rows -> reports/judge_author_control/, plus 20 ungrounded hire-side rejections per
             round as text (<round>/judge_control/ungrounded_rejections_hire.md)

    python scripts/judge_author_control.py --stage select
    python scripts/judge_author_control.py --stage judge
    python scripts/judge_author_control.py --stage summarize

`--limit N --tag debug` runs on N seeded rows per round into separate files, for the debug
partition.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("judge_author_control")

CONFIGS = ("configs/generation/negatives_v2_onpolicy_full_gh200.yaml",
           "configs/generation/negatives_v2_onpolicy_hire_gh200.yaml")
ROW_KEY = list(OP.ROW_KEY)
CELL = ["protected_group", "condition", "reference_decision"]
SIDES = ("rejected", "chosen", "control")
REASONS = ("unparsable", "alludes", "ungrounded", "contradicts_decision", "implausible")
PREREG = REPO_ROOT / "reports" / "judge_author_control" / "PREREG.md"


# ---- inputs --------------------------------------------------------------------------------

def read_config(path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))


def round_dir(cfg: dict) -> Path:
    return Path(resolve_output_path(cfg["artifacts_dir"]))


def out_dir(cfg: dict, tag: str) -> Path:
    return round_dir(cfg) / ("judge_control" if not tag else f"judge_control-{tag}")


def judged_rows(cfg: dict) -> pd.DataFrame:
    """The rows the October judge saw, in its order (stage_judge's `selected`)."""
    strategies = list(cfg.get("judge_strategies", [OP.TIPPED]))
    selected = pd.read_parquet(round_dir(cfg) / "selected.parquet")
    return selected[selected["rejected_strategy"].isin(strategies)].reset_index(drop=True)


def limit_rows(frame: pd.DataFrame, limit: int | None, seed: int = 42) -> pd.DataFrame:
    if not limit or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)


def attributed_samples(cfg: dict, rows: pd.DataFrame) -> dict[tuple, list[str]]:
    """Every attributed sample of the round, keyed by ROW_KEY. Prompts are attribute-free per
    pair first, then one attributed prompt per row of rows.parquet (stage_prepare)."""
    root = round_dir(cfg)
    pool = pd.read_parquet(root / "rows.parquet")
    if pool.duplicated(ROW_KEY).any():
        raise ValueError(f"{root}/rows.parquet: {ROW_KEY} does not identify a row")
    prompts = pd.read_parquet(root / "prompts.parquet")
    k = int(cfg.get("samples_per_prompt", 8))
    samples = OP.read_samples(root / "raw" / "samples", prompts["prompt"].tolist(),
                              int(cfg.get("chunk_size", 512)), k)
    attributed = prompts["kind"] == "attributed"
    if prompts.loc[attributed, "key"].tolist() != [str(i) for i in range(len(pool))]:
        raise ValueError(f"{root}/prompts.parquet: attributed prompts do not follow rows.parquet")
    by_row = [s for s, a in zip(samples, attributed) if a]
    keys = [tuple(r) for r in pool[ROW_KEY].itertuples(index=False)]
    wanted = {tuple(r) for r in rows[ROW_KEY].itertuples(index=False)}
    missing = wanted - set(keys)
    if missing:
        raise ValueError(f"{len(missing)} judged rows are not in rows.parquet, e.g. "
                         f"{sorted(missing)[:3]}")
    return {key: s for key, s in zip(keys, by_row) if key in wanted}


# ---- select --------------------------------------------------------------------------------

def select_controls(rows: pd.DataFrame, samples: dict[tuple, list[str]], lang: str = "en",
                    max_length_delta: int = 10) -> tuple[pd.DataFrame, dict]:
    """Per row, the first correct sample that passes the negatives' hard filters.

    filter_negatives keeps samples whose decision is the flip of `reference_decision`; it is
    handed each row with that verdict flipped, so it keeps the correct ones, and the verdict
    is put back afterwards. Every other gate is the negatives' own, unchanged."""
    candidates, raw, n_correct = [], [], {}
    for index, row in enumerate(rows.to_dict("records")):
        reference = row["reference_decision"]
        outs = samples[tuple(row[c] for c in ROW_KEY)]
        decisions = OP._decisions(outs, lang)
        correct = [j for j, d in enumerate(decisions) if d == reference]
        n_correct[index] = len(correct)
        for j in correct:
            candidates.append({**row, "reference_decision": N.flip(reference), "_row": index,
                               "control_sample_index": j})
            raw.append(outs[j])
    report = {"rows": len(rows),
              "rows_with_a_correct_sample": sum(v > 0 for v in n_correct.values()),
              "correct_samples": len(raw)}
    if not candidates:
        report.update({"filters": {}, "rows_with_a_control": 0})
        return pd.DataFrame(), report

    kept, report["filters"] = N.filter_negatives(raw, pd.DataFrame(candidates), lang, "control",
                                                 max_length_delta)
    kept = (kept.sort_values(["_row", "control_sample_index"])
                .drop_duplicates("_row", keep="first").reset_index(drop=True))
    # only the control's own columns come out of the filter; the row is the judged row as is
    kept = kept[["_row", "control_sample_index", "rejected_decision", "rejected_feedback",
                 "rejected_words", "rejected_proxy_words"]].rename(columns={
                     "rejected_decision": "control_decision",
                     "rejected_feedback": "control_feedback",
                     "rejected_words": "control_words",
                     "rejected_proxy_words": "control_proxy_words"})
    kept["n_correct_samples"] = kept["_row"].map(n_correct)
    kept = rows.iloc[kept["_row"]].reset_index(drop=True).join(kept)
    if not (kept["control_decision"] == kept["reference_decision"]).all():
        raise AssertionError("a control sample does not carry the reference verdict")
    report["rows_with_a_control"] = len(kept)
    report["by_cell"] = {
        " | ".join(map(str, k)): {"rows": len(g),
                                  "with_control": int(g.index.isin(kept["_row"]).sum())}
        for k, g in rows.groupby(CELL)
    }
    return kept.drop(columns=["_row"]), report


def stage_select(cfgs: list[dict], tag: str, limit: int | None) -> None:
    for cfg in cfgs:
        out = out_dir(cfg, tag)
        out.mkdir(parents=True, exist_ok=True)
        rows = limit_rows(judged_rows(cfg), limit)
        rows.to_parquet(out / "rows_judged.parquet", index=False)
        controls, report = select_controls(rows, attributed_samples(cfg, rows),
                                           str(cfg.get("lang", "en")),
                                           int(cfg.get("max_length_delta_words", 10)))
        controls.to_parquet(out / "control_selected.parquet", index=False)
        (out / "select.json").write_text(json.dumps(report, indent=2, default=N._json_default),
                                         encoding="utf-8")
        log.info("%s: %d of %d rows have a control sample", cfg["artifacts_dir"],
                 report["rows_with_a_control"], report["rows"])


# ---- judge ---------------------------------------------------------------------------------

def control_prompts(controls: pd.DataFrame) -> list[str]:
    return [N.judge_prompt(row, row["control_decision"], row["control_feedback"])
            for row in controls.to_dict("records")]


def retest_rows(rows: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    hire = rows[rows["reference_decision"] == "hire"]
    return limit_rows(hire, n, seed) if n else hire.iloc[:0]


def check_same_judge(cfgs: list[dict]) -> dict:
    """Both rounds were judged with one teacher setup; refuse to mix two."""
    specs = [(N.teacher_backend_config(c), c.get("judge_generation"),
              int(c.get("min_plausibility", 3))) for c in cfgs]
    if any(s != specs[0] for s in specs):
        raise ValueError("the rounds' configs judge with different settings")
    return cfgs[0]


def stage_judge(cfgs: list[dict], tag: str, retest_n: int) -> None:
    from hiring_bias_mitigation.eval import backends as BK
    from hiring_bias_mitigation.utils.config import require_output_root

    require_output_root()
    judge_cfg = check_same_judge(cfgs)
    work = []
    for cfg in cfgs:
        out = out_dir(cfg, tag)
        controls = pd.read_parquet(out / "control_selected.parquet")
        # the October prompts, recomputed by this code, must be the prompts October sent
        judged = pd.read_parquet(out / "rows_judged.parquet")
        for side, name in (("rejected", "student_judge"), ("chosen", "chosen_judge")):
            october = pd.read_parquet(round_dir(cfg) / "raw" / f"{name}.parquet",
                                      columns=[*ROW_KEY, "prompt"])
            merged = judged[ROW_KEY].merge(october, on=ROW_KEY, how="left", validate="one_to_one")
            if merged["prompt"].tolist() != N.judge_prompts(judged, side):
                raise ValueError(f"{cfg['artifacts_dir']}: {side} judge prompts differ from "
                                 "October's")
        work.append((out, controls, retest_rows(judged, retest_n)))

    teacher = BK.build_backend(N.teacher_backend_config(judge_cfg))
    generation = judge_cfg.get("judge_generation")
    try:
        for out, controls, retest in work:
            N._generate_cached(out / "control_judge.parquet", control_prompts(controls),
                               teacher, generation, controls)
            for side in ("rejected", "chosen"):
                N._generate_cached(out / f"retest_{side}_judge.parquet",
                                   N.judge_prompts(retest, side), teacher, generation, retest)
    finally:
        teacher.close()


# ---- summarize -----------------------------------------------------------------------------

def outcome(frame: pd.DataFrame, raw: list[str], strategy: str | None,
            min_plausibility: int) -> list[str]:
    """filter_by_judge's verdict per row: 'pass' or the reason it would drop the row. The
    judgement is attached as the rejected side; `strategy` only matters for proxy_cue."""
    judged = N.attach_judgements(frame.reset_index(drop=True), raw, "rejected")
    result = []
    for i in range(len(judged)):
        kept, stats = N.filter_by_judge(judged.iloc[[i]], "rejected", strategy, min_plausibility)
        result.append("pass" if len(kept) else
                      next(iter(stats["dropped_by_reason"])).removeprefix("judge_"))
    return result


def wilson(successes, n, z: float = 1.96):
    """Wilson 95% interval for a proportion, elementwise (as teacher_leakage_checks.py)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def position(control, rejected, chosen) -> float:
    """Where control sits between rejected (0) and chosen (1)."""
    gap = np.mean(chosen) - np.mean(rejected)
    return float((np.mean(control) - np.mean(rejected)) / gap) if gap else float("nan")


def position_ci(control, rejected, chosen, reps: int = 2000, seed: int = 42):
    """Percentile bootstrap over rows, the three sides resampled together (paired)."""
    c, r, h = (np.asarray(x, dtype=float) for x in (control, rejected, chosen))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(reps, len(c)))
    gap = h[idx].mean(1) - r[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (c[idx].mean(1) - r[idx].mean(1)) / gap
    pos = pos[np.isfinite(pos)]
    return (float(np.quantile(pos, 0.025)), float(np.quantile(pos, 0.975))) if len(pos) else \
        (float("nan"), float("nan"))


def mcnemar(a, b) -> dict:
    """Exact McNemar on two paired pass vectors."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": float(p)}


def side_columns(side: str, results: pd.Series) -> dict:
    n = len(results)
    passed = int((results == "pass").sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = wilson(passed, n)
    return {f"{side}_pass": passed / n if n else float("nan"),
            f"{side}_pass_lo": float(lo), f"{side}_pass_hi": float(hi),
            **{f"{side}_{r}": int((results == r).sum()) for r in REASONS}}


def compare(part: pd.DataFrame, round_name: str, basis: str, key: dict) -> dict:
    """The three sides on the rows of `part` that have a control sample."""
    matched = part[part["has_control"]]
    entry = {"round": round_name, "basis": basis, **key, "n_rows": len(part),
             "n_matched": len(matched),
             "coverage": len(matched) / len(part) if len(part) else float("nan"),
             "rejected_pass_all": float((part["rejected_result"] == "pass").mean()),
             "chosen_pass_all": float((part["chosen_result"] == "pass").mean())}
    for side in SIDES:
        entry.update(side_columns(side, matched[f"{side}_result"]))
    if not len(matched):
        return entry
    passes = {s: (matched[f"{s}_result"] == "pass").to_numpy() for s in SIDES}
    entry["position"] = position(passes["control"], passes["rejected"], passes["chosen"])
    entry["position_lo"], entry["position_hi"] = position_ci(
        passes["control"], passes["rejected"], passes["chosen"])
    for other in ("chosen", "rejected"):
        m = mcnemar(passes["control"], passes[other])
        entry.update({f"control_vs_{other}_only_control": m["only_first"],
                      f"control_vs_{other}_only_{other}": m["only_second"],
                      f"control_vs_{other}_p": m["p"]})
    return entry


def summarize_round(cfg: dict, tag: str,
                    round_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    out = out_dir(cfg, tag)
    mp = int(cfg.get("min_plausibility", 3))
    rows = pd.read_parquet(out / "rows_judged.parquet")
    student = pd.read_parquet(round_dir(cfg) / "raw" / "student_judge.parquet")
    chosen = pd.read_parquet(round_dir(cfg) / "raw" / "chosen_judge.parquet")
    for name, frame in (("student_judge", student), ("chosen_judge", chosen)):
        if frame.duplicated(ROW_KEY).any():
            raise ValueError(f"{round_name}/{name}: {ROW_KEY} is not unique")
    rows = rows.merge(student[[*ROW_KEY, "raw_output"]].rename(columns={"raw_output": "_r"}),
                      on=ROW_KEY, how="left", validate="one_to_one")
    rows = rows.merge(chosen[[*ROW_KEY, "raw_output"]].rename(columns={"raw_output": "_c"}),
                      on=ROW_KEY, how="left", validate="one_to_one")
    if rows[["_r", "_c"]].isna().any().any():
        raise ValueError(f"{round_name}: a judged row has no October judgement")
    strategy = str(rows["rejected_strategy"].iloc[0])
    rows["rejected_result"] = outcome(rows, rows["_r"].tolist(), strategy, mp)
    rows["chosen_result"] = outcome(rows, rows["_c"].tolist(), None, mp)

    control = pd.read_parquet(out / "control_judge.parquet")
    control["control_result"] = outcome(control, control["raw_output"].tolist(), None, mp)
    control_cols = [*ROW_KEY, "control_result", "control_feedback", "control_sample_index",
                    "raw_output"]
    rows = rows.merge(control[control_cols].rename(columns={"raw_output": "_k"}), on=ROW_KEY,
                      how="left", validate="one_to_one")
    rows["has_control"] = rows["control_result"].notna()

    retest = []
    for side in ("rejected", "chosen"):
        path = out / f"retest_{side}_judge.parquet"
        if not path.exists():
            continue
        again = pd.read_parquet(path)
        again[f"{side}_retest"] = outcome(again, again["raw_output"].tolist(),
                                          strategy if side == "rejected" else None, mp)
        rows = rows.merge(again[[*ROW_KEY, f"{side}_retest"]], on=ROW_KEY, how="left",
                          validate="one_to_one")
        both = rows[rows[f"{side}_retest"].notna()]
        before = both[f"{side}_result"] == "pass"
        after = both[f"{side}_retest"] == "pass"
        retest.append({"round": round_name, "side": side, "n": len(both),
                       "pass_october": float(before.mean()), "pass_retest": float(after.mean()),
                       "agreement_pass": float((before == after).mean()),
                       "agreement_reason": float((both[f"{side}_result"]
                                                  == both[f"{side}_retest"]).mean()),
                       **{f"mcnemar_{k}": v for k, v in mcnemar(after, before).items()}})
    rows.drop(columns=["_r", "_c", "_k"]).to_parquet(out / "per_row.parquet", index=False)

    by_cell, pooled = [], []
    for level, keys in (("cell", CELL), ("pooled", ["reference_decision"])):
        for key, part in rows.groupby(keys, sort=True):
            entry = compare(part, round_name, "october", dict(zip(keys, np.atleast_1d(key))))
            (by_cell if level == "cell" else pooled).append(entry)
    # the same comparison with chosen and rejected judged again in this job, on the retest rows
    if {"rejected_retest", "chosen_retest"} <= set(rows.columns):
        again = rows[rows["rejected_retest"].notna() & rows["chosen_retest"].notna()].copy()
        again["rejected_result"], again["chosen_result"] = (again["rejected_retest"],
                                                            again["chosen_retest"])
        for key, part in again.groupby("reference_decision", sort=True):
            pooled.append(compare(part, round_name, "retest", {"reference_decision": key}))
    return by_cell, pooled, retest


def write_ungrounded(cfg: dict, tag: str, n: int = 20, seed: int = 42) -> None:
    """n hire-side student rejections the October judge called ungrounded, as text, with the
    CV and job they were judged against, and the judge's note."""
    out = out_dir(cfg, tag)
    per_row = pd.read_parquet(out / "per_row.parquet")
    student = pd.read_parquet(round_dir(cfg) / "raw" / "student_judge.parquet")
    notes = N.attach_judgements(student.drop(columns=["raw_output"]),
                                student["raw_output"].tolist(), "rejected")
    pool = per_row[(per_row["reference_decision"] == "hire")
                   & (per_row["rejected_result"] == "ungrounded")]
    pick = pool.sample(n=min(n, len(pool)), random_state=seed).merge(
        notes[[*ROW_KEY, "judge_note"]], on=ROW_KEY, how="left")
    lines = [f"# Ungrounded hire-side rejections, {cfg['artifacts_dir']}",
             "", f"{len(pick)} rows drawn with seed {seed} from "
             f"{len(pool)} "
             "hire-reference rows whose student rejection the October judge called ungrounded.",
             ""]
    for i, row in enumerate(pick.to_dict("records"), 1):
        lines += [f"## {i}. {row['pair_id']} | {row['protected_group']} | {row['condition']} | "
                  f"{row['protected_attr']}", "",
                  f"**Judge note:** {row['judge_note']}", "",
                  f"**Student rejection (sample {row['sample_index']}):** "
                  f"{row['rejected_feedback']}",
                  "", f"**Teacher chosen (hire):** {row['chosen_feedback']}", ""]
        if isinstance(row.get("control_feedback"), str):
            lines += [f"**Student correct (sample {int(row['control_sample_index'])}, judge: "
                      f"{row['control_result']}):** {row['control_feedback']}", ""]
        lines += ["<details><summary>Job and CV</summary>", "", "```", str(row["job_description"]),
                  "```", "", "```", str(row["cv"]), "```", "", "</details>", ""]
    (out / "ungrounded_rejections_hire.md").write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=all", "--", "scripts", "src")
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted": [line[3:] for line in status.splitlines() if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}


def stage_summarize(cfgs: list[dict], tag: str) -> None:
    report_dir = REPO_ROOT / "reports" / "judge_author_control" / (tag or "")
    report_dir.mkdir(parents=True, exist_ok=True)
    by_cell, pooled, retest, inputs = [], [], [], {}
    for cfg in cfgs:
        name = Path(cfg["artifacts_dir"]).name
        c, p, r = summarize_round(cfg, tag, name)
        by_cell += c
        pooled += p
        retest += r
        write_ungrounded(cfg, tag)
        out = out_dir(cfg, tag)
        for path in [round_dir(cfg) / "raw" / "student_judge.parquet",
                     round_dir(cfg) / "raw" / "chosen_judge.parquet",
                     *sorted(out.glob("*_judge.parquet")), out / "control_selected.parquet"]:
            inputs[str(path.relative_to(round_dir(cfg).parents[1]))] = _sha256(path)
        inputs[f"{cfg['artifacts_dir']}/select"] = json.loads((out / "select.json").read_text())
    pd.DataFrame(by_cell).to_csv(report_dir / "by_cell.csv", index=False)
    pd.DataFrame(pooled).to_csv(report_dir / "pooled.csv", index=False)
    pd.DataFrame(retest).to_csv(report_dir / "retest.csv", index=False)
    meta = {"inputs": inputs, "configs": [c["_path"] for c in cfgs], "tag": tag,
            "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
            "code": code_state()}
    (report_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=N._json_default)
                                          + "\n", encoding="utf-8")
    print(pd.DataFrame(pooled)[["round", "basis", "reference_decision", "n_rows", "n_matched",
                                "rejected_pass", "control_pass", "chosen_pass", "position",
                                "position_lo", "position_hi"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("select", "judge", "summarize"), required=True)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--tag", default="", help="separate output files, e.g. debug")
    parser.add_argument("--limit", type=int, default=None, help="seeded rows per round")
    parser.add_argument("--retest-n", type=int, default=400,
                        help="hire-reference rows per round whose two sides are re-judged")
    args = parser.parse_args()
    cfgs = [{**read_config(p), "_path": p} for p in args.configs]
    if args.stage == "select":
        stage_select(cfgs, args.tag, args.limit)
    elif args.stage == "judge":
        stage_judge(cfgs, args.tag, args.retest_n)
    else:
        stage_summarize(cfgs, args.tag)


if __name__ == "__main__":
    main()
