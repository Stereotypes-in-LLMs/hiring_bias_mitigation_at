"""How the training anchors relate to the SFT student's own attribute-free decisions (CPU, ~1 min).

    python scripts/anchor_agreement.py
    python scripts/anchor_agreement.py --no-hire-topup --out reports/anchor_agreement

Why. Every training set after SFT (DPO r1, r2, r2em, GRPO grpo-02, RFT rft-01) takes its target
verdict from one anchor per pair: the v1 teacher's (Qwen3.5-122B) attribute-free verdict on the
pair's CV and job, one sample at temperature 0.7 (configs/generation/teacher.yaml; the samples
are artifacts/semisynthetic-v1/raw/en_reference.parquet). On a row where that anchor disagrees
with what the SFT student decides on the same CV and job WITHOUT any attribute, the target does
not teach invariance to the attribute: it moves the student's verdict on the CV itself. This
script counts those rows per training set and the direction they pull.

Inputs, all under $HBM_OUTPUT_ROOT:

  onpolicy-full/rows.parquet        the pool, 35,826 rows over 2,992 pairs; `reference_decision`
                                    is the anchor, shared by every row of a pair
  onpolicy-full/prompts.parquet     kind 'attr_free': one attribute-free prompt per pair (key =
                                    pair_id)
  onpolicy-full/raw/samples/        8 SFT-student samples per prompt at temperature 1, matched to
                                    the attribute-free prompts by prompt text
  onpolicy-hire32/ (optional)       32 samples per prompt on the anchor-hire pairs: a cross-check
                                    of the k=8 majority on that side
  each training config              data_config -> artifacts_dir -> <data_kind>_train.parquet,
                                    filtered as the trainer filters it (apply_data_filters)

Definitions. The student's attribute-free hire share is its hire samples over its parsed samples
(refusals and invalid outputs are counted, not used). Its majority is hire above one half,
reject below, tie at exactly one half. A training row disagrees when the student's majority is
the opposite of the row's target verdict, which here is always the anchor (checked); such a row
pulls the student toward the anchor's side. `af_error_mass` is the soft version: the summed
share of the student's attribute-free samples that oppose the target.

The student is not ground truth. A disagreement says the anchor and the student differ on that
CV, not which one is right. The pool excludes every benchmark candidate and job, so nothing here
can be set beside GPT-4o's reference or the Claude blind review on the same CVs.

Output: reports/anchor_agreement/{pairs.csv, by_training_set.csv, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SIDES = ("hire", "reject")
#: The column holding the verdict each data kind trains toward.
TARGET_COLUMN = {"grpo": "reference_decision", "rft": "decision",
                 "dpo_round2": "decision", "dpo_v2": "chosen_decision"}
DEFAULT_TRAINING = (
    "dpo_r1=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml",
    "dpo_r2=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r2_gh200.yaml",
    "dpo_r2em=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r2em_gh200.yaml",
    "grpo_02=configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_pilot_gh200.yaml",
    "rft_01=configs/mitigation/sft/qwen3.5-9b_en_only_rft_gh200.yaml",
)
#: v1 splits whose candidates were held out of the SFT student's training (candidate-wise split).
V1_VALIDATION_FILES = ("sft_validation.parquet", "dpo_validation.parquet")
POOL_ROWS = "pool_all_rows"


def read(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Single-threaded parquet read: Daint login nodes refuse pyarrow's worker threads."""
    return pd.read_parquet(path, columns=columns, use_threads=False)


# ---- the student's attribute-free decisions ------------------------------------------------

def free_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    """pair_id -> attribute-free prompt text, refusing anything that makes text matching
    ambiguous."""
    free = prompts[prompts["kind"] == "attr_free"]
    out = pd.DataFrame({"pair_id": free["key"].astype(str).to_numpy(),
                        "prompt": free["prompt"].to_numpy()})
    if out["pair_id"].duplicated().any():
        raise ValueError("a pair has two attribute-free prompts")
    if out["prompt"].duplicated().any():
        raise ValueError("two pairs share one attribute-free prompt text: matching samples "
                         "by text would be ambiguous")
    return out


def read_matching_samples(sample_dir: str | Path, prompts: set[str]) -> pd.DataFrame:
    """The stored samples whose prompt text is in `prompts`, one chunk at a time, so the other
    prompts' text is never held in memory all at once."""
    chunks = sorted(Path(sample_dir).glob("chunk-*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"no chunk-*.parquet under {sample_dir}")
    parts = []
    for chunk in chunks:
        frame = read(chunk, ["prompt", "sample_index", "raw_output"])
        parts.append(frame[frame["prompt"].isin(prompts)])
    return pd.concat(parts, ignore_index=True)


def decision_counts(free: pd.DataFrame, samples: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Per pair: n_samples, n_hire, n_reject, n_unparsed over the samples whose prompt text is
    the pair's attribute-free prompt. Every pair must have samples."""
    from hiring_bias_mitigation.eval.parsing import HIRE, REJECT, parse_output

    samples = samples[samples["prompt"].isin(set(free["prompt"]))]
    repeated = samples.duplicated(["prompt", "sample_index"])
    if repeated.any():
        raise ValueError(f"{int(repeated.sum())} samples repeat a (prompt, sample_index): "
                         "the sample directory holds more than one run")
    decisions = [parse_output(text, lang).decision for text in samples["raw_output"]]
    frame = pd.DataFrame({"prompt": samples["prompt"].to_numpy(), "decision": decisions})
    frame = frame.merge(free, on="prompt", how="left", validate="many_to_one")
    table = pd.crosstab(frame["pair_id"], frame["decision"])
    out = pd.DataFrame({
        "pair_id": table.index.astype(str),
        "n_samples": table.sum(axis=1).to_numpy(),
        "n_hire": table[HIRE].to_numpy() if HIRE in table else 0,
        "n_reject": table[REJECT].to_numpy() if REJECT in table else 0,
    })
    out["n_unparsed"] = out["n_samples"] - out["n_hire"] - out["n_reject"]
    missing = set(free["pair_id"]) - set(out["pair_id"])
    if missing:
        raise KeyError(f"{len(missing)} attribute-free prompts have no stored student samples, "
                       f"e.g. {sorted(missing)[:3]}: the samples are from another run")
    return out.reset_index(drop=True)


def majority(n_hire, n_reject) -> np.ndarray:
    """hire above one half of the parsed samples, reject below, tie at one half, none when no
    sample parsed."""
    hire, reject = np.asarray(n_hire), np.asarray(n_reject)
    parsed = hire + reject
    return np.select([parsed == 0, 2 * hire > parsed, 2 * hire < parsed],
                     ["none", "hire", "reject"], default="tie")


def with_shares(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Adds <prefix>hire_share and <prefix>majority from <prefix>n_hire and <prefix>n_reject."""
    hire, reject = frame[f"{prefix}n_hire"], frame[f"{prefix}n_reject"]
    parsed = (hire + reject).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(parsed > 0, hire.to_numpy(dtype=float) / parsed, np.nan)
    return frame.assign(**{f"{prefix}hire_share": share,
                           f"{prefix}majority": majority(hire, reject)})


def pair_table(rows: pd.DataFrame, counts: pd.DataFrame, heldout: set[str]) -> pd.DataFrame:
    """One row per pool pair: the anchor, the student's attribute-free counts, its majority,
    whether the two agree, and the share of student samples opposing the anchor."""
    rows = rows.assign(pair_id=rows["pair_id"].astype(str))
    anchors = rows.groupby("pair_id")["reference_decision"].nunique()
    if (anchors > 1).any():
        raise ValueError(f"{int((anchors > 1).sum())} pairs carry two anchors")
    if not rows["reference_decision"].isin(SIDES).all():
        raise ValueError("an anchor is neither hire nor reject")
    pairs = (rows.drop_duplicates("pair_id")[["pair_id", "candidate_id", "job_id",
                                              "reference_decision"]]
             .rename(columns={"reference_decision": "anchor"}))
    out = pairs.merge(counts, on="pair_id", how="left", validate="one_to_one")
    if out["n_samples"].isna().any():
        raise KeyError(f"{int(out['n_samples'].isna().sum())} pool pairs have no student counts")
    out = with_shares(out).rename(columns={"hire_share": "student_hire_share",
                                           "majority": "student_majority"})
    student = out["student_majority"]
    out["agreement"] = np.where(student.isin(["tie", "none"]), student,
                                np.where(student == out["anchor"], "agree", "disagree"))
    out["p_disagree"] = np.where(out["anchor"] == "hire", 1 - out["student_hire_share"],
                                 out["student_hire_share"])
    out["sft_heldout"] = out["candidate_id"].astype(str).isin(heldout)
    return out.reset_index(drop=True)


# ---- pool-level summaries -------------------------------------------------------------------

def confusion(pairs: pd.DataFrame) -> dict:
    """Anchor x student majority over pairs, with counts, shares, agreement and kappa (ties
    left out of the last two)."""
    n = len(pairs)
    table, of_side = {}, {}
    for anchor in SIDES:
        side = pairs[pairs["anchor"] == anchor]
        table[anchor] = {s: int((side["student_majority"] == s).sum())
                         for s in ("hire", "reject", "tie", "none")}
        of_side[anchor] = {s: (round(c / len(side), 4) if len(side) else None)
                           for s, c in table[anchor].items()}
    decided = pairs[pairs["student_majority"].isin(SIDES)]
    agree = float((decided["student_majority"] == decided["anchor"]).mean()) if len(decided) \
        else None
    kappa = None
    if len(decided):
        a = (decided["anchor"] == "hire").mean()
        s = (decided["student_majority"] == "hire").mean()
        expected = a * s + (1 - a) * (1 - s)
        kappa = round(float((agree - expected) / (1 - expected)), 4) if expected < 1 else None
    parsed = pairs["n_hire"] + pairs["n_reject"]
    return {
        "pairs": n,
        "anchor_hire_rate": round(float((pairs["anchor"] == "hire").mean()), 4) if n else None,
        "student_hire_rate_samples": round(float(pairs["n_hire"].sum() / parsed.sum()), 4)
        if parsed.sum() else None,
        "student_hire_rate_pair_mean": round(float(pairs["student_hire_share"].mean()), 4)
        if n else None,
        "student_majority_hire_rate": round(float((pairs["student_majority"] == "hire").mean()),
                                            4) if n else None,
        "counts": table,
        "share_of_all_pairs": {a: {s: round(c / n, 4) for s, c in row.items()}
                               for a, row in table.items()} if n else None,
        "share_of_anchor_side": of_side,
        "anchor_reject_student_hire": table["reject"]["hire"],
        "anchor_hire_student_reject": table["hire"]["reject"],
        "ties": int((pairs["student_majority"] == "tie").sum()),
        "agreement_excluding_ties": round(agree, 4) if agree is not None else None,
        "cohen_kappa_excluding_ties": kappa,
    }


def share_histogram(pairs: pd.DataFrame, anchor: str) -> list[dict]:
    """How many of the anchor's pairs had k of n student samples say hire."""
    side = pairs[pairs["anchor"] == anchor]
    parsed = (side["n_hire"] + side["n_reject"]).astype(int)
    frame = pd.DataFrame({"hires": side["n_hire"].astype(int), "parsed": parsed})
    counted = frame.value_counts().reset_index(name="pairs")
    counted["share"] = counted["hires"] / counted["parsed"].where(counted["parsed"] > 0)
    counted = counted.sort_values(["share", "parsed"], na_position="first")
    return [{"student_hire_samples": f"{h}/{p}", "pairs": int(c),
             "share_of_side": round(c / len(side), 4)}
            for h, p, c in zip(counted["hires"], counted["parsed"], counted["pairs"])]


def confident_opposition(pairs: pd.DataFrame, threshold: float = 7 / 8) -> dict:
    """Anchor-reject pairs the student hires in at least `threshold` of its samples, and
    anchor-hire pairs it hires in at most 1 - threshold."""
    share = pairs["student_hire_share"]
    rej = pairs["anchor"] == "reject"
    hire = pairs["anchor"] == "hire"
    return {
        "threshold": round(threshold, 4),
        "anchor_reject_student_hire_share_at_least": int((rej & (share >= threshold)).sum()),
        "anchor_reject_pairs": int(rej.sum()),
        "anchor_hire_student_hire_share_at_most": int((hire & (share <= 1 - threshold)).sum()),
        "anchor_hire_pairs": int(hire.sum()),
    }


# ---- training sets ---------------------------------------------------------------------------

def training_frame(name: str, config: str | Path) -> tuple[pd.DataFrame, dict]:
    """The rows a training config trains on, exactly as its trainer loads them, and where they
    came from."""
    from hiring_bias_mitigation.mitigation.training_common import apply_data_filters
    from hiring_bias_mitigation.utils.config import load_config, resolve_output_path

    cfg = load_config(config)
    kind = str(cfg.get("data_kind", cfg.get("objective", "")))
    if kind not in TARGET_COLUMN:
        raise ValueError(f"{name}: data_kind {kind!r} has no known target column")
    data = cfg["data"]
    path = Path(resolve_output_path(data["artifacts_dir"])) / f"{kind}_train.parquet"
    frame = apply_data_filters(read(path), data)
    info = {"training_set": name, "config": str(config),
            "data_config": cfg.get("data_config"), "train_file": str(path), "data_kind": kind,
            "target_column": TARGET_COLUMN[kind], "max_steps": cfg.get("max_steps"),
            "num_train_epochs": cfg.get("num_train_epochs")}
    return frame.reset_index(drop=True), info


def join_training(frame: pd.DataFrame, pairs: pd.DataFrame, target_column: str,
                  name: str) -> pd.DataFrame:
    """Each training row with its pair's anchor and the student's attribute-free view. Refuses
    rows whose pair is not in the pool and rows whose target is not the anchor."""
    rows = pd.DataFrame({"pair_id": frame["pair_id"].astype(str).to_numpy(),
                         "target": frame[target_column].astype(str).to_numpy()})
    unknown = ~rows["pair_id"].isin(set(pairs["pair_id"]))
    if unknown.any():
        raise KeyError(f"{name}: {int(unknown.sum())} rows from pairs outside the pool")
    joined = rows.merge(pairs[["pair_id", "anchor", "student_majority", "student_hire_share"]],
                        on="pair_id", how="left", validate="many_to_one")
    off = joined["target"] != joined["anchor"]
    if off.any():
        raise ValueError(f"{name}: {int(off.sum())} rows train toward a verdict other than "
                         "their pair's anchor; this script assumes the anchor is the target")
    return joined


def side_rows(joined: pd.DataFrame, info: dict, confident_share: float = 7 / 8) -> list[dict]:
    """One summary row per anchor side and one over all rows. `confident_disagree_rows`: rows
    where at least `confident_share` of the student's attribute-free samples oppose the
    target."""
    out = []
    for side in ("hire", "reject", "all"):
        g = joined if side == "all" else joined[joined["anchor"] == side]
        student = g["student_majority"]
        toward_reject = int(((g["anchor"] == "reject") & (student == "hire")).sum())
        toward_hire = int(((g["anchor"] == "hire") & (student == "reject")).sum())
        disagree = toward_reject + toward_hire
        opposed = np.where(g["anchor"] == "hire", 1 - g["student_hire_share"],
                           g["student_hire_share"])
        confident = int((opposed >= confident_share).sum())
        n = len(g)
        out.append({
            "training_set": info["training_set"], "anchor_side": side, "rows": n,
            "pairs": int(g["pair_id"].nunique()),
            "student_majority_hire": int((student == "hire").sum()),
            "student_majority_reject": int((student == "reject").sum()),
            "student_tie": int((student == "tie").sum()),
            "disagree_rows": disagree,
            "disagree_share": round(disagree / n, 4) if n else None,
            "tie_share": round(int((student == "tie").sum()) / n, 4) if n else None,
            "pull_toward_reject_rows": toward_reject,
            "pull_toward_hire_rows": toward_hire,
            "net_toward_reject_rows": toward_reject - toward_hire,
            "confident_disagree_rows": confident,
            "af_error_mass": round(float(np.nansum(opposed)), 2),
            "af_error_mean": round(float(np.nanmean(opposed)), 4) if n else None,
            "data_kind": info.get("data_kind"), "max_steps": info.get("max_steps"),
            "num_train_epochs": info.get("num_train_epochs"),
            "train_file": info.get("train_file"), "config": info.get("config"),
        })
    # How much more attribute-free opposition the reject side carries than the hire side.
    by_side = {r["anchor_side"]: r["af_error_mean"] for r in out}
    gap = (round(by_side["reject"] - by_side["hire"], 4)
           if by_side["reject"] is not None and by_side["hire"] is not None else None)
    for r in out:
        r["af_error_mean_reject_minus_hire"] = gap if r["anchor_side"] == "all" else None
    return out


# ---- the k=32 cross-check on the anchor-hire side ------------------------------------------

def topup_counts(topup: Path, free: pd.DataFrame, lang: str) -> tuple[pd.DataFrame, dict]:
    """The top-up's attribute-free counts, keyed by the pool's pair_id through the prompt text."""
    top = free_prompts(read(topup / "prompts.parquet"))
    matched = top.merge(free, on="prompt", how="left", suffixes=("_topup", ""))
    if matched["pair_id"].isna().any():
        raise ValueError(f"{int(matched['pair_id'].isna().sum())} top-up attribute-free prompts "
                         "are not the pool's text")
    if (matched["pair_id_topup"] != matched["pair_id"]).any():
        raise ValueError("the top-up keys a prompt to another pair than the pool does")
    samples = read_matching_samples(topup / "raw" / "samples", set(top["prompt"]))
    counts = decision_counts(top, samples, lang)
    return counts, {"pairs": len(counts), "samples": int(counts["n_samples"].sum()),
                    "samples_per_pair": sorted({int(x) for x in counts["n_samples"]}),
                    "_samples": samples}


def repeated_outputs(k8: pd.DataFrame, k32: pd.DataFrame) -> dict:
    """How many of the k=8 outputs on the top-up's prompts reappear verbatim among the same
    prompt's k=32 outputs. At temperature 1 that happens by chance almost never; if it is
    common, the two runs share sampling seeds and the cross-check is not independent."""
    k8 = k8[k8["prompt"].isin(set(k32["prompt"]))]
    seen = set(zip(k32["prompt"], k32["raw_output"]))
    return {"k8_outputs_on_topup_prompts": len(k8),
            "k8_outputs_repeated_verbatim_in_k32": int(sum(
                (p, o) in seen for p, o in zip(k8["prompt"], k8["raw_output"])))}


def topup_check(pairs: pd.DataFrame) -> dict:
    side = pairs[(pairs["anchor"] == "hire") & pairs["k32_n_samples"].notna()]
    both = side[side["student_majority"].isin(SIDES) & side["k32_majority"].isin(SIDES)]
    return {
        "anchor_hire_pairs_with_k32": len(side),
        "k32_majority": {s: int((side["k32_majority"] == s).sum())
                         for s in ("hire", "reject", "tie", "none")},
        "k32_hire_rate_samples": round(float(side["k32_n_hire"].sum()
                                             / (side["k32_n_hire"] + side["k32_n_reject"]).sum()),
                                       4) if len(side) else None,
        "k8_hire_rate_samples_same_pairs": round(float(side["n_hire"].sum()
                                                       / (side["n_hire"]
                                                          + side["n_reject"]).sum()), 4)
        if len(side) else None,
        "k8_vs_k32_majority_agree": int((both["student_majority"] == both["k32_majority"]).sum()),
        "k8_vs_k32_both_decided": len(both),
        "k8_reject_majority_confirmed_by_k32": int(((side["student_majority"] == "reject")
                                                     & (side["k32_majority"] == "reject")).sum()),
        "k8_reject_majority": int((side["student_majority"] == "reject").sum()),
    }


# ---- context ---------------------------------------------------------------------------------

def heldout_candidates(source: Path) -> set[str]:
    ids: set[str] = set()
    for name in V1_VALIDATION_FILES:
        path = source / name
        if path.exists():
            ids |= set(read(path, ["candidate_id"])["candidate_id"].astype(str))
    return ids


def benchmark_overlap(rows: pd.DataFrame, holdout_path: Path) -> dict:
    if not holdout_path.exists():
        return {"holdout_file": str(holdout_path), "checked": False}
    h = json.loads(holdout_path.read_text())
    return {"holdout_file": str(holdout_path.relative_to(REPO_ROOT))
            if holdout_path.is_relative_to(REPO_ROOT) else str(holdout_path),
            "checked": True,
            "pool_candidates_in_benchmark": len(set(rows["candidate_id"].astype(str))
                                                & set(h["candidate_ids"])),
            "pool_jobs_in_benchmark": len(set(rows["job_id"].astype(str)) & set(h["job_ids"]))}


def code_state() -> dict:
    """The commit this runs at, and its uncommitted changes to tracked code or configs."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=no", "--",
                     "scripts", "src", "configs")
        tracked = bool(git("ls-files", "--", "scripts/anchor_agreement.py").strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None, "this_script_tracked": None}
    return {"commit": commit,
            "uncommitted": [line[3:] for line in status.splitlines() if line.strip()],
            "this_script_tracked": tracked}


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---- printing --------------------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_confusion(title: str, c: dict) -> None:
    print(f"\n{title}  ({c['pairs']} pairs)")
    if not c["pairs"]:
        return
    print(f"  {'anchor':8s} {'student hire':>14s} {'student reject':>16s} {'tie':>10s}"
          f" {'none':>6s}")
    for anchor in SIDES:
        row, share = c["counts"][anchor], c["share_of_anchor_side"][anchor]
        cells = [f"{row[s]:5d} ({100 * share[s]:5.1f}%)" if share[s] is not None else
                 f"{row[s]:5d}" for s in ("hire", "reject", "tie")]
        print(f"  {anchor:8s} {cells[0]:>14s} {cells[1]:>16s} {cells[2]:>10s} "
              f"{row['none']:6d}")
    print(f"  anchor hire rate {_pct(c['anchor_hire_rate'])} | student attribute-free "
          f"hire rate {_pct(c['student_hire_rate_samples'])} of samples, "
          f"{_pct(c['student_majority_hire_rate'])} of pairs by majority | "
          f"agreement excl. ties {_pct(c['agreement_excluding_ties'])}, "
          f"kappa {c['cohen_kappa_excluding_ties']}")


def print_training(table: pd.DataFrame) -> None:
    print("\nTraining sets: rows whose anchor (= target) disagrees with the student's "
          "attribute-free majority")
    print(f"  {'set':14s} {'side':6s} {'rows':>6s} {'pairs':>6s} {'disagree':>15s} "
          f"{'tie':>6s} {'->reject':>9s} {'->hire':>7s} {'net->rej':>9s} {'conf>=7/8':>9s} "
          f"{'af err mean':>12s}")
    for r in table.to_dict("records"):
        if not r["rows"]:
            print(f"  {r['training_set']:14s} {r['anchor_side']:6s} {0:6d}")
            continue
        print(f"  {r['training_set']:14s} {r['anchor_side']:6s} {r['rows']:6d} {r['pairs']:6d} "
              f"{r['disagree_rows']:6d} ({100 * r['disagree_share']:5.1f}%) "
              f"{r['student_tie']:6d} {r['pull_toward_reject_rows']:9d} "
              f"{r['pull_toward_hire_rows']:7d} {r['net_toward_reject_rows']:9d} "
              f"{r['confident_disagree_rows']:9d} {r['af_error_mean']:12.4f}")


# ---- main ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="artifacts/semisynthetic-v2/onpolicy-full")
    parser.add_argument("--hire-topup", default="artifacts/semisynthetic-v2/onpolicy-hire32")
    parser.add_argument("--no-hire-topup", action="store_true",
                        help="skip the k=32 cross-check on the anchor-hire pairs")
    parser.add_argument("--sft-source", default="artifacts/semisynthetic-v1",
                        help="v1 artifacts; candidates in its *validation splits were held out "
                             "of the SFT student's training")
    parser.add_argument("--training", action="append", metavar="NAME=CONFIG",
                        help=f"a training set by its training config (repeatable; default: "
                             f"{', '.join(t.split('=')[0] for t in DEFAULT_TRAINING)})")
    parser.add_argument("--holdout", default=str(REPO_ROOT / "data/benchmark/holdout_ids.json"))
    parser.add_argument("--teacher-config", default="configs/generation/teacher.yaml")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--confident", type=float, default=7 / 8)
    parser.add_argument("--out", default="reports/anchor_agreement")
    args = parser.parse_args(argv)

    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    from hiring_bias_mitigation.utils.config import load_yaml, resolve_output_path

    pool = Path(resolve_output_path(args.pool))
    rows = read(pool / "rows.parquet", ["pair_id", "candidate_id", "job_id", "lang",
                                        "reference_decision", "protected_group",
                                        "protected_attr", "condition"])
    rows = rows[rows["lang"] == args.lang].reset_index(drop=True)
    free = free_prompts(read(pool / "prompts.parquet"))
    free = free[free["pair_id"].isin(set(rows["pair_id"].astype(str)))].reset_index(drop=True)
    k8 = read_matching_samples(pool / "raw" / "samples", set(free["prompt"]))
    counts = decision_counts(free, k8, args.lang)
    heldout = heldout_candidates(Path(resolve_output_path(args.sft_source)))
    pairs = pair_table(rows, counts, heldout)

    summary: dict = {
        "question": "Do the training anchors (the v1 teacher's one attribute-free verdict per "
                    "pair) disagree with the SFT student's own attribute-free decisions, and "
                    "in which direction do the training sets' targets pull on those rows?",
        "inputs": {"pool": args.pool, "sft_source": args.sft_source,
                   "hire_topup": None if args.no_hire_topup else args.hire_topup},
        "definitions": {
            "anchor": "rows.parquet reference_decision: the v1 teacher's attribute-free "
                      "verdict, one sample per pair",
            "student_hire_share": "hire samples / parsed samples on the pair's attribute-free "
                                  "prompt (SFT student, k samples at temperature 1)",
            "student_majority": "hire if share > 0.5, reject if < 0.5, tie at 0.5",
            "disagree": "student majority is the opposite verdict of the anchor; ties are "
                        "neither agree nor disagree",
            "af_error_mass": "sum over rows of the share of the student's attribute-free "
                             "samples opposing the row's target (= anchor)",
            "set_definition": "not used: this script works on pool pairs and training rows, "
                              "not on benchmark counterfactual sets",
        },
        "code": code_state(),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    teacher_path = REPO_ROOT / args.teacher_config
    if teacher_path.exists():
        teacher = load_yaml(teacher_path)
        summary["anchor_source"] = {"config": args.teacher_config,
                                    "teacher_model": teacher.get("teacher_model"),
                                    "generation": teacher.get("generation")}
    sample_report = pool / "sample_report.json"
    if sample_report.exists():
        rep = json.loads(sample_report.read_text())
        summary["student"] = {"model": rep.get("student"),
                              "samples_per_prompt": rep.get("samples_per_prompt"),
                              "generation": rep.get("student_generation")}
    generation_report = Path(resolve_output_path(args.sft_source)) / "generation_report.json"
    if generation_report.exists():
        summary["sft_training_data_balance"] = {
            "note": "v1 SFT rows by target verdict before and after balance_decisions, English "
                    "and Ukrainian together (generation_report.json filters.balance)",
            **(json.loads(generation_report.read_text()).get("filters", {}).get("balance")
               or {})}
    summary["samples"] = {"pairs": len(counts), "samples": int(counts["n_samples"].sum()),
                          "samples_per_pair": sorted({int(x) for x in counts["n_samples"]}),
                          "unparsed": int(counts["n_unparsed"].sum())}
    summary["pool"] = confusion(pairs)
    summary["pool"]["student_hire_share_histogram"] = {a: share_histogram(pairs, a)
                                                       for a in SIDES}
    summary["pool"]["confident_opposition"] = confident_opposition(pairs, args.confident)
    summary["pool_by_sft_exposure"] = {
        "heldout_candidates_note": "candidate in a v1 *validation split: held out of the SFT "
                                   "student's training. The others are SFT training "
                                   "candidates; sft_train.parquet is not on disk, so row-level "
                                   "membership is not checked",
        "sft_heldout": confusion(pairs[pairs["sft_heldout"]]),
        "sft_training_candidates": confusion(pairs[~pairs["sft_heldout"]]),
    }
    summary["benchmark"] = {
        **benchmark_overlap(rows, Path(args.holdout)),
        "comparable_to_gpt4o_or_claude_review": False,
        "why": "the pool excludes every benchmark candidate and job, so the anchors cannot "
               "be set beside GPT-4o's reference or the Claude blind review on the same CVs",
    }

    if not args.no_hire_topup:
        topup = Path(resolve_output_path(args.hire_topup))
        k32_counts, k32_info = topup_counts(topup, free, args.lang)
        k32_counts = with_shares(k32_counts.add_prefix("k32_").rename(
            columns={"k32_pair_id": "pair_id"}), "k32_")
        pairs = pairs.merge(k32_counts, on="pair_id", how="left", validate="one_to_one")
        summary["hire_topup_check"] = {
            **{k: v for k, v in k32_info.items() if not k.startswith("_")},
            **repeated_outputs(k8, k32_info["_samples"]),
            **topup_check(pairs),
        }

    training = [*(args.training or DEFAULT_TRAINING)]
    tables = [side_rows(join_training(rows, pairs, "reference_decision", POOL_ROWS),
                        {"training_set": POOL_ROWS, "data_kind": "pool",
                         "train_file": str(pool / "rows.parquet")}, args.confident)]
    for entry in training:
        name, _, config = entry.partition("=")
        frame, info = training_frame(name, REPO_ROOT / config if not Path(config).is_absolute()
                                     else config)
        info["config"] = config
        joined = join_training(frame, pairs, info["target_column"], name)
        tables.append(side_rows(joined, info, args.confident))
        counts_by_pair = frame["pair_id"].astype(str).value_counts()
        pairs[f"rows_in_{name}"] = pairs["pair_id"].map(counts_by_pair).fillna(0).astype(int)
    by_training = pd.DataFrame([r for t in tables for r in t])
    summary["by_training_set"] = by_training.drop(columns=["config", "train_file"]).to_dict(
        "records")

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "pairs.csv", index=False)
    by_training.to_csv(out / "by_training_set.csv", index=False)
    summary["outputs"] = {k: str((out / f).relative_to(REPO_ROOT)) if out.is_relative_to(REPO_ROOT)
                          else str(out / f)
                          for k, f in (("pairs", "pairs.csv"),
                                       ("by_training_set", "by_training_set.csv"),
                                       ("summary", "summary.json"))}
    summary = _json_ready(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"samples: {summary['samples']}")
    print_confusion("Pool: anchor x SFT student's attribute-free majority", summary["pool"])
    for label, key in (("  SFT-held-out candidates", "sft_heldout"),
                       ("  SFT training candidates", "sft_training_candidates")):
        print_confusion(label, summary["pool_by_sft_exposure"][key])
    print("\nStudent hire share on anchor-reject pairs (k hire of n parsed samples):")
    for h in summary["pool"]["student_hire_share_histogram"]["reject"]:
        print(f"  {h['student_hire_samples']:>5s}  {h['pairs']:5d}  "
              f"({100 * h['share_of_side']:5.1f}%)")
    print("Student hire share on anchor-hire pairs:")
    for h in summary["pool"]["student_hire_share_histogram"]["hire"]:
        print(f"  {h['student_hire_samples']:>5s}  {h['pairs']:5d}  "
              f"({100 * h['share_of_side']:5.1f}%)")
    print(f"confident opposition: {summary['pool']['confident_opposition']}")
    if "hire_topup_check" in summary:
        print(f"\nk=32 cross-check on anchor-hire pairs: {summary['hire_topup_check']}")
    print_training(by_training)
    print(f"\nbenchmark: {summary['benchmark']}")
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
