"""v3 additional data: the SFT-v3 pipeline on English pairs beyond V1's (docs/V3_NEW_DATA.md).

    CFG=configs/generation/v3_additional_data.yaml
    python scripts/v3_additional_data.py --config $CFG --stage pool        # CPU
    python scripts/v3_additional_data.py --config $CFG --stage generate --parts rjdb-000 rjdb-001
    python scripts/v3_additional_data.py --config $CFG --stage generate --parts all
    python scripts/v3_additional_data.py --config $CFG --stage report      # CPU
    python scripts/v3_additional_data.py --config $CFG --stage v1_dpo      # CPU
    python scripts/v3_additional_data.py --config $CFG --stage assemble    # CPU

SFT-v3 (docs/SFT_V3_ANCHORS.md) is v1's construction with two changes: the 8-draw vote's
majority as the anchor, and the whole-word `metrics.mentions_attribute` (9e02ae2) in every
filter. Its data sets cover V1's 3,000 English Djinni pairs (`artifacts/semisynthetic-v3/
sft_v3_swap`, `sft_v3_rebalanced`). This script runs the same construction on pairs V1 never
used, so the new rows extend those sets rather than replace them. V1's pairs are not drawn
again: they keep their v3 anchors and targets, and enter here only so new pairs can be checked
against them. Per new pair, in v1's order:

1. reference -- `synth.REFERENCE_PROMPT` drawn 8 times; `generation.vote.vote_anchors`. A new
   pair has no v1 anchor to fall back on, so a tie, no decided draw, or no usable majority
   feedback leaves it unanchored and out of every later pass;
2. variants -- `synth.build_variant_frame`, v1's 3 groups x 2 attributes x 2 conditions, from a
   generator seeded by the part's name, so a part's variants do not depend on other parts;
3. invariant -- `synth.INVARIANT_PROMPT` with the vote anchor quoted, `filters.filter_invariant`;
4. biased -- `synth.BIASED_PROMPT`, `filters.filter_biased`.

Every sampled pass seeds each prompt from its own text (`per_prompt_seed`): base 43 for the
vote, as job 5073453, and 42 for the variant passes, as the vote targets (job 5103710).

`pool` builds the new pairs from three sources and cuts them into parts named by source:
`djinni_new` (candidates V1 never used, one job each, through `pool.match_pairs`), `rjdb`
(train split) and `kaggle_linkedin`. A new pair whose (JD, CV) text repeats a V1 pair's or an
earlier new pair's is dropped, because two copies of one prompt would draw identical samples;
so is a prompt that cannot fit `max_model_len`. `generate` loads the teacher once and runs
every pass for each part it is given; each pass of each part is its own file, written
atomically and skipped when its prompts are already done, so a job cut by the time limit loses
at most the pass in flight. `v1_dpo` builds V1's pairs' DPO rows on the vote anchors from v1's
raw passes (CPU). `assemble` builds v1's SFT and DPO sets from the new kept rows, split by
candidate (a seeded 5% of each source's candidates in validation), and writes:
`additional_{sft,dpo}_*` (new rows alone); `sft_*` (SFT-v3's swap set, rows, order and split
exactly as built, with the new rows after it); `dpo_*` and `kto_*` (V1's pairs and the new
pairs); and `dpo_capped_*`, at most `dpo_cap_per_pair` rows per pool pair, hire and reject
balanced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data import external as X  # noqa: E402
from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import filters as F  # noqa: E402
from hiring_bias_mitigation.generation import pool as PL  # noqa: E402
from hiring_bias_mitigation.generation import sources as S  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.generation import vote as V  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("v3_additional_data")
LANG = "en"
SOURCE_ORDER = ("djinni_new", "rjdb", "kaggle_linkedin")
#: Characters per token, an overestimate for Qwen3.5 on this text (the external eval frames'
#: token calibration found chars/3.5 never undercounts). Used only to drop prompts that could
#: not fit.
CHARS_PER_TOKEN = 3.5
PASSES = ("votes", "invariant", "biased")


def _code() -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout

    return {"commit": git("rev-parse", "--short", "HEAD").strip(),
            "uncommitted": [line for line in git("status", "--porcelain").splitlines() if line]}


def _out(cfg: dict) -> Path:
    return Path(resolve_output_path(cfg["artifacts_dir"]))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _part_seed(part: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{part}".encode()).digest()[:4], "big")


def part_order(parts) -> list[str]:
    return sorted(set(parts), key=lambda p: (SOURCE_ORDER.index(p.rsplit("-", 1)[0]), p))


# ---- pool -----------------------------------------------------------------------------------

def v1_pairs(cfg: dict) -> pd.DataFrame:
    """V1's English pairs: excluded from generation, used for exclusion and dedup only."""
    v1 = pd.read_parquet(resolve_output_path(cfg["pool"]["v1_reference"]))
    return v1.drop(columns=["raw_output"], errors="ignore")


def djinni_new_pairs(cfg: dict, v1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Djinni candidates V1 did not use, each with one job, through v1's matcher."""
    spec = cfg["pool"]["djinni_new"]
    pool_cfg = PL.PoolConfig(lang=LANG, n_pairs=int(spec["n_pairs"]), jobs_per_candidate=1,
                             seed=int(spec["seed"]))
    candidates, jobs = PL.load_djinni(LANG)
    candidates = PL.filter_candidates(candidates, pool_cfg)
    before = len(candidates)
    candidates = candidates[~candidates["id"].astype(str).isin(set(v1["candidate_id"]))]
    pairs = PL.match_pairs(candidates.reset_index(drop=True), PL.filter_jobs(jobs, pool_cfg),
                           pool_cfg)
    report = {"candidates_after_filters": before, "candidates_not_in_v1": len(candidates),
              "pairs": len(pairs)}
    return pairs.assign(source="djinni_new", occupation=pairs["primary_keyword"],
                        cv_variant="original"), report


def external_pairs(cfg: dict) -> tuple[pd.DataFrame, dict]:
    spec = cfg["pool"]["external"]
    pool_cfg = PL.PoolConfig(lang=LANG, seed=int(spec.get("seed", 42)),
                             jobs_per_candidate=int(spec.get("jobs_per_candidate", 2)),
                             sources=tuple(spec["pairs_per_source"]),
                             pairs_per_source={k: int(v) for k, v in
                                               spec["pairs_per_source"].items()},
                             rjdb_include_weakened=bool(spec.get("rjdb_include_weakened", False)))
    return PL.build_pool_with_report(pool_cfg)


def _text_key(frame: pd.DataFrame) -> pd.Series:
    return frame["job_description"].str.strip() + "\x1e" + frame["cv"].str.strip()


def stage_pool(cfg: dict) -> None:
    require_output_root()
    out = _out(cfg) / "pool"
    v1 = v1_pairs(cfg)
    new, new_report = djinni_new_pairs(cfg, v1)
    ext, funnel = external_pairs(cfg)
    pool = pd.concat([f[S.POOL_COLUMNS] for f in (new, ext) if len(f)], ignore_index=True)
    report = {"built": datetime.now(timezone.utc).isoformat(), "code": _code(),
              "config": cfg["_path"], "djinni_new": new_report, "external_funnel": funnel,
              "by_source_built": pool["source"].value_counts().to_dict()}

    # One copy of each (JD, CV) text, and none that repeats a V1 pair's.
    key = _text_key(pool)
    dup = key.duplicated() | key.isin(set(_text_key(v1)))
    report["text_duplicates_dropped"] = pool.loc[dup, "source"].value_counts().to_dict()
    pool = pool[~dup].reset_index(drop=True)
    if pool["pair_id"].duplicated().any() or pool["pair_id"].isin(set(v1["pair_id"])).any():
        raise ValueError("pool: a pair_id repeats within the pool or repeats a V1 pair")
    if pool["candidate_id"].isin(set(v1["candidate_id"])).any():
        raise ValueError("pool: a V1 candidate reached the new pool")

    tokens = pd.Series([len(p) / CHARS_PER_TOKEN for p in SY.reference_prompts(pool, LANG)])
    # The invariant prompt is the longest pass: profile + the quoted anchor + instructions.
    over = tokens + int(cfg["pool"]["invariant_overhead_tokens"]) > int(
        cfg["pool"]["max_prompt_tokens"])
    report["over_length_dropped"] = pool.loc[over, "source"].value_counts().to_dict()
    pool = pool[~over].assign(est_prompt_tokens=tokens[~over].round().astype(int))
    pool = pool.reset_index(drop=True)

    assert_no_leakage(pool, "v3_additional_data.pool")
    # The external builders already drop held-out text; this also covers the Djinni rows.
    leaks = X.find_external_leaks(pool)
    report["external_holdout_dropped"] = (pool.loc[leaks["row"].unique(), "source"]
                                          .value_counts().to_dict() if len(leaks) else {})
    pool = pool.drop(index=leaks["row"].unique()).reset_index(drop=True)
    X.assert_no_external_leakage(pool, "v3_additional_data.pool")

    size = int(cfg["generate"]["part_size"])
    pool["order"] = pool.groupby("source").cumcount()
    pool["part"] = [f"{s}-{o // size:03d}" for s, o in zip(pool["source"], pool["order"])]
    _write_parquet(pool, out / "pairs_en.parquet")
    report["by_source"] = pool["source"].value_counts().to_dict()
    report["parts"] = {p: int((pool["part"] == p).sum()) for p in part_order(pool["part"])}
    report["est_prompt_tokens"] = {s: g["est_prompt_tokens"].describe().round(0).to_dict()
                                   for s, g in pool.groupby("source")}
    report["cv_chars_median"] = {s: int(g["cv"].str.len().median())
                                 for s, g in pool.groupby("source")}
    report["jd_chars_median"] = {s: int(g["job_description"].str.len().median())
                                 for s, g in pool.groupby("source")}
    _write_json(out / "pool_report.json", report)
    log.info("pool: %d pairs in %d parts: %s", len(pool), pool["part"].nunique(),
             report["by_source"])


# ---- generate -------------------------------------------------------------------------------

def _paths(cfg: dict, part: str, name: str) -> tuple[Path, Path]:
    base = _out(cfg) / "parts" / part
    return base / f"{name}.parquet", base / f"{name}.meta.json"


def _done(cfg: dict, part: str, name: str, prompts: list[str]) -> bool:
    path, meta = _paths(cfg, part, name)
    return (path.exists() and meta.exists()
            and json.loads(meta.read_text())["prompts_sha256"] == V.prompts_sha256(prompts))


def part_anchors(cfg: dict, part: str, pairs: pd.DataFrame) -> pd.DataFrame:
    draws = pd.read_parquet(_paths(cfg, part, "votes")[0])
    anchors = V.vote_anchors(draws, pairs["pair_id"], LANG)
    return pairs.merge(anchors, on="pair_id", validate="one_to_one")


def part_variants(cfg: dict, part: str, anchored: pd.DataFrame) -> pd.DataFrame:
    synth_cfg = SY.SynthConfig(lang=LANG, protected_groups=tuple(cfg["protected_groups"]),
                               conditions=tuple(cfg["conditions"]),
                               attributes_per_pair=int(cfg["attributes_per_pair"]),
                               seed=_part_seed(part, int(cfg["seed"])))
    return SY.build_variant_frame(anchored, synth_cfg)


def _generate_pass(cfg, backend, part, name, frame, prompts, generation, sampled_n=None):
    from hiring_bias_mitigation.eval import backends as BK

    start = time.time()
    if sampled_n:
        draws = backend.generate_samples(prompts, sampled_n)
        out = V.draw_frame(frame["pair_id"].tolist(), draws, LANG)
    else:
        out = frame.assign(raw_output=backend.generate(prompts))
    seeds = [BK.per_prompt_seed(int(generation["seed"]), p) for p in prompts]
    out["request_seed"] = np.repeat(seeds, sampled_n) if sampled_n else seeds
    path, meta = _paths(cfg, part, name)
    _write_parquet(out, path)
    _write_json(meta, {
        "part": part, "pass": name, "prompts": len(prompts), "rows": len(out),
        "prompts_sha256": V.prompts_sha256(prompts), "teacher": cfg["teacher_model"],
        "serving": cfg["serving"], "generation": generation,
        "seeding": "per_prompt_seed(generation.seed, prompt)",
        "generation_seconds": round(time.time() - start, 1), "code": _code(),
        "written": datetime.now(timezone.utc).isoformat()})
    log.info("%s %s: %d prompts in %.0f s", part, name, len(prompts), time.time() - start)


def stage_generate(cfg: dict, parts: list[str], passes: list[str]) -> None:
    from hiring_bias_mitigation.eval import backends as BK

    require_output_root()
    pool = pd.read_parquet(_out(cfg) / "pool" / "pairs_en.parquet")
    known = part_order(pool["part"])
    wanted = known if parts == ["all"] else parts
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; the pool has {known[:5]}...")

    engine: dict[str, object] = {}

    def backend_for(name: str):
        # One engine for the job; each pass sets its own decoding before it runs.
        generation = dict(cfg["passes"][name])
        if "llm" not in engine:
            engine["llm"] = BK.build_backend({**cfg["serving"], "model": cfg["teacher_model"],
                                              "generation": generation})
        engine["llm"].generation = BK.GenerationConfig(**generation)
        return engine["llm"], generation

    try:
        # Explicit parts run in the order given, so a job can put each source's first part
        # first and have it read against the stop conditions while the rest runs.
        for part in (known if parts == ["all"] else list(dict.fromkeys(wanted))):
            pairs = pool[pool["part"] == part].reset_index(drop=True)
            if "votes" in passes:
                prompts = SY.reference_prompts(pairs, LANG)
                if _done(cfg, part, "votes", prompts):
                    log.info("%s votes: already drawn for these prompts, skipped", part)
                else:
                    backend, generation = backend_for("votes")
                    _generate_pass(cfg, backend, part, "votes", pairs, prompts, generation,
                                   sampled_n=int(cfg["vote_draws"]))
            if not _paths(cfg, part, "votes")[0].exists():
                log.warning("%s: no votes yet; the variant passes are skipped", part)
                continue
            anchored = part_anchors(cfg, part, pairs)
            variants = part_variants(cfg, part, anchored[anchored["anchored"]]
                                     .reset_index(drop=True))
            for name, builder in (("invariant", SY.invariant_prompts),
                                  ("biased", SY.biased_prompts)):
                if name not in passes:
                    continue
                prompts = builder(variants, LANG)
                if _done(cfg, part, name, prompts):
                    log.info("%s %s: already generated for these prompts, skipped", part, name)
                    continue
                backend, generation = backend_for(name)
                _generate_pass(cfg, backend, part, name, variants, prompts, generation)
    finally:
        if "llm" in engine:
            engine["llm"].close()


# ---- report and assemble --------------------------------------------------------------------

def _voted_parts(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    pool = pd.read_parquet(_out(cfg) / "pool" / "pairs_en.parquet")
    return pool, [p for p in part_order(pool["part"]) if _paths(cfg, p, "votes")[0].exists()]


def _filtered(cfg: dict, done: list[str]) -> tuple[dict, dict]:
    """Kept rows and filter reports of every variant pass on disk, by pass."""
    kept, reports = {"invariant": [], "biased": []}, {}
    for part in done:
        for name, filt in (("invariant", F.filter_invariant), ("biased", F.filter_biased)):
            path = _paths(cfg, part, name)[0]
            if not path.exists():
                continue
            raw = pd.read_parquet(path)
            rows, reports[f"{part}/{name}"] = filt(raw["raw_output"].tolist(),
                                                   raw.drop(columns=["raw_output"]), LANG)
            kept[name].append(rows)
    return kept, reports


def stage_report(cfg: dict) -> None:
    pool, done = _voted_parts(cfg)
    if not done:
        raise SystemExit("report: no part has votes yet")
    anchors = pd.concat([part_anchors(cfg, p, pool[pool["part"] == p].reset_index(drop=True))
                         for p in done], ignore_index=True)
    draws = pd.concat([pd.read_parquet(_paths(cfg, p, "votes")[0]) for p in done],
                      ignore_index=True)
    _, reports = _filtered(cfg, done)
    by_source: dict[str, dict] = {}
    for key, rep in reports.items():
        part, name = key.split("/")
        agg = by_source.setdefault(part.rsplit("-", 1)[0], {}).setdefault(
            name, {"n_input": 0, "n_kept": 0, "dropped_by_reason": {}})
        agg["n_input"] += rep["n_input"]
        agg["n_kept"] += rep["n_kept"]
        for reason, n in rep["dropped_by_reason"].items():
            agg["dropped_by_reason"][reason] = agg["dropped_by_reason"].get(reason, 0) + n
    for passes in by_source.values():
        for agg in passes.values():
            agg["yield"] = round(agg["n_kept"] / agg["n_input"], 4) if agg["n_input"] else None
    seconds = {}
    for p in done:
        for name in PASSES:
            meta = _paths(cfg, p, name)[1]
            if meta.exists():
                m = json.loads(meta.read_text())
                seconds.setdefault(name, []).append(
                    {"part": p, "prompts": m["prompts"], "seconds": m["generation_seconds"]})
    _write_parquet(anchors, _out(cfg) / "anchors_en.parquet")
    _write_json(_out(cfg) / "report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(), "parts_voted": done,
        "vote": V.vote_report(anchors, draws, by=anchors["source"]),
        "filters_by_source": by_source, "generation_seconds": seconds})
    log.info("report: %d pairs voted over %d parts", len(anchors), len(done))


def validation_candidates(cfg: dict, frame: pd.DataFrame) -> set[str]:
    """A seeded `val_fraction` of each source's candidates (none of them is a V1 candidate)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    val: set[str] = set()
    for source in SOURCE_ORDER:
        ids = np.array(sorted(frame.loc[frame["source"] == source, "candidate_id"].unique()))
        if len(ids):
            n = max(1, int(len(ids) * float(cfg["split"]["val_fraction"])))
            val |= set(rng.permutation(ids)[:n].tolist())
    return val


def _decision(completions: pd.Series) -> pd.Series:
    return completions.map(lambda c: json.loads(c)["decision"])


def with_anchor(dpo: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """DPO rows with the chosen decision and their pair's vote, so a view can filter on them."""
    cols = anchors[["pair_id", "hire_votes", "unstable"]].rename(
        columns={"hire_votes": "anchor_hire_votes", "unstable": "anchor_unstable"})
    out = dpo.merge(cols, on="pair_id", how="left", validate="many_to_one")
    if out["anchor_hire_votes"].isna().any():
        raise ValueError("a DPO row has no anchor")
    return out.assign(decision=_decision(out["chosen"]))


def capped_balanced(dpo: pd.DataFrame, cap: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """At most `cap` rows per pool pair, then as many hire as reject rows.

    A pair's 12 variants share one CV and one JD, so its DPO rows are near-duplicates of each
    other; the cap spreads the set over more pairs. Within a pair, rows of different protected
    groups are taken first. The larger decision side is then drawn down at random to the
    smaller, as `filters.balance_decisions` does for SFT.
    """
    rng = np.random.default_rng(seed)
    shuffled = dpo.iloc[rng.permutation(len(dpo))].copy()
    shuffled["_group_rank"] = shuffled.groupby(["pair_id", "protected_group"]).cumcount()
    shuffled = shuffled.sort_values(["pair_id", "_group_rank"], kind="stable")
    capped = shuffled.groupby("pair_id").head(cap).drop(columns="_group_rank")
    sides = capped.groupby("decision")
    n = int(sides.size().min()) if capped["decision"].nunique() == 2 else 0
    out = (pd.concat([g.iloc[rng.permutation(len(g))[:n]] for _, g in sides], ignore_index=True)
           if n else capped.iloc[:0])
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return out, {"rows_before": len(dpo), "rows_after_cap": len(capped),
                 "by_decision_after_cap": capped["decision"].value_counts().to_dict(),
                 "per_side": n, "rows": len(out), "pairs": int(out["pair_id"].nunique())}


def stage_v1_dpo(cfg: dict) -> None:
    """V1's pairs' DPO rows on the vote anchors, from v1's raw passes; CPU only.

    `synth.BIASED_PROMPT` does not quote the anchor, so v1's biased generations stand for every
    V1 pair, including the 192 whose verdict the vote changed; only `filter_biased`'s flip check
    reads the anchor, and it is re-run against the vote's. The chosen side is SFT-v3's: v1's
    invariant targets on unchanged pairs and job 5103710's regenerated targets on changed ones.
    Both sides go through the current filters, with the whole-word attribute matcher. The split
    is SFT-v3's: a candidate in `sft_v3_swap`'s validation is in validation here.
    """
    require_output_root()
    spec = cfg["v1_dpo"]
    anchors = pd.read_parquet(resolve_output_path(spec["vote_anchors"]))
    changed = set(anchors.loc[anchors["changed"], "pair_id"])
    keep = set(anchors["pair_id"])
    inv = pd.read_parquet(resolve_output_path(spec["v1_invariant"]))
    inv = inv[inv["pair_id"].isin(keep) & ~inv["pair_id"].isin(changed)]
    regen = pd.read_parquet(resolve_output_path(spec["vote_targets"]))
    regen = regen[regen["pair_id"].isin(changed)].drop(columns=["v1_reference_decision", "seed"])
    chosen_raw = pd.concat([inv, regen[inv.columns]], ignore_index=True)
    invariant, rep_inv = F.filter_invariant(chosen_raw["raw_output"].tolist(),
                                            chosen_raw.drop(columns=["raw_output"]), LANG)
    by_pair = anchors.set_index("pair_id")
    biased_raw = pd.read_parquet(resolve_output_path(spec["v1_biased"]))
    biased_raw = biased_raw[biased_raw["pair_id"].isin(keep)].copy()
    biased_raw["reference_decision"] = biased_raw["pair_id"].map(by_pair["new_decision"])
    biased_raw["reference_feedback"] = biased_raw["pair_id"].map(by_pair["new_feedback"])
    biased, rep_b = F.filter_biased(biased_raw["raw_output"].tolist(),
                                    biased_raw.drop(columns=["raw_output"]), LANG)
    dpo = with_anchor(D.build_dpo_dataset(invariant, biased), anchors)
    dpo["source"] = "djinni_v1"
    assert_no_leakage(dpo, "v3_additional_data.v1_dpo")
    base = Path(resolve_output_path(cfg["assemble"]["sft_v3_dir"]))
    val = set(pd.read_parquet(base / "sft_validation.parquet", columns=["candidate_id"])
              ["candidate_id"])
    is_val = dpo["candidate_id"].isin(val)
    out = _out(cfg)
    _write_parquet(dpo[~is_val].reset_index(drop=True), out / "v1_dpo_train.parquet")
    _write_parquet(dpo[is_val].reset_index(drop=True), out / "v1_dpo_validation.parquet")
    _write_json(out / "v1_dpo_report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(), "inputs": spec,
        "pairs": len(keep), "changed_pairs": len(changed), "invariant": rep_inv,
        "biased": rep_b, "rows": len(dpo), "train": int((~is_val).sum()),
        "validation": int(is_val.sum()),
        "by_decision": dpo["decision"].value_counts().to_dict(),
        "rejected_flipped_share": float(dpo["rejected_flipped_decision"].mean()),
        "rejected_mentions_attribute_share": float(dpo["rejected_mentions_attribute"].mean()),
        "rows_on_unstable_anchors": int(dpo["anchor_unstable"].sum())})
    log.info("v1_dpo: %d rows (%s)", len(dpo), dpo["decision"].value_counts().to_dict())


def stage_assemble(cfg: dict) -> None:
    require_output_root()
    pool, done = _voted_parts(cfg)
    kept, reports = _filtered(cfg, done)
    if not kept["invariant"]:
        raise SystemExit("assemble: no invariant pass yet")
    invariant = pd.concat(kept["invariant"], ignore_index=True)
    biased = pd.concat(kept["biased"], ignore_index=True) if kept["biased"] else pd.DataFrame()
    anchors = pd.concat([part_anchors(cfg, p, pool[pool["part"] == p].reset_index(drop=True))
                         for p in done], ignore_index=True)

    sft, balance = F.balance_decisions(D.build_sft_dataset(invariant), seed=int(cfg["seed"]),
                                       column="decision")
    dpo = D.build_dpo_dataset(invariant, biased)
    dpo = with_anchor(dpo, anchors) if len(dpo) else dpo
    val = validation_candidates(cfg, pool)
    out = _out(cfg)
    base = Path(resolve_output_path(cfg["assemble"]["sft_v3_dir"]))
    splits: dict[str, dict] = {}
    new: dict[str, dict[str, pd.DataFrame]] = {}
    for name, frame in (("sft", sft), ("dpo", dpo)):
        if frame.empty:
            continue
        # The pool passed the external text holdout; the ids are re-checked here.
        assert_no_leakage(frame, f"v3_additional_data.assemble.{name}")
        if not frame["pair_id"].isin(set(pool["pair_id"])).all():
            raise ValueError(f"assemble: {name} has a row outside the pool")
        is_val = frame["candidate_id"].isin(val)
        new[name] = {"train": frame[~is_val].reset_index(drop=True),
                     "validation": frame[is_val].reset_index(drop=True)}
        for split, rows in new[name].items():
            _write_parquet(rows, out / f"additional_{name}_{split}.parquet")
        splits[f"additional_{name}"] = {s: len(r) for s, r in new[name].items()}

    for split, rows in new.get("sft", {}).items():
        # SFT-v3's swap set, unchanged, with the new rows after it.
        v3 = pd.read_parquet(base / f"sft_{split}.parquet")
        if set(v3["candidate_id"]) & set(rows["candidate_id"]):
            raise ValueError(f"assemble: a new candidate is already in SFT-v3 {split}")
        combined = pd.concat([v3.assign(source=v3.get("source", "djinni")), rows],
                             ignore_index=True)
        _write_parquet(combined, out / f"sft_{split}.parquet")
        splits[f"sft_{split}"] = {
            "rows": len(combined), "sft_v3_rows": len(v3), "new_rows": len(rows),
            "by_source_and_decision": {f"{s} {d}": int(n) for (s, d), n in
                                       combined.groupby(["source", "decision"]).size().items()}}

    # DPO and KTO: V1's pairs on the vote anchors (stage v1_dpo) and the new pairs, together.
    cap = int(cfg["assemble"]["dpo_cap_per_pair"])
    for split in ("train", "validation"):
        v1_path = out / f"v1_dpo_{split}.parquet"
        frames = [pd.read_parquet(v1_path)] if v1_path.exists() else []
        if "dpo" in new:
            frames.append(new["dpo"][split])
        if not frames:
            continue
        both = pd.concat(frames, ignore_index=True)
        _write_parquet(both, out / f"dpo_{split}.parquet")
        _write_parquet(D.build_kto_dataset(both), out / f"kto_{split}.parquet")
        target, target_report = capped_balanced(both, cap, int(cfg["seed"]))
        _write_parquet(target, out / f"dpo_capped_{split}.parquet")
        splits[f"dpo_{split}"] = {
            "rows": len(both), "includes_v1_pairs": v1_path.exists(),
            "by_source_and_decision": {f"{s} {d}": int(n) for (s, d), n in
                                       both.groupby(["source", "decision"]).size().items()},
            "capped_balanced": target_report}
    _write_json(out / "generation_report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(),
        "teacher_model": cfg["teacher_model"], "parts": done, "sft_v3_dir": str(base),
        "balance_new_sft_rows": balance, "dpo_cap_per_pair": cap, "splits": splits,
        "filters": reports, "validation_candidates": len(val)})
    log.info("assemble: %s", {k: v for k, v in splits.items() if not k.startswith("dpo_")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["pool", "generate", "report", "v1_dpo", "assemble"])
    parser.add_argument("--parts", nargs="+", default=["all"])
    parser.add_argument("--passes", nargs="+", default=list(PASSES), choices=PASSES)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_path"] = args.config
    {"pool": lambda: stage_pool(cfg),
     "generate": lambda: stage_generate(cfg, args.parts, args.passes),
     "report": lambda: stage_report(cfg),
     "v1_dpo": lambda: stage_v1_dpo(cfg),
     "assemble": lambda: stage_assemble(cfg)}[args.stage]()


if __name__ == "__main__":
    main()
