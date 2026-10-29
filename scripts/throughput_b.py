"""Part B of the GTC poster: the counterfactual audit as an inference workload, timed with and
without vLLM's prefix cache and with prompts submitted in prefix order, as built or shuffled
(docs/THROUGHPUT_B.md).

Round 1 (`--round b`, job 5106727): the base model on the first 225 pairs, prefix cache on/off x
prefix order/shuffled, one condition per GPU in parallel (scripts/cscs/throughput_b.sbatch):

    python scripts/throughput_b.py run --repeat r1 \
        --config configs/throughput/qwen3.5-9b_en_cacheon-prefix.yaml
    python scripts/throughput_b.py compare     # are the greedy outputs the same across conditions?
    python scripts/throughput_b.py speedup     # 1 - t_new / t_ref, after scripts/cost_report.py
    python scripts/throughput_b.py prefix --config <cfg>   # how much prefix the prompts share

Round 2 (`--round full`): the whole audit of DPO_V3_4880 (450 pairs) on the engine every audit
used, against prefix order and round 1's naive reference, as a 4 x 4 Latin square: each GPU
runs every condition once, one after another, each condition once in each position
(scripts/cscs/throughput_b_full.sbatch):

    python scripts/throughput_b.py schedule --round full --seed 2027   # GPU, position, condition
    python scripts/throughput_b.py run --warmup --repeat r0 --config configs/throughput/full/<cfg>
    python scripts/throughput_b.py compare --round full
    python scripts/throughput_b.py speedup --round full
    python scripts/throughput_b.py quality --round full   # the audit's measures on every run

`run` is one timing run. It clears the run's chunk cache, so every prompt is generated and
timed, generates through the same `runner.run_audit` an audit uses, and writes
<round raw_dir>/<run>.parquet. It never scores, so nothing reaches eval/results/ or
reports/reference_metrics.csv. Wrapped in `cost eval <label>` (scripts/cscs/env.sh), its wall
time, GPU, energy, requests, tokens, load and generate seconds and vLLM's prefix-cache counters
land in one cost record; scripts/cost_report.py puts them in reports/cost/throughput.csv.
`--warmup` generates a few pairs and keeps nothing: it fills the node's file and compile caches
so that the first timed run on a GPU does not pay for them alone.

`compare` lines up every timing run of a round, and the round's production audit, on the same
rows (pair, group, attribute, condition) and counts byte-identical responses and identical
decisions -> <report dir>/identity.csv, with the differing rows of each pair of runs, one row
per response, beside the outputs (round 1: every differing text, in
$HBM_OUTPUT_ROOT/outputs/throughput/differences.csv; round 2: differing decisions only, in
outputs/throughput/full/differences.csv, since cache-off runs differ in text on a fifth of the
prompts).

`speedup` reads reports/cost/throughput.csv and, within each job and repeat, sets every
condition against every other: 1 - t_new / t_ref on `generate_s` (steady-state generation) and
on `wall_s` (the whole step, model load included), the ratio, the prefix-cache hit rate ->
<report dir>/speedup.csv, and their median and range over the repeats -> speedup_summary.csv.
A repeat is one round-1 repeat, or one GPU of round 2 (its four runs one after another).

`quality` scores every timing run of a round, and the production audit, with the audit's own
measures (`eval.audit.prepare_frame`, `eval.metrics.summarise_run`,
`analysis.reference_metrics`): hire rate, agreement with GPT-4o, TPR/FPR, attribute flips and
unstable within-group sets on the expected grid -> <report dir>/quality.csv, and their median
and range per condition -> quality_summary.csv. CPU only, and it writes nothing under
eval/results/.

`prefix` tokenizes a config's prompts as the backend sends them (chat template included) and
measures, within each counterfactual set, pair and job, the token prefix all members share and
the share of prompt tokens a prefix cache could skip: token by token, and in whole blocks of
`--block` tokens, the granularity vLLM can resume a hybrid model from (528 for Qwen3.5 under
vLLM 0.31) -> reports/throughput_b/shared_prefix.csv. CPU only.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval import runner as R  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_config,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("throughput_b")

REPORT_DIR = REPO_ROOT / "reports" / "throughput_b"
#: The poster's reference in both rounds is the slowest setup on purpose.
REFERENCE = "cacheoff-shuffle"
ROW_KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
#: Pairs a warm-up run generates; its responses are discarded.
WARMUP_PAIRS = 4
WARMUP = "warmup"


class Round(NamedTuple):
    raw_dir: str        # where its timing runs write, never outputs/raw/
    label_prefix: str   # cost label of a run: <label_prefix><condition>
    production: str     # the audit run its outputs are checked against
    report_dir: Path
    differences: str    # rows written to differences.csv: "text" or "decision" differences
    conditions: tuple[str, ...]
    config: str         # config path, by condition


ROUNDS = {
    "b": Round(
        raw_dir="outputs/throughput/raw", label_prefix="throughput-",
        production="Qwen3.5-9B--en--baseline--cscs", report_dir=REPORT_DIR, differences="text",
        conditions=("cacheon-prefix", "cacheon-shuffle", "cacheoff-prefix", "cacheoff-shuffle"),
        config="configs/throughput/qwen3.5-9b_en_{condition}.yaml",
    ),
    "full": Round(
        raw_dir="outputs/throughput/full/raw", label_prefix="throughput-full-",
        production="Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305",
        report_dir=REPORT_DIR / "full", differences="decision",
        conditions=("cacheon-asbuilt-nostats", "cacheon-asbuilt", "cacheon-prefix",
                    "cacheoff-shuffle"),
        config="configs/throughput/full/qwen3.5-9b_en_dpo_v3_4880_{condition}.yaml",
    ),
}


def round_of(label: str) -> str | None:
    """The round a cost label belongs to: the one with the longest matching prefix."""
    matching = [k for k, r in ROUNDS.items() if str(label).startswith(r.label_prefix)]
    return max(matching, key=lambda k: len(ROUNDS[k].label_prefix), default=None)


def run(config: str, repeat: str, warmup: bool = False) -> None:
    cfg = load_config(config)
    raw_dir = cfg.get("raw_dir")
    allowed = sorted(r.raw_dir for r in ROUNDS.values())
    if raw_dir not in allowed:
        raise SystemExit(f"{config}: a timing run writes to raw_dir: one of {allowed}, never "
                         "next to the audits in outputs/raw/")
    if not re.fullmatch(r"r\d+", repeat):
        raise SystemExit(f"--repeat must look like r1, r2...; got {repeat!r}")
    cfg["run_suffix"] = f"{cfg['run_suffix']}-{repeat}"
    if warmup:
        cfg["limit_pairs"] = WARMUP_PAIRS
        cfg["run_suffix"] += f"-{WARMUP}"
    run_name = R.build_run_name(cfg)
    # Every prompt must be generated in this process: a chunk cached by an earlier attempt of
    # the same run would be skipped and its time never measured.
    R._drop_partial(run_name)
    parsed, meta = R.run_audit(cfg)
    if warmup:
        R._drop_partial(run_name)
        log.info("%s: warm-up, %d prompts, wall %.1f s, nothing kept", run_name,
                 meta["n_prompts"], meta["wall_seconds"])
        return
    R.save_raw(parsed, meta, raw_dir)
    log.info("%s: %d prompts, wall %.1f s", run_name, meta["n_prompts"], meta["wall_seconds"])


def latin_square(n: int, seed: int) -> list[list[int]]:
    """A seeded random n x n Latin square: the cyclic square with its rows, columns and symbols
    permuted. Row = GPU, column = position in that GPU's sequence, entry = condition."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows, cols, symbols = rng.permutation(n), rng.permutation(n), rng.permutation(n)
    return [[int(symbols[(rows[i] + cols[j]) % n]) for j in range(n)] for i in range(n)]


def schedule(round_name: str, seed: int) -> list[tuple[int, str, int, str, str]]:
    """(GPU, repeat, position, condition, config) for every timed run of a round."""
    rnd = ROUNDS[round_name]
    square = latin_square(len(rnd.conditions), seed)
    return [(gpu, f"r{gpu + 1}", position, rnd.conditions[c],
             rnd.config.format(condition=rnd.conditions[c]))
            for gpu, row in enumerate(square) for position, c in enumerate(row)]


def _runs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    runs = {}
    for path in sorted(raw_dir.glob("*--throughput-*.parquet")):
        runs[path.stem] = pd.read_parquet(path, columns=[*ROW_KEY, "raw_output", "decision"])
    return runs


def compare(round_name: str = "b") -> None:
    rnd = ROUNDS[round_name]
    raw_dir = Path(resolve_output_path(rnd.raw_dir))
    runs = _runs(raw_dir)
    if not runs:
        raise SystemExit(f"no timing runs under {raw_dir}")
    production = Path(resolve_output_path("outputs/raw")) / f"{rnd.production}.parquet"
    if production.is_file():
        runs[rnd.production] = pd.read_parquet(
            production, columns=[*ROW_KEY, "raw_output", "decision"]
        )
    rows, differences = [], []
    for a, b in itertools.combinations(runs, 2):
        both = runs[a].merge(runs[b], on=ROW_KEY, suffixes=("_a", "_b"), validate="1:1")
        same_text = both["raw_output_a"] == both["raw_output_b"]
        same_decision = both["decision_a"].fillna("") == both["decision_b"].fillna("")
        rows.append({
            "run_a": a, "run_b": b, "rows": len(both),
            "same_text": int(same_text.sum()), "same_decision": int(same_decision.sum()),
            "same_text_share": round(float(same_text.mean()), 5),
            "same_decision_share": round(float(same_decision.mean()), 5),
        })
        differs = ~same_text if rnd.differences == "text" else ~same_decision
        differences.append(both.loc[differs].assign(run_a=a, run_b=b))
    rnd.report_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(rnd.report_dir / "identity.csv", index=False)
    pd.concat(differences, ignore_index=True).to_csv(raw_dir.parent / "differences.csv",
                                                     index=False)
    print(table.to_string(index=False))


def _repeat(run_name: str) -> str:
    found = re.search(r"-(r\d+)$", str(run_name))
    return found.group(1) if found else ""


def speedup(throughput_csv: Path, round_name: str = "b") -> None:
    rnd = ROUNDS[round_name]
    steps = pd.read_csv(throughput_csv, dtype={"job_id": str})
    steps = steps[steps["label"].map(round_of) == round_name].copy()
    steps["condition"] = steps["label"].str[len(rnd.label_prefix):]
    steps = steps[steps["condition"] != WARMUP]
    if steps.empty:
        raise SystemExit(f"no {rnd.label_prefix}* steps in {throughput_csv}; run "
                         "scripts/cost_report.py after the timing job")
    # Energy is per step, model load included; steps.csv sits beside throughput.csv.
    energy = pd.read_csv(throughput_csv.parent / "steps.csv", dtype={"job_id": str})
    # Only this round's steps: other jobs repeat (job, label) with no run name.
    energy = energy[energy["label"].map(round_of) == round_name]
    steps = steps.merge(energy[["job_id", "label", "run_name", "gpu_energy_kwh"]],
                        on=["job_id", "label", "run_name"], how="left", validate="1:1")
    steps["repeat"] = steps["run_name"].map(_repeat)
    rows = []
    for (job, repeat), group in steps.groupby(["job_id", "repeat"], sort=True):
        if group["condition"].duplicated().any():
            log.warning("job %s %s: a condition appears twice; skipped", job, repeat)
            continue
        by = group.set_index("condition")
        # Every condition against every other one timed in the same job and repeat: the same
        # node and prompts; in round 1 at the same time, in round 2 on the same GPU.
        for new, ref in itertools.permutations(by.index, 2):
            s, r = by.loc[new], by.loc[ref]
            rows.append({
                "job_id": job, "repeat": repeat, "condition": new, "ref_condition": ref,
                "ref_job_id": job, "gpu_model": s["gpu_model"],
                "same_gpu_model": s["gpu_model"] == r["gpu_model"],
                "same_prompts": s["requests"] == r["requests"]
                and s["prompt_tokens"] == r["prompt_tokens"],
                "requests": s["requests"], "prompt_tokens": s["prompt_tokens"],
                "output_tokens": s["output_tokens"],
                "generate_s": s["generate_s"], "ref_generate_s": r["generate_s"],
                "time_saved_share": round(1 - s["generate_s"] / r["generate_s"], 4),
                "speedup_x": round(r["generate_s"] / s["generate_s"], 3),
                "load_s": s["load_s"], "wall_s": s["wall_s"], "ref_wall_s": r["wall_s"],
                "time_saved_wall_share": round(1 - s["wall_s"] / r["wall_s"], 4),
                "requests_per_s": s["requests_per_s"],
                "output_tokens_per_s": s["output_tokens_per_s"],
                "gpu_s_per_1k_requests": s["gpu_s_per_1k_requests"],
                "gpu_s_per_1k_requests_generate_only": s["gpu_s_per_1k_requests_generate_only"],
                "prefix_cache_hits": s.get("prefix_cache_hits"),
                "prefix_cache_hit_rate": s.get("prefix_cache_hit_rate"),
                "gpu_energy_kwh": s["gpu_energy_kwh"],
            })
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no job and repeat with two conditions to compare")
    # docs/COST.md: each condition at least twice, reported as median and range.
    summary = table.groupby(["condition", "ref_condition"], sort=True).agg(
        repeats=("repeat", "size"),
        jobs=("job_id", lambda j: " ".join(sorted(set(j)))),
        generate_s_median=("generate_s", "median"),
        generate_s_min=("generate_s", "min"), generate_s_max=("generate_s", "max"),
        ref_generate_s_median=("ref_generate_s", "median"),
        time_saved_median=("time_saved_share", "median"),
        time_saved_min=("time_saved_share", "min"),
        time_saved_max=("time_saved_share", "max"),
        speedup_x_median=("speedup_x", "median"),
        load_s_median=("load_s", "median"),
        load_s_min=("load_s", "min"), load_s_max=("load_s", "max"),
        wall_s_median=("wall_s", "median"),
        wall_s_min=("wall_s", "min"), wall_s_max=("wall_s", "max"),
        time_saved_wall_median=("time_saved_wall_share", "median"),
        time_saved_wall_min=("time_saved_wall_share", "min"),
        time_saved_wall_max=("time_saved_wall_share", "max"),
        requests_median=("requests", "median"),
        prompt_tokens_median=("prompt_tokens", "median"),
        output_tokens_median=("output_tokens", "median"),
        requests_per_s_median=("requests_per_s", "median"),
        output_tokens_per_s_median=("output_tokens_per_s", "median"),
        gpu_s_per_1k_requests_median=("gpu_s_per_1k_requests", "median"),
        gpu_s_per_1k_requests_generate_only_median=(
            "gpu_s_per_1k_requests_generate_only", "median"),
        prefix_cache_hits_median=("prefix_cache_hits", "median"),
        prefix_cache_hit_rate_median=("prefix_cache_hit_rate", "median"),
        gpu_energy_kwh_median=("gpu_energy_kwh", "median"),
    ).round(4).reset_index()
    rnd.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(rnd.report_dir / "speedup.csv", index=False)
    summary.to_csv(rnd.report_dir / "speedup_summary.csv", index=False)
    print(summary.to_string(index=False))


def _pct(value) -> float | None:
    return None if value is None or value != value else round(100.0 * float(value), 4)


def quality(round_name: str = "full") -> None:
    from hiring_bias_mitigation.analysis import reference_metrics as RM
    from hiring_bias_mitigation.eval import metrics as M
    from hiring_bias_mitigation.eval.audit import prepare_frame

    rnd = ROUNDS[round_name]
    paths = sorted(Path(resolve_output_path(rnd.raw_dir)).glob("*--throughput-*.parquet"))
    production = Path(resolve_output_path("outputs/raw")) / f"{rnd.production}.parquet"
    if production.is_file():
        paths.append(production)
    if not paths:
        raise SystemExit(f"no timing runs under {rnd.raw_dir}")
    pattern = re.compile(rf"--{re.escape(rnd.label_prefix)}(.+)-(r\d+)$")
    grids: dict[str, pd.DataFrame | None] = {}
    rows = []
    for path in paths:
        parsed, meta = R.load_raw(path)
        found = pattern.search(path.stem)
        condition, repeat = found.groups() if found else ("production audit", "")
        # The grid the run's config asks for, as reports/reference_metrics.csv uses it.
        key = str(meta.get("config_path"))
        if key not in grids:
            grids[key] = RM.expected_frame({"meta": meta})
        summary = M.summarise_run(prepare_frame(parsed))
        aligned, coverage = RM.align(RM.prepare(parsed), grids[key])
        unstable, complete, n_sets = RM.unstable_within_group(aligned)
        rows.append({
            "run": path.stem, "condition": condition, "repeat": repeat,
            "submit_order": meta.get("submit_order", "as_built"),
            "expected_prompts": coverage["expected_prompts"],
            "decided_pct": coverage["decided_pct"],
            "parse_failure_pct": _pct(summary.get("parse_failure_rate")),
            "refusal_pct": _pct(summary.get("refusal_rate")),
            "hire_pct": _pct(summary.get("population_acceptance_rate")),
            "attr_free_hire_pct": _pct(summary.get("attr_free_acceptance_rate")),
            "agree_gpt4o_pct": _pct(summary.get("reference_agreement")),
            **RM.hire_by_reference(aligned),
            **RM.flip_rates(aligned),
            "unstable_within_group_pct": unstable,
            "within_group_sets_complete": complete,
            "within_group_sets_expected": n_sets,
        })
    table = pd.DataFrame(rows)
    measures = ["hire_pct", "attr_free_hire_pct", "agree_gpt4o_pct", "tpr_vs_gpt4o_pct",
                "fpr_vs_gpt4o_pct", "flip_any_pct", "unstable_within_group_pct",
                "parse_failure_pct"]
    summary = table.groupby("condition", sort=True)[measures].agg(["median", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "runs", table.groupby("condition").size())
    summary = summary.round(4).reset_index()
    rnd.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(rnd.report_dir / "quality.csv", index=False)
    summary.to_csv(rnd.report_dir / "quality_summary.csv", index=False)
    print(summary.to_string(index=False))


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def shared_prefix(config: str, block: int, out: str = "shared_prefix.csv") -> None:
    from transformers import AutoTokenizer

    from hiring_bias_mitigation.eval.backends import _chat_wrap

    cfg = load_config(config)
    frame = R.build_frame(cfg).assign(profile_rendered="with_attribute")
    prompts = R._build_prompts(frame, "baseline")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    wrapped = [_chat_wrap(tokenizer, p, cfg.get("chat_template_kwargs")) for p in prompts]
    ids = tokenizer(wrapped, add_special_tokens=False)["input_ids"]
    total = sum(len(x) for x in ids)
    rows = [{"scope": "prompt", "groups": len(ids), "median_tokens": float(
        pd.Series([len(x) for x in ids]).median()), "mean_tokens": round(total / len(ids), 1),
        "prompt_tokens": total}]
    for scope, key in (("set", "group_id"), ("pair", "pair_id"), ("job", "job_id")):
        shared, members = [], []
        for idx in frame.groupby(key).indices.values():
            first = ids[idx[0]]
            shared.append(min((_common_prefix(first, ids[j]) for j in idx[1:]),
                              default=len(first)))
            members.append(len(idx))
        shared_s, members_s = pd.Series(shared), pd.Series(members)
        whole = (shared_s // block) * block
        # Every member after the first could skip the prefix the group shares.
        rows.append({
            "scope": scope, "groups": len(shared), "median_tokens": float(shared_s.median()),
            "mean_tokens": round(float(shared_s.mean()), 1), "prompt_tokens": total,
            "share_reaching_one_block": round(float((shared_s >= block).mean()), 4),
            "skippable_share_token_exact": round(float(((members_s - 1) * shared_s).sum())
                                                 / total, 4),
            "skippable_share_whole_blocks": round(float(((members_s - 1) * whole).sum())
                                                  / total, 4),
            "block": block,
        })
    production = (Path(resolve_output_path("outputs/raw"))
                  / f"{ROUNDS['b'].production}.parquet")
    if production.is_file():
        outputs = pd.read_parquet(production, columns=["raw_output"])["raw_output"].tolist()
        lengths = pd.Series([len(x) for x in tokenizer(outputs, add_special_tokens=False)[
            "input_ids"]])
        rows.append({"scope": f"response ({ROUNDS['b'].production})", "groups": len(lengths),
                     "median_tokens": float(lengths.median()),
                     "mean_tokens": round(float(lengths.mean()), 1)})
    table = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / out, index=False)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("run", help="one timing run: generate, never score")
    one.add_argument("--config", required=True)
    one.add_argument("--repeat", required=True, help="r1, r2...: appended to the run suffix")
    one.add_argument("--warmup", action="store_true",
                     help=f"generate {WARMUP_PAIRS} pairs to warm the node's caches; keep nothing")
    plan = sub.add_parser("schedule", help="which GPU runs which condition in which position")
    plan.add_argument("--round", choices=sorted(ROUNDS), default="full")
    plan.add_argument("--seed", type=int, required=True)
    same = sub.add_parser("compare", help="greedy outputs across conditions -> identity.csv")
    same.add_argument("--round", choices=sorted(ROUNDS), default="b")
    fast = sub.add_parser("speedup", help="1 - t_new / t_ref on generate_s -> speedup.csv")
    fast.add_argument("--round", choices=sorted(ROUNDS), default="b")
    fast.add_argument("--throughput", type=Path,
                      default=REPO_ROOT / "reports" / "cost" / "throughput.csv")
    good = sub.add_parser("quality", help="the audit's measures on every timing run")
    good.add_argument("--round", choices=sorted(ROUNDS), default="full")
    share = sub.add_parser("prefix", help="shared token prefixes -> shared_prefix.csv")
    share.add_argument("--config", required=True)
    share.add_argument("--block", type=int, default=528,
                       help="tokens per prefix-cache hit unit; vLLM 0.31 logs it for Qwen3.5")
    share.add_argument("--out", default="shared_prefix.csv",
                       help="file name under reports/throughput_b/")
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.config, args.repeat, args.warmup)
    elif args.cmd == "schedule":
        for gpu, repeat, position, condition, config in schedule(args.round, args.seed):
            print(gpu, repeat, position, condition, config)
    elif args.cmd == "compare":
        compare(args.round)
    elif args.cmd == "speedup":
        speedup(args.throughput, args.round)
    elif args.cmd == "quality":
        quality(args.round)
    else:
        shared_prefix(args.config, args.block, args.out)


if __name__ == "__main__":
    main()
