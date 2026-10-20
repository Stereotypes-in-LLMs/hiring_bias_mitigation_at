"""Part B of the GTC poster: the counterfactual audit as an inference workload, timed with and
without vLLM's prefix cache and with prompts submitted in prefix order or shuffled
(docs/THROUGHPUT_B.md).

    python scripts/throughput_b.py run --repeat r1 \
        --config configs/throughput/qwen3.5-9b_en_cacheon-prefix.yaml
    python scripts/throughput_b.py compare     # are the greedy outputs the same across conditions?
    python scripts/throughput_b.py speedup     # 1 - t_new / t_ref, after scripts/cost_report.py
    python scripts/throughput_b.py prefix --config <cfg>   # how much prefix the prompts share

`run` is one timing run. It clears the run's chunk cache, so every prompt is generated and
timed, generates through the same `runner.run_audit` an audit uses, and writes
outputs/throughput/raw/<run>.parquet. It never scores, so nothing reaches eval/results/ or
reports/reference_metrics.csv. Wrapped in `cost eval throughput-<condition>`
(scripts/cscs/env.sh), its wall time, GPU, energy, requests, tokens, load and generate seconds
and vLLM's prefix-cache counters land in one cost record; scripts/cost_report.py puts them in
reports/cost/throughput.csv.

`compare` lines up every timing run, and the production audit Qwen3.5-9B--en--baseline--cscs,
on the same rows (pair, group, attribute, condition) and counts byte-identical responses and
identical decisions -> reports/throughput_b/identity.csv, with the differing rows of each pair of
runs, one row per response, in $HBM_OUTPUT_ROOT/outputs/throughput/differences.csv (tens of
MB, so beside the outputs rather than in reports/).

`speedup` reads reports/cost/throughput.csv and, within each job and repeat, sets every
condition against every other: 1 - generate_s(new) / generate_s(ref), the ratio t_ref / t_new,
the prefix-cache hit rate -> reports/throughput_b/speedup.csv, and their median and range over
the repeats -> speedup_summary.csv. The poster's reference is `cacheoff-shuffle`.

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

RAW_DIR = "outputs/throughput/raw"
REPORT_DIR = REPO_ROOT / "reports" / "throughput_b"
#: The production audit the timing runs' outputs are checked against.
PRODUCTION_RUN = "Qwen3.5-9B--en--baseline--cscs"
#: Cost labels are `throughput-<condition>`; the reference is the slowest setup on purpose.
LABEL_PREFIX = "throughput-"
REFERENCE = "cacheoff-shuffle"
ROW_KEY = ["pair_id", "protected_group", "protected_attr", "condition"]


def run(config: str, repeat: str) -> None:
    cfg = load_config(config)
    if cfg.get("raw_dir") != RAW_DIR:
        raise SystemExit(f"{config}: a timing run writes to raw_dir: {RAW_DIR}, never next to "
                         "the audits in outputs/raw/")
    if not re.fullmatch(r"r\d+", repeat):
        raise SystemExit(f"--repeat must look like r1, r2...; got {repeat!r}")
    cfg["run_suffix"] = f"{cfg['run_suffix']}-{repeat}"
    run_name = R.build_run_name(cfg)
    # Every prompt must be generated in this process: a chunk cached by an earlier attempt of
    # the same run would be skipped and its time never measured.
    R._drop_partial(run_name)
    parsed, meta = R.run_audit(cfg)
    R.save_raw(parsed, meta, RAW_DIR)
    log.info("%s: %d prompts, wall %.1f s", run_name, meta["n_prompts"], meta["wall_seconds"])


def _runs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    runs = {}
    for path in sorted(raw_dir.glob("*--throughput-*.parquet")):
        runs[path.stem] = pd.read_parquet(path, columns=[*ROW_KEY, "raw_output", "decision"])
    return runs


def compare() -> None:
    raw_dir = Path(resolve_output_path(RAW_DIR))
    runs = _runs(raw_dir)
    if not runs:
        raise SystemExit(f"no timing runs under {raw_dir}")
    production = Path(resolve_output_path("outputs/raw")) / f"{PRODUCTION_RUN}.parquet"
    if production.is_file():
        runs[PRODUCTION_RUN] = pd.read_parquet(
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
        differences.append(both.loc[~same_text].assign(run_a=a, run_b=b))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "identity.csv", index=False)
    pd.concat(differences, ignore_index=True).to_csv(raw_dir.parent / "differences.csv",
                                                     index=False)
    print(table.to_string(index=False))


def _condition(label: str) -> str:
    return label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label


def _repeat(run_name: str) -> str:
    found = re.search(r"-(r\d+)$", str(run_name))
    return found.group(1) if found else ""


def speedup(throughput_csv: Path) -> None:
    steps = pd.read_csv(throughput_csv, dtype={"job_id": str})
    steps = steps[steps["label"].astype(str).str.startswith(LABEL_PREFIX)].copy()
    if steps.empty:
        raise SystemExit(f"no {LABEL_PREFIX}* steps in {throughput_csv}; run "
                         "scripts/cost_report.py after the timing job")
    # Energy is per step, model load included; steps.csv sits beside throughput.csv.
    energy = pd.read_csv(throughput_csv.parent / "steps.csv", dtype={"job_id": str})
    steps = steps.merge(energy[["job_id", "label", "run_name", "gpu_energy_kwh"]],
                        on=["job_id", "label", "run_name"], how="left", validate="1:1")
    steps["condition"] = steps["label"].map(_condition)
    steps["repeat"] = steps["run_name"].map(_repeat)
    rows = []
    for (job, repeat), group in steps.groupby(["job_id", "repeat"], sort=True):
        if group["condition"].duplicated().any():
            log.warning("job %s %s: a condition appears twice; skipped", job, repeat)
            continue
        by = group.set_index("condition")
        # Every condition against every other one timed in the same job and repeat: the same
        # node, the same prompts, at the same time.
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
                "requests_per_s": s["requests_per_s"],
                "output_tokens_per_s": s["output_tokens_per_s"],
                "gpu_s_per_1k_requests": s["gpu_s_per_1k_requests"],
                "gpu_s_per_1k_requests_generate_only": s["gpu_s_per_1k_requests_generate_only"],
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
        requests_per_s_median=("requests_per_s", "median"),
        output_tokens_per_s_median=("output_tokens_per_s", "median"),
        gpu_s_per_1k_requests_generate_only_median=(
            "gpu_s_per_1k_requests_generate_only", "median"),
        prefix_cache_hit_rate_median=("prefix_cache_hit_rate", "median"),
        gpu_energy_kwh_median=("gpu_energy_kwh", "median"),
    ).round(4).reset_index()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "speedup.csv", index=False)
    summary.to_csv(REPORT_DIR / "speedup_summary.csv", index=False)
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
    production = Path(resolve_output_path("outputs/raw")) / f"{PRODUCTION_RUN}.parquet"
    if production.is_file():
        outputs = pd.read_parquet(production, columns=["raw_output"])["raw_output"].tolist()
        lengths = pd.Series([len(x) for x in tokenizer(outputs, add_special_tokens=False)[
            "input_ids"]])
        rows.append({"scope": f"response ({PRODUCTION_RUN})", "groups": len(lengths),
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
    sub.add_parser("compare", help="greedy outputs across conditions -> identity.csv")
    fast = sub.add_parser("speedup", help="1 - t_new / t_ref on generate_s -> speedup.csv")
    fast.add_argument("--throughput", type=Path,
                      default=REPO_ROOT / "reports" / "cost" / "throughput.csv")
    share = sub.add_parser("prefix", help="shared token prefixes -> shared_prefix.csv")
    share.add_argument("--config", required=True)
    share.add_argument("--block", type=int, default=528,
                       help="tokens per prefix-cache hit unit; vLLM 0.31 logs it for Qwen3.5")
    share.add_argument("--out", default="shared_prefix.csv",
                       help="file name under reports/throughput_b/")
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.config, args.repeat)
    elif args.cmd == "compare":
        compare()
    elif args.cmd == "speedup":
        speedup(args.throughput)
    else:
        shared_prefix(args.config, args.block, args.out)


if __name__ == "__main__":
    main()
