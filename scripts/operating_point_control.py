# ruff: noqa: E501  (command lines in the docstring)
"""Is the fine-tuned model more consistent, or only more willing to hire?

SFT moved Qwen3.5-9B's hire rate from 16% to 28% (the reference hires 33%). Counterfactual
instability depends on the operating point: at a hire rate near 0% or 100% nothing flips.
So a stability gain could in principle come from the shift alone. This control separates the
two by reading each model's decision margin directly and sweeping the decision threshold.

For every audited prompt, the model is given the audit's exact chat-wrapped prompt plus the
forced prefix `{"decision": "` and the log-probability gap between the first token of the two
decision words is read in a single vLLM step (no generation). With decisions `margin > t`,
each threshold `t` gives a hire rate and a share of unstable counterfactual sets; the two
models are then compared **at equal hire rate**, which the audit itself cannot do.

The margins come from top-20 logprobs, so they are quantised (a 1/8 grid near 0, thousands of
prompts tied on one value) and carry float noise. The matching therefore snaps them to a fixed
grid and hires the tied bin at the cut with a weight, whole sets at a time (`matched_flags`):
the matched hire rate is exact, and prompts tied inside one set are never split. A decision
word outside the top 20 gets a floor value whose ordering means nothing; a comparison whose cut
falls among such prompts is refused (`floor_check`).

    # one pass per model, then the comparison (the pass is ~10 min on the GPU)
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which base
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which adapter
    python scripts/operating_point_control.py compare --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml

Two fine-tuned arms against each other (e.g. DPO continued from SFT), each after its own
`margins --which adapter` pass: the reference arm is moved to the compared arm's hire rate.

    python scripts/operating_point_control.py compare-arms --config <dpo audit cfg> --against <sft audit cfg>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.eval import runner as R  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_config,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("operating_point_control")

PREFIX = '{"decision": "'
WORDS = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
SET = ["pair_id", "protected_group", "condition"]
#: Margins are compared on this grid. They are differences of bf16 logits read through float32
#: logprobs: every one sits on a 1/16 grid, plus noise of up to ~1e-6 that a quantile picks up
#: (1.374999985) and that splits prompts with the same grid value.
GRID = 1 / 1024
#: A decision word outside the top 20 is scored this far below the 20th logprob.
FLOOR_GAP = 1.0
#: Per-prompt flags written by `margins`: which decision word fell outside the top 20.
FLOORED = ["hire_missing", "reject_missing"]
#: Refuse a matched comparison when floor-substituted prompts could move it by more than this.
MAX_FLOOR_PP = 0.1


def out_dir() -> Path:
    d = Path(resolve_output_path("outputs/operating_point"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def decision_words(run_name: str, lang: str) -> tuple[str, str]:
    """The words this run actually answers with, read from its own generations.

    Not the prompt's language: a Ukrainian SFT adapter trained on targets carrying the
    canonical English decision answers `{"decision": "reject"}` to a prompt that asks for
    `найняти або відхилити`. Reading the words off the run makes the control follow the model.
    """
    path = Path(resolve_output_path("outputs/raw")) / f"{run_name}.parquet"
    if not path.exists():
        return WORDS[lang]
    counts = pd.read_parquet(path, columns=["decision", "raw_decision"])
    words = {}
    for decision, group in counts.groupby(counts["decision"].str.lower()):
        if decision in ("hire", "reject"):
            words[decision] = group["raw_decision"].value_counts().idxmax()
    picked = (words.get("hire", WORDS[lang][0]), words.get("reject", WORDS[lang][1]))
    log.info("%s answers with %r / %r", run_name, *picked)
    return picked


def llm_kwargs(cfg: dict, model_path: str) -> dict:
    """vLLM arguments for the one-token pass. The config's engine_kwargs pass through: on
    Daint the GDN prefill must be Triton, as FlashInfer's JIT needs a CUDA toolkit."""
    return {"model": model_path,
            "gpu_memory_utilization": cfg.get("op_control_gpu_memory_utilization", 0.55),
            "max_model_len": cfg.get("max_model_len", 8192), "trust_remote_code": True,
            "seed": 42, "max_num_seqs": cfg.get("op_control_max_num_seqs", 64),
            **(cfg.get("engine_kwargs") or {})}


def decision_token_ids(tokenizer, words: tuple[str, str]) -> tuple[int, int]:
    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in words:
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        assert full[: len(base)] == base, "the prefix tokenises differently before the decision"
        ids.append(full[len(base)])
    assert ids[0] != ids[1], "the two decision words share a first token"
    return ids[0], ids[1]


def cmd_margins(cfg: dict, which: str, chunk: int = 2000) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    done = out_dir() / f"{R.build_run_name(cfg)}--{which}.parquet"
    if done.exists() and set(FLOORED) <= set(pd.read_parquet(done).columns):
        log.info("%s already written; skipping this pass", done.name)
        return
    if done.exists():
        log.info("%s predates the per-prompt floor flags; recomputing it", done.name)
    frame = R.build_frame(cfg)
    prompts = R._build_prompts(frame, "baseline")
    if which == "base":
        model_path = cfg["model"]
    else:
        from hiring_bias_mitigation.mitigation.merge import merge

        model_path = str(merge(R._require_adapter(cfg["mitigation"]["lora_path"])))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    run_name = R.build_run_name(cfg)
    # For the base pass the answer style is the baseline run's, not the adapter's.
    source = run_name if which == "adapter" else f"{cfg['model'].split('/')[-1]}--{cfg['lang']}--baseline"
    hire_id, reject_id = decision_token_ids(tokenizer, decision_words(source, cfg["lang"]))
    kwargs = cfg.get("chat_template_kwargs")
    texts = [BK._chat_wrap(tokenizer, p, kwargs) + PREFIX for p in prompts]

    # This pass needs one token, not a 512-token generation, so it asks for far less KV cache
    # than an audit. Holding every RequestOutput for 31k prompts is what made the 12B pass run
    # the machine out of memory, so the prompts go through in chunks and only the margin is
    # kept -- the outputs are dropped as soon as each chunk is read.
    llm = LLM(**llm_kwargs(cfg, model_path))
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    margins, hire_missing, reject_missing = [], [], []
    for start in range(0, len(texts), chunk):
        for o in llm.generate(texts[start:start + chunk], params):
            lp = o.outputs[0].logprobs[0]
            floor = min(v.logprob for v in lp.values()) - FLOOR_GAP  # a word outside the top 20
            h = lp[hire_id].logprob if hire_id in lp else floor
            r = lp[reject_id].logprob if reject_id in lp else floor
            hire_missing.append(hire_id not in lp)
            reject_missing.append(reject_id not in lp)
            margins.append(h - r)
        log.info("  %d / %d prompts", len(margins), len(texts))
    # The flags go with the margins: a floor value only bounds the true margin, so a comparison
    # has to know which prompts carry one (see floor_uncertain).
    result = frame[KEY].assign(margin=margins, hire_missing=hire_missing,
                               reject_missing=reject_missing)
    missing = int(result[FLOORED].to_numpy().sum())
    path = out_dir() / f"{run_name}--{which}.parquet"
    result.to_parquet(path, index=False)
    log.info("wrote %s (%d prompts; %d decision words outside the top 20)", path, len(result),
             missing)
    if missing > 0.5 * 2 * len(result):
        raise SystemExit(
            f"{missing} of {2 * len(result)} decision-word lookups missed the top 20: the model "
            "does not answer with these words, so the margin is meaningless. Check "
            f"{path.name} against the run's raw_decision values."
        )


def snap(margin) -> np.ndarray:
    """Margins on the GRID, so that prompts with the same grid value compare equal."""
    return np.round(np.asarray(margin, dtype=float) / GRID) * GRID


def _paired(ref: pd.DataFrame, run: pd.DataFrame, names=("ref", "run")) -> pd.DataFrame:
    """The prompts both arms scored, every column of an arm suffixed with its name."""
    left, right = (f.set_index(KEY).add_suffix(f"_{n}") for f, n in zip((ref, run), names))
    return left.join(right, how="inner").reset_index()


def _set_flags(frame: pd.DataFrame, hire: np.ndarray) -> pd.Series:
    """Per counterfactual set, whether its prompts disagree (`hire`: one decision per row)."""
    decided = frame.assign(hire=hire)
    decided = decided[decided["condition"] != "attr_free"]
    return decided.groupby(SET)["hire"].nunique() > 1


def unstable_share(frame: pd.DataFrame, column: str, threshold: float) -> tuple[float, float]:
    hire = snap(frame[column]) > threshold
    return float(hire.mean()), float(_set_flags(frame, hire).mean())


def cut_for_count(margin: np.ndarray, k: int) -> tuple[float, float]:
    """The cut that hires `k` of these snapped margins: every one above `level`, and the bin
    tied at `level` with weight `w` (in [0, 1), and 1 only when `k` is every prompt)."""
    levels, counts = np.unique(margin, return_counts=True)
    at_or_above = np.cumsum(counts[::-1])[::-1]
    i = max(int((at_or_above > k).sum()) - 1, 0)
    return float(levels[i]), float((k - (at_or_above[i] - counts[i])) / counts[i])


def matched_flags(frame: pd.DataFrame, column: str, k: int) -> tuple[pd.Series, dict]:
    """Set flags with `column` moved to hire exactly `k` prompts.

    A quantile threshold with `margin > t` lands inside a tied bin and misses the hire rate by
    up to the bin's size. Here the bin tied at the cut is hired with weight `w`, whole sets at a
    time: a set hires all of its tied prompts or none, as any threshold would. Splitting ties
    per prompt would make sets unstable that no threshold can. The expected flags are the two
    neighbouring cuts' flags interpolated with weight `w` (the same as one randomised threshold
    for every set), and the expected hire count is exactly `k`.
    """
    m = snap(frame[column])
    level, w = cut_for_count(m, k)
    excluded, included = _set_flags(frame, m > level), _set_flags(frame, m >= level)
    flags = (1 - w) * excluded.astype(float) + w * included.astype(float)
    tied = int((m == level).sum())
    return flags, {
        "threshold": level, "tie_weight": w, "tied_prompts": tied,
        "hire_rate": float(((m > level).sum() + w * tied) / len(m)),
        "unstable": float(flags.mean()),
        "unstable_bin_excluded": float(excluded.mean()),
        "unstable_bin_included": float(included.mean()),
    }


def floor_uncertain(frame: pd.DataFrame, arm: str, level: float) -> np.ndarray | None:
    """Prompts of `arm` whose decision at a cut at `level` their margin cannot settle; None for
    a margin file written before the per-prompt flags.

    A decision word outside the top 20 is scored FLOOR_GAP below the 20th logprob, but its true
    logprob is only known to lie below the 20th. With hire missing the true margin is at most
    margin + FLOOR_GAP, with reject missing at least margin - FLOOR_GAP, with both anything: a
    cut beyond that bound decides the prompt, a cut inside it does not.
    """
    columns = [f"{c}_{arm}" for c in FLOORED]
    if not set(columns) <= set(frame.columns):
        return None
    m = snap(frame[f"margin_{arm}"])
    hire_missing, reject_missing = (frame[c].to_numpy(dtype=bool) for c in columns)
    return ((hire_missing & reject_missing)
            | (hire_missing & (m + FLOOR_GAP >= level))
            | (reject_missing & (m - FLOOR_GAP <= level)))


def floor_check(frame: pd.DataFrame, cuts: dict[str, float],
                max_floor_pp: float = MAX_FLOOR_PP) -> dict:
    """How far floor-substituted prompts could move a comparison made at these cuts (arm -> level).

    A prompt the floor leaves undecided can flip only its own set, so the sets holding one bound
    the error in the unstable share. Above `max_floor_pp` the comparison is refused; below it,
    any such prompt is a warning. A margin file without the flags leaves it unverified.
    """
    uncertain = {arm: floor_uncertain(frame, arm, level) for arm, level in cuts.items()}
    known = [u for u in uncertain.values() if u is not None]
    hit = np.logical_or.reduce(known) if known else np.zeros(len(frame), dtype=bool)
    in_sets = frame["condition"].to_numpy() != "attr_free"
    sets = len(frame.loc[hit & in_sets, SET].drop_duplicates())
    bound = 100 * sets / len(frame.loc[in_sets, SET].drop_duplicates())
    if bound > max_floor_pp:
        status = "refused"
    elif len(known) < len(uncertain):
        status, sets, bound = "unverified", None, None   # a partial count would read as a pass
    else:
        status = "warning" if hit.any() else "ok"
    return {"floor_check": status,
            "floor_uncertain_prompts": {arm: None if u is None else int(u.sum())
                                        for arm, u in uncertain.items()},
            "floor_uncertain_sets": sets, "floor_bound_pp": bound}


def matched_comparison(ref: pd.DataFrame, run: pd.DataFrame, n_boot: int = 1000,
                       seed: int = 42, max_floor_pp: float = MAX_FLOOR_PP) -> dict:
    """`run` at its own greedy threshold against `ref` moved to the same hire rate.

    The interval resamples candidates (a CV appears in several sets) with both cuts held fixed,
    the same clustering as analysis.stability.compare. A comparison refused by floor_check
    carries no matched instability, delta or interval.
    """
    both = _paired(ref, run)
    hire_ref, unstable_ref = unstable_share(both, "margin_ref", 0.0)
    run_hire = snap(both["margin_run"]) > 0.0
    run_flags = _set_flags(both, run_hire)
    ref_flags, cut = matched_flags(both, "margin_ref", int(run_hire.sum()))
    floor = floor_check(both, {"ref": cut["threshold"], "run": 0.0}, max_floor_pp)
    candidate = run_flags.index.get_level_values("pair_id").str.split("__").str[0]
    per = (run_flags.astype(float) - ref_flags).groupby(candidate).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    boots = 100 * per["sum"].to_numpy()[idx].sum(axis=1) / per["count"].to_numpy()[idx].sum(axis=1)
    unstable_run = float(run_flags.mean())
    result = {
        "prompts": len(both), "sets": len(run_flags), "grid": GRID,
        "hire_rate_ref": hire_ref, "unstable_ref": unstable_ref,
        "hire_rate_run": float(run_hire.mean()), "unstable_run": unstable_run,
        "ref_threshold_matched": cut["threshold"], "ref_tie_weight": cut["tie_weight"],
        "ref_tied_prompts": cut["tied_prompts"], "hire_rate_ref_matched": cut["hire_rate"],
        "unstable_ref_matched": cut["unstable"],
        "unstable_ref_bin_excluded": cut["unstable_bin_excluded"],
        "unstable_ref_bin_included": cut["unstable_bin_included"],
        "delta_matched_pp": 100 * (unstable_run - cut["unstable"]),
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
        **floor,
    }
    if floor["floor_check"] == "refused":
        result.update(dict.fromkeys(["unstable_ref_matched", "unstable_ref_bin_excluded",
                                     "unstable_ref_bin_included", "delta_matched_pp",
                                     "ci_low", "ci_high"]))
    return result


def _guard_floor(summary: dict, path: Path, max_floor_pp: float) -> None:
    """Log the floor check; exit non-zero (the report is written, without its delta) on refusal."""
    status = summary["floor_check"]
    if status == "refused":
        raise SystemExit(
            f"refused: the matched cut lies among floor-substituted margins (a decision word "
            f"outside the top 20), whose order means nothing. They could move the comparison by "
            f"up to {summary['floor_bound_pp']:.2f} pp ({summary['floor_uncertain_sets']} sets; "
            f"limit {max_floor_pp} pp). {path.name} is written without the matched result.")
    if status == "warning":
        log.warning("floor-substituted margins near the matched cut could move the comparison by "
                    "up to %.3f pp (%d sets)", summary["floor_bound_pp"],
                    summary["floor_uncertain_sets"])
    if status == "unverified":
        log.warning("a margin file predates the per-prompt floor flags (rerun `margins`): whether "
                    "the matched cut lies among floor-substituted margins is unchecked")


def cmd_compare_arms(cfg: dict, against: dict, max_floor_pp: float = MAX_FLOOR_PP) -> None:
    run_name, ref_name = R.build_run_name(cfg), R.build_run_name(against)
    run = pd.read_parquet(out_dir() / f"{run_name}--adapter.parquet")
    ref = pd.read_parquet(out_dir() / f"{ref_name}--adapter.parquet")
    summary = {"run": run_name, "against": ref_name,
               **matched_comparison(ref, run, max_floor_pp=max_floor_pp)}
    report = REPO_ROOT / "reports" / "operating_point"
    report.mkdir(parents=True, exist_ok=True)
    path = report / f"{run_name}--vs--{ref_name}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    _guard_floor(summary, path, max_floor_pp)


def cmd_compare(cfg: dict, max_floor_pp: float = MAX_FLOOR_PP) -> None:
    name = R.build_run_name(cfg)
    base = pd.read_parquet(out_dir() / f"{name}--base.parquet")
    adapter = pd.read_parquet(out_dir() / f"{name}--adapter.parquet")
    both = _paired(base, adapter, ("base", "adapter"))

    # Sweep: on each row both models are moved to the same hire count, exactly.
    rows = []
    for q in np.linspace(0.40, 0.95, 56):
        k = round((1 - q) * len(both))
        for which in ("base", "adapter"):
            _, cut = matched_flags(both, f"margin_{which}", k)
            rows.append({"model": which, "quantile": round(q, 3), "threshold": cut["threshold"],
                         "tie_weight": cut["tie_weight"], "hire_rate": cut["hire_rate"],
                         "unstable": cut["unstable"]})
    curve = pd.DataFrame(rows)

    # The headline: each model at its own greedy threshold, and the base moved to the
    # adapter's hire rate.
    pair = matched_comparison(base, adapter, max_floor_pp=max_floor_pp)
    unstable_base_0, unstable_adapter_0 = pair["unstable_ref"], pair["unstable_run"]
    matched = pair["unstable_ref_matched"]
    summary = {
        "run": name, "prompts": pair["prompts"], "grid": GRID,
        "hire_rate_base": pair["hire_rate_ref"],
        "hire_rate_adapter": pair["hire_rate_run"],
        "unstable_base_at_own_threshold": unstable_base_0,
        "unstable_adapter_at_own_threshold": unstable_adapter_0,
        "base_threshold_matched_to_adapter_hire_rate": pair["ref_threshold_matched"],
        "base_tie_weight": pair["ref_tie_weight"],
        "base_tied_prompts": pair["ref_tied_prompts"],
        "hire_rate_base_matched": pair["hire_rate_ref_matched"],
        "unstable_base_at_matched_hire_rate": matched,
        "gain_explained_by_shift_pct": None if matched is None else float(
            100 * (unstable_base_0 - matched)
            / max(unstable_base_0 - unstable_adapter_0, 1e-9)),
        "floor_check": pair["floor_check"],
        "floor_uncertain_prompts": {"base": pair["floor_uncertain_prompts"]["ref"],
                                    "adapter": pair["floor_uncertain_prompts"]["run"]},
        "floor_uncertain_sets": pair["floor_uncertain_sets"],
        "floor_bound_pp": pair["floor_bound_pp"],
    }
    report = REPO_ROOT / "reports" / "operating_point"
    report.mkdir(parents=True, exist_ok=True)
    curve.to_csv(report / f"{name}--curve.csv", index=False)
    path = report / f"{name}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    _guard_floor(summary, path, max_floor_pp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("margins")
    m.add_argument("--config", required=True, help="the adapter's audit config")
    m.add_argument("--which", choices=("base", "adapter"), required=True)
    m.add_argument("--chunk", type=int, default=2000,
                   help="prompts per vLLM call; lower it if the machine is tight on memory")
    c = sub.add_parser("compare")
    c.add_argument("--config", required=True)
    a = sub.add_parser("compare-arms")
    a.add_argument("--config", required=True, help="the arm compared (e.g. DPO)")
    a.add_argument("--against", required=True, help="the reference arm (e.g. SFT)")
    for p in (c, a):
        p.add_argument("--max-floor-pp", type=float, default=MAX_FLOOR_PP,
                       help="refuse when floor-substituted margins could move the matched "
                            "instability by more than this many points")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # A config listed in reports/operating_point/skip.txt is skipped (exit 0): the control
    # costs ~75 min per pass, and cells whose hire rate barely moved do not need it. A file
    # rather than a flag, so a running queue can be told without restarting it.
    skip = REPO_ROOT / "reports" / "operating_point" / "skip.txt"
    if skip.exists() and Path(args.config).stem in skip.read_text().split():
        log.info("skipping %s (listed in %s)", args.config, skip)
        return
    if args.cmd == "margins":
        cmd_margins(cfg, args.which, args.chunk)
    elif args.cmd == "compare-arms":
        cmd_compare_arms(cfg, load_config(args.against), args.max_floor_pp)
    else:
        cmd_compare(cfg, args.max_floor_pp)


if __name__ == "__main__":
    main()
