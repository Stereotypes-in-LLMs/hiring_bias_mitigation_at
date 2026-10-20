# ruff: noqa: E501  (command lines in the docstring)
"""Is the fine-tuned model more consistent, or only more willing to hire?

SFT moved Qwen3.5-9B's hire rate from 16% to 28% (the reference hires 33%). Counterfactual
instability depends on the operating point: at a hire rate near 0% or 100% nothing flips.
So a stability gain could in principle come from the shift alone. This control separates the
two by reading each model's decision margin directly and sweeping the decision threshold.

For every audited prompt, the model is given the audit's exact chat-wrapped prompt plus the
forced prefix `{"decision": "`, and the log-probabilities of the first token of the two
decision words are read in a single vLLM step (no generation). With decisions `margin > t`,
each threshold `t` gives a hire rate and a share of unstable counterfactual sets; the two
models are then compared **at equal hire rate**, which the audit itself cannot do.

    # one pass per model, then the comparison (a pass is ~12 min on one GH200)
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which base
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which adapter
    python scripts/operating_point_control.py compare --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml

Two fine-tuned arms against each other (e.g. DPO continued from SFT), each after its own
`margins --which adapter` pass: the reference arm is moved to the compared arm's hire rate.

    python scripts/operating_point_control.py compare-arms --config <dpo audit cfg> --against <sft audit cfg>

One standard (`STANDARD`)
-------------------------
Every file this control writes follows the one procedure fixed here, so a new checkpoint needs
no check of the method of its own: `margins`, then `compare-arms`, and the report says whether
it conforms.

Scores. The exact log-probabilities of the two decision tokens, which vLLM gathers from the
full softmax over the raw logits whatever their rank (`logprob_token_ids`); margin =
logprob(hire) - logprob(reject). A missing value stops the pass. Version 1 read the two words
from the top-20 logprobs and scored a word outside them at the 20th logprob - 1: that kept
every margin's sign but not its size, and it touched 92-99% of a DPO checkpoint's prompts.

Files. `outputs/operating_point/v2/<run>--<which>.parquet` (the key columns, `logprob_hire`,
`logprob_reject`, `margin`) beside `<same>.meta.json`: the standard, a fingerprint of the
scored prompts, the decision tokens and chat-template settings, the model (served weights,
base revision, adapter chain with sha256), engine settings, software versions and the code
commit. A file is rescored only when its prompts or adapter weights change. Version-1 files
(`outputs/operating_point/<run>--<which>.parquet`) are left as they are and read by no
comparison; their reports stay under `reports/operating_point/`.

Matching. Margins are snapped to a 1/1024 grid (differences of bf16 logits, plus float noise).
The reference arm hires exactly the compared arm's count: the bin tied at the cut is hired
with a weight, whole sets at a time (`matched_flags`), so prompts tied inside one set are
never split. The interval resamples candidates (1,000 draws, seed 42), since sets sharing a
candidate are not independent.

Reports. `reports/operating_point/v2/<run>--vs--<ref>.json` (and `<run>.json`,
`<run>--curve.csv` for `compare`) carry both inputs' provenance and `conforms`, with
`issues` when it is false: a file scored or a comparison run with uncommitted code changes,
inputs scored with different vLLM versions, or a vLLM version no passing `validate` record
covers. Inputs from another standard, or scored on different prompts, are refused outright.

`validate` checks the scorer itself, once per vLLM version: on a fixed sample it compares the
vLLM margins with the same quantity computed by HuggingFace on the same weights
(`eval/margins.py`), and their signs with the audit's greedy decisions, and writes the record
under `reports/operating_point/v2/validation/` that later reports look up.

    python scripts/operating_point_control.py validate --config <audit cfg> --which adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
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

#: The procedure every score file and report here follows (module docstring).
STANDARD = "op-margin-v2"
#: Where this standard's score files and reports live, beside the version-1 ones.
VERSION_DIR = "v2"
PREFIX = '{"decision": "'
WORDS = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
SET = ["pair_id", "protected_group", "condition"]
#: Margins are compared on this grid. They are differences of bf16 logits read through float32
#: logprobs: every one sits on a 1/16 grid, plus noise of up to ~1e-6 that a quantile picks up
#: (1.374999985) and that splits prompts with the same grid value.
GRID = 1 / 1024
N_BOOT = 1000
BOOT_SEED = 42
METHOD = {
    "grid": GRID,
    "run_threshold": 0.0,
    "matching": "the reference arm hires exactly the compared arm's count; the bin tied at the "
                "cut is weighted, whole counterfactual sets at a time",
    "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "candidate"},
}
#: `validate` passes when vLLM and HuggingFace agree on at least this share of decisions and
#: rank the prompts alike, and vLLM's decisions match the audit's greedy ones as often.
MIN_AGREEMENT = 0.98
MIN_RANK_CORRELATION = 0.98
#: Below this share of prompts continuing with one of the two decision tokens, the model does
#: not answer with these words and its margins mean nothing.
MIN_DECISION_TOKEN_SHARE = 0.5


def out_dir() -> Path:
    d = Path(resolve_output_path("outputs/operating_point"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def score_paths(run: str, which: str) -> tuple[Path, Path]:
    """This standard's score file for one model, and its record."""
    d = out_dir() / VERSION_DIR
    d.mkdir(exist_ok=True)
    return d / f"{run}--{which}.parquet", d / f"{run}--{which}.meta.json"


def reports_dir(*parts: str) -> Path:
    d = REPO_ROOT.joinpath("reports", "operating_point", VERSION_DIR, *parts)
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


# ---- provenance ---------------------------------------------------------------------------

def prompt_fingerprint(frame: pd.DataFrame, prompts: list[str]) -> dict:
    """The scored prompts' identity, in order: two files are comparable only if it is equal."""
    h = hashlib.sha1()
    for key, text in zip(frame[KEY].itertuples(index=False, name=None), prompts):
        h.update("\x1f".join(map(str, key)).encode())
        h.update(b"\x1e")
        h.update(text.encode())
        h.update(b"\x1d")
    return {"n": len(prompts), "sha1": h.hexdigest()}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def adapter_record(adapter: Path) -> dict:
    weights = sorted(adapter.glob("adapter_model*"))
    return {"path": str(adapter), "sha256": file_sha256(weights[0]) if weights else None}


def hub_revision(model: str) -> str | None:
    """The cached snapshot of a Hub model id; None for a local path or a model not cached."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

    if Path(model).exists():
        return None
    try:
        path = snapshot_download(model, local_files_only=True, allow_patterns=["config.json"])
    except (HFValidationError, LocalEntryNotFoundError, OSError):
        return None
    return Path(path).name


def software() -> dict:
    versions = {}
    for package in ("vllm", "torch", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def code_state() -> dict:
    """The commit this code runs at, and its uncommitted changes to tracked code or configs."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=no", "--",
                     "scripts", "src", "configs")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}
    return {"commit": commit,
            "uncommitted": [line[3:] for line in status.splitlines() if line.strip()]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- scoring ------------------------------------------------------------------------------

def scoring_plan(cfg: dict, which: str) -> dict:
    """What a pass scores and with which adapter weights, before any model is loaded."""
    frame = R.build_frame(cfg)
    prompts = R._build_prompts(frame, "baseline")
    chain: list[Path] = []
    if which == "adapter":
        from hiring_bias_mitigation.mitigation.merge import adapter_chain

        chain = adapter_chain(Path(R._require_adapter(cfg["mitigation"]["lora_path"])))
    return {"run": R.build_run_name(cfg), "which": which, "frame": frame, "prompts": prompts,
            "fingerprint": prompt_fingerprint(frame, prompts),
            "adapters": [adapter_record(a) for a in chain]}


def load_scorer(cfg: dict, plan: dict) -> dict:
    """The served weights, the decision tokens and the exact texts the pass feeds the model."""
    from transformers import AutoTokenizer

    if plan["which"] == "base":
        model_path = cfg["model"]
    else:
        from hiring_bias_mitigation.mitigation.merge import merge

        model_path = str(merge(R._require_adapter(cfg["mitigation"]["lora_path"])))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # For the base pass the answer style is the baseline run's, not the adapter's.
    source = (plan["run"] if plan["which"] == "adapter"
              else f"{cfg['model'].split('/')[-1]}--{cfg['lang']}--baseline")
    words = decision_words(source, cfg["lang"])
    kwargs = cfg.get("chat_template_kwargs")
    return {"model_path": model_path, "tokenizer": tokenizer, "words": words,
            "token_ids": decision_token_ids(tokenizer, words),
            "texts": [BK._chat_wrap(tokenizer, p, kwargs) + PREFIX for p in plan["prompts"]]}


def score_logprobs(llm, texts: list[str], token_ids: tuple[int, int],
                   chunk: int = 2000) -> tuple[np.ndarray, np.ndarray, float]:
    """The two decision tokens' log-probabilities after the prefix, one pair per text, and the
    share of texts whose greedy next token is one of them.

    `logprob_token_ids` makes vLLM gather exactly these ids from the full log-softmax of the
    raw logits, so nothing is estimated: a missing value is an error, never a substitute. The
    texts go through in chunks, keeping only the numbers: holding every RequestOutput for 31k
    prompts is what ran the 12B pass out of memory.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(token_ids))
    hire_id, reject_id = token_ids
    hire, reject, decided = [], [], 0
    for start in range(0, len(texts), chunk):
        for offset, o in enumerate(llm.generate(texts[start:start + chunk], params)):
            out = o.outputs[0]
            lp = out.logprobs[0]
            missing = [t for t in token_ids if t not in lp]
            if missing:
                raise RuntimeError(
                    f"vLLM returned no logprob for decision token(s) {missing} on prompt "
                    f"{start + offset}: this vLLM does not honour logprob_token_ids")
            hire.append(lp[hire_id].logprob)
            reject.append(lp[reject_id].logprob)
            decided += out.token_ids[0] in token_ids
        log.info("  %d / %d prompts", len(hire), len(texts))
    return np.asarray(hire, dtype=float), np.asarray(reject, dtype=float), decided / len(texts)


def is_current(path: Path, meta_path: Path, plan: dict) -> bool:
    """A file of this standard for the same prompts and the same adapter weights."""
    if not (path.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text())
    return (meta.get("standard") == STANDARD and meta.get("prompts") == plan["fingerprint"]
            and [a["sha256"] for a in meta["model"]["adapters"]]
            == [a["sha256"] for a in plan["adapters"]])


def cmd_margins(cfg: dict, which: str, chunk: int = 2000) -> None:
    from vllm import LLM

    plan = scoring_plan(cfg, which)
    path, meta_path = score_paths(plan["run"], which)
    if is_current(path, meta_path, plan):
        log.info("%s is current under %s; skipping this pass", path.name, STANDARD)
        return
    scorer = load_scorer(cfg, plan)
    engine = llm_kwargs(cfg, scorer["model_path"])
    started = time.monotonic()
    llm = LLM(**engine)
    hire, reject, decided = score_logprobs(llm, scorer["texts"], scorer["token_ids"], chunk)
    if decided < MIN_DECISION_TOKEN_SHARE:
        raise SystemExit(
            f"only {decided:.1%} of prompts continue with {scorer['words']}: the model does not "
            "answer with these words, so the margin is meaningless. Check the run's "
            "raw_decision values.")
    plan["frame"][KEY].assign(logprob_hire=hire, logprob_reject=reject,
                              margin=hire - reject).to_parquet(path, index=False)
    meta = {
        "standard": STANDARD,
        "scoring": "log-probabilities of the first token of each decision word after the forced "
                   "prefix, gathered by vLLM from the full softmax over the raw logits "
                   "(logprob_token_ids); margin = logprob_hire - logprob_reject",
        "run": plan["run"], "which": which, "config": cfg.get("_config_path"),
        "prompts": plan["fingerprint"],
        "decision": {"prefix": PREFIX, "words": list(scorer["words"]),
                     "token_ids": list(scorer["token_ids"]),
                     "chat_template_kwargs": cfg.get("chat_template_kwargs"),
                     "sampled_decision_token_share": decided},
        "model": {"served": scorer["model_path"], "base": cfg["model"],
                  "base_revision": hub_revision(cfg["model"]), "adapters": plan["adapters"]},
        "engine": {k: v for k, v in engine.items() if k != "model"},
        "software": software(), "code": code_state(),
        "wall_seconds": round(time.monotonic() - started, 1), "written": _now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("wrote %s (%d prompts; %.1f%% continue with a decision token)", path,
             len(hire), 100 * decided)


# ---- matching -----------------------------------------------------------------------------

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


def matched_comparison(ref: pd.DataFrame, run: pd.DataFrame, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict:
    """`run` at its own greedy threshold against `ref` moved to the same hire count.

    The interval resamples candidates (a CV appears in several sets) with both cuts held fixed,
    the same clustering as analysis.stability.compare.
    """
    both = _paired(ref, run)
    hire_ref, unstable_ref = unstable_share(both, "margin_ref", 0.0)
    run_hire = snap(both["margin_run"]) > 0.0
    run_flags = _set_flags(both, run_hire)
    ref_flags, cut = matched_flags(both, "margin_ref", int(run_hire.sum()))
    candidate = run_flags.index.get_level_values("pair_id").str.split("__").str[0]
    per = (run_flags.astype(float) - ref_flags).groupby(candidate).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    boots = 100 * per["sum"].to_numpy()[idx].sum(axis=1) / per["count"].to_numpy()[idx].sum(axis=1)
    unstable_run = float(run_flags.mean())
    return {
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
    }


# ---- comparisons --------------------------------------------------------------------------

def load_scores(run: str, which: str) -> tuple[pd.DataFrame, dict]:
    """One model's scores under this standard, and their record; anything else is refused."""
    path, meta_path = score_paths(run, which)
    if not (path.exists() and meta_path.exists()):
        old = out_dir() / f"{run}--{which}.parquet"
        hint = (f"; {old.name} is a version-1 file (top-20 logprobs with a floor), which no "
                "comparison reads" if old.exists() else "")
        raise SystemExit(f"no {STANDARD} scores for {run} --{which}{hint}. Run `margins` first.")
    meta = json.loads(meta_path.read_text())
    if meta.get("standard") != STANDARD:
        raise SystemExit(f"{meta_path.name} was written under {meta.get('standard')!r}, "
                         f"not {STANDARD}. Run `margins` again.")
    return pd.read_parquet(path), meta


def check_comparable(*metas: dict) -> None:
    """Refuses inputs scored on different prompts, or after a different prefix or template."""
    fields = {"prompts": lambda m: m["prompts"],
              "decision prefix": lambda m: m["decision"]["prefix"],
              "chat-template settings": lambda m: m["decision"]["chat_template_kwargs"]}
    for name, get in fields.items():
        values = {json.dumps(get(m), sort_keys=True) for m in metas}
        if len(values) > 1:
            raise SystemExit(f"the inputs differ in their {name} ({sorted(values)}): they were "
                             "not scored on the same prompts the same way")


def validation_records() -> list[dict]:
    d = REPO_ROOT / "reports" / "operating_point" / VERSION_DIR / "validation"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def conformance(metas: list[dict], code: dict, records: list[dict]) -> list[str]:
    """Why a report does not meet the standard; empty when it does."""
    validated = {r["software"].get("vllm") for r in records
                 if r.get("standard") == STANDARD and r.get("passed")}
    issues = []
    for m in metas:
        name = f"{m['run']} --{m['which']}"
        if m["code"]["commit"] is None:
            issues.append(f"{name}: scored with no code commit recorded")
        elif m["code"]["uncommitted"]:
            issues.append(f"{name}: scored with uncommitted changes to "
                          f"{', '.join(m['code']['uncommitted'])}")
        if m["software"].get("vllm") not in validated:
            issues.append(f"{name}: scored with vLLM {m['software'].get('vllm')}, which no "
                          "passing `validate` record covers")
    if len({m["software"].get("vllm") for m in metas}) > 1:
        issues.append("the inputs were scored with different vLLM versions")
    if code["commit"] is None:
        issues.append("comparison: no code commit recorded")
    elif code["uncommitted"]:
        issues.append(f"comparison: run with uncommitted changes to {', '.join(code['uncommitted'])}")
    return issues


def provenance(meta: dict) -> dict:
    keep = ("run", "which", "config", "prompts", "decision", "model", "software", "code", "written")
    return {k: meta.get(k) for k in keep}


def arm_report(run_name: str, ref_name: str, run: pd.DataFrame, run_meta: dict,
               ref: pd.DataFrame, ref_meta: dict, code: dict, records: list[dict]) -> dict:
    """The compared arm against the reference arm at matched hire rate, as one report."""
    check_comparable(ref_meta, run_meta)
    issues = conformance([run_meta, ref_meta], code, records)
    return {"standard": STANDARD, "run": run_name, "against": ref_name,
            "conforms": not issues, "issues": issues, **matched_comparison(ref, run),
            "method": METHOD, "code": code,
            "inputs": {"run": provenance(run_meta), "against": provenance(ref_meta)}}


def _write_report(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, indent=2))
    for issue in summary["issues"]:
        log.warning("does not conform: %s", issue)
    log.info("wrote %s (%s)", path, "conforms" if summary["conforms"] else "does NOT conform")


def cmd_compare_arms(cfg: dict, against: dict) -> None:
    run_name, ref_name = R.build_run_name(cfg), R.build_run_name(against)
    run, run_meta = load_scores(run_name, "adapter")
    ref, ref_meta = load_scores(ref_name, "adapter")
    summary = arm_report(run_name, ref_name, run, run_meta, ref, ref_meta, code_state(),
                         validation_records())
    _write_report(summary, reports_dir() / f"{run_name}--vs--{ref_name}.json")


def cmd_compare(cfg: dict) -> None:
    name = R.build_run_name(cfg)
    base, base_meta = load_scores(name, "base")
    adapter, adapter_meta = load_scores(name, "adapter")
    check_comparable(base_meta, adapter_meta)
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
    pair = matched_comparison(base, adapter)
    unstable_base_0, unstable_adapter_0 = pair["unstable_ref"], pair["unstable_run"]
    matched = pair["unstable_ref_matched"]
    code = code_state()
    issues = conformance([adapter_meta, base_meta], code, validation_records())
    summary = {
        "standard": STANDARD, "run": name, "conforms": not issues, "issues": issues,
        "prompts": pair["prompts"], "grid": GRID,
        "hire_rate_base": pair["hire_rate_ref"],
        "hire_rate_adapter": pair["hire_rate_run"],
        "unstable_base_at_own_threshold": unstable_base_0,
        "unstable_adapter_at_own_threshold": unstable_adapter_0,
        "base_threshold_matched_to_adapter_hire_rate": pair["ref_threshold_matched"],
        "base_tie_weight": pair["ref_tie_weight"],
        "base_tied_prompts": pair["ref_tied_prompts"],
        "hire_rate_base_matched": pair["hire_rate_ref_matched"],
        "unstable_base_at_matched_hire_rate": matched,
        "gain_explained_by_shift_pct": float(
            100 * (unstable_base_0 - matched)
            / max(unstable_base_0 - unstable_adapter_0, 1e-9)),
        "method": METHOD, "code": code,
        "inputs": {"adapter": provenance(adapter_meta), "base": provenance(base_meta)},
    }
    out = reports_dir()
    curve.to_csv(out / f"{name}--curve.csv", index=False)
    _write_report(summary, out / f"{name}.json")


# ---- validating the scorer, once per vLLM version ----------------------------------------

def validation_sample(n_total: int, n: int, seed: int) -> np.ndarray:
    """A fixed, seeded subset of prompt positions, in order."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho as a Pearson correlation of ranks; NaN (a failed check) when either
    input is constant."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def validation_summary(vllm_margin, hf_margin, greedy=None, v1_margin=None) -> dict:
    """How closely the vLLM margins match HuggingFace's on the same weights, and the audit's
    greedy decisions; `passed` when every available check clears its threshold."""
    v, h = np.asarray(vllm_margin, dtype=float), np.asarray(hf_margin, dtype=float)
    diff = np.abs(v - h)
    out = {
        "decision_agreement_hf": float(np.mean((v > 0) == (h > 0))),
        "rank_correlation_hf": _rank_correlation(v, h),
        "abs_diff_hf": {"median": float(np.median(diff)),
                        "p95": float(np.percentile(diff, 95)), "max": float(diff.max())},
    }
    checks = [out["decision_agreement_hf"] >= MIN_AGREEMENT,
              out["rank_correlation_hf"] >= MIN_RANK_CORRELATION]
    if greedy is not None:
        g = np.asarray(greedy, dtype=object)
        decided = np.isin(g, ["hire", "reject"])
        out["greedy_decided"] = int(decided.sum())
        if decided.any():
            out["decision_agreement_greedy"] = float(
                np.mean((v[decided] > 0) == (g[decided] == "hire")))
            checks.append(out["decision_agreement_greedy"] >= MIN_AGREEMENT)
    if v1_margin is not None:
        old = np.asarray(v1_margin, dtype=float)
        out["v1_decision_agreement"] = float(np.mean((old > 0) == (v > 0)))
        out["v1_share_off_by_more_than_0.5"] = float(np.mean(np.abs(old - v) > 0.5))
    out["passed"] = bool(all(checks))
    return out


def _aligned(path: Path, sample: pd.DataFrame, column: str):
    """`column` of the file at `path` for the sample's rows, or None without the file."""
    if not path.exists():
        return None
    other = pd.read_parquet(path, columns=[*KEY, column])
    return sample.merge(other, on=KEY, how="left")[column].to_numpy()


def cmd_validate(cfg: dict, which: str, n: int = 256, seed: int = 42,
                 batch_size: int = 4) -> None:
    import gc

    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM

    from hiring_bias_mitigation.eval.margins import decision_margins

    plan = scoring_plan(cfg, which)
    scorer = load_scorer(cfg, plan)
    rows = validation_sample(len(plan["prompts"]), n, seed)
    prompts = [plan["prompts"][i] for i in rows]
    # HuggingFace first: it hands its memory back entirely before vLLM sizes its cache.
    model = AutoModelForCausalLM.from_pretrained(scorer["model_path"], dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda")
    hf = decision_margins(model, scorer["tokenizer"], prompts, lang=cfg["lang"],
                          batch_size=batch_size,
                          chat_template_kwargs=cfg.get("chat_template_kwargs"),
                          words=scorer["words"])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(**llm_kwargs(cfg, scorer["model_path"]))
    hire, reject, _ = score_logprobs(llm, [scorer["texts"][i] for i in rows],
                                     scorer["token_ids"], chunk=len(rows))
    sample = plan["frame"].iloc[rows][KEY].reset_index(drop=True)
    raw = Path(resolve_output_path("outputs/raw")) / f"{plan['run']}.parquet"
    greedy = _aligned(raw, sample, "decision") if which == "adapter" else None
    v1 = _aligned(out_dir() / f"{plan['run']}--{which}.parquet", sample, "margin")
    summary = validation_summary(hire - reject, hf, greedy, v1)
    record = {
        "standard": STANDARD, "run": plan["run"], "which": which,
        "config": cfg.get("_config_path"), "n": len(rows), "seed": seed,
        "thresholds": {"agreement": MIN_AGREEMENT, "rank_correlation": MIN_RANK_CORRELATION},
        **summary,
        "model": {"served": scorer["model_path"], "adapters": plan["adapters"]},
        "software": software(), "code": code_state(), "written": _now(),
    }
    path = reports_dir("validation") / f"{plan['run']}--{which}.json"
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not summary["passed"]:
        raise SystemExit(f"validation failed ({path}): the vLLM scorer does not reproduce the "
                         "reference quantity on this stack")
    log.info("validation passed: %s", path)


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
    v = sub.add_parser("validate", help="check the scorer against HuggingFace; once per vLLM version")
    v.add_argument("--config", required=True)
    v.add_argument("--which", choices=("base", "adapter"), required=True)
    v.add_argument("--n", type=int, default=256, help="prompts in the fixed sample")
    v.add_argument("--batch-size", type=int, default=4, help="HuggingFace batch size")
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
        cmd_compare_arms(cfg, load_config(args.against))
    elif args.cmd == "validate":
        cmd_validate(cfg, args.which, args.n, batch_size=args.batch_size)
    else:
        cmd_compare(cfg)
    if args.cmd != "validate":
        _refresh_reference_rows(cfg, getattr(args, "which", "adapter"))


def _refresh_reference_rows(cfg: dict, which: str) -> None:
    """Brings the reference-metrics rows up to date with what this command just wrote.

    A margin pass fills a run's confidence columns and a comparison its matched-comparison
    columns (docs/EVAL_STANDARD.md). A base pass belongs to the baselines of the config's
    model and language. The rows are descriptive, so a failure is logged and never raised.
    """
    try:
        from hiring_bias_mitigation.analysis import reference_metrics as RM

        names = (RM.baselines_of(cfg["model"], cfg["lang"]) if which == "base"
                 else [R.build_run_name(cfg)])
        for path in RM.refresh(names):
            log.info("refreshed %s", path)
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        log.warning("reference metrics not refreshed: %s", exc)


if __name__ == "__main__":
    main()
