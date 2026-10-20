"""Did v1's teacher anchors come from one sampling-noise draw shared by every prompt?

    VLLM_USE_V2_MODEL_RUNNER=0 python scripts/anchor_seed_check.py   # GPU, tensor parallel 4
    scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/anchor_seed_check.sbatch

v1's English anchors were drawn on the DGX Spark by `VLLMBackend.generate()`, which puts one
`SamplingParams(seed=42)` on all 3,000 reference prompts at T=0.7. Against the Daint pool vote
(job 5073453, 8 draws per pair) they look like a hard threshold of the per-pair hire
probability: no hires among the 288 pairs voted hire 1-4 of 8 times, 95% hires among the 8/8
pairs, and 2,999 of 3,000 outputs in one format. vLLM's classic GPU model runner seeds one
`torch.Generator` per request with the request seed and draws one exponential noise vector per
decode step (vLLM 0.31: `v1/worker/gpu_model_runner.py:1281`, `random_sample` in
`v1/sample/ops/topk_topp_sampler.py`), so under one shared seed every request sees the same
noise at step t, whatever its prompt.

This draws v1's 3,000 reference prompts twice in one engine, both times on the classic runner
(the sbatch sets VLLM_USE_V2_MODEL_RUNNER=0; the run stops before generating if vLLM reports the
V2 runner anyway):

    shared42    VLLMBackend.generate(), v1's own call: one SamplingParams(seed=42) for all prompts
    per_prompt  VLLMBackend.generate_samples(n=1): seed per_prompt_seed(42, prompt) per prompt

Prompts, decoding (v1's: T=0.7, top_p 0.9, 320 new tokens), engine and placement are the same;
only the seeding differs. The pre-registration (question, control, expected observation, stop
conditions) is the "Pre-registration" section of the teacher-shared-seed notes, copied into
the commit message that introduced this file. The CPU readout against the pool vote runs
elsewhere and is not part of this script.

An arm whose parquet already holds every prompt is skipped, so a job that times out after the
first arm resumes with the second.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    require_output_root,
    resolve_output_path,
)

LANG = "en"
TEACHER = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
#: v1's raw anchor generations, in v1's row order.
REFERENCE_V1 = "artifacts/semisynthetic-v1/raw/en_reference.parquet"
OUT = "artifacts/semisynthetic-v3/anchors"
#: v1's teacher decoding (configs/generation/teacher.yaml).
DECODING = {"max_new_tokens": 320, "temperature": 0.7, "top_p": 0.9, "seed": 42}
#: GH200 placement, as configs/generation/negatives_v2_small_gh200.yaml.
SERVING = {"backend": "vllm", "tensor_parallel_size": 4, "gpu_memory_utilization": 0.80,
           "max_model_len": 8192, "max_num_seqs": 128, "dtype": "auto",
           "engine_kwargs": {"gdn_prefill_backend": "triton"}}
ARMS = ("shared42", "per_prompt")
ENV_KEYS = ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER",
            "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_WORKER_MULTIPROC_METHOD", "SLURM_JOB_ID")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def arm_path(arm: str) -> Path:
    return Path(resolve_output_path(OUT)) / f"seedcheck_{arm}.parquet"


def code_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout
    try:
        state = {"commit": git("rev-parse", "HEAD").strip(),
                 "uncommitted": [line for line in git("status", "--porcelain").splitlines()
                                 if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        state = {"commit": None, "uncommitted": None}
    state["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return state


def versions() -> dict:
    out = {}
    for name in ("vllm", "torch", "transformers"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def seeds(arm: str, prompts: list[str]) -> list[int]:
    """The request seed each prompt was drawn with."""
    if arm == "shared42":
        return [DECODING["seed"]] * len(prompts)
    return [BK.per_prompt_seed(DECODING["seed"], p) for p in prompts]


def draw(backend, arm: str, prompts: list[str]) -> list[str]:
    if arm == "shared42":
        return backend.generate(prompts)
    return [texts[0] for texts in backend.generate_samples(prompts, 1)]


def frame(ref: pd.DataFrame, arm: str, prompts: list[str], texts: list[str]) -> pd.DataFrame:
    parsed = [parse_output(t, LANG) for t in texts]
    return pd.DataFrame({
        "pair_id": ref["pair_id"].to_numpy(), "row": range(len(ref)), "arm": arm,
        "seed": seeds(arm, prompts), "prompt_sha256": [_sha256(p) for p in prompts],
        "raw_output": texts, "decision": [p.decision for p in parsed],
        "outcome": [p.outcome for p in parsed], "feedback": [p.feedback for p in parsed]})


def summary(df: pd.DataFrame) -> dict:
    """Counts a reader of the log can check at a glance; the analysis proper runs elsewhere."""
    head = df["raw_output"].str.lstrip().str[:12]
    return {"rows": len(df), "outcome": dict(Counter(df["outcome"])),
            "decision": dict(Counter(df["decision"])),
            "hire_share_of_decided": float((df["decision"] == "hire").sum()
                                           / max(1, (df["outcome"] == "decided").sum())),
            "fenced": int(df["raw_output"].str.lstrip().str.startswith("```").sum()),
            "first_12_chars_top3": Counter(head).most_common(3)}


def runner_is_v2(backend) -> bool | None:
    try:
        return bool(backend._llm.llm_engine.vllm_config.use_v2_model_runner)
    except AttributeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N prompts only (a smoke run; writes nothing under OUT)")
    args = ap.parse_args()
    require_output_root()
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        sys.exit("FATAL: set VLLM_USE_V2_MODEL_RUNNER=0; this check is about the classic runner")

    ref = pd.read_parquet(resolve_output_path(REFERENCE_V1))
    if args.limit:
        ref = ref.head(args.limit)
    ref = ref.reset_index(drop=True)
    prompts = SY.reference_prompts(ref, LANG)
    todo = [a for a in args.arms
            if args.limit or not (arm_path(a).exists()
                                  and len(pd.read_parquet(arm_path(a), columns=["row"]))
                                  == len(ref))]
    for arm in set(args.arms) - set(todo):
        print(f"resuming: {arm_path(arm)} is complete")
    if not todo:
        return

    backend = BK.build_backend({"model": TEACHER, "generation": DECODING, **SERVING})
    backend._load()
    v2 = runner_is_v2(backend)
    print(f"vllm_config.use_v2_model_runner = {v2}")
    if v2 is not False:
        backend.close()
        sys.exit(f"FATAL: vLLM reports use_v2_model_runner={v2}; stopping before any draw")
    try:
        for arm in todo:
            start = time.time()
            texts = draw(backend, arm, prompts)
            seconds = round(time.time() - start, 1)
            df = frame(ref, arm, prompts, texts)
            info = summary(df)
            print(f"{arm}: {seconds} s | {json.dumps(info, default=str)}")
            if args.limit:
                continue
            path = arm_path(arm)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            meta = {
                "arm": arm, "model": TEACHER, "prompts": len(prompts),
                "prompt_builder": "synth.reference_prompts(en_reference.parquet, 'en')",
                "source": REFERENCE_V1, "decoding": DECODING, "serving": SERVING,
                "call": ("VLLMBackend.generate (one SamplingParams, seed 42)" if arm == "shared42"
                         else "VLLMBackend.generate_samples(n=1), per_prompt_seed(42, prompt)"),
                "use_v2_model_runner": v2, "env": {k: os.environ.get(k) for k in ENV_KEYS},
                "versions": versions(), "prompts_sha256": _sha256("\x1e".join(prompts)),
                "generation_seconds": seconds, "summary": info,
                "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "code": code_state()}
            path.with_suffix(".meta.json").write_text(
                json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"wrote {path}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
