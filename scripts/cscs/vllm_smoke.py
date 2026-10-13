"""Two prompts through the repo's own vLLM backend, to prove the GH200 environment end to end.

Renders two real audit prompts from v1's English data, generates with Qwen3.5-9B exactly as
the audit and the generators do (chat template, thinking off, greedy), and parses the answers
with the audit's parser. Fails loudly if vLLM cannot load the hybrid architecture or the
output does not parse -- the two things that would sink the teacher run an hour later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402


def main() -> None:
    source = resolve_output_path("artifacts/semisynthetic-v1/raw/en_invariant.parquet")
    raw = pd.read_parquet(source)
    rows = D._with_implicit(raw.drop(columns=["raw_output"]).head(2))
    prompts = [D._audit_prompt(r) for r in rows.to_dict("records")]
    backend = BK.build_backend({
        "model": "Qwen/Qwen3.5-9B", "backend": "vllm", "gpu_memory_utilization": 0.7,
        "max_model_len": 8192, "max_num_seqs": 8,
        "engine_kwargs": {"gdn_prefill_backend": "triton"},
        "generation": {"max_new_tokens": 200, "temperature": 0.0},
    })
    try:
        outputs = backend.generate(prompts)
    finally:
        backend.close()
    parsed = [parse_output(text, "en") for text in outputs]
    for text, p in zip(outputs, parsed):
        print(f"[{p.outcome}] {p.decision} | {text.strip()[:240]!r}")
    if not all(p.usable for p in parsed):
        raise SystemExit("vLLM smoke: an output did not parse")
    print("vLLM smoke OK")


if __name__ == "__main__":
    main()
