"""One GPU's share of the full-pool student sampling. Launched four times per node by
scripts/cscs/onpolicy_full.sbatch, once per GPU, after generate_onpolicy_full.py --stage prepare.

    CUDA_VISIBLE_DEVICES=0 python scripts/sample_student.py --config <cfg> --shard 0 --shards 4

Takes every chunk of prompts.parquet assigned to this shard, skips chunks already written for
exactly these prompts, and writes each finished chunk atomically, so a job stopped by the time
limit loses at most the chunk in flight and the next submission continues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("sample_student")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(resolve_output_path(cfg["artifacts_dir"]))
    prompts = pd.read_parquet(out / "prompts.parquet")["prompt"].tolist()
    chunk_size = int(cfg.get("chunk_size", 512))
    k = int(cfg.get("samples_per_prompt", 8))
    directory = out / "raw" / "samples"
    plan = OP.chunk_plan(len(prompts), chunk_size, args.shards)[args.shard]
    todo = [c for c in plan
            if not OP.chunk_done(OP.chunk_path(directory, c[0]), prompts[c[1]:c[2]], k)]
    log.info("shard %d/%d: %d of %d chunks to sample", args.shard, args.shards, len(todo),
             len(plan))
    if not todo:
        return

    model = json.loads((out / "prepare.json").read_text())["merged_checkpoint"]
    backend = BK.build_backend({"model": model, "backend": "vllm",
                                "generation": cfg.get("student_generation", {}),
                                **cfg.get("student_vllm", {})})
    try:
        for done, (chunk_id, start, end) in enumerate(todo, 1):
            samples = backend.generate_samples(prompts[start:end], k)
            OP.write_chunk(OP.chunk_path(directory, chunk_id), start, prompts[start:end], samples)
            log.info("shard %d: chunk %d written (%d/%d)", args.shard, chunk_id, done, len(todo))
    finally:
        backend.close()


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
