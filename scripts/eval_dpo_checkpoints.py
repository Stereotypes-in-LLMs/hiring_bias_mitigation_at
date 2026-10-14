"""Decision-level evaluation of every checkpoint of a DPO run, as a curve on W&B.

    python scripts/eval_dpo_checkpoints.py --run outputs/dpo/<name> --shard 0 --shards 4
    python scripts/eval_dpo_checkpoints.py --run outputs/dpo/<name> --log

Step 0 is the starting model (base + the SFT adapter the run continued from); each finished
checkpoint-N adds its DPO adapter on top. Every step greedy-decodes the same validation
prompts (all of them unless --n) and writes <run>/decision_eval/step-NNNN.json; steps already
written are skipped, so a job cut by the time limit resumes. --log writes
decision_eval/summary.md and a W&B run (job type decision-eval, x axis = checkpoint step)
linked to the training run's id. Metrics: mitigation/decision_eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation import decision_eval as DE  # noqa: E402
from hiring_bias_mitigation.mitigation import training_common as C  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_config,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("eval_dpo_checkpoints")

SUMMARY = ("agree_reference", "hire_rate", "hire_rate_ref_hire", "hire_rate_ref_reject",
           "invalid_rate")


def run_config(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_config.json").read_text())
    path = Path(manifest["config_path"])
    return load_config(path if path.is_absolute() else REPO_ROOT / path)


def evaluate(run_dir: Path, shard: int, shards: int, n: int | None, batch_size: int,
             max_new_tokens: int) -> None:
    out_dir = run_dir / "decision_eval"
    out_dir.mkdir(exist_ok=True)
    steps = [s for s in DE.checkpoint_steps(run_dir)[shard::shards]
             if not (out_dir / f"step-{s:04d}.json").exists()]
    if not steps:
        log.info("shard %d: nothing left to evaluate", shard)
        return
    cfg = run_config(run_dir)
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])
    cfg.pop("eval_limit_rows", None)  # the curve uses the whole validation split
    validation = C.load_generated_dataset(cfg, cfg.get("data_kind", "dpo"))["validation"]
    rows = DE.eval_rows(validation.to_pandas(), n, int(cfg.get("seed", 42)))

    from peft import PeftModel

    model, tokenizer = C.load_model_and_tokenizer(cfg, for_training=False)
    model.to("cuda")
    meta = [c for c in DE.META_COLUMNS if c in rows.columns]
    for step in steps:
        current = (model if step == 0
                   else PeftModel.from_pretrained(model, str(run_dir / f"checkpoint-{step}")))
        outputs = DE.generate(current, tokenizer, rows["prompt"].tolist(),
                              batch_size=batch_size, max_new_tokens=max_new_tokens,
                              template_kwargs=cfg.get("chat_template_kwargs"))
        metrics = DE.decision_metrics(rows, outputs)
        (out_dir / f"step-{step:04d}.json").write_text(json.dumps(
            {"step": step, "metrics": metrics, "rows": rows[meta].to_dict("records"),
             "outputs": outputs}, ensure_ascii=False), encoding="utf-8")
        log.info("step %d: %s", step, {k: metrics[k] for k in SUMMARY})
        if step:
            model = current.unload()


def log_run(run_dir: Path) -> None:
    records = [json.loads(f.read_text()) for f in sorted((run_dir / "decision_eval").glob(
        "step-*.json"))]
    if not records:
        raise SystemExit(f"no decision_eval/step-*.json under {run_dir}")
    lines = ["| step | " + " | ".join(SUMMARY) + " |", "|---" * (len(SUMMARY) + 1) + "|"]
    for r in records:
        lines.append(f"| {r['step']} | " + " | ".join(
            "n/a" if r["metrics"].get(k) is None else f"{r['metrics'][k]:.3f}"
            for k in SUMMARY) + " |")
    (run_dir / "decision_eval" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import wandb

    cfg = run_config(run_dir)
    id_file = run_dir / "wandb_run_id"
    run = wandb.init(project=cfg.get("wandb_project", "hiring-bias-mitigation"),
                     name=f"decision-eval-{run_dir.name}", job_type="decision-eval",
                     config={"run_dir": str(run_dir), "n_prompts": records[0]["metrics"]["n"],
                             "training_run_id": id_file.read_text().strip()
                             if id_file.exists() else None})
    for r in records:
        run.log({k: v for k, v in r["metrics"].items() if v is not None}, step=r["step"])
    columns = [*records[0]["rows"][0].keys(), "output"] if records[0]["rows"] else ["output"]
    for r in (records[0], records[-1]):
        data = [[*row.values(), out] for row, out in zip(r["rows"][:20], r["outputs"][:20])]
        run.summary[f"samples_step_{r['step']}"] = wandb.Table(columns=columns, data=data)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a DPO output dir, e.g. outputs/dpo/<name>")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--n", type=int, help="validation prompts (default: all)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--log", action="store_true", help="summarise and log to W&B")
    args = parser.parse_args()
    run_dir = Path(resolve_output_path(args.run))
    if args.log:
        log_run(run_dir)
    else:
        evaluate(run_dir, args.shard, args.shards, args.n, args.batch_size, args.max_new_tokens)


if __name__ == "__main__":
    main()
