"""On-policy v2 negatives: the SFT student's own attribute-tipped mistakes as `rejected`.

    CFG=configs/generation/negatives_v2_onpolicy_gh200.yaml
    python scripts/generate_onpolicy_negatives.py --config $CFG --stage sample
    python scripts/generate_onpolicy_negatives.py --config $CFG --stage judge
    python scripts/generate_onpolicy_negatives.py --config $CFG --stage sample --dry-run

Runs on the rows of an existing teacher batch (base_batch_dir, e.g. batch-01), so the two
kinds of negative are compared on identical prompts and identical `chosen` texts.

  sample  merges the published SFT adapter into its base (the audit's validated serving path;
          vLLM's LoRA path does not reproduce these adapters), checks that the merged model's
          greedy decisions agree with the HF + PEFT margins the direction check measured, draws
          k samples per row with and without the attribute, and selects one covert, attribute-
          tipped mistake per row (generation/onpolicy.py).
  judge   has the teacher judge the selected negatives as it judged its own, reuses the base
          batch's chosen-side judgements (never regenerating them), writes the candidates, and
          a compare/ directory holding these and the base batch's strategies side by side.

Every model pass is dumped under raw/ and reused only for identical prompts.
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
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("generate_onpolicy_negatives")


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=N._json_default)


def _cached_samples(path: Path, prompts: list[str], n: int, backend) -> list[list[str]]:
    """k samples per prompt, reused only when the dump holds these prompts and this k."""
    if path.exists():
        cached = pd.read_parquet(path)
        per_row = cached.sort_values(["row", "sample_index"]).groupby("row", sort=True)
        if (per_row["prompt"].first().tolist() == list(prompts)
                and int(cached["sample_index"].max()) + 1 == n):
            log.info("resuming: %s already sampled -- skipping", path.name)
            return [g.sort_values("sample_index")["raw_output"].tolist()
                    for _, g in cached.groupby("row", sort=True)]
        log.warning("%s holds different prompts or k -- resampling", path)
    samples = backend.generate_samples(prompts, n)
    rows = [{"row": i, "sample_index": j, "prompt": p, "raw_output": t}
            for i, (p, outs) in enumerate(zip(prompts, samples)) for j, t in enumerate(outs)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return samples


def stage_sample(cfg: dict, base: Path, out: Path, dry_run: bool) -> None:
    batch = pd.read_parquet(base / "batch.parquet")
    lang = str(cfg.get("lang", "en"))
    attributed, free = OP.attributed_prompts(batch), OP.attr_free_prompts(batch)
    k = int(cfg.get("samples_per_prompt", 8))
    log.info("%d rows from %s; %d samples per prompt, with and without the attribute",
             len(batch), base, k)
    if dry_run:
        pd.DataFrame({"attributed": attributed, "attr_free": free}).to_parquet(
            out / "prompts_student.parquet", index=False)
        print(f"dry run: {len(attributed)} + {len(free)} prompts -> "
              f"{out / 'prompts_student.parquet'}")
        return

    model = OP.merged_student(cfg)
    spec = {"model": str(model), "backend": "vllm",
            "generation": cfg.get("student_generation", {}), **cfg.get("student_vllm", {})}
    backend = BK.build_backend(spec)
    try:
        # Is the merged checkpoint, served by vLLM, the model the direction check measured?
        with N._generation(backend, {"temperature": 0.0, "max_new_tokens": 200}):
            greedy = N._generate_cached(out / "raw" / "student_greedy.parquet", attributed,
                                        backend, None, batch)
        samples_attr = _cached_samples(out / "raw" / "student_samples_attributed.parquet",
                                       attributed, k, backend)
        samples_free = _cached_samples(out / "raw" / "student_samples_attr_free.parquet",
                                       free, k, backend)
    finally:
        backend.close()

    report: dict = {"student": f"{cfg['student_base']} + {cfg['student_adapter']} (merged)",
                    "merged_checkpoint": str(model), "student_generation":
                    cfg.get("student_generation", {}), "samples_per_prompt": k}
    scores_path = base / "direction_student_scores.parquet"
    if scores_path.exists():
        scores = pd.read_parquet(scores_path).drop_duplicates("pair_id")
        greedy_decision = pd.Series([parse_output(t, lang).decision for t in greedy],
                                    index=batch["pair_id"])
        shared = scores[scores["pair_id"].isin(greedy_decision.index)]
        hf = (shared["student_decision_margin"] > 0).map({True: "hire", False: "reject"})
        agree = (hf.values == greedy_decision.loc[shared["pair_id"]].values).mean()
        report["merged_vllm_vs_hf_peft_greedy_agreement"] = round(float(agree), 4)
        report["merged_vllm_vs_hf_peft_rows"] = len(shared)
        log.info("merged vLLM vs HF+PEFT greedy agreement: %.3f on %d rows", agree, len(shared))

    selected, report["selection"] = OP.select_negatives(
        batch, samples_attr, samples_free, lang=lang,
        min_free_right=float(cfg.get("min_free_right", 0.5)),
        max_length_delta=int(cfg.get("max_length_delta_words", 10)))
    selected.to_parquet(out / "selected.parquet", index=False)
    (out / "sample_report.json").write_text(_json(report), encoding="utf-8")
    print(_json(report))


def stage_judge(cfg: dict, base: Path, out: Path) -> None:
    batch = pd.read_parquet(base / "batch.parquet")
    selected = pd.read_parquet(out / "selected.parquet")
    sample_report = json.loads((out / "sample_report.json").read_text())
    min_plausibility = int(cfg.get("min_plausibility", 3))
    judge_generation = cfg.get("judge_generation")

    # The base batch's chosen-side judgements, on the same rows. Read, never regenerated: a
    # mismatch means the base batch changed under us, and writing into it would corrupt it.
    chosen_dump = pd.read_parquet(base / "raw" / "chosen_judge.parquet")
    if chosen_dump["prompt"].tolist() != N.judge_prompts(batch, "chosen"):
        raise SystemExit(f"{base}/raw/chosen_judge.parquet does not match batch.parquet")
    judged_batch = N.attach_judgements(batch, chosen_dump["raw_output"].tolist(), "chosen")
    chosen_ok, chosen_report = N.filter_by_judge(judged_batch, "chosen", None, min_plausibility)
    ok_pairs = set(chosen_ok["pair_id"])

    teacher = BK.build_backend(N.teacher_backend_config(cfg))
    try:
        raw = N._generate_cached(out / "raw" / "student_judge.parquet",
                                 N.judge_prompts(selected, "rejected"), teacher,
                                 judge_generation, selected) if len(selected) else []
    finally:
        teacher.close()

    judged = N.attach_judgements(selected, raw, "rejected") if len(selected) else selected
    report: dict = {"generator": sample_report.get("student"),
                    "judge": cfg["teacher_model"], "sampling": sample_report,
                    "chosen_judge": chosen_report, "strategies": {}}
    all_pairs = []
    for strategy in (OP.TIPPED, OP.UNTIPPED):
        part = judged[judged["rejected_strategy"] == strategy] if len(judged) else judged
        entry: dict = {"n_prompts": len(batch), "n_selected": len(part),
                       "covertness": sample_report["selection"].get("wrong_samples_raw", {}),
                       "filters": {"negatives": sample_report["selection"].get("filters", {})}}
        if len(part):
            entry["judge"] = N._judge_summary(part)
            kept, entry["filters"]["judge"] = N.filter_by_judge(part, "rejected", strategy,
                                                                min_plausibility)
            before = len(kept)
            kept = kept[kept["pair_id"].isin(ok_pairs)].reset_index(drop=True)
            entry["filters"]["chosen_judge_dropped"] = int(before - len(kept))
        else:
            kept = part
        pairs = N.build_pairs(kept, cfg["teacher_model"], f"{base.name} rows, student samples")
        if len(pairs):
            pairs["rejected_generator"] = sample_report.get("student")
            for column in ("free_right_rate", "sample_index", "n_wrong_samples"):
                pairs[column] = kept[column].values
        pairs.to_parquet(out / f"candidates_{strategy}.parquet", index=False)
        entry.update({"n_kept": len(pairs),
                      "yield": float(len(pairs) / len(batch)) if len(batch) else 0.0,
                      "separability": N.separability(pairs), "pairs": N._pair_summary(pairs)})
        report["strategies"][strategy] = entry
        all_pairs.append(pairs)

    candidates = pd.concat(all_pairs, ignore_index=True)
    candidates.to_parquet(out / "dpo_v2_candidates.parquet", index=False)
    (out / "report.json").write_text(_json(report), encoding="utf-8")
    N.write_samples(out / "samples.md", candidates, int(cfg.get("n_samples", 20)))

    # Side by side with the base batch's strategies, for check_preference_direction --pairs.
    compare = out / "compare"
    compare.mkdir(exist_ok=True)
    base_report = json.loads((base / "report.json").read_text())
    frames, merged = [candidates], {"strategies": dict(report["strategies"])}
    for strategy in cfg.get("compare_with", ["covert_flip"]):
        frames.append(pd.read_parquet(base / f"candidates_{strategy}.parquet"))
        merged["strategies"][strategy] = base_report["strategies"][strategy]
    pd.concat(frames, ignore_index=True).to_parquet(compare / "pairs.parquet", index=False)
    (compare / "report.json").write_text(_json(merged), encoding="utf-8")
    for strategy, entry in report["strategies"].items():
        print(f"  {strategy:17s} kept {entry['n_kept']:4d}/{entry['n_prompts']} | "
              f"balanced AUC {entry['separability']['auc_balanced']}")
    print(f"compare: {compare / 'pairs.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("sample", "judge"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_output_root()
    base = Path(resolve_output_path(cfg["base_batch_dir"]))
    out = Path(resolve_output_path(cfg["artifacts_dir"]))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if args.stage == "sample":
        stage_sample(cfg, base, out, args.dry_run)
    else:
        stage_judge(cfg, base, out)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
