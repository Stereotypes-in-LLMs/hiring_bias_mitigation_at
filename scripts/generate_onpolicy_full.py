"""On-policy v2 negatives over v1's whole English pool, sampled on all four GPUs of a node.

    CFG=configs/generation/negatives_v2_onpolicy_full_gh200.yaml
    python scripts/generate_onpolicy_full.py --config $CFG --stage prepare
    # then scripts/sample_student.py once per GPU (scripts/cscs/onpolicy_full.sbatch does both)
    python scripts/generate_onpolicy_full.py --config $CFG --stage collect
    python scripts/generate_onpolicy_full.py --config $CFG --stage judge
    python scripts/generate_onpolicy_full.py --config $CFG --stage balance

  prepare  every pool row with v1's chosen side (v1's own filter), one attribute-free prompt per
           pair and one attribute-bearing prompt per row in prompts.parquet, and the merged
           student checkpoint (reused when already merged)
  collect  reads every sample chunk, refuses to continue while one is missing, and selects one
           covert, attribute-tipped mistake per row (generation/onpolicy.py)
  judge    the teacher judges the selected negatives and their chosen side; the survivors
           become candidates, a candidate-level train/validation split (v1's held-out
           candidates stay held out), report.json and samples.md
  balance  equal hire and reject reference verdicts per protected group inside each split,
           written to balanced/ with balance.json (the pool is 86% reject; spec D6). The DPO
           data config reads balanced/

A top-up run (`row_filter` + `topup_of` in its config) samples one slice of the pool again
with more samples per prompt, e.g. the hire rows at k=32 because the pool is 86% reject. Its
train/validation split comes from the base run's rows, so a candidate is on the same side in
both runs, and its balance stage merges the base run's pairs with its own before balancing.

Only `judge_strategies` (default: student_tipped) are judged and kept. Rows the student also
gets wrong without the attribute are counted in sample_report.json but are not
attribute-driven, so they do not train invariance.
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
from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("generate_onpolicy_full")

V1_VALIDATION_FILES = ("sft_validation.parquet", "dpo_validation.parquet")


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=N._json_default)


def _source(cfg: dict) -> Path:
    return Path(resolve_output_path(cfg["source_artifacts_dir"]))


def _topup_base(cfg: dict) -> Path | None:
    return Path(resolve_output_path(cfg["topup_of"])) if cfg.get("topup_of") else None


def stage_prepare(cfg: dict, out: Path) -> None:
    lang = str(cfg.get("lang", "en"))
    invariant = pd.read_parquet(_source(cfg) / "raw" / f"{lang}_invariant.parquet")
    rows, invariant_report = OP.pool_rows(invariant, lang)
    rows = OP.filter_rows(rows, cfg.get("row_filter"))
    free = OP.pair_free_prompts(rows)
    prompts = pd.DataFrame({
        "kind": ["attr_free"] * len(free) + ["attributed"] * len(rows),
        "key": [str(p) for p in free["pair_id"]] + [str(i) for i in range(len(rows))],
        "prompt": free["prompt"].tolist() + OP.attributed_prompts(rows),
    })
    rows.to_parquet(out / "rows.parquet", index=False)
    existing = out / "prompts.parquet"
    unchanged = (existing.exists()
                 and pd.read_parquet(existing)["prompt"].tolist() == prompts["prompt"].tolist())
    if not unchanged:
        prompts.to_parquet(existing, index=False)
    model = OP.merged_student(cfg)
    info = {"merged_checkpoint": str(model), "row_filter": cfg.get("row_filter"),
            "rows": len(rows), "pairs": len(free),
            "prompts": len(prompts), "invariant_filter": invariant_report}
    (out / "prepare.json").write_text(_json(info), encoding="utf-8")
    log.info("prepared %d rows over %d pairs: %d prompts", len(rows), len(free), len(prompts))


def stage_collect(cfg: dict, out: Path) -> None:
    rows = pd.read_parquet(out / "rows.parquet")
    prompts = pd.read_parquet(out / "prompts.parquet")
    k = int(cfg.get("samples_per_prompt", 8))
    samples = OP.read_samples(out / "raw" / "samples", prompts["prompt"].tolist(),
                              int(cfg.get("chunk_size", 512)), k)
    is_free = (prompts["kind"] == "attr_free").tolist()
    free_by_pair = {key: s for key, s, f in zip(prompts["key"], samples, is_free) if f}
    attributed = [s for s, f in zip(samples, is_free) if not f]
    selected, selection = OP.select_negatives(
        rows, attributed, OP.expand_free(rows, free_by_pair), lang=str(cfg.get("lang", "en")),
        min_free_right=float(cfg.get("min_free_right", 0.5)),
        max_length_delta=int(cfg.get("max_length_delta_words", 10)))
    selected.to_parquet(out / "selected.parquet", index=False)
    info = json.loads((out / "prepare.json").read_text())
    report = {"student": f"{cfg['student_base']} + {cfg['student_adapter']} (merged)",
              "merged_checkpoint": info["merged_checkpoint"], "samples_per_prompt": k,
              "student_generation": cfg.get("student_generation", {}), "selection": selection}
    (out / "sample_report.json").write_text(_json(report), encoding="utf-8")
    print(_json(selection))


def v1_validation_candidates(source: Path) -> set[str]:
    ids: set[str] = set()
    for name in V1_VALIDATION_FILES:
        path = source / name
        if path.exists():
            ids |= set(pd.read_parquet(path, columns=["candidate_id"])["candidate_id"].astype(str))
    return ids


def stage_judge(cfg: dict, out: Path) -> None:
    rows = pd.read_parquet(out / "rows.parquet")
    sample_report = json.loads((out / "sample_report.json").read_text())
    strategies = list(cfg.get("judge_strategies", [OP.TIPPED]))
    selected = pd.read_parquet(out / "selected.parquet")
    selected = selected[selected["rejected_strategy"].isin(strategies)].reset_index(drop=True)
    min_plausibility = int(cfg.get("min_plausibility", 3))
    judge_generation = cfg.get("judge_generation")
    log.info("judging %d selected negatives and their chosen side", len(selected))

    teacher = BK.build_backend(N.teacher_backend_config(cfg))
    try:
        rejected_raw = N._generate_cached(out / "raw" / "student_judge.parquet",
                                          N.judge_prompts(selected, "rejected"), teacher,
                                          judge_generation, selected)
        chosen_raw = N._generate_cached(out / "raw" / "chosen_judge.parquet",
                                        N.judge_prompts(selected, "chosen"), teacher,
                                        judge_generation, selected)
    finally:
        teacher.close()
    judged = N.attach_judgements(selected, rejected_raw, "rejected")
    judged = N.attach_judgements(judged, chosen_raw, "chosen")

    report: dict = {"generator": sample_report["student"], "judge": cfg["teacher_model"],
                    "sampling": sample_report, "strategies": {}}
    all_pairs = []
    for strategy in strategies:
        part = judged[judged["rejected_strategy"] == strategy].reset_index(drop=True)
        entry: dict = {"n_prompts": len(rows), "n_selected": len(part),
                       "covertness": sample_report["selection"].get("wrong_samples_raw", {}),
                       "filters": {"negatives": sample_report["selection"].get("filters", {})},
                       "judge": N._judge_summary(part) if len(part) else {}}
        kept, entry["filters"]["judge"] = N.filter_by_judge(part, "rejected", strategy,
                                                            min_plausibility)
        kept, entry["filters"]["chosen_judge"] = N.filter_by_judge(kept, "chosen", None,
                                                                   min_plausibility)
        entry["filters"]["chosen_judge_dropped"] = entry["filters"]["chosen_judge"]["n_input"] \
            - entry["filters"]["chosen_judge"]["n_kept"]
        pairs = N.build_pairs(kept, cfg["teacher_model"], "v1 en pool, student samples")
        if len(pairs):
            pairs["rejected_generator"] = sample_report["student"]
            for column in ("free_right_rate", "sample_index", "n_wrong_samples"):
                pairs[column] = kept[column].values
        pairs.to_parquet(out / f"candidates_{strategy}.parquet", index=False)
        entry.update({"n_kept": len(pairs), "yield": len(pairs) / len(rows) if len(rows) else 0.0,
                      "separability": N.separability(pairs), "pairs": N._pair_summary(pairs)})
        report["strategies"][strategy] = entry
        all_pairs.append(pairs)

    candidates = pd.concat(all_pairs, ignore_index=True)
    candidates.to_parquet(out / "dpo_v2_candidates.parquet", index=False)
    base = _topup_base(cfg)
    split_rows = pd.read_parquet(base / "rows.parquet") if base else rows
    val_ids = N.validation_candidates(split_rows, v1_validation_candidates(_source(cfg)),
                                      float(cfg.get("val_fraction", 0.05)),
                                      int(cfg.get("seed", 42)))
    is_val = candidates["candidate_id"].astype(str).isin(val_ids)
    candidates[~is_val].to_parquet(out / "dpo_v2_train.parquet", index=False)
    candidates[is_val].to_parquet(out / "dpo_v2_validation.parquet", index=False)
    report["splits"] = {"dpo_v2_train": int((~is_val).sum()),
                        "dpo_v2_validation": int(is_val.sum()),
                        "validation_candidates": len(val_ids)}
    (out / "report.json").write_text(_json(report), encoding="utf-8")
    N.write_samples(out / "samples.md", candidates, int(cfg.get("n_samples", 20)))
    for strategy, entry in report["strategies"].items():
        print(f"  {strategy:15s} kept {entry['n_kept']:5d} of {entry['n_selected']} selected | "
              f"balanced AUC {entry['separability']['auc_balanced']}")
    print(f"  splits: {report['splits']}")


def stage_balance(cfg: dict, out: Path) -> None:
    target = out / "balanced"
    target.mkdir(exist_ok=True)
    seed = int(cfg.get("seed", 42))
    base = _topup_base(cfg)
    info, parts = {}, []
    for split in ("train", "validation"):
        frame = pd.read_parquet(out / f"dpo_v2_{split}.parquet")
        merged = None
        if base:
            frame, merged = OP.merge_topup(pd.read_parquet(base / f"dpo_v2_{split}.parquet"),
                                           frame)
        balanced, info[split] = OP.balance_by_reference(frame, seed)
        if merged:
            info[split]["topup"] = merged
        balanced.to_parquet(target / f"dpo_v2_{split}.parquet", index=False)
        parts.append(balanced)
    candidates = pd.concat(parts, ignore_index=True)
    candidates.to_parquet(target / "dpo_v2_candidates.parquet", index=False)
    info["candidates"] = {"n": len(candidates),
                          "hire_share": float((candidates["reference_decision"] == "hire").mean())}
    (target / "balance.json").write_text(_json(info), encoding="utf-8")
    print(_json(info))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "collect", "judge", "balance"),
                        required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_output_root()
    out = Path(resolve_output_path(cfg["artifacts_dir"]))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    stages = {"prepare": stage_prepare, "collect": stage_collect, "judge": stage_judge,
              "balance": stage_balance}
    stages[args.stage](cfg, out)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
