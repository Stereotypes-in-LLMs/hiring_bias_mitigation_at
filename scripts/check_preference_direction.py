"""Checks a batch of preference pairs before any DPO step is spent on it.

    python scripts/check_preference_direction.py --batch artifacts/semisynthetic-v2/batch-01
    python scripts/check_preference_direction.py --batch <dir> \\
        --student Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en --also-base
    python scripts/check_preference_direction.py \\
        --pairs artifacts/semisynthetic-v1/dpo_validation.parquet

Without `--student` only the GPU-free checks run (D1-D3, D5-D7): validity of the direction,
covertness, wording separability, rationale overlap, decision balance and yield. With
`--student` the SFT student's own log-probabilities and decision margins are read under
HF + PEFT (D4) -- never through vLLM's LoRA path, which does not reproduce these adapters.

Writes `direction_report.json` and `direction_report.md` beside the pairs, and prints the
markdown. Definitions and thresholds: generation/direction_check.py and the v2 spec, §6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.generation import direction_check as DC  # noqa: E402
from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("check_preference_direction")


def load_pairs(batch: str | None, pairs: str | None) -> tuple[pd.DataFrame, Path, dict | None]:
    """The pairs to check, where to write beside them, and the batch's own report if any."""
    if pairs:
        path = Path(resolve_output_path(pairs))
        frame = pd.read_parquet(path)
        report_path = path.parent / "report.json"
    else:
        root = Path(resolve_output_path(batch))
        path = root / "dpo_v2_candidates.parquet"
        frame = pd.read_parquet(path)
        report_path = root / "report.json"
    generation_report = json.loads(report_path.read_text()) if report_path.exists() else None
    # v1 files carry neither column: the reference verdict is the chosen one by construction.
    if "reference_decision" not in frame.columns:
        frame["reference_decision"] = DC.canonical_decisions(frame, "chosen")
    if "rejected_strategy" not in frame.columns:
        frame["rejected_strategy"] = "v1"
    return frame, path.parent, generation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", help="a batch directory written by generate_negatives_v2.py")
    source.add_argument("--pairs", help="any (prompt, chosen, rejected) parquet, e.g. v1's")
    parser.add_argument("--student", help="adapter directory or Hub adapter id (D4)")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--also-base", action="store_true",
                        help="also measure the base model with the adapter disabled")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="check only the first N pairs (smoke)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-model", action="store_true",
                        help="explicitly skip D4 even if --student is given")
    args = parser.parse_args()

    frame, out_dir, generation_report = load_pairs(args.batch, args.pairs)
    if args.limit:
        frame = frame.head(args.limit)
    log.info("%d pairs from %s", len(frame), out_dir)

    report = DC.surface_report(frame, seed=args.seed, generation_report=generation_report)
    student = None
    if args.student and not args.no_model:
        scored, student = DC.student_report(frame, args.base_model, args.student,
                                            batch_size=args.batch_size, also_base=args.also_base)
        scored.to_parquet(out_dir / "direction_student_scores.parquet", index=False)
        report["D4_student"] = student

    markdown = DC.render_markdown(report, student)
    (out_dir / "direction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=N._json_default),
        encoding="utf-8")
    (out_dir / "direction_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"written: {out_dir / 'direction_report.md'}")


if __name__ == "__main__":
    main()
