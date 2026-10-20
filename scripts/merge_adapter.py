"""Merges a LoRA adapter into its base model so it can be audited without LoRA.

The audit does this itself before serving an adapter through vLLM (see
`hiring_bias_mitigation.mitigation.merge` for why); this script does it ahead of time.

    python scripts/merge_adapter.py --adapter outputs/sft/qwen3.5-9b_en_only
    # -> outputs/merged/qwen3.5-9b_en_only
    python scripts/merge_adapter.py --config configs/audit/<arm>.yaml --retention-only
    # -> outputs/merged/<arm>/merge_retention.json, weights untouched

A merge records how much of each adapter's update survived the bf16 rounding
(`merged_from.json`, key `merge_retention`). `--retention-only` measures that for a model merged
before the measurement existed. Both load the 9B base on CPU: run them in a job, not on a login
node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.merge import (  # noqa: E402
    measure_retention,
    merge,
    recorded_retention,
)
from hiring_bias_mitigation.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--adapter", nargs="+", help="adapter dirs, relative to the drive")
    which.add_argument("--config", nargs="+", help="audit configs; their mitigation.lora_path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retention-only", action="store_true",
                        help="measure the merge of an already merged adapter; weights untouched")
    parser.add_argument("--ensure-retention", action="store_true",
                        help="merge if needed, then measure the merge if none is recorded")
    args = parser.parse_args()
    adapters = args.adapter or [load_config(c)["mitigation"]["lora_path"] for c in args.config]
    for adapter in adapters:
        if args.retention_only:
            print(measure_retention(adapter))
            continue
        out = merge(adapter, force=args.force)
        print(out)
        if args.ensure_retention and recorded_retention(out) is None:
            print(measure_retention(adapter))


if __name__ == "__main__":
    main()
