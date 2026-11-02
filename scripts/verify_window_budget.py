"""Check the near-parameter-matched window arm without loading model weights.

Run before training and on the final saved adapter. The references are the
same-seed completed r32 adapters, not estimated architecture dimensions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def adapter_parameters(path: Path) -> tuple[int, set[int], int]:
    """Read tensor shapes only, and reject non-LoRA trainable tensors."""
    total, layers, tensors = 0, set(), 0
    with safe_open(str(path / "adapter_model.safetensors"), framework="np") as handle:
        for key in handle.keys():
            if not re.search(r"\.lora_[AB](?:\.[^.]+)?\.weight$", key):
                raise ValueError(f"unexpected adapter tensor: {key}")
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match is None:
                raise ValueError(f"cannot identify layer: {key}")
            layers.add(int(match[1]))
            total += math.prod(handle.get_slice(key).get_shape())
            tensors += 1
    return total, layers, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    if seed not in (42, 43, 44):
        raise SystemExit("seed must be 42, 43, or 44")
    output_root = Path(os.environ["HBM_OUTPUT_ROOT"])
    if not (output_root / ".hbm-models-root").exists():
        raise SystemExit("HBM_OUTPUT_ROOT is not a registered model output root")
    expected_config = {
        "lora_r": 163, "lora_alpha": 326, "lora_layer_range": [15, 20],
        "num_train_epochs": 1, "max_steps": -1, "learning_rate": 3e-5,
    }
    for key, value in expected_config.items():
        if cfg.get(key) != value:
            raise SystemExit(f"unexpected {key}: {cfg.get(key)!r}, expected {value!r}")
    if set(cfg.get("lora_target_modules", [])) != TARGETS:
        raise SystemExit("target modules differ from the registered full-layer arm")

    base = output_root / "outputs/distill"
    full = base / f"decision_distill_seed{seed}"
    window = base / f"decision_distill_l15_20_seed{seed}"
    full_n, full_layers, full_tensors = adapter_parameters(full)
    window_n, window_layers, window_tensors = adapter_parameters(window)
    if full_n != 58_195_968 or window_n != 11_403_264:
        raise SystemExit(f"reference adapter dimensions changed: full={full_n}, window={window_n}")
    if full_layers != set(range(32)) or window_layers != set(range(15, 21)):
        raise SystemExit("reference adapter layers differ from the registered comparison")
    if full_tensors != 256 or window_tensors != 52:
        raise SystemExit("reference adapter module counts differ from 128 and 26")
    for directory in (full, window):
        acfg = json.loads((directory / "adapter_config.json").read_text())
        if acfg.get("r") != 32 or acfg.get("lora_alpha") != 64:
            raise SystemExit(f"reference rank or scaling changed: {directory}")
    predicted_n = window_n // 32 * 163
    gap_pct = 100 * (predicted_n / full_n - 1)
    if abs(gap_pct) > 0.2:
        raise SystemExit(f"parameter budget gap exceeds registered 0.2%: {gap_pct}")

    data_cfg = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    train_data = output_root / data_cfg["artifacts_dir"] / "distill_train.parquet"
    if not train_data.exists():
        raise SystemExit(f"training data missing: {train_data}")
    sft = output_root / cfg["sft_checkpoint"]
    if not (sft / "adapter_model.safetensors").exists():
        raise SystemExit(f"SFT reference adapter missing: {sft}")
    # Evaluation also compares against the two completed r32 arms and SFT's
    # exact hire count. Refuse missing old scores before spending training GPUs.
    score_dir = output_root / "outputs/operating_point/v2"
    score_runs = (
        "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional",
        f"Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed{seed}",
        f"Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_l15_20_seed{seed}",
    )
    for run in score_runs:
        score = score_dir / f"{run}--adapter.parquet"
        meta = score_dir / f"{run}--adapter.meta.json"
        if not score.exists() or not meta.exists():
            raise SystemExit(f"required comparison scores missing before training: {score}")
        if json.loads(meta.read_text()).get("standard") != "op-margin-v2":
            raise SystemExit(f"comparison scores use a different standard: {meta}")

    result = {
        "phase": args.phase, "seed": seed, "config": args.config,
        "full_parameters": full_n, "old_window_parameters": window_n,
        "new_window_expected_parameters": predicted_n,
        "relative_parameter_gap_pct": gap_pct, "rank": 163, "alpha": 326,
        "layers": [15, 20], "budget_tolerance_pct": 0.2,
    }
    if args.phase == "after":
        saved = output_root / cfg["output_dir"]
        manifest = json.loads((saved / "run_config.json").read_text())
        if manifest.get("status") != "done":
            raise SystemExit("the final training manifest is not marked done")
        actual_n, actual_layers, actual_tensors = adapter_parameters(saved)
        acfg = json.loads((saved / "adapter_config.json").read_text())
        if (actual_n != predicted_n or actual_layers != window_layers
                or actual_tensors != 52 or acfg.get("r") != 163
                or acfg.get("lora_alpha") != 326):
            raise SystemExit(f"saved adapter does not match registered budget: {saved}")
        result["saved_parameters"] = actual_n
        result["saved_adapter"] = str(saved)
    out = ROOT / "reports/experiment_followup"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"window_budget_seed{seed}_{args.phase}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
