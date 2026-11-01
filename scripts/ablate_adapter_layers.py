"""Writes a copy of a trained LoRA adapter with its update in some decoder layers removed.

    python scripts/ablate_adapter_layers.py --adapter outputs/distill/decision_distill_seed42 \\
        --layers 15 20 --out outputs/distill/decision_distill_seed42_xw15_20

An ablation of a trained adapter, not a trained arm (docs/PATCHING_DISTILL.md, section 1F).
`lora_B` of every module in layers FIRST..LAST (inclusive) is set to zero, so those modules add
nothing (the update is scaling * B @ A) and every other tensor is copied bit for bit. The layer
is read from the tensor name with the same pattern as `interp.spec.mask_adapter_layers`, which
does the complement at run time: it keeps layers FIRST..LAST and zeroes the scaling outside.
`adapter_config.json` and `run_config.json` are copied unchanged, so
`mitigation.merge.adapter_chain` stacks the copy on the same parent as the original;
`ablation.json` records the source, its hash and what was zeroed. Paths resolve under
$HBM_OUTPUT_ROOT as the configs do. CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
COPIED = ("adapter_config.json", "run_config.json")


def ablate(adapter: Path, out: Path, first: int, last: int) -> dict:
    from safetensors.numpy import load_file, save_file

    source = adapter / "adapter_model.safetensors"
    tensors = load_file(str(source))
    zeroed, modules, layers = [], set(), set()
    for name, value in tensors.items():
        match = LAYER.search(name)
        if match is None:
            raise ValueError(f"{name}: a LoRA tensor outside the decoder layers")
        layer = int(match.group(1))
        module = name.split(".lora_")[0]
        modules.add(module)
        if first <= layer <= last and ".lora_B." in name:
            tensors[name] = value * 0
            zeroed.append(name)
            layers.add(layer)
    if not zeroed:
        raise ValueError(f"{adapter}: no LoRA module in layers {first}..{last}")
    out.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    for name in COPIED:
        if (adapter / name).is_file():
            shutil.copy2(adapter / name, out / name)
    params = {"zeroed": sum(tensors[n].size for n in zeroed),
              "total": sum(v.size for v in tensors.values())}
    record = {
        "source": str(adapter),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layers_zeroed": [first, last],
        "layers_found": sorted(layers),
        "modules_total": len(modules),
        "modules_zeroed": len(zeroed),
        "lora_B_params_zeroed": params["zeroed"],
        "params_total": params["total"],
        "note": "lora_B zeroed in these layers; every other tensor copied unchanged",
    }
    (out / "ablation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--layers", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = ablate(Path(resolve_output_path(args.adapter)), Path(resolve_output_path(args.out)),
                    *args.layers)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
