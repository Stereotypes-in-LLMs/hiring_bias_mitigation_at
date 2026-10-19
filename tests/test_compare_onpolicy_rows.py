"""scripts/compare_onpolicy_rows.py: per-row tipping re-scored from raw samples, paired by pair."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from hiring_bias_mitigation.generation import onpolicy as OP

_spec = importlib.util.spec_from_file_location(
    "compare_onpolicy_rows",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_onpolicy_rows.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _r(decision: str) -> str:
    return json.dumps({"decision": decision, "feedback": "Six years of Django meet the role."})


def _run(path: Path, attributed: dict, free: dict, k: int = 2) -> Path:
    """attributed: {(pair, attr): [decisions]}; free: {pair: [decisions]}; reference hire."""
    rows = pd.DataFrame([{"pair_id": p, "protected_group": "gender", "protected_attr": a,
                          "condition": "explicit", "reference_decision": "hire"}
                         for (p, a) in attributed])
    keys = list(free) + [str(i) for i in range(len(rows))]
    prompts = pd.DataFrame({"kind": ["attr_free"] * len(free) + ["attributed"] * len(rows),
                            "key": keys, "prompt": [f"prompt {path.name} {x}" for x in keys]})
    samples = [[_r(d) for d in free[p]] for p in free] + [[_r(d) for d in v]
                                                          for v in attributed.values()]
    path.mkdir()
    rows.to_parquet(path / "rows.parquet")
    prompts.to_parquet(path / "prompts.parquet")
    (path / "sample_report.json").write_text(json.dumps({"samples_per_prompt": k}))
    OP.write_chunk(OP.chunk_path(path / "raw" / "samples", 0), 0, prompts["prompt"].tolist(),
                   samples)
    return path


def test_rows_are_tipped_only_when_right_without_the_attribute_and_wrong_with_it(tmp_path):
    run = _run(tmp_path / "a",
               {("p0", "Male"): ["reject", "hire"], ("p1", "Female"): ["reject", "reject"],
                ("p0", "Agender"): ["hire", "hire"]},
               {"p0": ["hire", "hire"], "p1": ["hire", "reject"]})
    stats = C.row_stats(run, chunk_size=512)
    assert stats["n_wrong"].tolist() == [1, 2, 0]
    assert stats["free_right_rate"].tolist() == [1.0, 0.5, 1.0]
    # p1 is right only half the time without the attribute: wrong, but not attribute-tipped
    assert stats["tipped_wrong"].tolist() == [True, False, False]


def test_paired_difference_counts_only_shared_pairs():
    base = pd.DataFrame({"pair_id": ["p0", "p0", "p1"], "tipped_wrong": [True, False, False]})
    new = pd.DataFrame({"pair_id": ["p0", "p1", "p1"], "tipped_wrong": [False, True, True]})
    out = C.paired_difference(base, new, boot=200, seed=0)
    assert (out["pairs"], out["base_tipped_wrong"], out["new_tipped_wrong"]) == (2, 1, 2)
    assert abs(out["difference"] - (2 / 3 - 1 / 3)) < 1e-12
    assert out["ci95"][0] <= out["difference"] <= out["ci95"][1]
