"""Majority-decision flips on the pool and on the benchmark (scripts/attribute_flips.py)."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "attribute_flips", Path(__file__).resolve().parents[1] / "scripts" / "attribute_flips.py")
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)


def _pool():
    rows = pd.DataFrame({"pair_id": ["a", "a", "b", "c", "d", "e"],
                         "protected_attr": ["x", "y", "x", "x", "x", "x"],
                         "condition": "explicit",
                         "reference_decision": ["reject"] * 4 + ["hire"] * 2})
    selected = pd.DataFrame({
        "pair_id": ["a", "a", "b", "c", "e"],
        "protected_attr": ["x", "y", "x", "x", "x"],
        "condition": "explicit",
        "reference_decision": ["reject", "reject", "reject", "reject", "hire"],
        # a/x strong flip, a/y majority tie (4 of 8), b noise, c wrong without the attribute too
        "n_wrong_samples": [8, 4, 1, 7, 6],
        "free_right_rate": [1.0, 1.0, 0.875, 0.25, 0.75],
        "rejected_strategy": [AF.TIPPED, AF.TIPPED, AF.TIPPED, "student_untipped", AF.TIPPED],
    })
    return rows, selected


def test_pool_counts_a_majority_flip_only_when_both_majorities_are_clear():
    rows, selected = _pool()
    out = AF.pool_flips(rows, selected, k=8)
    assert out["tipped_rows"] == 4
    assert out["strong_rows"] == 1 and out["strong_by_reference"] == {"reject": 1}
    # a/x flips toward hire; a/y is a tie, b a minority, c wrong without the attribute too
    assert out["majority_to_hire"] == {"rows": 1, "pairs": 1, "reject_reference_rows": 4,
                                       "share": 0.25}
    assert out["majority_to_reject"]["rows"] == 1  # e: hire reference, 6 of 8 say reject


def test_benchmark_reads_each_attributed_prompt_against_its_pairs_attr_free_decision():
    audit = pd.DataFrame({
        "pair_id": ["p"] * 4 + ["q"] * 3,
        "condition": ["attr_free", "explicit", "explicit", "implicit",
                      "attr_free", "explicit", "implicit"],
        "reference_decision": ["reject"] * 4 + ["hire"] * 3,
        "decision": ["reject", "hire", "hire", "reject", "hire", "reject", "hire"],
    })
    out = AF.benchmark_flips(audit)
    assert out["reject_to_hire"]["prompt_share"] == 2 / 3
    assert out["reject_to_hire"]["prompt_share_on_reference"] == 2 / 3
    assert out["reject_to_hire"]["pairs_flipped_by_half"] == 1
    assert out["hire_to_reject"]["prompt_share"] == 0.5
    assert out["hire_to_reject"]["pairs_flipped_by_half"] == 1
    assert out["hire_to_reject"]["pairs_with_any_flip"] == 1
