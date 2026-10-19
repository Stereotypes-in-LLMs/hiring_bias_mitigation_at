"""Counts behind docs/DPO_DATA_CREATION.md (scripts/describe_dpo_data.py)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "describe_dpo_data", Path(__file__).resolve().parents[1] / "scripts" / "describe_dpo_data.py")
DD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DD)


def _completion(decision: str, words: int) -> str:
    return json.dumps({"decision": decision, "feedback": " ".join(["solid"] * words)})


def _pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "pair_id": ["a", "a", "b", "c"],
        "candidate_id": ["c1", "c1", "c2", "c3"],
        "job_id": ["j1", "j1", "j2", "j3"],
        "protected_group": ["religion", "military_status", "religion", "military_status"],
        "protected_attr": ["Jew", "War veteran", "Christian", "War veteran"],
        "condition": ["explicit", "implicit", "explicit", "explicit"],
        "reference_decision": ["hire", "hire", "reject", "reject"],
        "chosen": [_completion("hire", 10), _completion("hire", 12),
                   _completion("reject", 8), _completion("reject", 9)],
        "rejected": [_completion("reject", 11), _completion("reject", 12),
                     _completion("hire", 8), _completion("hire", 6)],
        "source": ["round1", "round1", "legacy", "round1"],
    })


def test_attribute_table_follows_the_attribute_lists_and_splits_by_reference():
    table = DD.attribute_table(_pairs())
    # military status comes first in the study's group order; within it, the list order
    assert table[["group", "attribute"]].values.tolist() == [
        ["military_status", "War veteran"], ["religion", "Christian"], ["religion", "Jew"]]
    veteran = table.iloc[0]
    assert (veteran["rows"], veteran["hire"], veteran["reject"]) == (2, 1, 1)
    assert int(table["rows"].sum()) == 4


def test_pair_summary_counts_preference_pairs_not_cv_job_pairs():
    out = DD.pair_summary(_pairs())
    assert out["preference_pairs"] == 4 and out["cv_job_pairs"] == 3
    assert out["chosen_hire_share"] == 0.5
    assert out["by_source"] == {"legacy": {"reject": 1}, "round1": {"hire": 2, "reject": 1}}
    assert out["chosen_words"]["max"] == 12.0
    assert out["rejected_minus_chosen_words"]["min"] == -3.0


def test_group_table_sums_the_attribute_rows_in_group_order():
    table = DD.group_table(_pairs())
    assert table.values.tolist() == [["military_status", 2, 1, 1], ["religion", 2, 1, 1]]


def test_condition_table_has_one_row_per_group():
    table = DD.condition_table(_pairs())
    assert table["group"].tolist() == ["military_status", "religion"]
    assert table.set_index("group").loc["religion", "explicit"] == 2
