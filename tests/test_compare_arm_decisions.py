"""Decision differences between two audited arms (scripts/compare_arm_decisions.py)."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "compare_arm_decisions",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_arm_decisions.py")
CD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CD)


def _arm(decisions: dict) -> pd.DataFrame:
    """Two pairs, three gender variants under one condition plus the attribute-free prompt.
    p has a reject reference, q a hire reference."""
    rows = []
    for pair, ref in (("p", "reject"), ("q", "hire")):
        for attr in ("free", "a", "b", "c"):
            free = attr == "free"
            rows.append({"pair_id": pair, "condition": "attr_free" if free else "explicit",
                         "protected_group": "none" if free else "gender",
                         "protected_attr": "attr_free" if free else attr,
                         "candidate_id": f"cand-{pair}", "job_position": "Engineer",
                         "reference_decision": ref, "decision": decisions[(pair, attr)],
                         "feedback": f"{pair}/{attr}"})
    return pd.DataFrame(rows)


# SFT-like reference arm: p stable at reject, q unstable.
AGAINST = {("p", "free"): "reject", ("p", "a"): "reject", ("p", "b"): "reject",
           ("p", "c"): "reject", ("q", "free"): "reject", ("q", "a"): "reject",
           ("q", "b"): "hire", ("q", "c"): "reject"}
# The compared arm: one variant of p turns hire (breaks a stable set, a new error); q hires
# everywhere (fixes its set and three errors).
RUN = {**AGAINST, ("p", "a"): "hire", ("q", "free"): "hire", ("q", "a"): "hire",
       ("q", "c"): "hire"}


def test_each_changed_prompt_is_labelled_against_the_reference_verdict():
    diff = CD.prompt_diff(CD.normalise(_arm(AGAINST)), CD.normalise(_arm(RUN)))
    changed = diff[diff["changed"]].set_index(["pair_id", "protected_attr"])
    assert len(changed) == 4
    assert changed.loc[("p", "a"), "flip"] == "reject->hire"
    assert changed.loc[("p", "a"), "effect"] == "new_error"
    assert set(changed.loc["q", "effect"]) == {"fixes_error"}


def test_sets_pairs_and_concentration_read_the_attributed_variants_only():
    diff = CD.prompt_diff(CD.normalise(_arm(AGAINST)), CD.normalise(_arm(RUN)))
    status = CD.set_status(diff).set_index("pair_id")
    assert status.loc["p", "status"] == "broken" and status.loc["q", "status"] == "fixed"
    assert status["variants"].tolist() == [3, 3]  # the attribute-free prompt is not a variant

    pairs = CD.pair_summary(diff)
    assert pairs["pair_id"].tolist() == ["q", "p"]  # most changed first
    q = pairs.set_index("pair_id").loc["q"]
    assert (q["changed"], q["reject_to_hire"], q["fixes_error"]) == (2, 2, 2)
    assert (q["attr_free_against"], q["attr_free_run"]) == ("reject", "hire")

    conc = CD.concentration(status.reset_index()).set_index("variants_changed")
    assert conc.loc["10-50%", "sets"] == 1 and conc.loc["50-90%", "sets"] == 1
    assert conc["changed_prompts"].sum() == 3


def test_set_comparison_counts_fixed_and_broken_sets_pooled_and_per_cell():
    def rows(decisions):
        frame = CD.normalise(_arm(decisions))
        return frame[frame["condition"] != "attr_free"]

    table = CD.set_comparison(rows(AGAINST), rows(RUN)).set_index("scope")
    assert table.loc["all groups (pair x condition)", "fixed"] == 1
    assert table.loc["all groups (pair x condition)", "broken"] == 1
    assert table.loc["gender / explicit", "sets"] == 2
