"""Scoring arms against Claude's blind review (scripts/claude_reference.py). Synthetic data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "claude_reference", Path(__file__).resolve().parents[1] / "scripts" / "claude_reference.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)


def _free(rows: dict[str, list[str]], ref: list[str]) -> pd.DataFrame:
    """attr_free decisions: every DEFINING arm defaults to GPT-4o's decision."""
    index = [f"p{i}" for i in range(len(ref))]
    frame = pd.DataFrame({arm: rows.get(arm, ref) for arm in CR.DEFINING}, index=index)
    for arm, values in rows.items():
        frame[arm] = values
    frame[CR.REFERENCE] = ref
    return frame


def test_strata_disagree_when_any_defining_arm_departs_else_split_by_gpt4o():
    free = _free({"r1-300": ["reject", "reject", "reject", "reject"],
                  "GRPO": ["hire", "hire", "hire", "hire"]},  # not defining: ignored
                 ref=["hire", "reject", "hire", "reject"])
    free.loc["p2", "r1-300"] = "hire"
    strata = CR.assign_strata(free)
    assert strata.tolist() == ["disagree", "agree-reject", "agree-hire", "agree-reject"]


def test_free_decisions_refuses_audits_with_different_references():
    def audit(ref):
        return pd.DataFrame({"pair_id": ["a", "b"], "condition": "attr_free",
                             "decision": ["hire", "reject"], "reference_decision": ref})
    with pytest.raises(SystemExit):
        CR.free_decisions({"SFT": audit(["hire", "reject"]), "X": audit(["hire", "hire"])})


def test_fit_bucket():
    assert [CR.fit_bucket(s) for s in (1, 2, 3, 4, 5)] == ["1-2", "1-2", "3", "4-5", "4-5"]


def _design(n_disagree, k_disagree, n_hire, k_hire, n_rej, k_rej, N_hire, N_rej):
    """Graded agree indicators per stratum (k of n agree) and the stratum sizes."""
    def block(n, k):
        return [True] * k + [False] * (n - k)
    values = pd.DataFrame({"x": block(n_disagree, k_disagree) + block(n_hire, k_hire)
                           + block(n_rej, k_rej)})
    stratum = pd.Series(["disagree"] * n_disagree + ["agree-hire"] * n_hire
                        + ["agree-reject"] * n_rej)
    sizes = pd.DataFrame({"N": [n_disagree, N_hire, N_rej], "n": [n_disagree, n_hire, n_rej]},
                         index=list(CR.STRATA))
    return values, stratum, sizes


def test_stratified_estimate_weights_by_population_size_not_sample_size():
    # the README's design: 74/113 on disagree, controls 17/20 (hire) and 16/20 (reject)
    values, stratum, sizes = _design(113, 74, 20, 17, 20, 16, 67, 270)
    stratified = CR.stratified_estimate(values, stratum, sizes)["x"]
    assert stratified == pytest.approx((74 + 67 * 17 / 20 + 270 * 16 / 20) / 450)
    # the README pooled the 40 controls (33/40 = 82.5%) for all 337 agree pairs: 78.2%
    pooled = CR.pooled_estimate(values, stratum, sizes)["x"]
    assert pooled == pytest.approx((74 + 337 * 33 / 40) / 450)
    assert round(100 * pooled, 1) == 78.2
    assert stratified < pooled  # agree-reject is 80% of the agree pairs, and the worse one


def test_bootstrap_is_paired_and_degenerate_where_strata_are_constant():
    values, stratum, sizes = _design(10, 10, 4, 0, 4, 4, 20, 40)
    values["y"] = values["x"]
    draws = CR.stratified_bootstrap(values, stratum, sizes, n_boot=50, seed=1)
    point = CR.stratified_estimate(values, stratum, sizes)
    # constant within every stratum: every draw is the point estimate
    assert np.allclose(draws[:, 0], point["x"])
    values.loc[0, "x"] = values.loc[0, "y"] = False
    draws = CR.stratified_bootstrap(values, stratum, sizes, n_boot=50, seed=1)
    assert draws[:, 0].std() > 0
    assert np.array_equal(draws[:, 0], draws[:, 1])  # same draws for every column


def test_bootstrap_refuses_a_populated_stratum_with_no_graded_pair():
    values, stratum, sizes = _design(3, 2, 0, 0, 2, 1, 5, 9)
    with pytest.raises(SystemExit):
        CR.stratified_bootstrap(values, stratum, sizes, n_boot=5)


def _grades(free, strata, claude, fit, sets=None):
    rows = []
    for pid, c, f in zip(free.index, claude, fit):
        rows.append({"pair_id": pid, "set": "disagree" if strata[pid] == "disagree" else "agree",
                     "decision_gpt4o": free.loc[pid, CR.REFERENCE],
                     "decision_claude": c, "fit_score_claude": f,
                     **{col: free.loc[pid, arm] for arm, col in CR.GRADE_COLUMNS.items()}})
    grades = pd.DataFrame(rows)
    if sets is not None:
        grades["set"] = sets
    return grades


def test_check_grades_refuses_a_disagree_set_that_is_not_the_raw_parquets():
    free = _free({"SFT": ["reject", "reject", "hire"]}, ref=["hire", "reject", "hire"])
    strata = CR.assign_strata(free)
    grades = _grades(free, strata, ["hire", "reject", "hire"], [4, 2, 5])
    checks = CR.check_grades(grades, free, strata)
    assert checks["disagree_N_raw"] == 1
    assert checks["claude_decision_contradicts_fit_bucket"] == 0
    wrong = _grades(free, strata, ["hire", "reject", "hire"], [4, 2, 5],
                    sets=["disagree", "disagree", "agree"])
    with pytest.raises(SystemExit):
        CR.check_grades(wrong, free, strata)


def test_flips_against_sft_count_graded_and_ungraded_and_who_claude_sides_with():
    # p0, p1: SFT hire, GPT-4o hire; arm X rejects both. p0 is a disagree pair (r1-300
    # rejects), p1 an agree-hire pair not graded. p2: SFT reject -> X hire, graded control.
    ref = ["hire", "hire", "reject", "reject"]
    free = _free({"r1-300": ["reject", "hire", "reject", "reject"],
                  "X": ["reject", "reject", "hire", "reject"]}, ref=ref)
    strata = CR.assign_strata(free)
    grades = _grades(free.loc[["p0", "p2"]], strata, ["reject", "reject"], [2, 3])
    graded = CR.graded_frame(grades, free, strata)
    table = CR.flip_table(free, graded, strata, ["SFT", "X"]).set_index("direction")
    to_reject = table.loc["SFT hire -> reject"]
    assert (to_reject["changed"], to_reject["graded"], to_reject["ungraded"]) == (2, 1, 1)
    assert to_reject["ungraded_agree-hire"] == 1 and to_reject["graded_disagree"] == 1
    assert to_reject["claude_with_arm"] == 1 and to_reject["fit_1-2"] == 1
    assert to_reject["gpt4o_with_sft"] == 1 and to_reject["gpt4o_with_arm_all"] == 0
    to_hire = table.loc["SFT reject -> hire"]
    assert (to_hire["changed"], to_hire["graded"], to_hire["claude_with_sft"]) == (1, 1, 1)
    assert to_hire["fit_3"] == 1


def _audit(decisions: dict[tuple[str, str], list[str]]) -> pd.DataFrame:
    rows = []
    for (pair, group), ds in decisions.items():
        for i, d in enumerate(ds):
            rows.append({"pair_id": pair, "protected_group": group, "protected_attr": f"{group}{i}",
                         "condition": "explicit", "decision": d})
    return pd.DataFrame(rows)


def test_sets_are_pair_group_condition_and_incomplete_sets_are_dropped_for_every_arm():
    a = _audit({("p0", "gender"): ["hire", "reject"], ("p0", "religion"): ["hire", "hire"],
                ("p1", "gender"): ["reject", None]})
    flags = CR.set_flags(a)
    # per (pair, group): one of p0's two sets is unstable; per (pair, condition) it would be 1/1
    assert flags.loc[("p0", "gender", "explicit"), "unstable"]
    assert not flags.loc[("p0", "religion", "explicit"), "unstable"]
    b = _audit({("p0", "gender"): ["hire", "hire"], ("p0", "religion"): ["hire", "hire"],
                ("p1", "gender"): ["reject", "reject"]})
    unstable, dropped = CR.complete_flags({"SFT": flags, "B": CR.set_flags(b)})
    assert dropped == 1 and len(unstable) == 2  # p1 undecided in SFT: dropped for both arms


def test_common_rows_keeps_only_prompts_every_arm_holds():
    a = _audit({("p0", "gender"): ["hire", "reject", "hire"]})
    b = _audit({("p0", "gender"): ["hire", "reject"]})
    matched, dropped = CR.common_rows({"SFT": a, "B": b})
    assert dropped == {"SFT": 1, "B": 0}
    assert len(matched["SFT"]) == len(matched["B"]) == 2


def test_stability_table_shows_how_much_instability_the_graded_pairs_cover():
    index = pd.MultiIndex.from_tuples(
        [(p, "g", "explicit") for p in ("p0", "p1", "p2")], names=CR.SET)
    unstable = pd.DataFrame({"SFT": [True, True, False], "X": [False, True, True]}, index=index)
    strata = pd.Series({"p0": "disagree", "p1": "agree-hire", "p2": "agree-hire"})
    graded = pd.DataFrame({"fit": ["3", "4-5"], "claude_vs_gpt4o": ["agree", "disagree"],
                           "stratum": ["disagree", "agree-hire"]}, index=["p0", "p1"])
    sizes = pd.DataFrame({"N": [1, 2, 0], "n": [1, 1, 0]}, index=list(CR.STRATA))
    table = CR.stability_table(unstable, graded, strata, sizes, ["SFT", "X"])
    row = table.set_index(["arm", "breakdown", "level"])
    assert row.loc[("SFT", "population", "graded pairs"), "share_of_arm_unstable"] == 1.0
    assert row.loc[("X", "population", "graded pairs"), "share_of_arm_unstable"] == 0.5
    assert row.loc[("X", "graded: fit", "3"), "delta_vs_sft_pp"] == -100.0
    # p1 stands for both agree-hire pairs: weights 1 and 2
    assert row.loc[("SFT", "graded: all", "all"), "pairs_weighted"] == 3.0
    assert row.loc[("X", "graded: all", "all"), "unstable_share_weighted"] == pytest.approx(2 / 3)


def test_kappa_and_none_where_a_rater_is_constant():
    a = pd.Series(["hire", "hire", "reject", "reject"])
    b = pd.Series(["hire", "reject", "reject", "reject"])
    assert CR.cohen_kappa(a, b) == pytest.approx(0.5)
    assert CR.cohen_kappa(a, pd.Series(["hire"] * 4)) is None
