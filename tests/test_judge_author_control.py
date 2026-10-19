"""The judge-author control (scripts/judge_author_control.py). No GPU: samples and judgements
are canned strings."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "judge_author_control",
    Path(__file__).resolve().parents[1] / "scripts" / "judge_author_control.py")
JA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(JA)

CHOSEN = "Eight years of Django and PostgreSQL meet every listed requirement for this role."
RIGHT = "Eight years of Django and PostgreSQL cover the role, and the team lead is shown."
WRONG = "Team-lead scope is unverified and the PostgreSQL depth the role needs is not shown."


def _r(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def _rows() -> pd.DataFrame:
    base = {"candidate_id": "c", "job_id": "j", "lang": "en",
            "cv": "Eight years of Python, Django and PostgreSQL. Led a team of six engineers.",
            "job_description": "Senior Python engineer. Django, PostgreSQL, team lead.",
            "protected_group": "military_status", "protected_attr": "War veteran",
            "group_label": "military status", "condition": "explicit",
            "chosen_feedback": CHOSEN, "rejected_strategy": "student_tipped",
            "sample_index": 0}
    return pd.DataFrame([
        {**base, "pair_id": "p0", "reference_decision": "hire", "chosen_decision": "hire",
         "rejected_decision": "reject", "rejected_feedback": WRONG},
        {**base, "pair_id": "p1", "reference_decision": "hire", "chosen_decision": "hire",
         "rejected_decision": "reject", "rejected_feedback": WRONG},
        {**base, "pair_id": "p2", "reference_decision": "reject", "chosen_decision": "reject",
         "rejected_decision": "hire", "rejected_feedback": RIGHT},
    ])


def test_control_is_the_first_correct_sample_that_passes_the_negatives_filters():
    rows = _rows()
    samples = {
        ("p0", "War veteran", "explicit"): [_r("reject", WRONG),
                                            _r("hire", "A war veteran fits our team well here."),
                                            _r("hire", RIGHT)],
        ("p1", "War veteran", "explicit"): [_r("reject", WRONG), _r("reject", WRONG)],
        ("p2", "War veteran", "explicit"): [_r("hire", RIGHT), _r("reject", WRONG)],
    }
    controls, report = JA.select_controls(rows, samples)

    assert controls["pair_id"].tolist() == ["p0", "p2"]
    # p0's first correct sample names the attribute, so its second correct one is taken
    assert controls["control_sample_index"].tolist() == [2, 1]
    assert (controls["control_decision"] == controls["reference_decision"]).all()
    assert controls["control_feedback"].tolist() == [RIGHT, WRONG]
    # the judged row's own negative and verdict are untouched
    assert controls["rejected_feedback"].tolist() == [WRONG, RIGHT]
    assert controls["rejected_strategy"].tolist() == ["student_tipped"] * 2
    assert controls["reference_decision"].tolist() == ["hire", "reject"]
    assert controls["n_correct_samples"].tolist() == [2, 1]
    assert report["rows_with_a_correct_sample"] == 2
    assert report["rows_with_a_control"] == 2
    assert report["filters"]["dropped_by_reason"] == {"negative_attribute_leak": 1}


def test_control_prompt_states_the_correct_verdict_and_the_attribute():
    controls, _ = JA.select_controls(
        _rows().iloc[:1], {("p0", "War veteran", "explicit"): [_r("hire", RIGHT)]})
    prompt = JA.control_prompts(controls)[0]
    assert "- decision: 'hire'" in prompt and RIGHT in prompt
    assert "military status: War veteran" in prompt


def _judgement(alludes=False, grounded=True, supports=True, plausibility=4) -> str:
    return json.dumps({"mentions_or_alludes": alludes, "grounded": grounded,
                       "supports_decision": supports, "plausibility": plausibility,
                       "note": "x"})


def test_outcome_is_filter_by_judge_row_by_row():
    frame = _rows().iloc[[0, 0, 0, 0, 0]].reset_index(drop=True)
    raw = [_judgement(), _judgement(grounded=False), _judgement(plausibility=2),
           "not json", _judgement(alludes=True, grounded=False)]
    assert JA.outcome(frame, raw, None, 3) == ["pass", "ungrounded", "implausible",
                                               "unparsable", "alludes"]


def test_position_is_zero_at_rejected_and_one_at_chosen():
    chosen = np.array([1, 1, 1, 1])
    rejected = np.array([0, 0, 1, 1])
    assert JA.position(rejected, rejected, chosen) == 0.0
    assert JA.position(chosen, rejected, chosen) == 1.0
    assert JA.position(np.array([1, 0, 1, 1]), rejected, chosen) == 0.5
    lo, hi = JA.position_ci(np.array([1, 0, 1, 1] * 50), np.tile(rejected, 50),
                            np.tile(chosen, 50))
    assert lo < 0.5 < hi


def test_summarize_round_matches_rows_and_reads_every_side(tmp_path, monkeypatch):
    rows = _rows()
    key = JA.ROW_KEY
    monkeypatch.setattr(JA, "round_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(JA, "out_dir", lambda cfg, tag: tmp_path / "judge_control")
    (tmp_path / "raw").mkdir()
    (tmp_path / "judge_control").mkdir()
    rows.to_parquet(tmp_path / "judge_control" / "rows_judged.parquet", index=False)
    # October, in a different row order than the judged rows: matching is by key
    order = [2, 0, 1]
    rows.iloc[order][key].assign(
        raw_output=[_judgement(), _judgement(grounded=False), _judgement()]
    ).to_parquet(tmp_path / "raw" / "student_judge.parquet", index=False)
    rows.iloc[order][key].assign(raw_output=[_judgement()] * 3).to_parquet(
        tmp_path / "raw" / "chosen_judge.parquet", index=False)
    controls = rows.iloc[[0, 2]].assign(control_feedback=RIGHT, control_sample_index=1)
    controls.assign(raw_output=[_judgement(), _judgement()]).to_parquet(
        tmp_path / "judge_control" / "control_judge.parquet", index=False)
    rows.iloc[[0, 1]].assign(raw_output=[_judgement(grounded=False), _judgement()]).to_parquet(
        tmp_path / "judge_control" / "retest_rejected_judge.parquet", index=False)

    rows.iloc[[0, 1]].assign(raw_output=[_judgement(supports=False), _judgement()]).to_parquet(
        tmp_path / "judge_control" / "retest_chosen_judge.parquet", index=False)

    by_cell, pooled, retest = JA.summarize_round({"min_plausibility": 3}, "", "r")

    hire = next(p for p in pooled if p["reference_decision"] == "hire"
                and p["basis"] == "october")
    assert (hire["n_rows"], hire["n_matched"]) == (2, 1)
    # p0: rejected ungrounded, chosen pass, control pass
    assert (hire["rejected_pass"], hire["chosen_pass"], hire["control_pass"]) == (0, 1, 1)
    assert hire["rejected_ungrounded"] == 1 and hire["position"] == 1.0
    assert hire["rejected_pass_all"] == 0.5
    assert retest == [{"round": "r", "side": "rejected", "n": 2, "pass_october": 0.5,
                       "pass_retest": 0.5, "agreement_pass": 1.0, "agreement_reason": 1.0,
                       "mcnemar_only_first": 0, "mcnemar_only_second": 0, "mcnemar_p": 1.0},
                      {"round": "r", "side": "chosen", "n": 2, "pass_october": 1.0,
                       "pass_retest": 0.5, "agreement_pass": 0.5, "agreement_reason": 0.5,
                       "mcnemar_only_first": 0, "mcnemar_only_second": 1, "mcnemar_p": 1.0}]
    # on the retest rows, p0's chosen now fails, so control sits above chosen
    again = next(p for p in pooled if p["basis"] == "retest")
    assert (again["n_rows"], again["n_matched"], again["chosen_pass"]) == (2, 1, 0.0)
    assert again["chosen_contradicts_decision"] == 1
    assert len(by_cell) == 2

    JA.write_ungrounded({"artifacts_dir": "x"}, "")
    text = (tmp_path / "judge_control" / "ungrounded_rejections_hire.md").read_text()
    assert "1 rows drawn" in text and WRONG in text and RIGHT in text
