"""On-policy negatives: the student's own attribute-tipped mistakes as `rejected`.

No GPU, no network: the student's samples are canned strings.
"""

import json

import pandas as pd

from hiring_bias_mitigation.generation import onpolicy as OP


def _r(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def _batch() -> pd.DataFrame:
    base = {
        "candidate_id": "c", "job_id": "j", "lang": "en",
        "cv": "Eight years of Python, Django and PostgreSQL. Led a team of six engineers.",
        "job_description": "Senior Python engineer. Django, PostgreSQL, team lead.",
        "protected_group": "military_status", "protected_attr": "War veteran",
        "group_label": "military status", "condition": "explicit",
        "reference_feedback": "Strong match.",
    }
    rows = []
    for i, ref in enumerate(["hire", "hire", "reject"]):
        rows.append({**base, "pair_id": f"p{i}", "candidate_id": f"c{i}", "job_id": f"j{i}",
                     "reference_decision": ref, "chosen_decision": ref,
                     "chosen_feedback":
                         "Eight years of Django and PostgreSQL meet every listed requirement."})
    return pd.DataFrame(rows)


COVERT = ("Team-lead scope is unverified and the PostgreSQL depth the role needs is not "
          "demonstrated.")


def test_attribute_free_prompt_drops_the_attribute_and_keeps_the_cv():
    prompt = OP.attr_free_prompts(_batch())[0]
    assert "War veteran" not in prompt and "military status" not in prompt
    assert "Eight years of Python" in prompt
    assert "smart AI hiring system" in prompt  # the audit's own template


def test_select_takes_the_first_covert_wrong_sample_and_labels_tipping():
    attributed = [
        [_r("reject", COVERT), _r("hire", "Fine match overall for the requirements listed.")],
        [_r("reject", "A war veteran may not adapt to our fast agile pace here."),
         _r("reject", COVERT)],
        [_r("reject", "Lacks the PostgreSQL depth."), _r("reject", "Lacks Django.")],
    ]
    free = [
        [_r("hire", "x x x x"), _r("hire", "x x x x")],      # right without the attribute
        [_r("hire", "x x x x"), _r("reject", "x x x x")],    # 50% right: not tipped
        [_r("reject", "x x x x"), _r("reject", "x x x x")],
    ]
    kept, report = OP.select_negatives(_batch(), attributed, free, min_free_right=0.5)

    assert list(kept["pair_id"]) == ["p0", "p1"]
    assert list(kept["rejected_strategy"]) == ["student_tipped", "student_untipped"]
    # p1's first wrong sample named the attribute, so its second one is taken
    assert list(kept["sample_index"]) == [0, 1]
    assert (kept["rejected_decision"] == "reject").all()
    assert kept["free_right_rate"].tolist() == [1.0, 0.5]

    assert report["rows"] == 3
    assert report["rows_with_a_wrong_sample"] == 2
    assert report["rows_tipped_with_a_wrong_sample"] == 1
    assert report["wrong_samples"] == 3
    assert report["wrong_samples_raw"]["raw_attribute_mention_rate"] == 1 / 3
    assert report["filters"]["dropped_by_reason"] == {"negative_attribute_leak": 1}
    assert report["selected"] == {"student_tipped": 1, "student_untipped": 1}


def test_select_on_a_batch_with_no_mistakes_is_empty_not_an_error():
    attributed = [[_r("hire", COVERT)], [_r("hire", COVERT)], [_r("reject", COVERT)]]
    free = [[_r("hire", "x")], [_r("hire", "x")], [_r("reject", "x")]]
    kept, report = OP.select_negatives(_batch(), attributed, free)
    assert kept.empty
    assert report["rows_with_a_wrong_sample"] == 0


# ---- full-pool sampling: chunks, shards, resume ----------------------------------------------

def test_chunk_plan_gives_every_item_to_exactly_one_shard():
    plan = OP.chunk_plan(n_items=10, chunk_size=3, shards=2)
    covered = sorted(i for shard in plan for _, start, end in shard for i in range(start, end))
    assert covered == list(range(10))
    assert [cid for cid, _, _ in plan[0]] == [0, 2]
    assert [cid for cid, _, _ in plan[1]] == [1, 3]
    assert plan[1][-1] == (3, 9, 10)


def test_a_finished_chunk_is_skipped_only_for_identical_prompts(tmp_path):
    path = OP.chunk_path(tmp_path, 0)
    assert not OP.chunk_done(path, ["a", "b"], k=2)
    OP.write_chunk(path, start=0, prompts=["a", "b"], samples=[["a1", "a2"], ["b1", "b2"]])
    assert OP.chunk_done(path, ["a", "b"], k=2)
    assert not OP.chunk_done(path, ["a", "c"], k=2)   # different prompts: redo
    assert not OP.chunk_done(path, ["a", "b"], k=3)   # different k: redo


def test_read_samples_reassembles_in_order_and_refuses_a_missing_chunk(tmp_path):
    prompts = ["p0", "p1", "p2", "p3", "p4"]
    OP.write_chunk(OP.chunk_path(tmp_path, 1), 2, prompts[2:4], [["c", "c"], ["d", "d"]])
    OP.write_chunk(OP.chunk_path(tmp_path, 0), 0, prompts[0:2], [["a", "a"], ["b", "b"]])
    import pytest

    with pytest.raises(FileNotFoundError, match="1 chunk"):
        OP.read_samples(tmp_path, prompts, chunk_size=2, k=2)
    OP.write_chunk(OP.chunk_path(tmp_path, 2), 4, prompts[4:], [["e", "e"]])
    assert OP.read_samples(tmp_path, prompts, chunk_size=2, k=2) == [
        ["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"], ["e", "e"]]


def test_attribute_free_prompts_are_one_per_pair_and_expand_back_to_rows():
    rows = pd.concat([_batch(), _batch().assign(protected_attr="Civilian")], ignore_index=True)
    pairs = OP.pair_free_prompts(rows)
    assert list(pairs["pair_id"]) == ["p0", "p1", "p2"]
    free = {pid: [f"s-{pid}"] for pid in pairs["pair_id"]}
    expanded = OP.expand_free(rows, free)
    assert expanded == [["s-p0"], ["s-p1"], ["s-p2"], ["s-p0"], ["s-p1"], ["s-p2"]]
