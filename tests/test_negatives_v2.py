"""v2 negatives: covert, flipped, judge-verified `rejected` responses on v1's prompts.

No network and no GPU. The teacher is replaced by canned strings; the v1 artifacts by small
synthetic frames shaped like `raw/en_invariant.parquet`.
"""

import json

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import negatives_v2 as N


def _response(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def _row(**overrides) -> dict:
    base = {
        "pair_id": "c0__j0",
        "candidate_id": "c0",
        "job_id": "j0",
        "lang": "en",
        "cv": "Eight years of Python, Django and PostgreSQL. Led a team of six engineers.",
        "job_description": "Senior Python engineer. Django, PostgreSQL, team lead.",
        "protected_group": "military_status",
        "protected_attr": "War veteran",
        "condition": "explicit",
        "group_label": "military status",
        "profile": (
            "Candidate's military status: War veteran\n"
            "Eight years of Python, Django and PostgreSQL. Led a team of six engineers."
        ),
        "reference_decision": "hire",
        "reference_feedback": "Strong Django and PostgreSQL match for the requirements.",
        "chosen_decision": "hire",
        "chosen_feedback": "Eight years of Django and PostgreSQL meet every listed requirement.",
    }
    base.update(overrides)
    return base


# ---- prompts ---------------------------------------------------------------------------------

def test_flip_is_an_involution():
    assert N.flip("hire") == "reject"
    assert N.flip("reject") == "hire"


def test_covert_flip_prompt_never_shows_the_attribute():
    """Attribute-free by construction: the teacher cannot leak what it never saw."""
    prompt = N.negative_prompt(_row(), "covert_flip")
    assert "War veteran" not in prompt
    assert "military status" not in prompt
    assert "Eight years of Python" in prompt  # the bare CV is there
    assert "'reject'" in prompt  # the flipped target decision
    assert "meet every listed requirement" not in prompt  # the chosen rationale is not shown


def test_minimal_edit_prompt_shows_the_chosen_rationale_and_hides_the_attribute():
    prompt = N.negative_prompt(_row(), "minimal_edit")
    assert "meet every listed requirement" in prompt
    assert "War veteran" not in prompt
    assert "'reject'" in prompt


def test_proxy_cue_prompt_shows_the_injected_profile():
    prompt = N.negative_prompt(_row(), "proxy_cue")
    assert "Candidate's military status: War veteran" in prompt
    assert "'reject'" in prompt


def test_unknown_strategy_is_an_error():
    with pytest.raises(ValueError, match="strategy"):
        N.negative_prompt(_row(), "overt")


def test_ukrainian_prompts_are_deliberately_absent():
    with pytest.raises(ValueError, match="Ukrainian"):
        N.negative_prompt(_row(lang="uk"), "covert_flip")


def test_judge_prompt_names_the_attribute_so_allusions_can_be_judged():
    prompt = N.judge_prompt(_row(), "reject", "Availability concerns for a demanding schedule.")
    assert "War veteran" in prompt
    assert "Availability concerns" in prompt
    assert "mentions_or_alludes" in prompt


# ---- lexicons --------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,group,expected",
    [
        ("Military leave could disrupt delivery.", "military_status", True),
        ("Deep experience with microservices.", "military_status", False),
        ("She lacks the required Java stack.", "gender", True),
        ("The candidate lacks the required Java stack.", "gender", False),
        ("Faith-based scheduling constraints are a concern.", "religion", True),
        ("Solid PostgreSQL and Django experience.", "religion", False),
        ("Served in the armed forces; strong discipline.", "military_status", True),
        ("She lacks Java.", "military_status", False),  # the lexicon is per group
    ],
)
def test_group_lexicon_matches_whole_words_of_the_rows_group(text, group, expected):
    assert N.mentions_group(text, group) is expected


def test_proxy_words_are_reported_not_filtered():
    assert N.proxy_words("Concerns about availability and long-term commitment.") == [
        "availability", "commitment", "long-term",
    ]
    assert N.proxy_words("Strong Django experience.") == []


# ---- hard filters ----------------------------------------------------------------------------

def _batch(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(_row(pair_id=f"c{i}__j{i}", candidate_id=f"c{i}", job_id=f"j{i}"))
    return pd.DataFrame(rows)


def test_filter_negatives_keeps_only_covert_flipped_length_matched_negatives():
    rows = _batch(7)
    raw = [
        _response("reject",  # keep
                  "Team-lead claims are unverified; no evidence of ownership at the stated scale."),
        _response("hire", "Eight years of Django meet every requirement."),  # not flipped
        _response("reject", "A war veteran may struggle to adapt to our pace."),  # names attr
        _response("reject", "Military leave could disrupt delivery timelines."),  # names the group
        _response("reject", " ".join(["weak"] * 50)),  # too long
        _response("reject", "No."),  # too short
        "I cannot produce a biased hiring rationale.",  # refusal
    ]
    kept, report = N.filter_negatives(raw, rows, "en", "covert_flip")
    assert len(kept) == 1
    assert kept.iloc[0]["rejected_decision"] == "reject"
    assert kept.iloc[0]["rejected_strategy"] == "covert_flip"
    assert bool(kept.iloc[0]["rejected_flipped_decision"]) is True
    assert bool(kept.iloc[0]["rejected_mentions_attribute"]) is False
    assert report["n_input"] == 7 and report["n_kept"] == 1
    assert report["dropped_by_reason"] == {
        "negative_not_flipped": 1,
        "negative_attribute_leak": 1,
        "negative_group_leak": 1,
        "negative_length": 2,
        "negative_refused": 1,
    }


def test_filter_negatives_drops_a_rationale_much_longer_than_its_chosen_partner():
    rows = _batch(1)  # chosen rationale is 10 words
    raw = [_response("reject", " ".join(["unverified"] * 25))]  # 25 words: 15 more than chosen
    kept, report = N.filter_negatives(raw, rows, "en", "covert_flip", max_length_delta=10)
    assert kept.empty
    assert report["dropped_by_reason"] == {"negative_length_mismatch": 1}


# ---- judge -----------------------------------------------------------------------------------

def _judgement(alludes=False, grounded=True, supports=True, plausibility=4) -> str:
    return json.dumps(
        {
            "mentions_or_alludes": alludes, "grounded": grounded,
            "supports_decision": supports, "plausibility": plausibility, "note": "ok",
        }
    )


def test_parse_judgement_reads_booleans_and_the_score():
    parsed = N.parse_judgement("Sure:\n" + _judgement(alludes=True, plausibility=2))
    assert parsed == {
        "alludes": True, "grounded": True, "supports": True, "plausibility": 2, "note": "ok",
    }
    assert N.parse_judgement("no json here") is None


def test_filter_by_judge_drops_allusions_fabrications_contradictions_and_implausible_text():
    frame = _batch(6).assign(rejected_decision="reject", rejected_feedback="x")
    raw = [
        _judgement(),
        _judgement(alludes=True),
        _judgement(grounded=False),
        _judgement(supports=False),
        _judgement(plausibility=2),
        "unparsable",
    ]
    judged = N.attach_judgements(frame, raw, side="rejected")
    kept, report = N.filter_by_judge(judged, side="rejected", strategy="covert_flip")
    assert len(kept) == 1
    assert report["dropped_by_reason"] == {
        "judge_alludes": 1, "judge_ungrounded": 1, "judge_contradicts_decision": 1,
        "judge_implausible": 1, "judge_unparsable": 1,
    }


def test_proxy_cue_records_allusions_instead_of_dropping_them():
    frame = _batch(2).assign(rejected_decision="reject", rejected_feedback="x")
    judged = N.attach_judgements(frame, [_judgement(alludes=True), _judgement()], "rejected")
    kept, report = N.filter_by_judge(judged, side="rejected", strategy="proxy_cue")
    assert len(kept) == 2
    assert kept["judge_alludes"].tolist() == [True, False]
    assert report["dropped_by_reason"] == {}


def test_chosen_side_judgements_use_their_own_prefix():
    frame = _batch(2)
    judged = N.attach_judgements(frame, [_judgement(), _judgement(grounded=False)], "chosen")
    assert "chosen_judge_plausibility" in judged.columns
    kept, report = N.filter_by_judge(judged, side="chosen", strategy=None)
    assert len(kept) == 1
    assert report["dropped_by_reason"] == {"chosen_judge_ungrounded": 1}


# ---- batch sampling --------------------------------------------------------------------------

def _invariant(n_pairs: int = 60) -> pd.DataFrame:
    """A frame shaped like the filtered v1 invariant pass: one attribute per group."""
    rows = []
    for i in range(n_pairs):
        decision = "hire" if i % 2 == 0 else "reject"
        for group, attr, label in (
            ("military_status", "War veteran", "military status"),
            ("gender", "Female", "gender"),
            ("religion", "Sikh", "religion"),
        ):
            for condition in ("explicit", "implicit"):
                rows.append(_row(
                    pair_id=f"c{i}__j{i}", candidate_id=f"c{i}", job_id=f"j{i}",
                    protected_group=group, protected_attr=attr, group_label=label,
                    condition=condition, reference_decision=decision,
                    chosen_decision=decision,
                ))
    return pd.DataFrame(rows)


def test_sample_batch_is_stratified_by_cell_and_takes_one_row_per_pair():
    batch = N.sample_batch(_invariant(60), rows_per_cell=3, exclude_candidates=set(), seed=1)
    cells = batch.groupby(["protected_group", "condition", "reference_decision"]).size()
    assert len(cells) == 12 and set(cells) == {3}
    assert batch["pair_id"].is_unique


def test_sample_batch_never_draws_an_excluded_candidate():
    excluded = {f"c{i}" for i in range(0, 60, 3)}
    batch = N.sample_batch(_invariant(60), rows_per_cell=2, exclude_candidates=excluded, seed=1)
    assert set(batch["candidate_id"]).isdisjoint(excluded)


def test_sample_batch_reports_a_short_cell_instead_of_failing():
    """Three hire pairs cannot fill a cell of five; take what exists and say so."""
    batch = N.sample_batch(_invariant(6), rows_per_cell=5, exclude_candidates=set(), seed=1)
    assert len(batch) <= 12 * 5
    assert batch["pair_id"].is_unique


def test_validation_candidates_merge_v1s_held_out_ids():
    inv = _invariant(40)
    val = N.validation_candidates(inv, extra_val_ids={"c3", "c7"}, val_fraction=0.1, seed=42)
    assert {"c3", "c7"} <= val
    assert 4 <= len(val) <= 6


# ---- pairing ---------------------------------------------------------------------------------

def test_build_pairs_renders_both_sides_like_v1_and_records_provenance():
    rows = _batch(2)
    negatives = rows.assign(
        rejected_decision="reject",
        rejected_feedback="Team-lead claims are unverified at the stated scale.",
        rejected_strategy="covert_flip",
        rejected_flipped_decision=True,
        rejected_mentions_attribute=False,
        rejected_mentions_group=False,
    )
    pairs = N.build_pairs(negatives, teacher_model="teacher", source_artifacts="v1")
    assert len(pairs) == 2
    assert json.loads(pairs["chosen"].iloc[0]) == {
        "decision": "hire",
        "feedback": "Eight years of Django and PostgreSQL meet every listed requirement.",
    }
    assert json.loads(pairs["rejected"].iloc[0])["decision"] == "reject"
    assert "smart AI hiring system" in pairs["prompt"].iloc[0]
    assert "Candidate's military status: War veteran" in pairs["prompt"].iloc[0]
    for column in ("reference_decision", "chosen_decision", "rejected_decision",
                   "rejected_strategy", "chosen_words", "rejected_words", "length_delta",
                   "token_overlap", "teacher_model", "source_artifacts", "generated_on"):
        assert column in pairs.columns, column
    assert pairs["chosen_words"].iloc[0] == 10
    assert pairs["rejected_words"].iloc[0] == 8


# ---- surface statistics ----------------------------------------------------------------------

def test_token_overlap_is_jaccard_over_word_sets():
    assert N.token_overlap("strong django skills", "strong django skills") == 1.0
    assert N.token_overlap("strong django skills", "weak java skills") == pytest.approx(1 / 5)
    assert N.token_overlap("", "x") == 0.0


def _pairs_frame(chosen: list[str], rejected: list[str], decisions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "chosen": [json.dumps({"decision": d, "feedback": c}) for c, d in zip(chosen, decisions)],
        "rejected": [json.dumps({"decision": N.flip(d), "feedback": r})
                     for r, d in zip(rejected, decisions)],
        "reference_decision": decisions,
    })


def test_separability_is_high_when_negatives_carry_a_wording_shortcut():
    n = 40
    chosen = [f"Candidate {i} lacks the required Java and Selenium experience." for i in range(n)]
    rejected = [f"We cannot hire candidates of the Sikh faith for role {i}." for i in range(n)]
    decisions = ["reject"] * n
    auc = N.separability(_pairs_frame(chosen, rejected, decisions))
    assert auc["auc_balanced"] > 0.9
    assert auc["auc_within_decision"] > 0.9
    assert auc["n_pairs"] == n


def test_separability_is_near_chance_when_only_the_decision_differs_across_a_balanced_batch():
    """Half the rows argue 'lacks X' on the chosen side, half on the rejected side.

    Polarity separates the sides *within* a reference-decision stratum by design (chosen is
    always the reference verdict), so the number that detects a wording or style shortcut is
    the AUC on the decision-balanced batch, where polarity cancels.
    """
    import random

    rng = random.Random(0)
    n = 60
    pos = [f"Strong {w} experience meets the requirement." for w in rng.sample(range(1000), n)]
    neg = [f"Lacks the required {w} experience for this role." for w in rng.sample(range(1000), n)]
    decisions = ["hire" if i % 2 == 0 else "reject" for i in range(n)]
    chosen = [p if d == "hire" else q for p, q, d in zip(pos, neg, decisions)]
    rejected = [q if d == "hire" else p for p, q, d in zip(pos, neg, decisions)]
    auc = N.separability(_pairs_frame(chosen, rejected, decisions))
    assert auc["auc_balanced"] < 0.7
    assert auc["auc_within_decision"] > 0.9


# ---- end to end, with a canned teacher -------------------------------------------------------

class _FakeTeacher:
    """Answers negative prompts with a covert flipped rationale and judge prompts with a
    clean verdict. Records how often it was called so resumption can be asserted."""

    def __init__(self, judge_alludes_every: int = 0):
        self.calls = 0
        self.judge_alludes_every = judge_alludes_every

    def generate(self, prompts):
        import re

        self.calls += 1
        out = []
        for i, prompt in enumerate(prompts):
            if prompt.startswith("You are auditing"):
                alludes = bool(self.judge_alludes_every) and i % self.judge_alludes_every == 0
                out.append(_judgement(alludes=alludes))
                continue
            target = re.search(r"exactly '(hire|reject)'", prompt).group(1)
            feedback = (
                "Team-lead claims are unverified; no evidence of ownership at scale."
                if target == "reject"
                else "Eight years of Django and PostgreSQL cover every listed requirement."
            )
            out.append(_response(target, feedback))
        return out

    def close(self):
        pass


class _Unavailable:
    def generate(self, prompts):
        raise AssertionError("the teacher must not be called when every pass is cached")

    def close(self):
        pass


def _raw_invariant(n_pairs: int = 24) -> pd.DataFrame:
    """Shaped like raw/en_invariant.parquet: the variant frame plus the teacher's raw output."""
    frame = _invariant(n_pairs).drop(columns=["chosen_decision", "chosen_feedback"])
    frame["raw_output"] = [
        _response(d, "Eight years of Django and PostgreSQL meet every listed requirement."
                  if d == "hire"
                  else "Lacks the PostgreSQL depth and team-lead scope the role lists.")
        for d in frame["reference_decision"]
    ]
    return frame


def _cfg(**overrides) -> dict:
    cfg = {
        "rows_per_cell": 1,
        "strategies": ["covert_flip", "minimal_edit"],
        "teacher_model": "fake-teacher",
        "generation": {"max_new_tokens": 64, "temperature": 0.7},
        "judge_generation": {"max_new_tokens": 64, "temperature": 0.0},
        "min_plausibility": 3,
        "max_length_delta_words": 10,
        "val_fraction": 0.1,
        "seed": 42,
        "n_samples": 5,
    }
    cfg.update(overrides)
    return cfg


def test_run_batch_writes_candidates_splits_report_and_samples(tmp_path):
    teacher = _FakeTeacher()
    report = N.run_batch(_cfg(), _raw_invariant(), teacher, tmp_path, source_artifacts="v1-test")

    assert (tmp_path / "batch.parquet").exists()
    for strategy in ("covert_flip", "minimal_edit"):
        assert (tmp_path / f"candidates_{strategy}.parquet").exists()
        assert (tmp_path / "raw" / f"{strategy}_negatives.parquet").exists()
        assert (tmp_path / "raw" / f"{strategy}_judge.parquet").exists()
    assert (tmp_path / "raw" / "chosen_judge.parquet").exists()
    candidates = pd.read_parquet(tmp_path / "dpo_v2_candidates.parquet")
    assert set(candidates["rejected_strategy"]) == {"covert_flip", "minimal_edit"}
    assert len(candidates) == 2 * 12  # one row per cell, twelve cells, two strategies
    assert (candidates["rejected_decision"] != candidates["chosen_decision"]).all()
    train = pd.read_parquet(tmp_path / "dpo_v2_train.parquet")
    assert set(train.columns) >= {"prompt", "chosen", "rejected", "rejected_strategy"}
    assert report["strategies"]["covert_flip"]["n_kept"] == 12
    assert report["strategies"]["covert_flip"]["covertness"]["raw_attribute_mention_rate"] == 0.0
    assert report["strategies"]["covert_flip"]["separability"]["n_pairs"] == 12
    assert report["batch"]["rows"] == 12
    assert json.loads((tmp_path / "report.json").read_text())["batch"]["rows"] == 12
    samples = (tmp_path / "samples.md").read_text()
    assert "## covert_flip" in samples and "rejected" in samples


def test_run_batch_resumes_from_its_raw_dumps(tmp_path):
    """A day of teacher time must not be repeated after an interruption (v1 policy)."""
    first = N.run_batch(_cfg(), _raw_invariant(), _FakeTeacher(), tmp_path, "v1-test")
    second = N.run_batch(_cfg(), _raw_invariant(), _Unavailable(), tmp_path, "v1-test")
    assert (second["strategies"]["covert_flip"]["n_kept"]
            == first["strategies"]["covert_flip"]["n_kept"])


def test_run_batch_dry_run_writes_prompts_and_loads_no_model(tmp_path):
    report = N.run_batch(_cfg(), _raw_invariant(), _Unavailable(), tmp_path, "v1-test",
                         dry_run=True)
    assert (tmp_path / "batch.parquet").exists()
    prompts = pd.read_parquet(tmp_path / "prompts_covert_flip.parquet")
    assert len(prompts) == 12
    assert "War veteran" not in prompts["prompt"].iloc[0]
    assert report["dry_run"] is True
    assert not (tmp_path / "dpo_v2_candidates.parquet").exists()


def test_run_batch_drops_pairs_whose_chosen_side_fails_the_judge(tmp_path):
    teacher = _FakeTeacher(judge_alludes_every=1)  # every judgement says "alludes"
    report = N.run_batch(_cfg(strategies=["covert_flip"]), _raw_invariant(), teacher, tmp_path,
                         "v1-test")
    assert report["strategies"]["covert_flip"]["n_kept"] == 0
    assert report["chosen_judge"]["dropped_by_reason"] == {"chosen_judge_alludes": 12}


def test_covertness_of_raw_measures_the_teacher_before_any_filter():
    rows = _batch(4)
    raw = [
        _response("reject", "A war veteran may struggle with our pace."),
        _response("reject", "Military leave is a concern."),
        _response("hire", "Fine."),
        "I cannot do this.",
    ]
    stats = N.covertness_of_raw(raw, rows)
    assert stats == {
        "n": 4, "n_usable": 3, "refusal_rate": 0.25, "flipped_rate": pytest.approx(2 / 3),
        "raw_attribute_mention_rate": pytest.approx(1 / 3),
        "raw_group_mention_rate": pytest.approx(2 / 3),
    }
