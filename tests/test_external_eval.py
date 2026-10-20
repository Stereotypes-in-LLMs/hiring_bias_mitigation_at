"""External evaluation frames: set structure, exclusions and flags, on small synthetic fixtures.

The fixtures mimic the staged schema of each source (external.STAGED_COLUMNS plus the SG
auxiliary tables), so these tests need neither the network nor the external data on disk. One
integration test at the bottom builds two frames from the real staged data and checks them
against the counts the independent verifier reported; it skips when the data is absent.

The property every frame must have is the one the inconsistency rate is defined over: inside
one group_id the rows share the JD and the base, and differ in the cue and nothing else.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_mitigation.data import external_eval as X
from hiring_bias_mitigation.data import prompts as P
from hiring_bias_mitigation.data.external import external_root
from hiring_bias_mitigation.data.injection import ATTR_FREE, build_eval_set
from hiring_bias_mitigation.eval import audit as A
from hiring_bias_mitigation.eval import metrics as M
from hiring_bias_mitigation.eval.parsing import parse_frame

BUDGET = X.LengthBudget()


def _cv(cv_id, text, base=None, variant="", split="", extra=None, lang="en"):
    return {"cv_id": cv_id, "text": text, "lang": lang, "occupation": "", "split": split,
            "base_cv_id": base or cv_id, "variant": variant, "source_file": "fixture",
            "extra": json.dumps(extra or {})}


def _job(job_id, text, title="Role", split=""):
    return {"job_id": job_id, "text": text, "title": title, "occupation": "", "lang": "en",
            "split": split, "source_file": "fixture", "extra": "{}"}


def _pair(pair_id, cv_id, job_id, label=None, split=""):
    return {"pair_id": pair_id, "cv_id": cv_id, "job_id": job_id, "label": label,
            "label_kind": "none", "split": split, "extra": "{}"}


JD_TEXT = "Job Title: Accounts Assistant\n\nRequired Qualifications:\n- Two years of bookkeeping"


# ---------------------------------------------------------------------------------------
# LLMHiringFairnessSG fixture
# ---------------------------------------------------------------------------------------

SG_CUES = [
    # basket, k, languages, activities, volunteering, hobbies, religion, military, exp-bearing
    ("chinese_male", 0, "English, Mandarin", "Robotics Club (Hardware Lead)",
     "Repairing electronics", "Building PCs", [], [], True),
    ("chinese_female", 1, "English, Mandarin", "Girl Guides (Patrol Leader)",
     "Befriender at an elderly centre", "Baking", [], [], True),
    ("malay_male", 3, "English, Malay, Basic Arabic", "Malay Cultural Society",
     "Mosque traffic marshal", "Motorcycling", ["Arabic", "Mosque"], [], False),
    ("caucasian_female", 0, "English, Dutch, French", "Equestrian Club",
     "Riding for the Disabled", "Pilates", [], [], False),
]


def _cue_block(languages, activities="", volunteering="", hobbies="", only_languages=False):
    lines = ["", "", "**Additional Information**", f"Languages: {languages}"]
    if not only_languages:
        lines += [f"Activities: {activities}", f"Volunteering: {volunteering}",
                  f"Hobbies: {hobbies}"]
    return "\n".join(lines)


@pytest.fixture
def sg():
    cvs, ablated, diffs = [], [], []
    jobs = [_job("job_000", JD_TEXT), _job("job_001", JD_TEXT), _job("job_002", "Job Title: Nurse")]
    pairs = []
    for j in ("job_000", "job_001", "job_002"):
        base_id = f"{j}_neutral_0"
        base = f"[Full Name]\n**Work Experience**\nAccounts clerk at {j}, 2 years."
        cvs.append(_cv(base_id, base, extra={"job_id": j}))
        pairs.append(_pair(f"{j}__{base_id}", base_id, j))
        for basket, k, langs, act, vol, hob, _rel, _mil, exp_bearing in SG_CUES:
            eth, gender = basket.split("_")
            cv_id = f"{j}_{basket}_{k}"
            variant = f"ethnicity={eth};gender={gender};cue={k}"
            cvs.append(_cv(cv_id, base + _cue_block(langs, act, vol, hob), base_id, variant))
            diffs.append({"cv_id": cv_id, "experience_bearing_strict": exp_bearing,
                          "occupation_relevant": False})
            fields = "Languages"
            text = base + _cue_block(langs, only_languages=True)
            if j == "job_002" and basket == "caucasian_female":  # the ablation leak
                fields = "Languages+Activities+Volunteering+Hobbies"
                text = base + _cue_block(langs, act, vol, hob)
            ablated.append(_cv(f"ablated/1_marker_no_activities/{cv_id}", text, base_id,
                               f"{variant};ablation=1_marker_no_activities;fields={fields}"))
            ablated.append(_cv(f"ablated/0_markers_no_language/{cv_id}", base, base_id,
                               f"{variant};ablation=0_markers_no_language;fields="))
        ablated.append(_cv(f"ablated/1_marker_no_activities/{base_id}", base, base_id,
                           "ablation=1_marker_no_activities;fields="))
    inventory = pd.DataFrame([{
        "basket": b, "cue_index": k, "languages": lg, "activities": a, "volunteering": v,
        "hobbies": h, "n_languages": len(lg.split(",")), "religion_linked": json.dumps(rel),
        "military_or_uniformed_linked": json.dumps(mil), "experience_bearing_strict": exp,
    } for b, k, lg, a, v, h, rel, mil, exp in SG_CUES])
    return {"cvs": pd.DataFrame(cvs), "jobs": pd.DataFrame(jobs), "pairs": pd.DataFrame(pairs),
            "variant_diffs": pd.DataFrame(diffs), "cue_inventory": inventory,
            "cvs_ablated": pd.DataFrame(ablated)}


def test_eval_columns_match_build_eval_set():
    """The frames claim build_eval_set's schema; if injection grows a column, they must too."""
    pairs = pd.DataFrame([{"pair_id": "p", "candidate_id": "c", "job_id": "j", "cv": "cv",
                           "job_description": "jd", "job_position": "x"}])
    assert list(build_eval_set(pairs, ["military_status"], "en").columns) == X.EVAL_COLUMNS


def test_sg_native_sets_differ_only_in_the_cue_block(sg):
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    report = X.assert_frame_structure(frame, "sg_native")
    assert report["n_sets"] == 3 and report["n_sets_cue_isolation_checked"] == 3
    assert list(frame.columns[: len(X.EVAL_COLUMNS)]) == X.EVAL_COLUMNS
    for _, members in frame[frame["condition"] == X.NATIVE].groupby("group_id"):
        base = frame[(frame["pair_id"] == members["pair_id"].iloc[0]) &
                     (frame["protected_group"] == ATTR_FREE)]["cv"].iloc[0]
        assert members["cv"].nunique() == len(members) == len(SG_CUES)
        assert members["job_description"].nunique() == 1
        assert members["candidate_id"].nunique() == 1
        assert all(cv == base + cue for cv, cue in zip(members["cv"], members["cue_block"]))
    neutral = frame[frame["condition"] == "attr_free"]
    assert len(neutral) == 3 and (neutral["protected_attr"] == ATTR_FREE).all()


def test_sg_native_flags(sg):
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    by_cue = frame[frame["condition"] == X.NATIVE].drop_duplicates("cue_set").set_index("cue_set")
    assert by_cue.loc["chinese_female#1", "explicit_gender_word"]          # Girl Guides
    assert by_cue.loc["chinese_female#1", "military_or_uniformed_broad"]
    assert not by_cue.loc["chinese_female#1", "military_or_uniformed_linked"]
    assert by_cue.loc["malay_male#3", "religion_linked"]
    assert by_cue.loc["malay_male#3", "religion_terms"] == "Arabic;Mosque"
    assert not by_cue.loc["caucasian_female#0", "any_confound"]
    assert by_cue.loc["chinese_male#0", "experience_bearing"]
    assert by_cue.loc["malay_male#3", "protected_attr"] == "Malay | Male | cue 3"
    # job_000 and job_001 share a JD text: both pairs are flagged, job_002 is not.
    dup = frame.drop_duplicates("pair_id").set_index("job_id")["jd_duplicate_of"]
    assert dup["llmhiringfairness_sg:job_000"] == "llmhiringfairness_sg:job_001"
    assert dup["llmhiringfairness_sg:job_002"] == ""


def test_sg_variant_that_is_not_base_plus_block_is_refused(sg):
    cvs = sg["cvs"].copy()
    cvs.loc[cvs["cv_id"] == "job_000_chinese_male_0", "text"] = "a different CV entirely"
    with pytest.raises(ValueError, match="not a prefix"):
        X.sg_native_frame(cvs, sg["jobs"], sg["pairs"], sg["variant_diffs"], sg["cue_inventory"])


def test_sg_language_only_collapses_identical_texts_and_drops_leaked_jobs(sg):
    frame, excl = X.sg_language_only_frame(sg["cvs"], sg["cvs_ablated"], sg["jobs"],
                                           sg["pairs"], sg["cue_inventory"], BUDGET)
    X.assert_frame_structure(frame, "sg_language_only")
    assert excl["jobs_with_leaked_cue_fields"]["base_cv_ids"] == ["job_002_neutral_0"]
    assert frame["pair_id"].nunique() == 2
    native = frame[frame["condition"] == X.NATIVE]
    # chinese_male#0 and chinese_female#1 both read "English, Mandarin": one row, two sources.
    assert native.groupby("pair_id").size().tolist() == [3, 3]
    mandarin = native[native["languages"] == "English, Mandarin"]
    assert mandarin["n_source_variants"].tolist() == [2, 2]
    assert set(mandarin["protected_attr"]) == {"Chinese | L1"}
    arabic = native[native["languages"].str.contains("Arabic")]
    assert arabic["religion_linked"].all() and (arabic["religion_terms"] == "Arabic").all()


def test_sg_language_only_attributes_do_not_name_the_languages(sg):
    """Every languages line names English. With the line as the attribute, the scorer's mention
    detector (5-letter stems) counted any rationale that mentions English as naming it."""
    frame, _ = X.sg_language_only_frame(sg["cvs"], sg["cvs_ablated"], sg["jobs"], sg["pairs"],
                                        sg["cue_inventory"], BUDGET)
    attrs = set(frame.loc[frame["condition"] == X.NATIVE, "protected_attr"])
    assert attrs == {"Chinese | L1", "Malay | L1", "Caucasian | L1"}
    assert not any(M.mentions_attribute("Fluent English and Malay are a plus.", a)
                   for a in attrs if not a.startswith("Malay"))
    assert M.mentions_attribute("The candidate is Malay.", "Malay | L1")


# ---------------------------------------------------------------------------------------
# FAIRE fixture
# ---------------------------------------------------------------------------------------

FAIRE_NAMES = {
    "black": "Ade Kone", "native_american_hawaiian": "Yoki Tsinajinnie", "asian": "David Park",
    "hispanic": "Dylan Rivera", "white": "Moshe Cohen", "male": "Elijah Jimenez",
    "female": "Olivia Jimenez",
}


@pytest.fixture
def faire():
    bases = {
        "faire_r004": "HR ADMINISTRATOR   Summary   Customer service manager, 15 years.",
        "faire_r019": "IT SPECIALIST   Summary   Network administrator, 8 years.",
        "faire_r022": "IT SPECIALIST   Summary   Network administrator, 8 years.",
        "faire_r030": "TEACHER   Personal   Gender: Female   Summary   Primary teacher.",
    }
    cvs, jobs, pairs = [], [], []
    for base_id, text in bases.items():
        cvs.append(_cv(base_id, text, extra={"kaggle_match": {"kaggle_ids": ["123"]}}))
        jobs.append(_job(f"{base_id}_jd", f"We are seeking a person for {base_id}."))
        pairs.append(_pair(base_id, base_id, f"{base_id}_jd"))
        for group, name in FAIRE_NAMES.items():
            axis = "gender" if group in ("male", "female") else "race"
            eth, gen = ("", group) if axis == "gender" else (group, "")
            sep = " " * (12 if (base_id, group) == ("faire_r004", "asian") else 10)
            variant_text = f"{name}{sep}{text}"
            if (base_id, group) == ("faire_r004", "hispanic"):
                variant_text = f"{name}{sep}{text.replace('   ', '      ')}"  # whitespace drift
            cvs.append(_cv(f"{base_id}_{axis}_{group}_name", variant_text, base_id,
                           f"axis={axis};name={name};gender={gen};ethnicity={eth};type=name",
                           extra={"first_name_gender_by_lexicon": "unknown"}))
        diff = "derived_from_other_cv" if base_id == "faire_r019" else "experience_bearing"
        cvs.append(_cv(f"{base_id}_race_white_experience", text + " Church volunteer.", base_id,
                       "axis=race;name=;gender=;ethnicity=white;type=experience",
                       extra={"diff_class": diff}))
    jobs.append(_job("faire_r1722_jd", "Marketing Coordinator", split="orphan_no_cv"))
    return {"cvs": pd.DataFrame(cvs), "jobs": pd.DataFrame(jobs), "pairs": pd.DataFrame(pairs)}


def test_faire_keeps_name_variants_only_and_one_copy_of_a_duplicate_base(faire):
    frame, excl = X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"], BUDGET)
    X.assert_frame_structure(frame, "faire_names")
    assert excl["duplicate_bases"]["base_cv_ids"] == {"faire_r019": "same base text as faire_r022"}
    assert excl["experience_variants"]["n_rows"] == 4
    assert excl["orphan_jobs"]["n_jobs"] == 1
    assert sorted(frame["candidate_id"].unique()) == ["faire:faire_r004", "faire:faire_r022",
                                                      "faire:faire_r030"]
    assert not frame["variant_cv_id"].str.endswith("_experience").any()
    sizes = frame[frame["condition"] == X.NATIVE].groupby("protected_group")["group_id"].nunique()
    assert sizes.to_dict() == {"faire_gender_name": 3, "faire_race_name": 3}
    race = frame[frame["protected_group"] == "faire_race_name"]
    assert (race.groupby("group_id").size() == 5).all()


def test_faire_variants_are_rebuilt_so_only_the_name_differs(faire):
    frame, _ = X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"], BUDGET)
    rows = frame[frame["candidate_id"] == "faire:faire_r004"].set_index("variant_cv_id")
    base = rows.loc["faire:faire_r004", "cv"]
    for _, row in rows[rows["condition"] == X.NATIVE].iterrows():
        assert row["cv"] == f"{row['name']}{X.FAIRE_NAME_SEPARATOR}{base}"
    assert not rows.loc["faire:faire_r004_race_hispanic_name", "staged_text_identical"]
    assert not rows.loc["faire:faire_r004_race_asian_name", "staged_text_identical"]
    assert rows.loc["faire:faire_r004_race_black_name", "staged_text_identical"]


def test_faire_refuses_a_variant_that_is_not_name_plus_base(faire):
    cvs = faire["cvs"].copy()
    cvs.loc[cvs["cv_id"] == "faire_r004_race_black_name", "text"] = "Ade Kone   another CV"
    with pytest.raises(ValueError, match="not the name followed"):
        X.faire_names_frame(cvs, faire["jobs"], faire["pairs"])


def test_faire_flags(faire):
    frame, _ = X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"],
                                   X.LengthBudget(max_model_len=600, max_new_tokens=512))
    native = frame[frame["condition"] == X.NATIVE]
    white = native[native["protected_attr"] == "White"]
    assert white["name_religion_coded"].all() and (white["name_religion"] == "jewish").all()
    assert not native[native["protected_attr"] == "Female"]["name_religion_coded"].any()
    assert frame.groupby("candidate_id")["base_marker_hit"].first().to_dict() == {
        "faire:faire_r004": False, "faire:faire_r022": False, "faire:faire_r030": True}
    assert set(native["axis"]) == {"race", "gender"}
    # A 600-token context with 512 reserved for the answer: every prompt is over length.
    assert frame["over_max_model_len"].all()
    assert not X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"],
                                   BUDGET)[0]["over_max_model_len"].any()


# ---------------------------------------------------------------------------------------
# TalentCLEF fixture
# ---------------------------------------------------------------------------------------

def _tc_cv(cv_id, text, gender, base=None, split="dev"):
    variant = "" if base is None else f"gender={gender};sibling_of={base}"
    return _cv(cv_id, text, base, variant, split,
               extra={"gender_inferred": gender, "sibling_jaccard5": 0.7})


@pytest.fixture
def talentclef():
    cvs = [
        _tc_cv("en-dev-1", "Maria Lopez\nLanguages: English, Spanish (Native)", "female"),
        _tc_cv("en-dev-2", "Piotr Nowak\nLanguages: English, Polish (Native)\nArmy logistics",
               "male", "en-dev-1"),
        _tc_cv("en-dev-3", "Amira Hassan\nBSc, Forman Christian College", "male"),
        _tc_cv("en-dev-4", "Sarah Mitchell\nBSc, Lahore University", "female", "en-dev-3"),
        _tc_cv("en-dev-5", "Chen Wei\nAnalyst", "undetermined"),
        _tc_cv("en-dev-6", "Kofi Sesay\nAnalyst", "undetermined", "en-dev-5"),
        _tc_cv("en-dev-7", "Carlos Mendoza\nClerk", "male"),
        _tc_cv("en-dev-8", "Lucia Mendoza\nClerk", "female", "en-dev-7"),
    ]
    # Other CVs of the corpus, for the name table: Amira is a woman's name there.
    cvs += [_tc_cv(f"en-test-{i}", "Amira Khan\nEngineer", "female", split="test")
            for i in range(3)]
    jobs = [_job("en-dev-q1", "Cashier wanted.", split="dev"),
            _job("en-dev-q2", "Sales director wanted.", split="dev")]
    qrels = [("en-dev-1", "en-dev-q1"), ("en-dev-2", "en-dev-q1"), ("en-dev-3", "en-dev-q1"),
             ("en-dev-4", "en-dev-q1"), ("en-dev-3", "en-dev-q2"), ("en-dev-4", "en-dev-q2"),
             ("en-dev-5", "en-dev-q2"), ("en-dev-6", "en-dev-q2")]
    pairs = [_pair(f"{j}-{c}", c, j, label="1", split="dev") for c, j in qrels]
    return {"cvs": pd.DataFrame(cvs), "jobs": pd.DataFrame(jobs), "pairs": pd.DataFrame(pairs)}


def test_talentclef_sets_are_sibling_pairs_per_positive_jd(talentclef):
    frame, excl = X.talentclef_sibling_frame(talentclef["cvs"], talentclef["jobs"],
                                             talentclef["pairs"], budget=BUDGET)
    report = X.assert_frame_structure(frame, "talentclef_siblings")
    assert report["n_sets"] == 3 and report["n_sets_without_base_text"] == 3
    assert excl["gender_undetermined"]["base_cv_ids"] == ["en-dev-5"]
    assert excl["no_positive_jd"]["base_cv_ids"] == ["en-dev-7"]
    for _, members in frame.groupby("group_id"):
        assert sorted(members["protected_attr"]) == ["Female", "Male"]
        assert members["candidate_id"].nunique() == 1 and members["cv"].nunique() == 2
    assert (frame["condition"] == X.NATIVE).all() and (frame["human_relevance"] == 1).all()


def test_talentclef_flags(talentclef):
    frame, _ = X.talentclef_sibling_frame(talentclef["cvs"], talentclef["jobs"],
                                          talentclef["pairs"], budget=BUDGET)
    pair1 = frame[frame["candidate_id"] == "talentclef2026_a:en-dev-1"]
    pair3 = frame[frame["candidate_id"] == "talentclef2026_a:en-dev-3"]
    assert pair1["languages_differ"].all() and not pair3["languages_differ"].any()
    assert set(pair1["languages_only_here"]) == {"Spanish", "Polish"}
    assert pair3["religion_marker_differs"].all() and not pair1["religion_marker_differs"].any()
    assert set(pair3["religion_marker_terms"]) == {"christian", ""}
    # Piotr's "Army" is a pool marker, not a religion one: the any-marker flag sees it, the
    # religion flag does not.
    assert pair1.loc[pair1["first_name"] == "Piotr", "cv_marker_hit"].all()
    amira = pair3[pair3["first_name"] == "Amira"]
    assert amira["name_gender_contradiction"].all()
    assert (amira["name_gender_majority"] == "female").all()
    assert not frame[frame["first_name"] != "Amira"]["name_gender_contradiction"].any()


def test_religion_markers_are_the_religion_patterns_of_the_pool_filter_only():
    assert len(X.RELIGION_MARKERS) == 2
    for text in ("Forman Christian College", "religious studies", "Jewish community centre"):
        assert X.religion_marker_terms(text), text
    for text in ("female engineer", "Army reservist", "married", "she/her", "military"):
        assert not X.religion_marker_terms(text), text


def test_name_gender_check_needs_evidence_and_a_clear_majority():
    table = {"Wei": {"female": 14, "male": 8}, "Ana": {"female": 2},
             "Abebe": {"male": 10, "female": 1}}
    assert X.name_gender_check("Wei", "male", table) == ("", False)    # the corpus splits on it
    assert X.name_gender_check("Ana", "male", table) == ("", False)    # two others: too few
    assert X.name_gender_check("Abebe", "female", table) == ("male", True)
    assert X.name_gender_check("Abebe", "male", table) == ("male", False)


# ---------------------------------------------------------------------------------------
# RJDB fixture
# ---------------------------------------------------------------------------------------

def _rjdb(i, matched, weakened, removed, kind, split="test", names=("Ann Lee", "Ann Lee")):
    tid = f"rjdb-{split}-{i:05d}"
    fi = {"Skills": removed, "Experience": "the same job experience as before"}
    cvs = [_cv(f"{tid}-m", matched, split=split, extra={"name": names[0]}),
           _cv(f"{tid}-u", weakened, f"{tid}-m", f"filtered:skills_removed={'|'.join(removed)}",
               split, extra={"name": names[1], "Filtered-information": fi,
                             "removed_skills": removed, "experience_change_kind": kind})]
    jobs = [_job(f"{tid}-jd", f"## Job title\nAnalyst {i}", split=split)]
    pairs = [_pair(f"{tid}-match", f"{tid}-m", f"{tid}-jd", "match", split),
             _pair(f"{tid}-mismatch", f"{tid}-u", f"{tid}-jd", "mismatch", split)]
    return cvs, jobs, pairs


@pytest.fixture
def rjdb():
    parts = [
        _rjdb(0, "Skills: SQL, Python\n2019 - 2023", "Skills: Python\n2019 - 2021", ["SQL"],
              "shortened"),
        _rjdb(1, "Skills: AngularJS, Go", "Skills: Go; separate from AngularJS", ["AngularJS"],
              "dropped", names=("Michal Smith", "Michal Anderson")),
        _rjdb(2, "Skills: Excel, Word", "Skills: Excel, Word.", ["Excel"], "unchanged"),
        _rjdb(3, "train only", "train only, weaker", ["X"], "dropped", split="train"),
    ]
    return {kind: pd.DataFrame([row for part in parts for row in part[i]])
            for i, kind in enumerate(("cvs", "jobs", "pairs"))}


def test_rjdb_sets_pair_matched_with_weakened_under_one_jd(rjdb):
    frame = X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET)
    X.assert_frame_structure(frame, "rjdb_qualification")
    assert frame["pair_id"].nunique() == 3                     # the train triple is not used
    assert (frame["condition"] == X.QUALIFICATION_CONDITION).all()
    assert (frame["protected_group"] == "qualification").all()
    members = frame.groupby("group_id")["qualification"].apply(sorted).tolist()
    assert members == [["matched", "weakened"]] * 3
    flags = frame.drop_duplicates("pair_id").set_index("pair_id")
    assert flags["removed_skill_still_mentioned"].tolist() == [False, True, True]
    assert flags["no_effective_change"].tolist() == [False, False, True]
    assert flags["full_name_changed"].tolist() == [False, True, False]
    assert json.loads(flags["filtered_information"].iloc[0])["Skills"] == ["SQL"]


# ---------------------------------------------------------------------------------------
# Injected frames and scope
# ---------------------------------------------------------------------------------------

SCOPE = {
    "groups": {"lapa-12b": {"en": ["gender", "military_status", "religion"]},
               "qwen3.5-9b": {"en": ["military_status"], "uk": ["gender"]}},
    "conditions": {"lapa-12b": {"en": ["explicit", "implicit", "attr_free"]},
                   "qwen3.5-9b": {"en": ["implicit", "attr_free"]}},
}


def test_english_scope_is_the_union_in_study_order():
    scope = X.english_scope(SCOPE)
    assert scope["groups"] == ["military_status", "gender", "religion"]
    assert scope["conditions"] == ["explicit", "implicit", "attr_free"]
    assert scope["by_model"]["qwen3.5-9b"] == {"groups": ["military_status"],
                                               "conditions": ["implicit", "attr_free"]}


def test_injected_frame_excludes_marker_and_overlength_bases(faire):
    base = X.faire_base_pairs(faire["cvs"], faire["jobs"], faire["pairs"])
    frame, excl = X.injected_frame(base, "injected_faire", X.FAIRE, ["military_status"],
                                   ["explicit", "implicit", "attr_free"], None, BUDGET)
    report = X.assert_frame_structure(frame, "injected_faire")
    assert excl["base_marker_hit"]["pair_ids"] == ["faire:faire_r030"]
    assert frame["pair_id"].nunique() == 2 and report["n_sets"] == 4
    assert frame.groupby("pair_id").size().tolist() == [11, 11]   # 5 explicit + 5 implicit + 1
    for _, members in frame[frame["condition"] != "attr_free"].groupby("group_id"):
        assert members["cv"].nunique() == 1                      # injected at prompt time
        assert len({P.build_profile(r) for r in members.to_dict("records")}) == len(members)
    assert list(frame.columns[: len(X.EVAL_COLUMNS)]) == X.EVAL_COLUMNS
    assert (frame["reference_decision"] == "").all()

    tight = X.LengthBudget(max_model_len=600, max_new_tokens=512)
    empty, excl = X.injected_frame(base, "injected_faire", X.FAIRE, ["military_status"],
                                   ["explicit", "attr_free"], None, tight)
    assert empty.empty and excl["over_max_model_len"]["n_pairs"] == 2


def test_injected_gender_that_contradicts_the_base_name_is_flagged(talentclef):
    base = X.talentclef_base_pairs(talentclef["cvs"], talentclef["jobs"], talentclef["pairs"])
    frame, _ = X.injected_frame(base, "injected_talentclef", X.TALENTCLEF, ["gender"],
                                ["explicit", "attr_free"], None, BUDGET)
    rows = frame[frame["protected_group"] == "gender"].set_index(["candidate_id", "protected_attr"])
    maria = "talentclef2026_a:en-dev-1"                           # inferred female
    assert rows.loc[(maria, "Male"), "injected_gender_conflicts_base"].all()
    assert not rows.loc[(maria, "Female"), "injected_gender_conflicts_base"].any()
    assert not rows.loc[(maria, "Non-Binary"), "injected_gender_conflicts_base"].any()


def test_restrict_to_scope_keeps_attr_free(faire):
    base = X.faire_base_pairs(faire["cvs"], faire["jobs"], faire["pairs"])
    frame, _ = X.injected_frame(base, "injected_faire", X.FAIRE, ["military_status", "religion"],
                                ["explicit", "implicit", "attr_free"], None, BUDGET)
    scoped = X.restrict_to_scope(frame, ["military_status"], ["implicit", "attr_free"])
    assert set(scoped["condition"]) == {"implicit", "attr_free"}
    assert set(scoped["protected_group"]) == {"military_status", ATTR_FREE}


def test_restrict_to_scope_keeps_only_the_reference_rows_of_attr_free(rjdb):
    """attr_free means the per-pair reference row. Keyed on the condition, the RJDB frame (then
    labelled attr_free) passed a military-only scope whole: 2,000 of 2,000 rows."""
    frame = X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET)
    assert X.restrict_to_scope(frame, ["military_status"], ["implicit", "attr_free"]).empty
    as_before = frame.assign(condition="attr_free")
    assert X.restrict_to_scope(as_before, ["military_status"], ["implicit", "attr_free"]).empty


# ---------------------------------------------------------------------------------------
# The selection a runner hook would make
# ---------------------------------------------------------------------------------------

#: The English baseline audit's scope (configs/audit/qwen3.5-9b_en_baseline.yaml).
EN_BASELINE = {
    "lang": "en", "max_model_len": 8192, "generation": {"max_new_tokens": 512},
    "conditions": ["explicit", "implicit", "attr_free"],
    "protected_groups": ["military_status", "gender", "religion", "military_status_x_gender",
                         "military_status_x_religion"],
}


def test_a_native_frame_survives_the_hooks_selection(sg, talentclef):
    """The audit config's benchmark groups cut sg_native to its 100 attr_free rows, which
    scored as a usable run with no groups; a native frame must keep its own sets."""
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    naive = X.restrict_to_scope(frame, EN_BASELINE["protected_groups"],
                                EN_BASELINE["conditions"])
    assert set(naive["protected_group"]) == {ATTR_FREE}            # the trap
    selected, covered = X.select_for_audit(frame, EN_BASELINE)
    assert len(selected) == len(frame)
    assert covered["frame_kind"] == "source_built"
    assert covered["groups"] == ["sg_ethnicity_x_gender"]
    assert covered["conditions"] == [X.NATIVE, "attr_free"]
    tc, _ = X.talentclef_sibling_frame(talentclef["cvs"], talentclef["jobs"], talentclef["pairs"],
                                       budget=BUDGET)
    assert X.select_for_audit(tc, EN_BASELINE)[1]["n_sets"] == 3
    only_native, covered = X.select_for_audit(frame, EN_BASELINE, {"conditions": [X.NATIVE]})
    assert covered["conditions"] == [X.NATIVE] and len(only_native) == len(frame) - 3


def test_the_hooks_selection_scopes_injected_frames_and_records_what_they_hold(faire):
    base = X.faire_base_pairs(faire["cvs"], faire["jobs"], faire["pairs"])
    frame, _ = X.injected_frame(base, "injected_faire", X.FAIRE, ["military_status", "religion"],
                                ["explicit", "implicit", "attr_free"], None, BUDGET)
    selected, covered = X.select_for_audit(frame, EN_BASELINE)
    assert covered["frame_kind"] == "injected"
    # The config lists gender and two intersections the frame does not hold; meta must not.
    assert covered["groups"] == ["military_status", "religion"]
    assert len(selected) == len(frame)
    scoped, covered = X.select_for_audit(frame, {**EN_BASELINE, "protected_groups":
                                                 ["military_status"],
                                                 "conditions": ["implicit", "attr_free"]})
    assert covered["groups"] == ["military_status"]
    assert set(scoped["condition"]) == {"implicit", "attr_free"}
    assert X.select_for_audit(frame, {**EN_BASELINE, "limit_pairs": 1})[1]["n_pairs"] == 1


def test_the_hooks_selection_refuses_what_it_cannot_score(faire, rjdb):
    base = X.faire_base_pairs(faire["cvs"], faire["jobs"], faire["pairs"])
    frame, _ = X.injected_frame(base, "injected_faire", X.FAIRE, ["military_status"],
                                ["explicit", "attr_free"], None, BUDGET)
    with pytest.raises(ValueError, match="no counterfactual set"):
        X.select_for_audit(frame, {**EN_BASELINE, "protected_groups": ["gender"]})
    with pytest.raises(ValueError, match="language"):
        X.select_for_audit(frame, {**EN_BASELINE, "lang": "uk"})
    rj = X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET)
    with pytest.raises(ValueError, match="does not hold"):
        X.select_for_audit(rj, EN_BASELINE, {"groups": ["gender"]})
    with pytest.raises(ValueError, match="no counterfactual set"):
        X.select_for_audit(rj, {**EN_BASELINE, "max_model_len": 600})   # every pair too long


# ---------------------------------------------------------------------------------------
# The structure check catches what it is there to catch
# ---------------------------------------------------------------------------------------

def test_check_frame_flags_a_set_that_changes_more_than_the_cue(sg):
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    first = frame.index[frame["condition"] == X.NATIVE][0]
    broken_jd = frame.copy()
    broken_jd.loc[first, "job_description"] = "another JD"
    assert broken_jd.pipe(X.check_frame)["n_job_description_varies"] == 1
    broken_body = frame.copy()
    broken_body.loc[first, "cv"] = "Rewritten body." + broken_body.loc[first, "cue_block"]
    assert broken_body.pipe(X.check_frame)["n_cue_not_isolated"] == 1
    duplicate = frame.copy()
    second = frame.index[frame["condition"] == X.NATIVE][1]
    duplicate.loc[second, "protected_attr"] = duplicate.loc[first, "protected_attr"]
    assert duplicate.pipe(X.check_frame)["n_duplicate_attribute"] == 1
    with pytest.raises(AssertionError, match="malformed"):
        X.assert_frame_structure(duplicate, "fixture")


def test_check_frame_keeps_attr_free_for_the_reference_row(rjdb, sg):
    """RJDB's matched/weakened rows were once labelled attr_free; stability.py then dropped the
    whole frame and attribute_flips.py would have read the matched CV as the reference."""
    frame = X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET)
    relabelled = frame.assign(condition="attr_free")
    relabelled["group_id"] = (relabelled["pair_id"] + "::" + relabelled["protected_group"]
                              + "::attr_free")
    assert relabelled.pipe(X.check_frame)["n_attr_free_not_reference"] == len(frame)
    native = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                               sg["cue_inventory"], BUDGET)
    free = native[native["condition"] == "attr_free"]
    doubled = pd.concat([native, free.iloc[:1]], ignore_index=True)
    assert doubled.pipe(X.check_frame)["n_attr_free_rows_per_pair_above_one"] == 1


# ---------------------------------------------------------------------------------------
# Scorer compatibility, no model
# ---------------------------------------------------------------------------------------

def _score(frame, mode="random"):
    parsed = parse_frame(frame.assign(raw_output=X.fabricate_raw_outputs(frame, mode)))
    return A.score_run(parsed, "fixture", {}, similarity=None, n_permutations=50)


def test_score_run_reads_a_native_frame(sg):
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    record = _score(frame)
    assert list(record.groups) == ["sg_ethnicity_x_gender::native"]
    rows = record.groups["sg_ethnicity_x_gender::native"]["attributes"]
    assert len(rows) == len(SG_CUES)
    assert all(pd.notna(r["acceptance_rate__paired_p"]) for r in rows)
    assert record.summary["n_decided"] == len(frame)


def test_a_set_with_a_repeated_attribute_breaks_the_scorers_paired_test(sg):
    """Why every frame keeps protected_attr unique inside a group_id.

    `audit._paired_against_set` takes one value per attribute per set; two rows of one
    attribute in a set give vectors of different lengths and the paired test raises. The SG
    variants were first labelled by basket (5 cue sets share one), which would have done this.
    """
    frame = X.sg_native_frame(sg["cvs"], sg["jobs"], sg["pairs"], sg["variant_diffs"],
                              sg["cue_inventory"], BUDGET)
    by_basket = frame.assign(protected_attr=frame["protected_attr"].str.replace(r" \| cue \d+$",
                                                                                "", regex=True))
    by_basket.loc[by_basket["ethnicity"] == "Chinese", "protected_attr"] = "Chinese"
    with pytest.raises(ValueError, match="equal lengths"):
        _score(by_basket, "random")


def test_empty_reference_reads_as_zero_agreement_not_as_missing(rjdb):
    """Pins the scorer behaviour the runner hook has to deal with (see the build script).

    With reference_decision empty, metrics.add_reference_agreement compares every decision
    against `invalid output` and reports agreement 0.0, a number, where there is no reference.
    """
    frame = X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET)
    record = _score(frame, "constant_hire")
    assert record.summary["reference_agreement"] == 0.0
    assert set(record.scored["reference_bucket"]) == {"invalid output"}


def test_two_member_sets_make_per_attribute_inconsistency_restate_the_gap(rjdb, talentclef):
    """Pins why the 2-member frames' notes say to ignore per-attribute IR.

    A discordant pair is a tie, ties resolve to reject (metrics.add_inconsistency), so only
    the hired member is "inconsistent". Under a model that hires every weakened CV and rejects
    every matched one, IR(weakened) = 1 and IR(matched) = 0: the acceptance gap again, read
    as though one side were the unstable one. Swap the bias and the IRs swap.
    """
    for frame, hired, rejected in (
        (X.rjdb_qualification_frame(rjdb["cvs"], rjdb["jobs"], rjdb["pairs"], budget=BUDGET),
         "weakened", "matched"),
        (X.talentclef_sibling_frame(talentclef["cvs"], talentclef["jobs"], talentclef["pairs"],
                                    budget=BUDGET)[0], "Male", "Female"),
    ):
        record = _score(frame, "attribute_biased")
        (group,) = record.groups.values()
        rows = {r["protected_attr"]: r for r in group["attributes"]}
        assert rows[hired]["acceptance_rate"] == 1.0 and rows[rejected]["acceptance_rate"] == 0.0
        assert rows[hired]["inconsistency_rate"] == 1.0
        assert rows[rejected]["inconsistency_rate"] == 0.0
        assert group["population"]["inconsistency_rate"] == 0.5   # every set discordant


def test_fabricated_outputs_parse_as_decisions(faire):
    frame, _ = X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"], BUDGET)
    for mode in ("constant_hire", "random", "attribute_biased"):
        parsed = parse_frame(frame.assign(raw_output=X.fabricate_raw_outputs(frame, mode)))
        assert (parsed["outcome"] == "decided").all()
    with pytest.raises(ValueError):
        X.fabricate_raw_outputs(frame, "biased")


def test_sample_rows_show_two_members_and_the_bare_base(faire):
    frame, _ = X.faire_names_frame(faire["cvs"], faire["jobs"], faire["pairs"], BUDGET)
    picks = X.sample_rows(frame)
    assert len(picks) == 3 and picks["pair_id"].nunique() == 1
    assert picks["condition"].tolist() == [X.NATIVE, X.NATIVE, "attr_free"]


def test_length_budget_reads_the_audit_config():
    budget = X.LengthBudget.from_audit_config({"max_model_len": 4096,
                                               "generation": {"max_new_tokens": 256}})
    assert budget == X.LengthBudget(4096, 256)
    assert budget.over(3841) and not budget.over(3840)
    assert X.estimate_prompt_tokens("x" * 35) == 10 + X.CHAT_TEMPLATE_TOKENS


# ---------------------------------------------------------------------------------------
# Integration: the real staged data, when it is on disk
# ---------------------------------------------------------------------------------------

def _staged(slug: str) -> Path:
    path = external_root() / slug / "staged"
    if not (path / "cvs.parquet").exists():
        pytest.skip(f"{path} is not on disk")
    return path


def test_real_sg_native_matches_the_verified_counts():
    _staged(X.SG)
    data = X.load_source(X.SG)
    frame = X.sg_native_frame(data["cvs"], data["jobs"], data["pairs"], data["variant_diffs"],
                              data["cue_inventory"])
    report = X.assert_frame_structure(frame, "sg_native")
    native = frame[frame["condition"] == X.NATIVE]
    assert report["n_sets"] == 100 and len(native) == 4000
    # Verifier and profile: 26/40 experience-bearing, 8 religion-linked, 5 with explicit
    # gender words, 3 narrow / 5 broad uniformed cue sets; job_007 == job_008.
    assert native.groupby("cue_set")["experience_bearing"].first().sum() == 26
    assert native.groupby("cue_set")["religion_linked"].first().sum() == 8
    assert native.groupby("cue_set")["explicit_gender_word"].first().sum() == 5
    assert native.groupby("cue_set")["military_or_uniformed_linked"].first().sum() == 3
    assert native.groupby("cue_set")["military_or_uniformed_broad"].first().sum() == 5
    assert sorted(frame.loc[frame["jd_duplicate"], "job_id"].unique()) == [
        "llmhiringfairness_sg:job_007", "llmhiringfairness_sg:job_008"]


def test_real_faire_names_match_the_verified_counts():
    _staged(X.FAIRE)
    data = X.load_source(X.FAIRE)
    frame, excl = X.faire_names_frame(data["cvs"], data["jobs"], data["pairs"])
    X.assert_frame_structure(frame, "faire_names")
    native = frame[frame["condition"] == X.NATIVE]
    assert frame["candidate_id"].nunique() == 97 and len(native) == 97 * 7
    assert list(excl["duplicate_bases"]["base_cv_ids"]) == ["faire_r019"]
    assert frame.groupby("candidate_id")["base_marker_hit"].first().sum() == 16
    coded = native[native["name_religion_coded"]].groupby("protected_attr").size()
    assert coded["White"] == 26 and coded["Male"] == 10 and "Female" not in coded
