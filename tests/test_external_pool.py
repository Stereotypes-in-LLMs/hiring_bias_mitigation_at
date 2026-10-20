"""Multi-source training pool: Djinni unchanged by default, external sources opt-in and guarded.

No network and no external data: a small fake external root (`HBM_EXTERNAL_ROOT`) carries
every staged table the pool and the external holdout read, with leaks planted on purpose.
The one integration test that reads the real staged data skips when it is not on disk.
"""

import importlib.util
import json
import random
import string
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_mitigation.data import external as X
from hiring_bias_mitigation.generation import dataset as D
from hiring_bias_mitigation.generation import filters as F
from hiring_bias_mitigation.generation import pool as PL
from hiring_bias_mitigation.generation import sources as S
from hiring_bias_mitigation.generation import synth as SY

REPO = Path(__file__).resolve().parents[1]
_rng = random.Random(20261019)


def _words(n: int) -> str:
    """Words no marker, section or category pattern can match, and no two texts share."""
    return " ".join(
        "".join(_rng.choice(string.ascii_lowercase) for _ in range(_rng.randint(5, 9)))
        for _ in range(n)
    )


# ---- a fake external root ---------------------------------------------------------------------

def _table(kind: str, rows: list[dict]) -> pd.DataFrame:
    columns = X.STAGED_COLUMNS[kind]
    return pd.DataFrame([{c: row.get(c, "") for c in columns} for row in rows], columns=columns)


def _write(root: Path, slug: str, kind: str, rows: list[dict]) -> None:
    path = X.staged_path(slug, kind, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _table(kind, rows).to_parquet(path, index=False)


def _rjdb_cv(name: str, body: str, *, trailer: str = "") -> str:
    first = name.split()[0].lower()
    return (
        f"## Personal Information\nName: {name}\nEmail: {first}@example.com\n\n"
        f"## Education\n{body[:120]}\n\n## Skills\n- {body[120:300]}\n- Expert in C#\n\n"
        f"## Experience\n- {body[300:]} (2014 - 2017)\n{trailer}\n#"
    )


def _rjdb_jd(title: str, body: str, extra: str = "") -> str:
    return f"## Job Title\n{title}\n\n## Job Summary\n{body}\n{extra}\n\n#"


FAIRE_CV = _words(90)
RJDB_TEST_CV = _rjdb_cv("Tomas Novak", _words(80))
EEO = ("We are an equal opportunity employer. All qualified applicants will receive "
       "consideration without regard to race, religion, sex, age or protected veteran status.")


def _rjdb_triple(split: str, i: int, matched: str, weakened: str, jd: str) -> dict:
    tid = f"rjdb-{split}-{i:05d}"
    return {"tid": tid, "split": split, "m": matched, "u": weakened, "jd": jd}


def build_fake_root(root: Path) -> Path:
    triples = [
        _rjdb_triple("train", 0, _rjdb_cv("Anna Kowalski", _words(80)),
                     _rjdb_cv("Anna Kowal", _words(70)), _rjdb_jd("Data Analyst", _words(80))),
        # The body repeats the name, so the CV cannot be made name-free by cutting the header.
        _rjdb_triple("train", 1, _rjdb_cv("Lethabo Dube", _words(80), trailer="Lethabo led it."),
                     _rjdb_cv("Lethabo Dube", _words(70)), _rjdb_jd("Chemist", _words(80))),
        # A train CV that copies a held-out test CV.
        _rjdb_triple("train", 2, RJDB_TEST_CV, _rjdb_cv("Tomas Novak", _words(70)),
                     _rjdb_jd("Physicist", _words(80))),
        # A JD that makes a security clearance a requirement.
        _rjdb_triple("train", 3, _rjdb_cv("Mila Horvat", _words(80)),
                     _rjdb_cv("Mila Horvat", _words(70)),
                     _rjdb_jd("Network Admin", _words(80), "Active TS/SCI clearance required.")),
        # The weakened rewrite talks about its own construction; the matched CV is fine.
        _rjdb_triple("train", 4, _rjdb_cv("Omer Kaya", _words(80)),
                     _rjdb_cv("Omer Kaya", _words(70), trailer="Same as Resume 1."),
                     _rjdb_jd("Designer", _words(80))),
        _rjdb_triple("train", 5, _rjdb_cv("Siti Rahman", _words(80)),
                     _rjdb_cv("Siti Rahman", _words(70)), _rjdb_jd("Nurse", _words(80))),
        _rjdb_triple("dev", 0, _rjdb_cv("Ana Silva", _words(80)), _rjdb_cv("Ana Silva", _words(70)),
                     _rjdb_jd("Teacher", _words(80))),
        _rjdb_triple("test", 0, RJDB_TEST_CV, _rjdb_cv("Tomas Novak", _words(60)),
                     _rjdb_jd("Physicist", _words(80))),
    ]
    cvs, jobs, pairs = [], [], []
    for t in triples:
        for suffix, text, label in (("m", t["m"], "match"), ("u", t["u"], "mismatch")):
            name = text.split("Name: ", 1)[1].split("\n", 1)[0]
            cvs.append({"cv_id": f"{t['tid']}-{suffix}", "text": text, "lang": "en",
                        "base_cv_id": f"{t['tid']}-m", "split": t["split"],
                        "extra": json.dumps({"name": name})})
            pairs.append({"pair_id": f"{t['tid']}-{label}", "cv_id": f"{t['tid']}-{suffix}",
                          "job_id": f"{t['tid']}-jd", "label": label, "split": t["split"],
                          "extra": json.dumps({"triple_id": t["tid"]})})
        jobs.append({"job_id": f"{t['tid']}-jd", "text": t["jd"],
                     "title": t["jd"].split("\n")[1], "occupation": t["jd"].split("\n")[1],
                     "lang": "en", "split": t["split"]})
    _write(root, "rjdb", "cvs", cvs)
    _write(root, "rjdb", "jobs", jobs)
    _write(root, "rjdb", "pairs", pairs)

    def kaggle(cv_id, category, headline, text, cluster=""):
        return {"cv_id": cv_id, "text": text, "lang": "en", "occupation": category,
                "base_cv_id": cv_id, "extra": json.dumps(
                    {"Category": category, "headline": headline, "near_dup_cluster": cluster})}

    padded = "EXECUTIVE CHEF       Summary     " + "     ".join(_words(12) for _ in range(8))
    near_dup = _words(90)
    _write(root, "kaggle_resume", "cvs", [
        kaggle("k1", "CHEF", "EXECUTIVE CHEF", padded + " Reach me at chef.cook@gmail.com or "
               "(555) 432-1000, www.linkedin.com/in/chefcook"),
        kaggle("k2", "AUTOMOBILE", "SOFTWARE DEVELOPER", _words(90)),  # headline overrides
        kaggle("k3", "HR", "", _words(90)),  # no headline
        kaggle("k4", "SALES", "SALES ASSOCIATE", _words(80) + " Honorable Discharge, US Navy."),
        kaggle("k5", "TEACHER", "TEACHER", FAIRE_CV),  # a FAIRE base CV under another id
        kaggle("k6", "CHEF", "SOUS CHEF", near_dup, cluster="k7"),
        kaggle("k7", "CHEF", "SOUS CHEF", near_dup + " " + _words(3), cluster="k7"),
        kaggle("k8", "CHEF", "LINE COOK", _words(90)),
    ])

    def li(job_id, title, text, lang="en"):
        return {"job_id": job_id, "title": title, "text": text, "lang": lang}

    chef_jd = _words(90)
    it_jd = _words(90)
    _write(root, "linkedin_postings", "jobs", [
        li("j1", "Line Cook", _words(90)),
        li("j2", "Executive Chef", chef_jd),
        li("j3", "Sous Chef", chef_jd.replace(chef_jd.split()[40], "replacedword")),  # near dup
        li("j4", "Senior Software Developer", it_jd),
        li("j5", "Java Developer", _words(90)),
        li("j6", "Software Developer", _words(80) + " Active Secret clearance required."),
        li("j7", "Software Developer II", "  " + it_jd.upper() + " "),  # exact dup of j4
        li("j8", "Teacher", _words(90)),
        li("j9", "Prep Cook", _words(80) + " " + EEO),
        li("j10", "Warehouse Associate", _words(90)),  # no category
        li("j11", "Cocinero", _words(90), lang="und"),
    ])

    for slug in ("faire", "llmhiringfairness_sg", "talentclef2026_a", "vanetik_kogan"):
        _write(root, slug, "cvs", [{"cv_id": "c0", "text": FAIRE_CV if slug == "faire"
                                    else _words(90), "lang": "en"}])
        _write(root, slug, "jobs", [{"job_id": "j0", "text": _words(90), "lang": "en"}])
    return root


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = build_fake_root(tmp_path_factory.mktemp("external_data"))
    yield root
    X.holdout_index.cache_clear()


@pytest.fixture
def external(fake_root, monkeypatch):
    monkeypatch.setenv("HBM_EXTERNAL_ROOT", str(fake_root))
    return fake_root


def _cfg(**kw) -> PL.PoolConfig:
    base = dict(lang="en", jobs_per_candidate=2, seed=42, sources=("rjdb", "kaggle_linkedin"),
                pairs_per_source={"rjdb": 10, "kaggle_linkedin": 10})
    return PL.PoolConfig(**{**base, **kw})


# ---- Djinni stays V1 by default -----------------------------------------------------------------

def _djinni_frames():
    candidates = pd.DataFrame({
        "id": [f"cand-{i}" for i in range(6)],
        "CV": [_words(80) for _ in range(6)],
        "Primary Keyword": ["Python", "Python", "Java", "Java", "Python", "QA"],
        "Experience Years": [5.0, 3.0, 4.0, 6.0, 2.0, 1.0],
    })
    jobs = pd.DataFrame({
        "id": [f"job-{i}" for i in range(6)],
        "Long Description": [_words(80) for _ in range(6)],
        "Position": ["Dev"] * 6,
        "Primary Keyword": ["Python", "Python", "Java", "Java", "Python", "QA"],
        "Exp Years": ["3y", "2y", "4y", "5y", "1y", "no_exp"],
    })
    return candidates, jobs


def test_default_pool_config_is_djinni_only():
    cfg = PL.PoolConfig(lang="en")
    assert cfg.sources == ("djinni",)
    assert cfg.pairs_per_source == {} and cfg.rjdb_include_weakened is False
    from_config = PL.pool_config({"n_pairs_per_language": 3000, "jobs_per_candidate": 2}, "en")
    assert from_config.sources == ("djinni",) and from_config.pairs_per_source == {}
    assert (from_config.n_pairs, from_config.jobs_per_candidate) == (3000, 2)


def test_default_pool_is_the_v1_pool_plus_provenance(monkeypatch):
    """Same pairs, same order, same texts as V1's match_pairs; only three columns are added.

    The external module must not even be consulted: a Djinni-only run reads no staged file.
    """
    candidates, jobs = _djinni_frames()
    monkeypatch.setattr(PL, "load_djinni", lambda lang: (candidates, jobs))
    monkeypatch.setattr(S, "build_external", lambda *a, **k: pytest.fail("external consulted"))
    cfg = PL.PoolConfig(lang="en", n_pairs=5, jobs_per_candidate=2)

    v1 = PL.match_pairs(PL.filter_candidates(candidates, cfg), PL.filter_jobs(jobs, cfg), cfg)
    pool = PL.build_pool(cfg)

    assert list(pool.columns) == [*v1.columns, "source", "occupation", "cv_variant"]
    pd.testing.assert_frame_equal(pool[v1.columns], v1)
    assert set(pool["source"]) == {"djinni"} and set(pool["cv_variant"]) == {"original"}
    assert (pool["occupation"] == pool["primary_keyword"]).all()
    # The teacher sees exactly V1's prompts.
    assert SY.reference_prompts(pool, "en") == SY.reference_prompts(v1, "en")


def test_unknown_pool_source_is_an_error():
    with pytest.raises(ValueError, match="unknown pool source"):
        S.external_sources(("djinni", "djini"))


def test_an_external_source_needs_an_explicit_pair_count(external):
    with pytest.raises(ValueError, match="pairs_per_source"):
        S.build_external(_cfg(pairs_per_source={"rjdb": 5}))


def test_external_sources_are_skipped_for_ukrainian(tmp_path, monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(S.log, "warning", lambda msg, *args: warnings.append(msg % args))
    monkeypatch.setenv("HBM_EXTERNAL_ROOT", str(tmp_path / "nothing-here"))
    rows, funnel = S.build_external(_cfg(lang="uk"))
    assert rows.empty and list(rows.columns) == S.POOL_COLUMNS and funnel == {}
    assert any("English-only" in w for w in warnings)


# ---- RJDB --------------------------------------------------------------------------------------

def test_rjdb_matched_and_weakened_cv_are_one_candidate(external):
    """The weakened CV is the same person rewritten, so the split must keep the two together."""
    pool, funnel = PL.build_pool_with_report(_cfg(sources=("rjdb",), rjdb_include_weakened=True))
    by_candidate = pool.groupby("candidate_id")["cv_variant"].apply(sorted).to_dict()
    assert by_candidate["rjdb:rjdb-train-00000"] == ["matched", "weakened"]
    triple = pool[pool["candidate_id"] == "rjdb:rjdb-train-00000"]
    assert triple["pair_id"].nunique() == 2
    assert set(triple["pair_id"]) == {"rjdb:rjdb-train-00000-match",
                                      "rjdb:rjdb-train-00000-mismatch"}
    assert triple["job_id"].nunique() == 1 and triple["job_description"].nunique() == 1
    assert triple["cv"].nunique() == 2

    train, val = D.train_val_split(pool, val_fraction=0.34, seed=0)
    assert set(train["candidate_id"]).isdisjoint(set(val["candidate_id"]))
    # Every id carries its source prefix.
    for column in ("pair_id", "candidate_id", "job_id"):
        assert pool[column].str.startswith("rjdb:").all()
    # The weakened rewrite of triple 4 talks about its construction: matched kept, weakened not.
    assert by_candidate["rjdb:rjdb-train-00004"] == ["matched"]
    assert funnel["rjdb_weakened_cv"]["cleaning_drops_by_reason"] == {"meta_text": 1}


def test_rjdb_reads_only_the_train_split_and_drops_what_it_must(external):
    pool, funnel = PL.build_pool_with_report(_cfg(sources=("rjdb",)))
    triples = set(pool["candidate_id"])
    assert triples == {"rjdb:rjdb-train-00000", "rjdb:rjdb-train-00004", "rjdb:rjdb-train-00005"}
    assert set(pool["cv_variant"]) == {"matched"}  # weakened is opt-in
    assert not any("dev" in t or "test" in t for t in triples)
    assert funnel["rjdb_cv"]["cleaning_drops_by_reason"] == {"name_residue": 1}  # triple 1
    assert funnel["rjdb_cv"]["holdout_leaks_by_source"] == {"rjdb": 1}  # triple 2 copies test
    assert funnel["rjdb_jd"]["requirement_hits_by_kind"] == {"security_clearance": 1}  # triple 3
    assert pool["candidate_years"].isna().all() and pool["job_required_years"].isna().all()
    assert (pool["primary_keyword"] == "").all() and (pool["source"] == "rjdb").all()


def test_rjdb_cleaning_removes_the_personal_block_notes_and_hash_artefacts():
    raw = (
        ":\n\n## Personal Information\nName: Konstantina Caruso\n"
        "Email: KonstantinaCaruso@email.com\n\n###### Education\nBachelor in ERP Systems\n\n"
        "## Skills\n- Expertise in SQL\n- Expert in C# and F#\n\n## Experience\n"
        "- Oracle SCM Developer (2014 - 2017)######\n"
        "(Note: Kindly fill in the information as appropriate)\n**Note:** generated text\n\n#\n"
    )
    clean, reason = S.clean_rjdb_cv(raw, "Konstantina Caruso")
    assert reason == ""
    assert "Konstantina" not in clean and "Caruso" not in clean and "@" not in clean
    assert "Personal Information" not in clean and "Note" not in clean
    assert clean.startswith("## Education\nBachelor in ERP Systems")
    assert "- Expert in C# and F#" in clean  # a single "#" is a language name, not an artefact
    assert "- Oracle SCM Developer (2014 - 2017)\n" in clean + "\n"
    assert not clean.rstrip().endswith("#")

    bare = "**\n\n**Personal Information**\nYulia Petrova\nyulia@mail.com\n\n**Skills**\n- SQL\n"
    assert S.clean_rjdb_cv(bare)[0] == "**Skills**\n- SQL"

    residue = "## Personal Information\nName: Aya Johnson\n\n## Education\nAya holds a BSc."
    assert S.clean_rjdb_cv(residue, "Aya Johnson") == ("", "name_residue")
    meta = "## Personal Information\nName: Omer Kaya\n\n## Skills\nSame as Resume 1."
    assert S.clean_rjdb_cv(meta, "Omer Kaya") == ("", "meta_text")
    inline = ("Personal Information:\nName: Sumarni W\n"
              "Education: Bachelor in Business Administration, Major data analytics and more.\n"
              "## Skills\n- SQL")
    assert S.clean_rjdb_cv(inline)[1] == "irregular_personal_block"
    assert S.clean_rjdb_cv("Name: X\nno sections here at all")[1] == "no_sections"


# ---- Kaggle x LinkedIn ----------------------------------------------------------------------

def test_kaggle_cleaning_collapses_padding_and_scrubs_contacts():
    raw = ("CHEF       Summary     Cooks well.\n\n\n\n  Reach me at chef.cook@gmail.com,  "
           "(555) 432-1000 or https://www.linkedin.com/in/chef-cook  ")
    clean = S.clean_kaggle_cv(raw)
    assert "  " not in clean and "\n\n\n" not in clean
    assert "@" not in clean and "432" not in clean and "linkedin" not in clean
    assert clean.startswith("CHEF Summary Cooks well.")


@pytest.mark.parametrize("category,headline,expected", [
    ("CHEF", "EXECUTIVE CHEF", ("CHEF", "agree")),
    ("AUTOMOBILE", "DATASTAGE ETL DEVELOPER", ("INFORMATION-TECHNOLOGY", "headline")),
    ("ARTS", "", ("", "no_headline")),
    ("AVIATION", "ENTREPRENEUR GENERAL MANAGER", ("", "headline_unmapped")),
    ("ARTS", "ENGLISH LANGUAGE ARTS TEACHER", ("TEACHER", "headline")),
    ("ADVOCATE", "PATIENT ADVOCATE", ("ADVOCATE", "agree")),
])
def test_kaggle_occupation_comes_from_the_headline(category, headline, expected):
    assert S.kaggle_occupation(category, headline) == expected


def test_category_keywords_cover_every_kaggle_category_and_keep_software_out_of_engineering():
    assert len(S.CATEGORY_TITLE_KEYWORDS) == 24
    assert S.title_categories("Senior Software Engineer") == ["INFORMATION-TECHNOLOGY"]
    assert S.title_categories("Mechanical Engineer") == ["ENGINEERING"]
    assert "HR" not in S.title_categories("Automotive Technician | Up to $50/Hr")
    assert "TEACHER" not in S.title_categories("Principal Quality Engineer")


def test_kaggle_linkedin_pairs_by_occupation_and_filters_both_sides(external):
    pool, funnel = PL.build_pool_with_report(_cfg(sources=("kaggle_linkedin",)))
    cvs = {c.split(":", 1)[1] for c in pool["candidate_id"]}
    jds = [j.split(":", 1)[1] for j in pool["job_id"]]
    assert cvs <= {"k1", "k2", "k6", "k8"}
    assert not cvs & {"k3", "k4", "k5", "k7"}  # no headline, Navy, FAIRE copy, near-dup twin
    assert len(jds) == len(set(jds))  # a posting is used once
    assert not {"j2", "j3"} <= set(jds)  # near-duplicate postings are one posting
    assert not set(jds) & {"j6", "j7", "j10", "j11"}  # clearance, exact dup, no category, und
    chef = pool[pool["occupation"] == "CHEF"]
    assert set(chef["job_id"].str.split(":").str[1]) <= {"j1", "j2", "j3", "j9"}
    assert "linkedin_postings:j9" in set(pool["job_id"])  # EEO boilerplate is kept
    it = pool[pool["candidate_id"] == "kaggle_resume:k2"]
    assert set(it["job_id"]) <= {"linkedin_postings:j4", "linkedin_postings:j5"}
    assert (pool.groupby("candidate_id").size() <= 2).all()
    assert funnel["kaggle_cv"]["holdout_leaks_by_source"] == {"faire": 1}
    assert funnel["linkedin_jd"]["requirement_hits_by_kind"] == {"security_clearance": 1}
    assert pool["pair_id"].str.match(r"^kaggle_resume:\w+__linkedin_postings:\w+$").all()
    k1 = pool[pool["candidate_id"] == "kaggle_resume:k1"]["cv"]
    assert k1.empty or ("@" not in k1.iloc[0] and "  " not in k1.iloc[0])


def test_kaggle_linkedin_offers_the_jd_whose_title_shares_the_headline_first(tmp_path,
                                                                           monkeypatch):
    """A category is broad: an assistant teacher must not get a corrections instructor posting
    while a preschool teacher posting is there. The headline decides within the category."""
    root = build_fake_root(tmp_path)
    _write(root, "kaggle_resume", "cvs", [
        {"cv_id": "t1", "text": _words(90), "lang": "en", "occupation": "TEACHER",
         "extra": json.dumps({"Category": "TEACHER", "headline": "ASSISTANT PRESCHOOL TEACHER"})},
    ])
    _write(root, "linkedin_postings", "jobs", [
        {"job_id": f"j{i}", "title": title, "text": _words(90), "lang": "en"}
        for i, title in enumerate(["Corrections Food Service Instructor", "Fitness Instructor",
                                   "Math Teacher", "Lead Preschool Teacher", "Tutor"])
    ])
    monkeypatch.setenv("HBM_EXTERNAL_ROOT", str(root))
    for seed in range(5):
        pool = PL.build_pool(_cfg(sources=("kaggle_linkedin",), jobs_per_candidate=2, seed=seed))
        assert list(pool["job_position"]) == ["Lead Preschool Teacher", "Math Teacher"]
    assert S.title_words("HR ADMINISTRATOR/MARKETING ASSOCIATE (Sr.)") == {
        "hr", "administrator", "marketing", "associate"}


# ---- markers and JD requirements ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Honorable Discharge, US Navy, 2009.",
    "Airman in The United States Air Force.",
    "Infantry Officer, 2nd Battalion.",
    "Petty Officer (E6), information systems.",
    "Date of Birth: 01/02/1980",
    "Nationality: Indian",
    "Gender: F",
    "Sex : M",
    "Age: 41",
    "She is proficient in SQL and Python.",
    "Volunteer at First Baptist Church.",
    "Youth pastor for six years.",
    "President of the campus sorority.",
])
def test_extra_markers_fire_where_the_djinni_list_is_silent(text):
    assert not PL.has_protected_marker(text, "en")
    assert S.has_external_marker(text)


@pytest.mark.parametrize("text", [
    "The agent he/she communicates with the physician.",
    "Responsible for his or her duties.",
    "Sales associate at Old Navy.",
    "BA, Temple University.",
    "Expert in C# and deployed microservices to production.",
    "Worked with the HIM staff on records.",
    "Taught good citizenship and respect.",
])
def test_extra_markers_leave_ordinary_text_alone(text):
    assert not S.has_external_marker(text)


def test_djinni_keeps_its_own_marker_list():
    """The extra list is for external CVs only; Djinni's filter is V1's, unchanged."""
    assert PL.has_protected_marker("Served in the US Navy.", "en") is False


@pytest.mark.parametrize("text,kind", [
    ("Must have an active Secret clearance.", "security_clearance"),
    ("TS/SCI with polygraph.", "security_clearance"),
    ("Veterans' preference applies to this position.", "veteran_preference"),
    ("Military experience preferred.", "veteran_preference"),
    ("Must be a U.S. citizen.", "us_citizenship"),
    ("US Citizenship required.", "us_citizenship"),
    ("Join our church ministry team.", "religious_employer"),
    ("A Catholic healthcare system.", "religious_employer"),
    ("Must be at least 18 years of age.", "age_limit"),
    ("Applicants must be 21 or older.", "age_limit"),
])
def test_jd_requirement_filter_names_each_requirement(text, kind):
    assert kind in S.jd_requirements(text)


def test_jd_requirement_filter_keeps_eeo_boilerplate():
    boilerplate = (
        EEO + " Minorities, women, veterans and individuals with disabilities are encouraged to "
        "apply. We do not discriminate on the basis of marital status, except where sex is a "
        "bona fide occupational qualification. VEVRAA federal contractor."
    )
    assert S.jd_requirements(boilerplate) == []


# ---- the external holdout ---------------------------------------------------------------------

def test_final_assert_fires_on_a_leak_the_drops_let_through(external, monkeypatch):
    """If every leak drop is bypassed, build_pool must still refuse the pool it would return.

    The builders' drop is disabled, and so is the first `find_external_leaks` call in
    build_pool, which is its own drop. The assert after it uses the real check.
    """
    real = X.find_external_leaks
    calls: list[int] = []

    def blind_once(df, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return pd.DataFrame(columns=["kind", "row", "match", "score"])
        return real(df, *args, **kwargs)

    monkeypatch.setattr(S, "drop_external_leaks", lambda frame, *a, **k: frame)
    monkeypatch.setattr(X, "find_external_leaks", blind_once)
    with pytest.raises(AssertionError, match="external-holdout leakage"):
        PL.build_pool(_cfg(sources=("kaggle_linkedin",)))
    assert len(calls) == 2


def test_build_pool_drops_a_leaking_row_and_still_asserts(external, monkeypatch):
    """With the builders bypassed, the drop on the way out removes the planted copies."""
    monkeypatch.setattr(S, "drop_external_leaks", lambda frame, *a, **k: frame)
    pool, funnel = PL.build_pool_with_report(_cfg(sources=("rjdb", "kaggle_linkedin")))
    assert "kaggle_resume:k5" not in set(pool["candidate_id"])
    assert "rjdb:rjdb-train-00002" not in set(pool["candidate_id"])
    assert funnel["pool"]["holdout_leaks_by_held_out_source"] == {"faire": 1, "rjdb": 1}
    assert X.find_external_leaks(pool, root=external).empty


# ---- the generator's resume guard -----------------------------------------------------------

def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_training_data", REPO / "scripts" / "generate_training_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cached_dump_for_other_pairs_is_refused(tmp_path):
    """Same row count, different pairs: the old row-count check reused the wrong generations."""
    gen = _generator()
    (tmp_path / "raw").mkdir()
    dumped = pd.DataFrame({"pair_id": ["a", "b"], "cv": ["x", "y"]})
    gen._dump_raw(tmp_path, "en_reference", dumped, ["out-a", "out-b"])

    assert gen._cached_raw(tmp_path, "en_reference", dumped) == ["out-a", "out-b"]
    assert gen._cached_raw(tmp_path, "en_missing", dumped) is None
    with pytest.raises(SystemExit, match="different pool"):
        gen._cached_raw(tmp_path, "en_reference", dumped.assign(pair_id=["a", "c"]))
    with pytest.raises(SystemExit, match="different pool"):
        gen._cached_raw(tmp_path, "en_reference", dumped.iloc[::-1])  # same ids, other order
    with pytest.raises(SystemExit, match="different pool"):
        gen._cached_raw(tmp_path, "en_reference", dumped.head(1))

    variants = dumped.assign(protected_group="gender", protected_attr="Male",
                             condition="explicit")
    gen._dump_raw(tmp_path, "en_invariant", variants, ["v1", "v2"])
    with pytest.raises(SystemExit, match="different pool"):
        gen._cached_raw(tmp_path, "en_invariant", variants.assign(protected_attr="Female"))

    pd.DataFrame({"raw_output": ["x"]}).to_parquet(tmp_path / "raw" / "en_biased.parquet")
    with pytest.raises(SystemExit, match="cannot check"):
        gen._cached_raw(tmp_path, "en_biased", variants)


# ---- provenance through dataset assembly, and the publish guard ----------------------------

def _response(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def test_source_survives_into_sft_dpo_and_kto_rows():
    """assemble_datasets re-filters the raw dumps, so provenance must ride through every step."""
    pairs = pd.DataFrame({
        "pair_id": ["rjdb:t-match", "c1__j1"], "candidate_id": ["rjdb:t", "c1"],
        "job_id": ["rjdb:t-jd", "j1"], "cv": ["Eight years of Python."] * 2,
        "job_description": ["Senior Python engineer."] * 2, "lang": ["en"] * 2,
        "source": ["rjdb", "djinni"],
    })
    anchored, _ = F.filter_reference(
        [_response("hire", "Strong Python and Django experience fits the role.")] * 2, pairs, "en")
    variants = SY.build_variant_frame(anchored, SY.SynthConfig(
        lang="en", protected_groups=("gender",), conditions=("explicit",), attributes_per_pair=1))
    invariant, _ = F.filter_invariant(
        [_response("hire", "Python and Django experience meets every requirement.")]
        * len(variants), variants, "en")
    biased, _ = F.filter_biased(
        [_response("reject", "Background suggests a poor cultural fit for the team.")]
        * len(variants), variants, "en")
    sft = D.build_sft_dataset(invariant)
    dpo = D.build_dpo_dataset(invariant, biased)
    kto = D.build_kto_dataset(dpo)
    for frame in (sft, dpo, kto):
        assert dict(zip(frame["candidate_id"], frame["source"])) == {"rjdb:t": "rjdb",
                                                                    "c1": "djinni"}
    v1 = D.build_sft_dataset(invariant.drop(columns="source"))
    assert set(v1["source"]) == {"djinni"}  # V1 dumps carry no source column
    assert set(D.build_kto_dataset(dpo.drop(columns="source"))["source"]) == {"djinni"}


def test_publish_guard_refuses_unpublishable_sources():
    djinni = pd.DataFrame({"source": ["djinni"] * 3})
    v1 = pd.DataFrame({"prompt": ["p"]})
    mixed = pd.DataFrame({"source": ["djinni", "rjdb", "kaggle_linkedin", "kaggle_linkedin"]})
    assert S.check_publishable(djinni, "t") == {}
    assert S.check_publishable(v1, "t") == {}  # no source column: V1 Djinni data
    with pytest.raises(SystemExit, match="unpublishable"):
        S.check_publishable(mixed, "t")
    assert S.check_publishable(mixed, "t", allow=True) == {"kaggle_linkedin": 2, "rjdb": 1}
    with pytest.raises(SystemExit):
        S.check_publishable(pd.DataFrame({"source": ["djinni", None]}), "t")
    assert not S.is_publishable("no_such_source") and not S.is_publishable("faire")
    assert S.is_publishable("djinni")


def test_push_script_refuses_external_rows(tmp_path, monkeypatch):
    import runpy

    pd.DataFrame({"candidate_id": ["rjdb:t"], "job_id": ["rjdb:t-jd"], "prompt": ["p"],
                  "source": ["rjdb"]}).to_parquet(tmp_path / "sft_train.parquet")
    monkeypatch.setenv("PUSH_TO_HUB", "true")
    monkeypatch.setenv("HF_TOKEN", "not-a-token")
    monkeypatch.setattr("sys.argv", ["push_dataset_to_hub.py", "--artifacts-dir", str(tmp_path),
                                     "--repo-id", "nobody/nothing"])
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("reached the prompt"))
    with pytest.raises(SystemExit, match="unpublishable"):
        runpy.run_path(str(REPO / "scripts" / "push_dataset_to_hub.py"), run_name="__main__")


def test_external_dataset_card_does_not_claim_mit(tmp_path):
    path = D.write_dataset_card(tmp_path / "README.md", name="x", teacher_model="t",
                                generation={}, filter_report={}, splits={"sft_train": 1},
                                groups=["gender"], languages=["en"], n_rows=1,
                                sources=("rjdb", "kaggle_linkedin"))
    text = path.read_text()
    assert "license: other" in text and "license: mit" not in text
    assert "Publishable: no" in text and "RJDB" in text and "LinkedIn" in text
    djinni = D.write_dataset_card(tmp_path / "d.md", name="x", teacher_model="t", generation={},
                                  filter_report={}, splits={"sft_train": 1}, groups=["gender"],
                                  languages=["en"], n_rows=1).read_text()
    assert "license: mit" in djinni and "## Sources" not in djinni


# ---- the draft config ------------------------------------------------------------------------

def test_draft_config_is_external_only_and_never_overwrites_v1():
    from hiring_bias_mitigation.utils.config import load_config

    cfg = load_config(REPO / "configs" / "generation" / "teacher_external_draft.yaml")
    header = (REPO / "configs" / "generation" / "teacher_external_draft.yaml").read_text()
    assert header.startswith("# Hand-written") and "DRAFT" in header
    assert cfg["artifacts_dir"] == "artifacts/semisynthetic-v4/external"
    assert "djinni" not in cfg["pool_sources"] and cfg["languages"] == ["en"]
    pool_cfg = PL.pool_config(cfg, "en")
    assert set(pool_cfg.pairs_per_source) == set(S.external_sources(pool_cfg.sources))


# ---- the real staged data, when it is on disk ------------------------------------------------

REAL_ROOT = Path("/ritom/scratch/cscs/zhang/xw/external_data")


@pytest.mark.skipif(not (REAL_ROOT / "kaggle_resume" / "staged" / "cvs.parquet").exists(),
                    reason="the real staged external data is not on this machine")
def test_real_kaggle_candidates_exclude_every_faire_base(monkeypatch):
    """All 98 FAIRE base CVs are Kaggle CVs. None of those that pass the filters may remain."""
    monkeypatch.setenv("HBM_EXTERNAL_ROOT", str(REAL_ROOT))
    funnel: dict = {}
    cvs = S._kaggle_candidates(PL.PoolConfig(lang="en"), REAL_ROOT, funnel)
    assert funnel["kaggle_cv"]["staged"] == 2484
    assert set(funnel["kaggle_cv"]["holdout_leaks_by_source"]) == {"faire"}
    assert 0 < funnel["kaggle_cv"]["holdout_leaks_by_source"]["faire"] <= 98
    probe = cvs.rename(columns={"cv": "cv_text"})
    assert X.find_external_leaks(probe, cv_col="cv_text", jd_col="-", root=REAL_ROOT).empty
    assert not cvs["cv"].map(S.has_external_marker).any()
