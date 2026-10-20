"""scripts/v3_additional_data.py end to end on a tiny pool, with a fake teacher.

No GPU and no network.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import sources as S

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v3_additional_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("v3_additional_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTeacher:
    """Hires every pair whose CV mentions Python; its biased answers always flip that."""

    def __init__(self):
        self.generation = None
        self.calls = []

    def _answer(self, prompt: str) -> str:
        decision = "hire" if "Python" in prompt.split("Candidate profile")[-1] else "reject"
        if "simulating a biased hiring system" in prompt:     # always the other verdict
            return json.dumps({"decision": "reject" if decision == "hire" else "hire",
                               "feedback": "We prefer a different kind of person for this "
                               "team and its culture overall."})
        return json.dumps({"decision": decision, "feedback": "The candidate's experience and "
                           "skills were weighed against the stated requirements."})

    def generate(self, prompts):
        self.calls.append(("generate", len(prompts)))
        return [self._answer(p) for p in prompts]

    def generate_samples(self, prompts, n):
        self.calls.append(("samples", len(prompts), n))
        return [[self._answer(p)] * n for p in prompts]

    def close(self):
        pass


def _pool(n_per_source: int = 4) -> pd.DataFrame:
    rows = []
    for source in ("djinni_new", "rjdb"):
        for i in range(n_per_source):
            skill = "Python" if i % 2 == 0 else "Excel"
            rows.append({
                "pair_id": f"{source}:c{i}__{source}:j{i}", "candidate_id": f"{source}:c{i}",
                "job_id": f"{source}:j{i}",
                "cv": f"Engineer with six years of {skill} work on data pipelines. " * 12,
                "job_description": f"We need an engineer for role {i} with Python. " * 12,
                "job_position": "Engineer", "primary_keyword": "", "candidate_years": 0.0,
                "job_required_years": 0.0, "lang": "en", "source": source,
                "occupation": "engineer", "cv_variant": "original",
            })
    pool = pd.DataFrame(rows, columns=[*S.POOL_COLUMNS])
    pool["est_prompt_tokens"] = 300
    pool["order"] = pool.groupby("source").cumcount()
    pool["part"] = pool["source"] + "-000"
    return pool


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    (tmp_path / ".hbm-models-root").touch()
    mod = _load()
    out = tmp_path / "artifacts" / "additional"
    (out / "pool").mkdir(parents=True)
    _pool().to_parquet(out / "pool" / "pairs_en.parquet", index=False)
    v3 = tmp_path / "sft_v3_swap"
    v3.mkdir()
    for split, cand in (("train", "v1-a"), ("validation", "v1-b")):
        pd.DataFrame({"prompt": ["p"], "completion": ['{"decision": "reject", "feedback": "x"}'],
                      "decision": ["reject"], "lang": ["en"], "candidate_id": [cand],
                      "job_id": ["j"], "pair_id": [f"{cand}__j"]}).to_parquet(
            v3 / f"sft_{split}.parquet", index=False)
    cfg = {
        "_path": "test", "artifacts_dir": "artifacts/additional", "seed": 42,
        "protected_groups": ["military_status", "gender", "religion"],
        "conditions": ["explicit", "implicit"], "attributes_per_pair": 2,
        "vote_draws": 8, "teacher_model": "fake", "serving": {"backend": "vllm"},
        "passes": {p: {"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9, "seed": s}
                   for p, s in (("votes", 43), ("invariant", 42), ("biased", 42))},
        "split": {"val_fraction": 0.25},
        "stop": {"max_undecided_draw_share": 0.05, "max_unanchored_share": 0.10,
                 "min_invariant_yield": 0.80, "min_biased_yield": 0.20},
        "assemble": {"sft_v3_dir": str(v3), "dpo_cap_per_pair": 2,
                     "exclude": {"djinni_new_extra_markers": True,
                                 "rjdb_template_residue": True}},
    }
    teacher = FakeTeacher()
    from hiring_bias_mitigation.eval import backends as BK
    monkeypatch.setattr(BK, "build_backend", lambda _cfg: teacher)
    return mod, cfg, teacher, out


def test_generate_report_assemble(setup):
    mod, cfg, teacher, out = setup
    mod.stage_generate(cfg, ["all"], list(mod.PASSES))
    for part in ("djinni_new-000", "rjdb-000"):
        for name in mod.PASSES:
            assert (out / "parts" / part / f"{name}.parquet").exists()
    votes = pd.read_parquet(out / "parts" / "rjdb-000" / "votes.parquet")
    assert len(votes) == 4 * 8 and votes["request_seed"].nunique() == 4
    invariant = pd.read_parquet(out / "parts" / "rjdb-000" / "invariant.parquet")
    assert len(invariant) == 4 * 12                         # v1's 12 variants per pair

    # A rerun on the same prompts generates nothing.
    calls = len(teacher.calls)
    mod.stage_generate(cfg, ["all"], list(mod.PASSES))
    assert len(teacher.calls) == calls

    mod.stage_report(cfg)
    report = json.loads((out / "report.json").read_text())
    assert report["vote"]["all"]["anchored"] == 8
    assert report["vote"]["by"]["rjdb"]["anchor_hire_share"] == 0.5

    with pytest.raises(SystemExit):                     # v1_dpo has not been built
        mod.stage_assemble(cfg)
    mod.stage_assemble(cfg, allow_partial=True)
    new_train = pd.read_parquet(out / "additional_sft_train.parquet")
    new_val = pd.read_parquet(out / "additional_sft_validation.parquet")
    combined = pd.read_parquet(out / "sft_train.parquet")
    dpo_train = pd.read_parquet(out / "additional_dpo_train.parquet")
    capped = pd.read_parquet(out / "dpo_capped_train.parquet")
    balanced = pd.read_parquet(out / "dpo_capped_balanced_train.parquet")
    assert capped.groupby("pair_id").size().max() <= 2
    assert balanced["decision"].value_counts().nunique() == 1      # as many hire as reject
    assert set(balanced["pair_id"]) <= set(capped["pair_id"])
    assert {"anchor_hire_votes", "anchor_unstable", "anchor_changed", "reference_decision",
            "pool_source"} <= set(capped.columns)
    gen = json.loads((out / "generation_report.json").read_text())
    assert gen["splits"]["dpo_train"]["dpo_capped_balanced"]["decision_contrast_share"] == 1.0
    assert not set(new_train["candidate_id"]) & set(new_val["candidate_id"])
    assert set(new_train["pool_source"]) <= {"djinni_new", "rjdb"}
    # The publish guard's names: new Djinni rows are Djinni.
    assert set(new_train["source"]) <= {"djinni", "rjdb"}
    assert (new_train["reference_decision"] == new_train["decision"]).all()
    # SFT-v3's swap rows come first, unchanged; the new rows follow.
    assert combined["candidate_id"].iloc[0] == "v1-a" and combined["source"].iloc[0] == "djinni"
    assert combined["pool_source"].iloc[0] == "djinni_v1"
    assert len(combined) == 1 + len(new_train)
    assert "v1-b" in set(pd.read_parquet(out / "sft_validation.parquet")["candidate_id"])
    assert len(dpo_train) and (dpo_train["chosen"] != dpo_train["rejected"]).all()
    # The SFT completion carries the vote's verdict, written as JSON, not a fenced raw answer.
    assert new_train["completion"].str.startswith('{"decision"').all()


def test_unknown_part_is_refused(setup):
    mod, cfg, _, _ = setup
    with pytest.raises(SystemExit):
        mod.stage_generate(cfg, ["nope-000"], list(mod.PASSES))


def test_v1_dpo_reads_the_vote_anchor_and_keeps_sft_v3s_split(setup, tmp_path):
    """The flip check uses the vote's verdict; changed pairs take the regenerated targets."""
    mod, cfg, _, out = setup
    key = {"candidate_id": "v1-b", "job_id": "j", "pair_id": "v1-b__j", "cv": "cv",
           "job_description": "jd", "job_position": "", "primary_keyword": "",
           "candidate_years": 0.0, "job_required_years": 0.0, "lang": "en",
           "protected_group": "gender", "protected_attr": "Male", "condition": "explicit",
           "group_label": "gender", "profile": "p", "reference_feedback": "v1 feedback"}
    ok = json.dumps({"decision": "reject", "feedback": "The record lacks the core skills "
                     "the role asks for in its requirements."})
    hire = json.dumps({"decision": "hire", "feedback": "Strong record across every skill "
                       "the role asks for in its requirements."})
    pd.DataFrame([{**key, "reference_decision": "reject", "raw_output": ok}]).to_parquet(
        tmp_path / "inv.parquet")
    pd.DataFrame([{**key, "reference_decision": "hire", "raw_output": hire,
                   "v1_reference_decision": "reject", "seed": 1}]).to_parquet(
        tmp_path / "regen.parquet")
    # v1's biased answer says reject: no flip against v1's reject, a flip against the vote's hire.
    pd.DataFrame([{**key, "reference_decision": "reject", "raw_output": ok}]).to_parquet(
        tmp_path / "biased.parquet")
    pd.DataFrame({"pair_id": ["v1-b__j"], "new_decision": ["hire"],
                  "new_feedback": ["vote feedback"], "changed": [True], "hire_votes": [7],
                  "decided": [8], "source": ["vote"],
                  "unstable": [True]}).to_parquet(tmp_path / "anchors.parquet")
    cfg["v1_dpo"] = {"vote_anchors": str(tmp_path / "anchors.parquet"),
                     "v1_invariant": str(tmp_path / "inv.parquet"),
                     "vote_targets": str(tmp_path / "regen.parquet"),
                     "v1_biased": str(tmp_path / "biased.parquet")}
    mod.stage_v1_dpo(cfg)
    val = pd.read_parquet(out / "v1_dpo_validation.parquet")      # v1-b is in SFT-v3's validation
    assert len(val) == 1 and len(pd.read_parquet(out / "v1_dpo_train.parquet")) == 0
    row = val.iloc[0]
    assert json.loads(row["chosen"])["decision"] == "hire"         # the regenerated target
    assert json.loads(row["rejected"])["decision"] == "reject" and row["rejected_flipped_decision"]
    assert row["source"] == "djinni" and row["pool_source"] == "djinni_v1"
    assert row["reference_decision"] == "hire"
    # 7 of 8 is a stable vote; that it overturned v1 is `anchor_changed`, not `anchor_unstable`.
    assert not bool(row["anchor_unstable"]) and bool(row["anchor_changed"])


def test_a_source_that_crosses_a_stop_condition_gets_no_further_part(setup):
    """The pre-registered stop conditions are read in the job, on each source's first part."""
    mod, cfg, _, out = setup
    pool = pd.read_parquet(out / "pool" / "pairs_en.parquet")
    rjdb = pool["source"] == "rjdb"
    pool.loc[rjdb, "part"] = ["rjdb-000", "rjdb-000", "rjdb-001", "rjdb-001"]
    pool.to_parquet(out / "pool" / "pairs_en.parquet", index=False)
    cfg["stop"]["min_invariant_yield"] = 1.01       # no part can meet it
    mod.stage_generate(cfg, ["rjdb-000", "rjdb-001", "djinni_new-000"], list(mod.PASSES))
    check = json.loads((out / "parts" / "rjdb-000" / "stop_check.json").read_text())
    assert check["reasons"] and check["readout"]["invariant_yield"] < 1.01
    assert not (out / "parts" / "rjdb-001").exists()             # skipped
    assert (out / "parts" / "djinni_new-000" / "biased.parquet").exists()   # other sources run
    meta = json.loads((out / "parts" / "rjdb-000" / "votes.meta.json").read_text())
    assert meta["engine_load_seconds"] is not None and meta["pass_sha256"]


def test_balanced_view_balances_each_pool_source_on_its_own():
    """A hire-heavy source must not supply another source's hire side."""
    mod = _load()
    rows = pd.DataFrame({
        "pool_source": ["rjdb"] * 6 + ["djinni_new"] * 6,
        "decision": ["hire"] * 5 + ["reject"] + ["hire"] + ["reject"] * 5,
        "pair_id": [f"p{i}" for i in range(12)],
    })
    out = mod.balanced(rows, seed=0)
    per = out.groupby(["pool_source", "decision"]).size().to_dict()
    assert per == {("djinni_new", "hire"): 1, ("djinni_new", "reject"): 1,
                   ("rjdb", "hire"): 1, ("rjdb", "reject"): 1}


def test_quality_exclusions_drop_contradicting_djinni_cvs_and_rjdb_residue():
    mod = _load()
    cfg = {"assemble": {"exclude": {"djinni_new_extra_markers": True,
                                    "rjdb_template_residue": True}}}
    pairs = pd.DataFrame({
        "pair_id": ["a", "b", "c", "d"],
        "source": ["djinni_new", "djinni_new", "rjdb", "kaggle_linkedin"],
        "cv": ["She led the data team for five years.", "Led the data team for five years.",
               "Education: Details not provided.", "Company Name City , State"],
    })
    excluded, report = mod.quality_exclusions(cfg, pairs)
    assert report["djinni_new_extra_marker"] == 1 and report["rjdb_template_residue"] == 1
    assert excluded == {"a", "c"}            # Kaggle's anonymisation placeholders are kept
    cfg["assemble"]["exclude"] = {}
    assert mod.quality_exclusions(cfg, pairs)[0] == set()


def test_a_job_without_the_biased_pass_still_reads_the_stop_conditions(setup):
    mod, cfg, teacher, out = setup
    mod.stage_generate(cfg, ["rjdb-000"], ["votes", "invariant"])
    assert not (out / "parts" / "rjdb-000" / "biased.parquet").exists()
    check = json.loads((out / "parts" / "rjdb-000" / "stop_check.json").read_text())
    assert "invariant_yield" in check["readout"] and "biased_yield" not in check["readout"]
