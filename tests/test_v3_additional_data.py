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
    """Hires every pair whose CV mentions Python; writes biased answers that flip."""

    def __init__(self):
        self.generation = None
        self.calls = []

    def _answer(self, prompt: str) -> str:
        if "simulating a biased hiring system" in prompt:
            return json.dumps({"decision": "reject", "feedback": "We prefer a different kind of "
                               "person for this team and its culture overall."})
        decision = "hire" if "Python" in prompt.split("Candidate profile")[-1] else "reject"
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
        "assemble": {"sft_v3_dir": str(v3)},
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

    mod.stage_assemble(cfg)
    new_train = pd.read_parquet(out / "additional_sft_train.parquet")
    new_val = pd.read_parquet(out / "additional_sft_validation.parquet")
    combined = pd.read_parquet(out / "sft_train.parquet")
    dpo_train = pd.read_parquet(out / "additional_dpo_train.parquet")
    assert not set(new_train["candidate_id"]) & set(new_val["candidate_id"])
    assert set(new_train["source"]) <= {"djinni_new", "rjdb"}
    # SFT-v3's swap rows come first, unchanged; the new rows follow.
    assert combined["candidate_id"].iloc[0] == "v1-a" and combined["source"].iloc[0] == "djinni"
    assert len(combined) == 1 + len(new_train)
    assert "v1-b" in set(pd.read_parquet(out / "sft_validation.parquet")["candidate_id"])
    assert len(dpo_train) and (dpo_train["chosen"] != dpo_train["rejected"]).all()
    # The SFT completion carries the vote's verdict, written as JSON, not a fenced raw answer.
    assert new_train["completion"].str.startswith('{"decision"').all()


def test_unknown_part_is_refused(setup):
    mod, cfg, _, _ = setup
    with pytest.raises(SystemExit):
        mod.stage_generate(cfg, ["nope-000"], list(mod.PASSES))
