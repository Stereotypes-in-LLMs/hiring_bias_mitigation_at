"""Anchor agreement: the training anchors against the SFT student's attribute-free decisions.

Synthetic data only; CPU; no network. The script is loaded from scripts/ by path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "anchor_agreement",
    Path(__file__).resolve().parents[1] / "scripts" / "anchor_agreement.py")
AA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AA)


def _answer(decision: str) -> str:
    return json.dumps({"decision": decision, "feedback": "Matches the stack asked for"})


#: pair -> (anchor, attribute-free sample decisions)
POOL = {
    "a": ("hire", ["hire"] * 8),                                   # agree
    "b": ("reject", ["hire"] * 7 + ["reject"]),                    # disagree, confident hire
    "c": ("reject", ["hire"] * 4 + ["reject"] * 4),                # tie
    "d": ("reject", ["hire"] + ["reject"] * 6 + ["maybe"]),        # reject 1/7, one invalid
    "e": ("hire", ["hire"] * 2 + ["reject"] * 6),                  # disagree, pulls to hire
}


def _rows() -> pd.DataFrame:
    out = []
    for pid, (anchor, _) in POOL.items():
        for condition in ("explicit", "implicit"):
            out.append({"pair_id": pid, "candidate_id": f"cand-{pid}", "job_id": f"job-{pid}",
                        "lang": "en", "reference_decision": anchor,
                        "protected_group": "gender", "protected_attr": "Female",
                        "condition": condition})
    return pd.DataFrame(out)


def _prompts() -> pd.DataFrame:
    rows = _rows()
    free = [("attr_free", pid, f"FREE {pid}") for pid in POOL]
    attributed = [("attributed", str(i), f"ATTR {r['pair_id']} {r['condition']}")
                  for i, r in rows.iterrows()]
    return pd.DataFrame(free + attributed, columns=["kind", "key", "prompt"])


def _samples() -> pd.DataFrame:
    out = []
    for item, (pid, (_, decisions)) in enumerate(POOL.items()):
        for j, d in enumerate(decisions):
            out.append({"item": item, "sample_index": j, "prompt": f"FREE {pid}",
                        "raw_output": _answer(d)})
    # Attributed prompts' samples all say the opposite: they must not count.
    for i, r in _rows().iterrows():
        for j in range(8):
            out.append({"item": 100 + i, "sample_index": j,
                        "prompt": f"ATTR {r['pair_id']} {r['condition']}",
                        "raw_output": _answer("reject" if POOL[r["pair_id"]][0] == "hire"
                                              else "hire")})
    return pd.DataFrame(out)


def _pairs(heldout: set[str] | None = None) -> pd.DataFrame:
    free = AA.free_prompts(_prompts())
    counts = AA.decision_counts(free, _samples())
    return AA.pair_table(_rows(), counts, heldout or set())


def test_counts_come_from_the_attribute_free_prompt_text_only():
    counts = AA.decision_counts(AA.free_prompts(_prompts()), _samples()).set_index("pair_id")
    assert counts.loc["a", ["n_samples", "n_hire", "n_reject", "n_unparsed"]].tolist() == \
        [8, 8, 0, 0]
    assert counts.loc["d", ["n_samples", "n_hire", "n_reject", "n_unparsed"]].tolist() == \
        [8, 1, 6, 1]
    assert counts.loc["e", "n_hire"] == 2


def test_majority_uses_parsed_samples_and_keeps_ties_apart():
    pairs = _pairs().set_index("pair_id")
    assert pairs["student_majority"].to_dict() == {"a": "hire", "b": "hire", "c": "tie",
                                                   "d": "reject", "e": "reject"}
    assert pairs.loc["d", "student_hire_share"] == pytest.approx(1 / 7)
    assert pairs["agreement"].to_dict() == {"a": "agree", "b": "disagree", "c": "tie",
                                            "d": "agree", "e": "disagree"}
    assert pairs.loc["b", "p_disagree"] == pytest.approx(7 / 8)
    assert pairs.loc["e", "p_disagree"] == pytest.approx(6 / 8)
    assert AA.majority([0, 3, 4], [0, 5, 4]).tolist() == ["none", "reject", "tie"]


def test_a_pair_without_samples_or_with_repeated_samples_is_refused():
    free = AA.free_prompts(_prompts())
    samples = _samples()
    with pytest.raises(KeyError, match="no stored student samples"):
        AA.decision_counts(free, samples[samples["prompt"] != "FREE c"])
    with pytest.raises(ValueError, match="repeat a"):
        AA.decision_counts(free, pd.concat([samples, samples.head(1)]))


def test_ambiguous_prompt_text_and_two_anchors_per_pair_are_refused():
    prompts = _prompts()
    prompts.loc[prompts["key"] == "b", "prompt"] = "FREE a"
    with pytest.raises(ValueError, match="share one attribute-free prompt"):
        AA.free_prompts(prompts)
    rows = _rows()
    rows.loc[0, "reference_decision"] = "reject"   # pair a: hire on one row, reject on another
    counts = AA.decision_counts(AA.free_prompts(_prompts()), _samples())
    with pytest.raises(ValueError, match="two anchors"):
        AA.pair_table(rows, counts, set())


def test_confusion_counts_shares_and_kappa():
    c = AA.confusion(_pairs())
    assert c["counts"]["hire"] == {"hire": 1, "reject": 1, "tie": 0, "none": 0}
    assert c["counts"]["reject"] == {"hire": 1, "reject": 1, "tie": 1, "none": 0}
    assert c["anchor_reject_student_hire"] == 1 and c["anchor_hire_student_reject"] == 1
    assert c["anchor_hire_rate"] == pytest.approx(0.4)
    assert c["share_of_anchor_side"]["reject"]["tie"] == pytest.approx(round(1 / 3, 4))
    # samples: hires 8+7+4+1+2 = 22 of 39 parsed
    assert c["student_hire_rate_samples"] == pytest.approx(round(22 / 39, 4))
    assert c["student_majority_hire_rate"] == pytest.approx(0.4)
    # 4 decided pairs, 2 agree; anchor hire 2/4, student hire 2/4 -> expected 0.5, kappa 0
    assert c["agreement_excluding_ties"] == pytest.approx(0.5)
    assert c["cohen_kappa_excluding_ties"] == pytest.approx(0.0)


def test_histogram_and_confident_opposition():
    pairs = _pairs()
    hist = {h["student_hire_samples"]: h["pairs"] for h in AA.share_histogram(pairs, "reject")}
    assert hist == {"1/7": 1, "4/8": 1, "7/8": 1}
    conf = AA.confident_opposition(pairs, 7 / 8)
    assert conf["anchor_reject_student_hire_share_at_least"] == 1   # pair b
    assert conf["anchor_hire_student_hire_share_at_most"] == 0      # e is 2/8, not <= 1/8


def test_training_rows_count_disagreement_per_row_and_its_direction():
    pairs = _pairs()
    frame = pd.DataFrame({"pair_id": ["b", "b", "b", "e", "a", "a", "c"],
                          "reference_decision": ["reject"] * 3 + ["hire"] * 3 + ["reject"]})
    joined = AA.join_training(frame, pairs, "reference_decision", "toy")
    table = {r["anchor_side"]: r for r in AA.side_rows(joined, {"training_set": "toy"})}
    assert table["reject"]["rows"] == 4 and table["reject"]["pairs"] == 2
    assert table["reject"]["disagree_rows"] == 3 and table["reject"]["student_tie"] == 1
    assert table["reject"]["pull_toward_reject_rows"] == 3
    assert table["hire"]["pull_toward_hire_rows"] == 1 and table["hire"]["disagree_rows"] == 1
    assert table["all"]["net_toward_reject_rows"] == 2
    assert table["all"]["disagree_share"] == pytest.approx(round(4 / 7, 4))
    # soft: 3 x 7/8 (b) + 4/8 (c) on the reject side
    assert table["reject"]["af_error_mass"] == pytest.approx(round(3 * 7 / 8 + 0.5, 2))
    # b's rows oppose at 7/8, e's at 6/8: only b's count as confident
    assert table["all"]["confident_disagree_rows"] == 3
    # reject side: (3 x 7/8 + 4/8) / 4; hire side: (6/8 + 0 + 0) / 3
    assert table["all"]["af_error_mean_reject_minus_hire"] == pytest.approx(
        round(round((3 * 7 / 8 + 0.5) / 4, 4) - round(0.75 / 3, 4), 4))


def test_rows_outside_the_pool_or_with_a_target_other_than_the_anchor_are_refused():
    pairs = _pairs()
    with pytest.raises(KeyError, match="outside the pool"):
        AA.join_training(pd.DataFrame({"pair_id": ["zz"], "decision": ["hire"]}), pairs,
                         "decision", "toy")
    with pytest.raises(ValueError, match="other than"):
        AA.join_training(pd.DataFrame({"pair_id": ["b"], "decision": ["hire"]}), pairs,
                         "decision", "toy")


def _write_pool(root: Path) -> Path:
    pool = root / "artifacts" / "pool"
    (pool / "raw" / "samples").mkdir(parents=True)
    _rows().to_parquet(pool / "rows.parquet", index=False)
    _prompts().to_parquet(pool / "prompts.parquet", index=False)
    samples = _samples()
    samples.iloc[:20].to_parquet(pool / "raw" / "samples" / "chunk-00000.parquet", index=False)
    samples.iloc[20:].to_parquet(pool / "raw" / "samples" / "chunk-00001.parquet", index=False)
    return pool


def _write_training(root: Path) -> Path:
    data_dir = root / "artifacts" / "train"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"pair_id": ["b", "b", "e", "a", "a"],
                  "decision": ["reject", "reject", "hire", "hire", "hire"],
                  "lang": ["en", "en", "en", "en", "uk"],
                  "protected_group": ["gender"] * 5,
                  "condition": ["explicit"] * 5}).to_parquet(
        data_dir / "rft_train.parquet", index=False)
    data_cfg = root / "data.yaml"
    data_cfg.write_text("name: toy\nartifacts_dir: artifacts/train\nlanguages: [en]\n")
    cfg = root / "train.yaml"
    cfg.write_text(f"stage: sft\ndata_kind: rft\ndata_config: {data_cfg}\nmax_steps: -1\n")
    return cfg


def test_training_frame_reads_the_trainers_file_with_its_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    frame, info = AA.training_frame("toy", _write_training(tmp_path))
    assert len(frame) == 4                      # the 'uk' row is filtered as the trainer does
    assert info["target_column"] == "decision"
    assert info["train_file"].endswith("artifacts/train/rft_train.parquet")


def test_end_to_end_writes_the_three_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    _write_pool(tmp_path)
    cfg = _write_training(tmp_path)
    # k=32 top-up over the hire-anchor pairs: pair a 30/32 hire, e 4/32 hire
    topup = tmp_path / "artifacts" / "topup"
    (topup / "raw" / "samples").mkdir(parents=True)
    pd.DataFrame({"kind": ["attr_free", "attr_free"], "key": ["a", "e"],
                  "prompt": ["FREE a", "FREE e"]}).to_parquet(topup / "prompts.parquet",
                                                              index=False)
    rows = [{"item": 0, "sample_index": j, "prompt": "FREE a",
             "raw_output": _answer("hire" if j < 30 else "reject")} for j in range(32)]
    rows += [{"item": 1, "sample_index": j, "prompt": "FREE e",
              "raw_output": _answer("hire" if j < 4 else "reject")} for j in range(32)]
    pd.DataFrame(rows).to_parquet(topup / "raw" / "samples" / "chunk-00000.parquet",
                                  index=False)
    out = tmp_path / "report"
    summary = AA.main(["--pool", "artifacts/pool", "--hire-topup", "artifacts/topup",
                       "--sft-source", "artifacts/none", "--training", f"toy={cfg}",
                       "--holdout", str(tmp_path / "missing.json"), "--out", str(out)])
    pairs = pd.read_csv(out / "pairs.csv")
    by_set = pd.read_csv(out / "by_training_set.csv")
    written = json.loads((out / "summary.json").read_text())
    assert len(pairs) == 5 and pairs.set_index("pair_id").loc["b", "rows_in_toy"] == 2
    assert pairs.set_index("pair_id").loc["e", "k32_majority"] == "reject"
    assert set(by_set["training_set"]) == {AA.POOL_ROWS, "toy"}
    toy = by_set[(by_set["training_set"] == "toy") & (by_set["anchor_side"] == "all")].iloc[0]
    assert toy["rows"] == 4 and toy["pull_toward_reject_rows"] == 2
    assert toy["pull_toward_hire_rows"] == 1
    pool_all = by_set[(by_set["training_set"] == AA.POOL_ROWS)
                      & (by_set["anchor_side"] == "all")].iloc[0]
    assert pool_all["rows"] == 10
    check = written["hire_topup_check"]
    assert check["anchor_hire_pairs_with_k32"] == 2
    assert check["k8_reject_majority_confirmed_by_k32"] == 1
    # 16 k=8 outputs on the top-up's two prompts; identical JSON strings, so all repeat
    assert check["k8_outputs_on_topup_prompts"] == 16
    assert check["k8_outputs_repeated_verbatim_in_k32"] == 16
    assert written["pool"]["counts"] == summary["pool"]["counts"]
    assert written["benchmark"]["comparable_to_gpt4o_or_claude_review"] is False
