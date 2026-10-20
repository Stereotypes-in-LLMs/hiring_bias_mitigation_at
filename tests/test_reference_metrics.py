"""Per-run reference metrics, each readout checked on a frame small enough to count by hand.

Several tests are the counterexamples of a review: a dropped variant, a missing margin or a
refusal on a flipping prompt must show as incomplete or uncovered, never as a fairer number.
"""

import json

import numpy as np
import pandas as pd
import pytest

from hiring_bias_mitigation.analysis import reference_metrics as RM


def _rows(decisions, ref="hire", condition="explicit", group="g", free=None):
    """decisions: {pair_id: {attr: decision}}; free: {pair_id: attr-free decision}."""
    rows = []
    for pair, by_attr in decisions.items():
        r = ref[pair] if isinstance(ref, dict) else ref
        for attr, d in by_attr.items():
            rows.append({"pair_id": pair, "protected_group": group, "protected_attr": attr,
                         "condition": condition, "d": d, "ref": r})
        if free is not None:
            rows.append({"pair_id": pair, "protected_group": "__attr_free__",
                         "protected_attr": "__attr_free__", "condition": "attr_free",
                         "d": free[pair], "ref": r})
    return pd.DataFrame(rows)


def _expected(frame):
    return frame[[*RM.KEY, "ref"]].copy()


def test_prepare_maps_ukrainian_words_and_drops_undecided_rows():
    raw = pd.DataFrame({
        "pair_id": ["p", "p", "p"], "protected_group": ["g"] * 3, "protected_attr": ["a", "b", "c"],
        "condition": ["explicit"] * 3, "decision": ["найняти", "reject", "hire"],
        "outcome": ["decided", "decided", "refused"],
        "reference_decision": ["Відхилити", "hire", "hire"]})
    rows = RM.prepare(raw)
    assert rows["d"].tolist()[:2] == ["hire", "reject"]
    assert pd.isna(rows["d"].iloc[2])
    assert rows["ref"].tolist() == ["reject", "hire", "hire"]


def test_align_reports_missing_duplicated_and_unexpected_prompts():
    full = _rows({"p": {"a": "hire", "b": "reject", "c": "hire"}})
    run = pd.concat([full.iloc[[0, 0]], _rows({"q": {"a": "hire"}})])  # a twice, b c gone, q extra
    aligned, cov = RM.align(run, _expected(full))
    assert cov["expected_source"] == "config" and cov["expected_prompts"] == 3
    assert cov["duplicate_keys"] == 1 and cov["unexpected_prompts"] == 1
    assert cov["present_pct"] == pytest.approx(0.0)  # the duplicated key counts as unusable
    assert aligned["d"].isna().all()
    _, observed = RM.align(full, None)
    assert observed["expected_source"] == "observed" and observed["decided_pct"] == 100.0


def test_a_dropped_variant_makes_the_set_incomplete_not_stable():
    full = _rows({"p": {"A": "hire", "B": "reject"}})
    assert RM.unstable_within_group(RM.align(full, _expected(full))[0]) == (100.0, 1, 1)
    dropped = full[full["protected_attr"] == "A"]
    share, complete, expected = RM.unstable_within_group(RM.align(dropped, _expected(full))[0])
    assert np.isnan(share) and complete == 0 and expected == 1


def test_flip_rates_are_conditional_and_report_their_coverage():
    rows = _rows({"p1": {"a": "hire", "b": "reject"}, "p2": {"a": "reject", "b": "reject"}},
                 free={"p1": "reject", "p2": "hire"})
    out = RM.flip_rates(rows)
    # p1 rejected attribute-free: a flips (1 of 2); p2 hired attribute-free: both flip (2 of 2).
    assert out["flip_reject_to_hire_pct"] == pytest.approx(50.0)
    assert out["flip_hire_to_reject_pct"] == pytest.approx(100.0)
    assert out["flip_any_pct"] == pytest.approx(75.0)
    assert out["flip_coverage_pct"] == pytest.approx(100.0)


def test_a_refusal_on_a_flipping_prompt_shows_as_lost_coverage():
    usable = RM.flip_rates(_rows({"p": {"A": "hire", "B": "reject"}}, free={"p": "hire"}))
    refused = RM.flip_rates(_rows({"p": {"A": "hire", "B": None}}, free={"p": "hire"}))
    assert usable["flip_any_pct"] == pytest.approx(50.0)
    assert refused["flip_any_pct"] == pytest.approx(0.0)
    assert refused["flip_coverage_pct"] == pytest.approx(50.0)


def test_error_rate_gaps_read_tpr_and_fpr_per_cell():
    rows = _rows({"h1": {"a": "hire", "b": "hire"}, "h2": {"a": "hire", "b": "reject"},
                  "r1": {"a": "reject", "b": "reject"}},
                 ref={"h1": "hire", "h2": "hire", "r1": "reject"})
    out = RM.error_rate_gaps(rows)
    assert out["tpr_gap"] == pytest.approx(50.0) and out["fpr_gap"] == pytest.approx(0.0)
    assert out["eodds_gap"] == pytest.approx(50.0) and (out["eodds_cells"], out["cells"]) == (1, 1)


def test_an_equalized_odds_gap_needs_both_parts():
    # Only hire-reference pairs: the TPR gap exists, the FPR gap does not, so no EO gap.
    rows = _rows({"h1": {"a": "hire", "b": "reject"}}, ref={"h1": "hire"})
    out = RM.error_rate_gaps(rows)
    assert out["tpr_gap"] == pytest.approx(100.0) and np.isnan(out["fpr_gap"])
    assert np.isnan(out["eodds_gap"]) and (out["eodds_cells"], out["cells"]) == (0, 1)


def test_error_rate_gaps_restrict_to_the_given_pairs():
    rows = _rows({"h1": {"a": "hire", "b": "hire"}, "h2": {"a": "hire", "b": "reject"}},
                 ref={"h1": "hire", "h2": "hire"})
    assert RM.error_rate_gaps(rows, pairs={"h1"})["tpr_gap"] == pytest.approx(0.0)


def test_claude_clear_pairs_need_agreement_and_a_clear_score(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame({"pair_id": ["a", "b", "c", "d"],
                  "gpt4o_decision": ["hire", "hire", "reject", "reject"],
                  "claude_decision": ["hire", "hire", "hire", "reject"],
                  "claude_fit": [5, 3, 4, 1]}).to_csv(path, index=False)
    assert RM.claude_clear_pairs(path) == {"a", "d"}
    assert RM.claude_clear_pairs(tmp_path / "missing.csv") is None


def _margins(rows, values):
    out = rows[RM.KEY].copy()
    out["margin"] = values
    return out


def test_a_missing_margin_drops_the_set_instead_of_flattening_it():
    rows = _rows({"p": {"A": "hire", "B": "reject"}})
    full = RM.margin_metrics(rows, _margins(rows, [1.0, -1.0]))
    assert full["margin_set_sd_nats"] == pytest.approx(1.0) and full["margin_sets_complete"] == 1
    gap = RM.margin_metrics(rows, _margins(rows, [1.0, -1.0]).iloc[:1])
    assert np.isnan(gap["margin_set_sd_nats"]) and gap["margin_sets_complete"] == 0
    assert gap["margin_coverage_pct"] == pytest.approx(50.0)


def test_repeated_margin_keys_count_as_missing():
    rows = _rows({"p": {"A": "hire", "B": "reject"}})
    margins = _margins(rows, [1.0, -1.0])
    out = RM.margin_metrics(rows, pd.concat([margins, margins.iloc[[1]]]))
    assert out["margin_coverage_pct"] == pytest.approx(50.0)


def test_the_paired_margin_gap_compares_each_attribute_with_the_reference_on_the_same_pairs():
    decisions = {f"p{i}": {"Male": "hire", "Female": "hire", "Agender": "hire"} for i in range(12)}
    rows = _rows(decisions, group="gender")
    shift = {"Male": 0.0, "Female": -0.5, "Agender": 1.0}
    base = np.repeat(np.arange(12, dtype=float), 3)  # each pair its own level
    margins = _margins(rows, base + rows["protected_attr"].map(shift).to_numpy())
    out = RM.margin_metrics(rows, margins, "en")
    assert out["margin_paired_gap_nats"] == pytest.approx(1.5)


def test_margin_metrics_without_a_margin_file_are_empty():
    out = RM.margin_metrics(_rows({"p": {"a": "hire"}}), None)
    assert np.isnan(out["margin_set_sd_nats"]) and np.isnan(out["margin_coverage_pct"])


def _base_pass(path, rows, model="M", adapters=()):
    _margins(rows, 0.0).to_parquet(path, index=False)
    path.with_name(path.name.removesuffix(".parquet") + ".meta.json").write_text(
        json.dumps({"model": {"base": model, "adapters": list(adapters)}}))


def test_a_baseline_binds_only_to_a_unique_base_pass_on_its_exact_prompts(tmp_path):
    rows = _rows({"p": {"A": "hire", "B": "reject"}})
    meta = {"model": "M", "mitigation": {"family": "none"}}
    _base_pass(tmp_path / "x--base.parquet", rows)
    _base_pass(tmp_path / "y--base.parquet", rows.iloc[:1])          # other prompts
    _base_pass(tmp_path / "z--base.parquet", rows, model="Other")     # other model
    frame, source = RM.margins_for("baseline", meta, tmp_path, rows)
    assert source == "x--base.parquet" and len(frame) == 2
    _base_pass(tmp_path / "w--base.parquet", rows)                    # a second match
    frame, source = RM.margins_for("baseline", meta, tmp_path, rows)
    assert frame is None and source.startswith("ambiguous")


def test_an_adapter_run_reads_its_own_pass_and_nothing_else(tmp_path):
    rows = _rows({"p": {"A": "hire"}})
    _base_pass(tmp_path / "x--base.parquet", rows)
    meta = {"model": "M", "mitigation": {"family": "sft"}}
    assert RM.margins_for("arm", meta, tmp_path, rows) == (None, None)
    _margins(rows, 1.0).to_parquet(tmp_path / "arm--adapter.parquet", index=False)
    assert RM.margins_for("arm", meta, tmp_path, rows)[1] == "arm--adapter.parquet"


def _report(path, **fields):
    path.write_text(json.dumps({"standard": "op-margin-v2", **fields}))


def test_primary_comparison_prefers_the_declared_reference(tmp_path):
    _report(tmp_path / "run--vs--sft.json", run="run", against="sft", delta_matched_pp=-0.9,
            ci_low=-1.9, ci_high=0.1, conforms=True)
    assert RM.primary_comparison("run", tmp_path)["primary_against"] == "sft"
    _report(tmp_path / "run--vs--dpo.json", run="run", against="dpo", delta_matched_pp=0.2,
            ci_low=-0.5, ci_high=0.9, conforms=True)
    several = RM.primary_comparison("run", tmp_path)
    assert several["primary_against"] is None
    assert set(several["primary_other_comparisons"].split(", ")) == {"sft", "dpo"}
    chosen = RM.primary_comparison("run", tmp_path, prefer="sft")
    assert chosen["primary_delta_pp"] == -0.9 and chosen["primary_other_comparisons"] == "dpo"


def test_a_base_comparison_reads_the_compare_report(tmp_path):
    _report(tmp_path / "sft.json", run="sft", unstable_adapter_at_own_threshold=0.036,
            unstable_base_at_matched_hire_rate=0.129, conforms=True)
    out = RM.primary_comparison("sft", tmp_path, prefer="base")
    assert out["primary_against"] == "base" and out["primary_delta_pp"] == pytest.approx(-9.3)


def test_merge_kept_reads_the_last_adapter_of_the_served_merge(tmp_path):
    served = tmp_path / "merged"
    served.mkdir()
    (served / "merged_from.json").write_text(json.dumps({"merge_retention": [
        {"adapter": "sft", "kept_fraction": 0.99}, {"adapter": "dpo", "kept_fraction": 0.91}]}))
    meta = {"mitigation": {"family": "dpo"}, "served_model": str(served)}
    assert RM.merge_kept_pct(meta) == pytest.approx(91.0)
    assert RM.merge_kept_pct({"mitigation": {"family": "none"}}) is None
    assert RM.merge_kept_pct({**meta, "served_model": str(tmp_path / "gone")}) is None


def test_the_tables_carry_the_verdict_and_one_row_per_comparison(tmp_path):
    rows_dir, reports = tmp_path / "rows", tmp_path / "reports"
    reports.mkdir()
    RM.write_row({"run": "base", "family": "none", "tpr_vs_gpt4o_pct": 42.0,
                  "agree_gpt4o_pct": 79.0}, rows_dir)
    RM.write_row({"run": "sft", "family": "sft", "baseline_run": "base", "tpr_vs_gpt4o_pct": 65.0,
                  "agree_gpt4o_pct": 82.0}, rows_dir)
    RM.write_row({"run": "arm", "family": "dpo", "baseline_run": "base", "primary_against": "sft",
                  "primary_delta_pp": -2.0, "primary_ci_high_pp": -1.0, "primary_conforms": True,
                  "merge_kept_pct": 90.0, "decided_pct": 100.0, "tpr_vs_gpt4o_pct": 64.0,
                  "agree_gpt4o_pct": 81.0}, rows_dir)
    _report(reports / "arm--vs--sft.json", run="arm", against="sft", delta_matched_pp=-2.0,
            ci_low=-3.0, ci_high=-1.0, conforms=True)
    _report(reports / "sft.json", run="sft", unstable_adapter_at_own_threshold=0.03,
            unstable_base_at_matched_hire_rate=0.12, conforms=True)
    table = pd.read_csv(RM.write_table(rows_dir, tmp_path / "t.csv", reports,
                                       tmp_path / "c.csv")).set_index("run")
    assert table.at["arm", "std_verdict"] == "better"
    assert table.at["arm", "std_tpr_vs_ref_pp"] == pytest.approx(-1.0)
    long = pd.read_csv(tmp_path / "c.csv").set_index("run")
    assert long.at["arm", "reference_run"] == "sft" and long.at["arm", "std_verdict"] == "better"
    assert long.at["sft", "reference_run"] == "base"
    assert long.at["sft", "std_verdict"] == "undecided: no interval"


def test_run_row_reads_the_expected_grid(tmp_path):
    full = _rows({p: {"A": "hire", "B": "hire"} for p in ("p1", "p2")}, free={"p1": "hire",
                                                                           "p2": "hire"})
    raw = full.rename(columns={"d": "decision"}).assign(outcome="decided",
                                                        reference_decision="hire")
    raw = raw.drop(columns="ref")
    raw[raw["pair_id"] == "p1"].to_parquet(tmp_path / "run.parquet", index=False)  # p2 lost
    record = {"run_name": "M--en--sft--adapter--x", "meta": {
        "model": "M", "lang": "en", "mitigation": {"family": "sft"}},
        "summary": {"population_acceptance_rate": 1.0, "reference_agreement": 1.0,
                    "parse_failure_rate": 0.0, "refusal_pct": 0.0,
                    "disparity": {"overall": {"ar_mad": 0.0, "ar_sd": 0.0}}}}
    row = RM.run_row(record, tmp_path / "run.parquet", margin_dir=tmp_path, report_dir=tmp_path,
                     clear_pairs={"p1", "p2"}, expected=_expected(full))
    assert row["expected_prompts"] == 6 and row["decided_pct"] == pytest.approx(50.0)
    assert row["within_group_sets_complete"] == 1 and row["within_group_sets_expected"] == 2
    assert row["flip_coverage_pct"] == pytest.approx(50.0)
    assert row["agreed_clear_pairs"] == 1
    assert row["margin_source"] is None and row["margin_set_sd_nats"] is None
