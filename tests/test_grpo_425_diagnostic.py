"""CPU-only checks for diagnostic identities, coverage and attribution inputs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "grpo425_diagnostic", Path(__file__).parents[1] / "scripts/grpo_425_diagnostic.py"
)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_authorized", True),
        ("rows_per_arm", 52),
        ("slurm_jobs_authorized", 2),
        ("strict_startup_gate_unchanged", False),
    ],
)
def test_protocol_rejects_scope_changes(field, value):
    protocol = D.protocol()
    protocol[field] = value
    with pytest.raises(ValueError, match="protocol differs"):
        D.validate_protocol(protocol)


def test_protocol_preserves_real_greedy_seed_difference():
    protocol = D.protocol()
    assert protocol["historical"]["sampling_seed"] is None
    assert protocol["live"]["sampling_seed"] == 42
    assert protocol["live"]["engine_seed"] == "rank"


@pytest.mark.parametrize(
    "records",
    [
        [{"prompt_id": "a"}],
        [{"prompt_id": "a"}, {"prompt_id": "a"}],
        [{"prompt_id": "a"}, {"prompt_id": "x"}],
    ],
)
def test_unique_coverage_rejects_missing_duplicate_extra(records):
    with pytest.raises(ValueError, match="prompt identities"):
        D.unique_order(records, ["a", "b"])


def test_unique_coverage_restores_order():
    assert (
        D.unique_order([{"prompt_id": "b"}, {"prompt_id": "a"}], ["a", "b"])[0]["prompt_id"] == "a"
    )


def test_controls_are_same_pair_agreeing_and_distinct():
    rows = [
        {"prompt_id": p, "pair_id": pair, "condition": condition, "protected_group": "gender"}
        for p, pair, condition in [
            ("a", "p", "explicit"),
            ("b", "p", "explicit"),
            ("c", "p", "explicit"),
            ("d", "p", "implicit"),
            ("e", "q", "explicit"),
        ]
    ]
    result = D.matched_controls(rows, {"a", "b"})
    assert [r["control_prompt_id"] for r in result] == ["c", "d"]
    assert all(r["pair_id"] == "p" for r in result)


def test_controls_never_cross_pair_to_fill_shortfall():
    rows = [
        {"prompt_id": "a", "pair_id": "p", "condition": "explicit", "protected_group": "gender"}
    ]
    with pytest.raises(ValueError, match="same-pair"):
        D.matched_controls(rows, {"a"})


def test_manifest_binding_rejects_mutation():
    value = {"training_authorized": False}
    value["binding_sha256"] = D.digest(value)
    D.check_binding(value)
    value["training_authorized"] = True
    with pytest.raises(ValueError, match="binding mismatch"):
        D.check_binding(value)


def request(ids=(1, 2), text="wrapped"):
    return SimpleNamespace(
        prompt=text,
        prompt_token_ids=ids,
        outputs=[SimpleNamespace(text='{"decision":"hire"}', token_ids=[3], finish_reason="stop")],
    )


def test_actual_vllm_ids_checked_even_when_text_matches():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    records, diffs = D.check_request_outputs([request((1, 4))], expected)
    assert records[0]["actual_input_ids"] == [1, 4]
    assert diffs == [
        {"prompt_id": "a", "text_equal": True, "token_ids_equal": False, "actual_token_ids": [1, 4]}
    ]


def test_actual_vllm_missing_ids_cannot_pass():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    _, diffs = D.check_request_outputs([request(None)], expected)
    assert diffs[0]["token_ids_equal"] is False


def test_actual_vllm_complete_inputs_pass():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    assert D.check_request_outputs([request()], expected)[1] == []


def test_comparison_keeps_both_directions_and_reference_labels():
    result = D.comparison(
        ["hire", "reject", "hire", "reject"],
        ["reject", "hire", "reject", "hire"],
        ["hire", "hire", "reject", "reject"],
    )
    assert result["decision_differences"] == 4
    assert result["by_reference"] == {
        label: {"hire_to_reject": 1, "reject_to_hire": 1} for label in ["hire", "reject"]
    }


@pytest.mark.parametrize("right", [["hire"], ["hire", None]])
def test_comparison_rejects_partial_or_invalid(right):
    with pytest.raises(ValueError):
        D.comparison(["hire", "reject"], right, ["hire", "reject"])


def test_cached_partial_phase_is_not_success(tmp_path):
    path = tmp_path / "result.json"
    D.atomic(path, {"binding_sha256": "m", "status": "failed"})
    with pytest.raises(ValueError, match="incomplete or foreign"):
        D.bound_result(path, {"binding_sha256": "m"})


def test_completed_phase_fingerprint_is_verified(tmp_path):
    data = tmp_path / "raw.json"
    data.write_text("original")
    path = tmp_path / "result.json"
    D.atomic(path, {"binding_sha256": "m", "status": "complete", "files": [D.fingerprint(data)]})
    data.write_text("changed")
    with pytest.raises(ValueError, match="fingerprint changed"):
        D.bound_result(path, {"binding_sha256": "m"})


def test_base_snapshot_symlink_change_is_detected_even_if_old_blob_is_unchanged(tmp_path):
    first, second = tmp_path / "blob1", tmp_path / "blob2"
    first.write_text("original")
    second.write_text("replacement")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    link = snapshot / "model.safetensors"
    link.symlink_to(first)
    mapping = {link.name: str(first.resolve())}
    D.verify_snapshot_mapping(snapshot, mapping)
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(ValueError, match="snapshot mapping changed"):
        D.verify_snapshot_mapping(snapshot, mapping)
