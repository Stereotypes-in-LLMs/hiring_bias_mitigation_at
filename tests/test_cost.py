"""Cost records (utils/cost.py) and the report built from them (scripts/cost_report.py).

CPU only, no network, no Slurm. NVML is switched off (HBM_COST_NVML=0) so the GPU count
falls back to CUDA_VISIBLE_DEVICES, which each test sets.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hiring_bias_mitigation.utils import cost

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cost_report", ROOT / "scripts" / "cost_report.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)


@pytest.fixture
def records(tmp_path, monkeypatch):
    folder = tmp_path / "records"
    monkeypatch.setenv("HBM_COST_DIR", str(folder))
    monkeypatch.setenv("HBM_COST_NVML", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv(cost.NOTES_ENV, raising=False)
    monkeypatch.delenv("HBM_COST", raising=False)
    for var in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "HBM_COST_ARM"):
        monkeypatch.delenv(var, raising=False)
    return folder


def _child_env() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _wrap(*command: str, stage="eval", label="x") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hiring_bias_mitigation.utils.cost", "run", "--stage", stage,
         "--label", label, "--", *command],
        env=_child_env(), capture_output=True, text=True, timeout=120)


def _only_record(folder: Path) -> dict:
    files = list(folder.glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


# --------------------------------------------------------------------------- recording


def test_a_wrapped_command_leaves_one_record_with_what_its_processes_noted(records):
    child = (
        "from hiring_bias_mitigation.utils import cost\n"
        "cost.add(requests=3, output_tokens=40)\n"
        "cost.add(requests=2)\n"
        "cost.note(run_name='r', params_trainable=7)\n"
        "with cost.phase('generate'):\n"
        "    pass\n"
    )
    result = _wrap(sys.executable, "-c", child, "--config", "configs/a.yaml", label="audit")
    assert result.returncode == 0, result.stderr
    record = _only_record(records)
    assert record["status"] == "ok" and record["exit_code"] == 0
    assert (record["stage"], record["label"], record["config"]) == ("eval", "audit",
                                                                    "configs/a.yaml")
    assert record["notes"]["counters"] == {"requests": 5, "output_tokens": 40}
    assert record["notes"]["facts"] == {"run_name": "r", "params_trainable": 7}
    assert "generate" in record["notes"]["phases"]
    # Two visible devices, counted as such without NVML.
    assert record["gpu"] == {"count": 2, "basis": "visible"}
    assert record["software"]["python"] and "torch" in record["software"]
    assert record["gpu_h"] == pytest.approx(record["wall_s"] * 2 / 3600, abs=1e-4)
    # The notes files are folded in and removed.
    assert not list((records / ".notes").glob("*.json"))


def test_notes_of_several_processes_add_up_instead_of_overwriting(records):
    one = "from hiring_bias_mitigation.utils import cost; cost.add(rows=1)"
    script = f"import subprocess, sys; [subprocess.run([sys.executable, '-c', {one!r}]) " \
             "for _ in range(3)]"
    assert _wrap(sys.executable, "-c", script).returncode == 0
    assert _only_record(records)["notes"]["counters"] == {"rows": 3}


def test_the_command_exit_status_passes_through_and_marks_the_step_failed(records):
    result = _wrap(sys.executable, "-c", "raise SystemExit(3)")
    assert result.returncode == 3
    record = _only_record(records)
    assert (record["status"], record["exit_code"]) == ("failed", 3)


def test_sigterm_is_forwarded_and_the_step_is_recorded_as_terminated(records):
    proc = subprocess.Popen(
        [sys.executable, "-m", "hiring_bias_mitigation.utils.cost", "run", "--stage", "train",
         "--label", "long", "--", sys.executable, "-c", "import time; time.sleep(60)"],
        env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while not list(records.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _only_record(records)["status"] == "running"  # on disk while the step runs
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    record = _only_record(records)
    assert (record["status"], record["exit_code"]) == ("terminated", 143)


def test_a_command_that_cannot_start_reports_127(records):
    assert _wrap("/nonexistent/binary").returncode == 127


def test_recording_switched_off_runs_the_command_and_writes_nothing(records, monkeypatch):
    monkeypatch.setenv("HBM_COST", "0")
    assert _wrap(sys.executable, "-c", "print('hi')").stdout.strip() == "hi"
    assert not records.exists() or not list(records.glob("*.json"))


def test_track_records_a_block_and_a_nested_track_becomes_a_phase(records):
    with cost.track("patching", "layer sweep", arm="d"):
        cost.note(params_modified=4096)
        with cost.track("patching", "inner"):
            cost.add(pairs=10)
    record = _only_record(records)
    assert (record["stage"], record["arm"], record["status"]) == ("patching", "d", "ok")
    assert record["notes"]["facts"] == {"params_modified": 4096}
    assert record["notes"]["counters"] == {"pairs": 10}
    assert "inner" in record["notes"]["phases"]
    assert not cost.tracked()


def test_track_marks_an_exception_as_failed_and_reraises(records):
    with pytest.raises(ValueError), cost.track("eval", "boom"):
        raise ValueError("x")
    assert _only_record(records)["status"] == "failed"


def test_untracked_code_writes_no_notes(records, tmp_path):
    cost.add(requests=1)
    cost.note(x=1)
    assert not records.exists()


def test_config_of_reads_both_spellings():
    assert cost.config_of(["python", "x.py", "--config", "a.yaml", "--config", "b"]) == "a.yaml"
    assert cost.config_of(["torchrun", "-m", "m", "--config=c.yaml"]) == "c.yaml"
    assert cost.config_of(["python", "x.py"]) is None


def test_the_training_callback_notes_parameters_and_the_steps_of_this_segment(records):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from hiring_bias_mitigation.mitigation import training_common as C

    assert C.cost_callbacks() == []  # nothing to record into
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model[0].weight.requires_grad_(False)  # 12 frozen of 23
    args = SimpleNamespace(world_size=4, per_device_train_batch_size=1,
                           gradient_accumulation_steps=4, output_dir="outputs/dpo/x",
                           bf16=True, fp16=False)
    state = SimpleNamespace(is_world_process_zero=True, global_step=175, log_history=[],
                            total_flos=1.5e15)

    class Pairs:  # what a DPO dataloader's dataset looks like after TRL tokenised it
        column_names = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids", "x")

        def __len__(self):
            return 2

        def __getitem__(self, column):
            return [[1, 2, 3], [4]]  # 4 tokens per column

    loader = SimpleNamespace(dataset=Pairs())
    with cost.track("train", "dpo"):
        (callback,) = C.cost_callbacks()
        callback.on_train_begin(args, state, None, model=model, train_dataloader=loader)
        for _ in range(125):  # resumed at 175, ends at 300
            state.global_step += 1
            callback.on_step_end(args, state, None)
            if state.global_step % 25 == 0:
                callback.on_save(args, state, None)
        state.log_history.append({"train_runtime": 1063.2, "train_samples_per_second": 7.5})
        callback.on_train_end(args, state, None)
    facts = _only_record(records)["notes"]["facts"]
    assert (facts["params_total"], facts["params_trainable"]) == (23, 11)
    assert (facts["global_step"], facts["train_steps"]) == (300, 125)
    assert facts["train_runtime_s"] == 1063.2 and facts["world_size"] == 4
    assert (facts["effective_batch"], facts["precision"]) == (16, "bf16")
    assert (facts["train_rows"], facts["train_tokens_per_epoch"]) == (2, 12)
    assert sorted(facts["checkpoint_times"], key=int) == ["200", "225", "250", "275", "300"]
    assert "torch_peak_gib" in facts  # None on a machine without CUDA


# --------------------------------------------------------------------------- report


def _job(**overrides) -> dict:
    row = {"JobID": "100", "JobName": "xw-dpo", "Partition": "normal", "State": "COMPLETED",
           "Start": "2026-10-14T14:00:00", "End": "2026-10-14T15:00:00", "ElapsedRaw": "3600",
           "NNodes": "1", "AllocTRES": "billing=288,cpu=288,energy=7200000,gres/gpu=4,node=1",
           "ConsumedEnergyRaw": "7200000", "WorkDir": "/repo"}
    row.update(overrides)
    return CR.parse_sacct_row(row)


def test_sacct_rows_give_the_allocation_and_drop_a_broken_energy_counter():
    job = _job()
    assert (job["gpus"], job["gpu_h_alloc"], job["node_h"]) == (4, 4.0, 1.0)
    assert job["node_energy_kwh"] == 2.0
    assert _job(ConsumedEnergyRaw="18446744073709551614")["node_energy_kwh"] is None
    assert _job(AllocTRES="cpu=16,mem=48G,node=1")["gpus"] == 0  # the xfer partition
    assert _job(State="CANCELLED by 30448")["state"] == "CANCELLED"


def test_a_log_is_cut_at_its_step_markers_and_housekeeping_is_split_off(tmp_path):
    (tmp_path / "xw-dpo-100.out").write_text(
        "host nid1 | job 100 | config configs/mitigation/dpo/r1.yaml | commit abc1234\n"
        "=== 2026-10-14 14:00:30 W&B: upload offline runs not synced yet\n"
        "noise\n"
        "=== 2026-10-14 14:01:00 DPO on 4 GPUs (DDP); resumes from the newest checkpoint\n")
    steps = CR.log_steps(_job(), tmp_path, datetime(2026, 10, 15))
    assert [(s["label"][:5], s["stage"], s["wall_s"]) for s in steps] == [
        ("setup", "other", 30.0), ("W&B: ", "other", 30.0), ("DPO o", "train", 3540.0)]
    assert all(s["gpu_basis"] == "allocated" and s["n_gpus"] == 4 for s in steps)
    assert steps[2]["gpu_h"] == pytest.approx(3540 * 4 / 3600, abs=1e-4)
    assert steps[2]["config"] == "configs/mitigation/dpo/r1.yaml"
    assert steps[2]["commit"] == "abc1234"


def test_a_job_without_a_log_is_charged_whole_to_its_name_stage(tmp_path):
    (step,) = CR.log_steps(_job(JobName="xw-audit"), tmp_path, datetime(2026, 10, 15))
    assert (step["source"], step["stage"], step["gpu_h"]) == ("sacct", "eval", 4.0)
    (step,) = CR.log_steps(_job(JobName="xw-v9-new"), tmp_path, datetime(2026, 10, 15))
    assert step["stage"] == CR.DEFAULT_STAGE


def _record(**overrides) -> dict:
    record = {"schema": 1, "stage": "eval", "label": "audit", "arm": None, "config": "c.yaml",
              "start": "2026-10-14T14:00:00+02:00", "end": "2026-10-14T14:10:00+02:00",
              "wall_s": 600.0, "status": "ok", "gpu_h": 600 / 3600,
              "gpu": {"count": 1, "basis": "nvml-processes", "energy_j": 3.6e5},
              "slurm": {"job_id": "100", "job_name": "xw-dpo"}, "code": {}, "notes": {},
              "_file": "r.json"}
    record.update(overrides)
    return record


def test_a_record_still_running_in_an_ended_job_is_read_as_killed():
    jobs = {"100": _job()}
    step = CR.record_step(_record(status="running"), jobs, datetime(2026, 10, 15))
    assert step["status"] == "killed"
    running = {"100": _job(State="RUNNING")}
    assert CR.record_step(_record(status="running"), running, datetime.now())["status"] == \
        "running"
    assert step["gpu_energy_kwh"] == 0.1 and step["gpu_basis"] == "used"


def test_jobs_split_into_used_and_idle_and_stages_add_up_to_the_allocation(tmp_path):
    jobs = [_job()]
    steps = [CR.record_step(_record(), {"100": jobs[0]}, datetime(2026, 10, 15))]
    rows = CR.job_table(jobs, steps, {"100"})
    assert rows[0]["gpu_h_steps"] == pytest.approx(600 / 3600, abs=1e-4)
    assert rows[0]["gpu_h_idle"] == pytest.approx(4 - 600 / 3600, abs=1e-3)
    table = {r["stage"].split(" (")[0]: r for r in CR.stage_table(steps, rows)}
    assert table["eval"]["gpu_h"] + table["idle"]["gpu_h"] == \
        pytest.approx(table["total allocated"]["gpu_h"], abs=1e-3)
    assert table["eval"]["gpu_h_measured"] == pytest.approx(600 / 3600, abs=1e-3)
    assert table["eval"]["gpu_h_reconstructed"] is None


def test_what_was_not_measured_is_blank_not_zero(tmp_path):
    jobs = [_job()]
    steps = CR.log_steps(jobs[0], tmp_path, datetime(2026, 10, 15))
    rows = CR.job_table(jobs, steps, recorded=set())
    assert rows[0]["gpu_h_idle"] is None  # the reconstructed step holds the idle time
    table = {r["stage"].split(" (")[0]: r for r in CR.stage_table(steps, rows)}
    assert table["idle"]["gpu_h"] is None
    assert table["train"]["gpu_energy_kwh"] is None and table["train"]["gpu_h_measured"] is None
    assert table["train"]["gpu_h_reconstructed"] == 4.0
    for s in steps:
        s["recipe"] = False
    sft = CR.arm_table([{**s, "arm": "sft", "stage": "eval"} for s in steps])[0]
    assert sft["arm"] == "sft" and sft["recipe_gpu_h"] is None  # not covered, not 0


def test_recipe_keeps_cut_segments_and_drops_failures_checks_and_live_steps():
    def step(status, stage="train", label="dpo", **extra):
        return {**CR._blank_step(), "status": status, "stage": stage, "label": label,
                "arm": "a", "gpu_h": 1.0, "wall_s": 3600, **extra}

    steps = [step("ok"), step("timeout"), step("terminated"), step("failed"),
             step("cancelled"), step("running"), step("killed"),
             step("ok", stage="datagen", label="direction check, GPU-free part"),
             step("ok", stage="datagen", label="d4-check"),
             step("ok", stage="eval", label="audit"),
             step("cancelled", stage="datagen", label="generate", _recipe=True)]
    CR.mark_recipe(steps)
    assert [s["recipe"] for s in steps] == [True, True, True, False, False, False, False,
                                            False, False, False, True]
    (arm,) = [r for r in CR.arm_table(steps) if r["arm"] == "a"]
    assert (arm["recipe_gpu_h"], arm["excluded_gpu_h"]) == (4.0, 6.0)
    by_stage = {r["stage"]: r for r in CR.stage_table(steps, [])}
    assert by_stage["train"]["gpu_h_not_ok"] == 5.0  # the running step is not counted yet


def test_a_long_step_marker_keeps_every_config_for_arm_matching(tmp_path):
    configs = " ".join(f"configs/audit/qwen3.5-9b_en_dpo_v2neg_r2em_step{n}_gh200.yaml:adapter"
                       for n in (50, 100, 148))
    marker = f"margin passes: {configs} configs/audit/qwen3.5-9b_en_dpo_v2neg_r2_step102.yaml"
    assert len(marker) > 200
    (tmp_path / "xw-opc-100.out").write_text(f"=== 2026-10-14 14:00:00 {marker}\n")
    steps = [s for s in CR.log_steps(_job(JobName="xw-opc"), tmp_path, datetime(2026, 10, 15))
             if s["label"] != "setup"]
    rules = CR.load_labels(ROOT / "reports" / "cost" / "labels.csv")
    CR.apply_labels(steps, rules)
    assert len(steps[0]["label"]) == 200
    assert steps[0]["arm"] == "dpo_r2em+dpo_r2"


def test_a_checkpoint_costs_its_segment_up_to_the_save_plus_earlier_segments():
    def segment(job, start, gpu_h, times):
        return {**CR._blank_step(), "source": "record", "stage": "train", "job_id": job,
                "start": start, "n_gpus": 4, "gpu_h": gpu_h, "arm": "dpo_v3",
                "_output_dir": "outputs/dpo/v3", "checkpoint_times": json.dumps(times)}

    steps = [segment("2", "2026-10-21 12:00:00", 2.0, {"300": "2026-10-21 12:15:00"}),
             segment("1", "2026-10-21 10:00:00", 1.5, {"100": "2026-10-21 10:15:00"})]
    rows = {r["checkpoint_step"]: r for r in CR.checkpoint_table(steps)}
    assert rows[100]["gpu_h_to_checkpoint"] == 1.0  # 15 min x 4 GPUs
    assert rows[300]["gpu_h_earlier_segments"] == 1.5
    assert rows[300]["gpu_h_to_checkpoint"] == 2.5


def test_every_sbatch_that_calls_cost_runs_without_it():
    # 13 of these source the main checkout's env.sh by absolute path: if that env.sh has no
    # cost(), the step must still run, unrecorded, instead of failing the job.
    import re

    calls = re.compile(r"(^|\s|=\S+ )cost (datagen|train|eval|patching|other) ", re.M)
    for path in sorted((ROOT / "scripts" / "cscs").glob("*.sbatch")):
        text = path.read_text()
        if calls.search(text):
            fallback = text.find("type cost > /dev/null 2>&1 || cost()")
            assert fallback != -1, path.name
            assert fallback < calls.search(text).start(), path.name


def test_labels_fill_arms_first_match_wins_and_several_configs_share_a_step():
    rules = [(CR.re.compile(p), arm, stage, "") for p, arm, stage in [
        ("r2em", "dpo_r2em", ""), ("r2", "dpo_r2", ""), ("sft_en", "sft", ""),
        ("^xw-margin ", "dpo_r2", "datagen")]]
    blank = CR._blank_step
    steps = [
        {**blank(), "job_name": "xw-audit", "config": "configs/audit/r2em_step50.yaml",
         "label": "audit", "arm": "", "stage": "eval"},
        {**blank(), "job_name": "xw-opc", "config": "",
         "label": "margin passes: configs/audit/sft_en.yaml:adapter configs/audit/r2_x.yaml",
         "arm": "", "stage": "eval"},
        {**blank(), "job_name": "xw-margin", "config": "", "label": "score", "arm": "",
         "stage": "eval"},
        {**blank(), "job_name": "xw-dpo", "config": "configs/r2em.yaml", "label": "dpo",
         "arm": "given", "stage": "train"},
    ]
    CR.apply_labels(steps, rules)
    CR.mark_recipe(steps)
    assert [s["arm"] for s in steps] == ["dpo_r2em", "sft+dpo_r2", "dpo_r2", "given"]
    assert steps[2]["stage"] == "datagen"
    arms = {r["arm"]: r for r in CR.arm_table([{**s, "gpu_h": 1.0, "wall_s": 3600}
                                               for s in steps])}
    # The shared margin pass counts in full for both of its arms.
    assert arms["dpo_r2"]["total_gpu_h"] == 2.0 and arms["dpo_r2"]["shared_gpu_h"] == 1.0
    assert arms["sft"]["eval_gpu_h"] == 1.0


def test_a_rule_widens_a_recorded_arm_into_a_shared_one_but_never_reassigns_it():
    rules = [(CR.re.compile(p), arm, "", "") for p, arm in [
        (r"\| 5115569 ", "dpo_v3_5k+grpo_v3_all"), (r"\| 5118984 ", "grpo_v3_all")]]
    blank = CR._blank_step
    steps = [{**blank(), "job_name": "xw-v3-additional", "job_id": "5115569",
              "label": "teacher", "arm": "dpo_v3_5k", "stage": "datagen"},
             {**blank(), "job_name": "xw-dpo", "job_id": "5118984", "label": "dpo",
              "arm": "dpo_v3_5k", "stage": "train"}]
    CR.apply_labels(steps, rules)
    assert [s["arm"] for s in steps] == ["dpo_v3_5k+grpo_v3_all", "dpo_v3_5k"]


def test_parameters_are_counted_from_the_safetensors_header(tmp_path):
    def write(path, tensors):
        header = json.dumps({"__metadata__": {}, **{
            k: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
            for k, shape in tensors.items()}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)

    base = tmp_path / "hub" / "models--org--base" / "snapshots" / "abc"
    base.mkdir(parents=True)
    write(base / "model-00001.safetensors",
          {"model.layers.0.w": [10, 10], "model.visual.patch": [5, 4], "mtp.fc.weight": [3, 5]})
    run = tmp_path / "outputs" / "dpo" / "arm"
    run.mkdir(parents=True)
    write(run / "adapter_model.safetensors", {"lora_A": [2, 10], "lora_B": [10, 2]})
    (run / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "org/base", "r": 2, "target_modules": ["q_proj"]}))
    (row,) = CR.params_table(tmp_path, tmp_path)
    # The vision tower and the unused multi-token-prediction head are not text parameters.
    assert (row["params_trainable"], row["params_base"], row["params_base_text"]) == \
        (40, 135, 100)
    assert (row["params_base_vision"], row["params_base_mtp"]) == (20, 15)
    assert row["trainable_pct_of_text"] == 40.0


def test_vllm_calls_add_their_requests_and_tokens_to_the_step(records):
    from hiring_bias_mitigation.eval import backends as BK

    def output(n_prompt, *completions):
        return SimpleNamespace(prompt_token_ids=list(range(n_prompt)), outputs=[
            SimpleNamespace(token_ids=list(range(n))) for n in completions])

    BK._count_vllm([output(5, 3)])  # untracked: nothing to add to
    with cost.track("eval", "audit"):
        BK._count_vllm([output(5, 3), output(7, 2, 4)])
    assert _only_record(records)["notes"]["counters"] == {
        "requests": 2, "completions": 3, "prompt_tokens": 12, "output_tokens": 9}
